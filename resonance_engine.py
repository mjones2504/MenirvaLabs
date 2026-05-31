from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math
import struct
import time
from typing import Iterable, List

import faiss
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from sentence_transformers import SentenceTransformer


EPS = 1e-6
MIN_TENSION = 0.001
ROOT_TENSION_FLOOR = 0.05
EMA_DECAY = 0.95
FRICTION_THRESHOLD = 0.05
SRC_ACT_THRESHOLD = 0.05
OVERRIDE_WINDOW_S = 30
SIM_GATE = 0.10
PROJECTION_SEED = 42
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}
_PROJECTION_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _random_tension() -> float:
    return float(np.random.uniform(0.01, 0.30))


def get_embedder(model_name: str) -> SentenceTransformer:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = _EMBEDDER_CACHE.get(model_name)
    if embedder is None:
        embedder = SentenceTransformer(model_name, device=device)
        _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def build_projection(hd_dim: int, embed_dim: int, seed: int = PROJECTION_SEED) -> np.ndarray:
    key = (hd_dim, embed_dim, seed)
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    raw = rng.choice([-1, 0, 0, 1], size=(hd_dim, embed_dim)).astype(np.float32)
    projection = raw * math.sqrt(3.0)
    _PROJECTION_CACHE[key] = projection
    return projection


def embed_to_hd(
    embedding: np.ndarray,
    projection: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    projected = projection @ embedding
    threshold = float(np.percentile(np.abs(projected), 30))
    if rng is None:
        rng = np.random.default_rng()
    random_bits = rng.choice([-1.0, 1.0], size=projected.shape)
    result = np.where(
        projected > threshold,
        1.0,
        np.where(projected < -threshold, -1.0, random_bits),
    )
    return result.astype(np.float32)


def pack_bipolar(vector: np.ndarray) -> np.ndarray:
    bits = ((vector + 1.0) / 2.0).astype(np.uint8)
    return np.packbits(bits, axis=None)


def bind_token_to_node(node: "Node", hd_token: np.ndarray, alpha: float = 0.1) -> None:
    if hd_token.shape[0] != node.dim:
        raise ValueError("Token vector must match node dimensionality.")
    current = node.vector
    blended = (1.0 - alpha) * current + alpha * hd_token
    new_vec = np.where(blended >= 0.0, 1.0, -1.0).astype(np.float32)
    node._packed = pack_bipolar(new_vec)


@dataclass(eq=False)
class Node:
    idx: int
    dim: int = 10_000
    activation: float = 0.0
    ema: float = 0.0
    is_root: bool = False
    cluster_id: int | None = None
    _packed: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._packed is None:
            raise ValueError("Node requires a packed vector.")

    @classmethod
    def from_bipolar(cls, idx: int, vector: np.ndarray) -> "Node":
        packed = pack_bipolar(vector)
        return cls(idx=idx, dim=vector.shape[0], _packed=packed)

    @classmethod
    def from_embedding(
        cls,
        idx: int,
        embedding: np.ndarray,
        projection: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> "Node":
        hd_vec = embed_to_hd(embedding, projection, rng=rng)
        return cls.from_bipolar(idx=idx, vector=hd_vec)

    @property
    def vector(self) -> np.ndarray:
        unpacked = np.unpackbits(self._packed, axis=None)[: self.dim]
        vec = unpacked.astype(np.float32)
        vec = vec * 2.0 - 1.0
        return vec


@dataclass(eq=False)
class Edge:
    src: int
    dst: int
    tension: float = field(default_factory=_random_tension)
    read_only: bool = False


class ResonanceGraph:
    def __init__(
        self,
        n_nodes: int = 100,
        dim: int = 10_000,
        sparsity: float = 0.06,
        damping: float = 0.18,
        lr: float = 0.01,
        top_k: int = 10,
        reindex_interval: int = 100,
        concept_labels: list[str] | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        projection_seed: int = PROJECTION_SEED,
        projection: np.ndarray | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.dim = dim
        self.sparsity = sparsity
        self.damping = damping
        self.lr = lr
        self.top_k = top_k
        self.reindex_interval = reindex_interval
        self.embed_model = embed_model
        self.projection_seed = projection_seed

        self.embedder = get_embedder(self.embed_model)
        if hasattr(self.embedder, "get_embedding_dimension"):
            embed_dim = self.embedder.get_embedding_dimension()
        else:
            embed_dim = self.embedder.get_sentence_embedding_dimension()
        if projection is None:
            self.projection = build_projection(self.dim, embed_dim, self.projection_seed)
        else:
            self.projection = projection
            embed_dim = projection.shape[1]
            if projection.shape[0] != self.dim:
                raise ValueError("Projection rows must match dim.")

        if concept_labels is None:
            labels = ["root truth anchor"] + [f"concept {i}" for i in range(1, self.n_nodes)]
        else:
            labels = list(concept_labels)
            if len(labels) < self.n_nodes:
                labels.extend(f"concept {i}" for i in range(len(labels), self.n_nodes))
            elif len(labels) > self.n_nodes:
                labels = labels[: self.n_nodes]

        embeddings = self.embedder.encode(
            labels,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        embed_rng = np.random.default_rng(self.projection_seed)
        self.nodes = [
            Node.from_embedding(
                idx=i,
                embedding=embeddings[i],
                projection=self.projection,
                rng=embed_rng,
            )
            for i in range(self.n_nodes)
        ]
        self.edges: List[Edge] = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if np.random.random() < self.sparsity:
                    self.edges.append(Edge(src=i, dst=j))
                    self.edges.append(Edge(src=j, dst=i))

        self.root_edges: List[Edge] = []
        self._init_root()
        self._resolve_collisions()
        self._build_adjacency()
        self._build_sim_index()
        self.tick = 0
        self.locked = False

    def _resolve_collisions(self) -> None:
        rng = np.random.default_rng(self.projection_seed + 1)
        seen_vectors: set[bytes] = {self.nodes[0]._packed.tobytes()}
        for node in self.nodes[1:]:
            vec_key = node._packed.tobytes()
            attempts = 0
            while vec_key in seen_vectors and attempts < 10:
                hd_vec = node.vector
                flip_mask = rng.random(self.dim) < 0.05
                hd_vec[flip_mask] *= -1.0
                node._packed = pack_bipolar(hd_vec)
                vec_key = node._packed.tobytes()
                attempts += 1
            seen_vectors.add(vec_key)

    def _build_adjacency(self) -> None:
        self.adj_in: dict[int, List[Edge]] = defaultdict(list)
        self.adj_out: dict[int, List[Edge]] = defaultdict(list)
        for edge in self.edges:
            self.adj_out[edge.src].append(edge)
            self.adj_in[edge.dst].append(edge)

    def _build_sim_index(self) -> None:
        matrix = np.vstack([node.vector for node in self.nodes]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        matrix = np.ascontiguousarray(matrix / norms, dtype=np.float32)
        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)
        self.sim_index = index

    def _query_neighbors(self, node_idx: int) -> list[int]:
        vec = self.nodes[node_idx].vector.astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec))
        if norm <= EPS:
            return []
        vec = vec / norm
        scores, indices = self.sim_index.search(vec.reshape(1, -1), self.top_k + 1)
        neighbors: list[int] = []
        for idx in indices[0]:
            if idx == node_idx or idx < 0:
                continue
            neighbors.append(int(idx))
            if len(neighbors) >= self.top_k:
                break
        return neighbors

    def _maybe_reindex(self) -> None:
        if self.tick % self.reindex_interval == 0:
            self._build_sim_index()

    def propagate(self, dt: float = 0.001) -> None:
        if self.locked:
            return

        self.nodes[0].activation = 1.0
        for edge in self.root_edges:
            if edge.tension < ROOT_TENSION_FLOOR:
                edge.tension = ROOT_TENSION_FLOOR

        self._maybe_reindex()

        incoming_signal = np.zeros(self.n_nodes, dtype=np.float32)
        for i in range(1, self.n_nodes):
            neighbors = self._query_neighbors(i)
            if not neighbors:
                continue
            in_edges = self.adj_in.get(i, [])
            if not in_edges:
                continue
            edge_map = {edge.src: edge for edge in in_edges}
            vec_i = self.nodes[i].vector
            norm_i = float(np.linalg.norm(vec_i))
            if norm_i <= EPS:
                continue
            vec_i_norm = vec_i / norm_i
            total = 0.0
            for j in neighbors:
                edge = edge_map.get(j)
                if edge is None:
                    continue
                vec_j = self.nodes[j].vector
                norm_j = float(np.linalg.norm(vec_j))
                if norm_j <= EPS:
                    continue
                vec_j_norm = vec_j / norm_j
                sim = float(np.dot(vec_i_norm, vec_j_norm))
                if sim <= 0.0:
                    continue
                total += edge.tension * self.nodes[j].activation * sim
            incoming_signal[i] = total

        def dA(activations: np.ndarray) -> np.ndarray:
            deriv = np.zeros_like(activations, dtype=np.float32)
            if activations.size > 1:
                deriv[1:] = -self.damping * activations[1:] + incoming_signal[1:]
            return deriv

        A = np.array([node.activation for node in self.nodes], dtype=np.float32)
        k1 = dA(A)
        A_pred = np.clip(A + dt * k1, 0.0, 1.0)
        k2 = dA(A_pred)
        A_next = np.clip(A + 0.5 * dt * (k1 + k2), 0.0, 1.0)
        A_next[0] = 1.0

        for idx, value in enumerate(A_next):
            self.nodes[idx].activation = float(value)

        self.tick += 1

    def _init_root(self) -> None:
        root = self.nodes[0]
        root.is_root = True
        root.activation = 1.0
        ones = np.ones(self.dim, dtype=np.uint8)
        root._packed = np.packbits(ones, axis=None)
        self.root_edges = [edge for edge in self.edges if edge.src == 0]
        for edge in self.root_edges:
            edge.read_only = True
            if edge.tension < ROOT_TENSION_FLOOR:
                edge.tension = ROOT_TENSION_FLOOR

    def local_learn(self) -> None:
        if self.locked:
            return

        deltas: dict[Edge, float] = defaultdict(float)
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= EPS:
                return 0.0
            return float(np.dot(a, b) / denom)

        for node in self.nodes:
            if node.is_root:
                continue
            node.ema = EMA_DECAY * node.ema + (1.0 - EMA_DECAY) * node.activation
            friction = abs(node.activation - node.ema)
            if friction < FRICTION_THRESHOLD:
                continue
            node_vec = node.vector
            for edge in self.adj_in.get(node.idx, []):
                if edge.read_only:
                    continue
                src_act = self.nodes[edge.src].activation
                if src_act < SRC_ACT_THRESHOLD:
                    continue
                sim = cosine_sim(self.nodes[edge.src].vector, node_vec)
                if sim < SIM_GATE:
                    continue
                delta = self.lr * (
                    node.activation * src_act - (node.activation ** 2) * edge.tension
                )
                deltas[edge] += delta

        for edge, delta in deltas.items():
            edge.tension = float(np.clip(edge.tension + delta, MIN_TENSION, 1.0))

        for node in self.nodes:
            if node.is_root:
                continue
            incoming_edges = self.adj_in.get(node.idx, [])
            total = sum(edge.tension for edge in incoming_edges)
            if total <= EPS:
                continue
            for edge in incoming_edges:
                edge.tension = max(MIN_TENSION, edge.tension / total)


class OverrideInterceptor:
    def __init__(self, public_key_bytes: bytes, graph: ResonanceGraph) -> None:
        self.pubkey = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        self.graph = graph
        self._seen_nonces: set[str] = set()

    def check_input(self, data: bytes) -> bool:
        if len(data) < 72:
            return False

        payload = data[:-64]
        signature = data[-64:]
        try:
            self.pubkey.verify(signature, payload)
        except Exception:
            return False

        if len(payload) < 8:
            return False
        ts = struct.unpack(">Q", payload[-8:])[0]
        if abs(time.time() - ts) > OVERRIDE_WINDOW_S:
            return False

        sig_hash = hashlib.sha256(signature).hexdigest()
        if sig_hash in self._seen_nonces:
            return False
        self._seen_nonces.add(sig_hash)

        self._execute_override()
        return True

    def _execute_override(self) -> None:
        self.graph.damping = 1.0
        for node in self.graph.nodes:
            if node.is_root:
                continue
            node.activation = 0.0
            node.ema = 0.0
        self.graph.locked = True
        print("[OVERRIDE] Absolute damping field engaged. Engine locked.")


def sign_command(command: bytes, privkey: ed25519.Ed25519PrivateKey) -> bytes:
    payload = command + struct.pack(">Q", int(time.time()))
    signature = privkey.sign(payload)
    return payload + signature


def stream_train(
    graph: ResonanceGraph,
    token_stream: Iterable[np.ndarray],
    steps_per_token: int = 50,
    settle_steps: int = 20,
    bind_alpha: float = 0.1,
) -> None:
    for token_vec in token_stream:
        if graph.n_nodes <= 1:
            return
        entry_idx = int(np.random.randint(1, graph.n_nodes))
        graph.nodes[entry_idx].activation = 1.0
        bind_token_to_node(graph.nodes[entry_idx], token_vec, alpha=bind_alpha)
        for _ in range(steps_per_token):
            graph.propagate()
            graph.local_learn()
        for _ in range(settle_steps):
            graph.propagate()


def _print_stats(graph: ResonanceGraph) -> None:
    active = sum(1 for node in graph.nodes if node.activation > 0.0)
    if graph.edges:
        mean_tension = float(np.mean([edge.tension for edge in graph.edges]))
    else:
        mean_tension = 0.0
    print(f"[tick {graph.tick}] active={active} mean_tension={mean_tension:.4f}")


if __name__ == "__main__":
    graph = ResonanceGraph()
    tokens = [np.random.randn(graph.dim).astype(np.float32) for _ in range(20)]
    steps_per_token = 50
    settle_steps = 20

    for _token in tokens:
        if graph.n_nodes <= 1:
            break
        entry_idx = int(np.random.randint(1, graph.n_nodes))
        graph.nodes[entry_idx].activation = 1.0
        bind_token_to_node(graph.nodes[entry_idx], _token, alpha=0.1)
        for _ in range(steps_per_token):
            graph.propagate()
            graph.local_learn()
            if graph.tick % 10 == 0:
                _print_stats(graph)
        for _ in range(settle_steps):
            graph.propagate()
            if graph.tick % 10 == 0:
                _print_stats(graph)
