from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import resonance_engine as re


def build_corpus(per_topic: int = 20, seed: int = 0) -> List[Tuple[str, str]]:
    rng = np.random.default_rng(seed)
    topics = {
        "astronomy": {
            "nouns": ["galaxy", "telescope", "planet", "nebula", "comet"],
            "verbs": ["observes", "tracks", "studies", "maps", "detects"],
            "adjs": ["distant", "bright", "cold", "massive", "faint"],
        },
        "cooking": {
            "nouns": ["recipe", "stove", "sauce", "pan", "herbs"],
            "verbs": ["simmers", "whisks", "seasons", "sautees", "bakes"],
            "adjs": ["savory", "fresh", "spicy", "rich", "golden"],
        },
        "finance": {
            "nouns": ["market", "portfolio", "bond", "dividend", "price"],
            "verbs": ["rises", "drops", "hedges", "rebalances", "yields"],
            "adjs": ["volatile", "steady", "liquid", "risky", "stable"],
        },
        "sports": {
            "nouns": ["team", "coach", "stadium", "tournament", "defense"],
            "verbs": ["scores", "practices", "competes", "passes", "defends"],
            "adjs": ["fast", "strong", "agile", "focused", "tough"],
        },
        "music": {
            "nouns": ["melody", "guitar", "rhythm", "chorus", "tempo"],
            "verbs": ["plays", "sings", "harmonizes", "improvises", "records"],
            "adjs": ["smooth", "loud", "soft", "bright", "steady"],
        },
        "programming": {
            "nouns": ["function", "compiler", "thread", "database", "API"],
            "verbs": ["executes", "optimizes", "connects", "parses", "deploys"],
            "adjs": ["robust", "efficient", "secure", "modular", "scalable"],
        },
        "health": {
            "nouns": ["workout", "nutrition", "sleep", "heart", "balance"],
            "verbs": ["improves", "supports", "strengthens", "restores", "stabilizes"],
            "adjs": ["healthy", "daily", "consistent", "calm", "active"],
        },
        "travel": {
            "nouns": ["journey", "train", "passport", "city", "coast"],
            "verbs": ["explores", "visits", "boards", "navigates", "returns"],
            "adjs": ["long", "busy", "quiet", "scenic", "historic"],
        },
        "gardening": {
            "nouns": ["garden", "soil", "seed", "harvest", "irrigation"],
            "verbs": ["grows", "waters", "plants", "prunes", "fertilizes"],
            "adjs": ["green", "fertile", "dry", "sunny", "lush"],
        },
        "history": {
            "nouns": ["archive", "treaty", "kingdom", "artifact", "timeline"],
            "verbs": ["records", "preserves", "explains", "documents", "links"],
            "adjs": ["ancient", "modern", "regional", "cultural", "notable"],
        },
    }

    templates = [
        "The {adj} {noun} {verb} over the region.",
        "A {adj} {noun} {verb} every morning.",
        "This {noun} {verb} with a {adj} result.",
        "Experts say the {adj} {noun} {verb} this season.",
        "We noticed the {noun} was {adj} as it {verb}.",
    ]

    sentences: List[Tuple[str, str]] = []
    for topic, words in topics.items():
        for i in range(per_topic):
            template = templates[i % len(templates)]
            sentence = template.format(
                adj=rng.choice(words["adjs"]),
                noun=rng.choice(words["nouns"]),
                verb=rng.choice(words["verbs"]),
            )
            sentences.append((sentence, topic))

    rng.shuffle(sentences)
    return sentences


def embed_sentences(sentences: Iterable[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(list(sentences), convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def random_projection_bipolar(
    embeddings: np.ndarray,
    dim: int,
    seed: int = 123,
    projection: np.ndarray | None = None,
) -> np.ndarray:
    if projection is None:
        projection = re.build_projection(dim, embeddings.shape[1], seed=seed)
    rng = np.random.default_rng(seed)
    vectors = [re.embed_to_hd(emb, projection, rng=rng) for emb in embeddings]
    return np.stack(vectors).astype(np.float32)


def pick_entry_indices(num_tokens: int, n_nodes: int, seed: int = 99) -> List[int]:
    rng = np.random.RandomState(seed)
    return rng.randint(1, n_nodes, size=num_tokens).tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= re.EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_node_centroids(
    embeddings: np.ndarray,
    entries: List[int],
    topics: List[str],
) -> tuple[dict[int, np.ndarray], dict[int, Counter[str]], dict[int, List[int]]]:
    node_to_indices: dict[int, List[int]] = defaultdict(list)
    for idx, node_idx in enumerate(entries):
        node_to_indices[node_idx].append(idx)

    centroids: dict[int, np.ndarray] = {}
    topic_counts: dict[int, Counter[str]] = {}
    for node_idx, indices in node_to_indices.items():
        centroids[node_idx] = embeddings[indices].mean(axis=0)
        topic_counts[node_idx] = Counter(topics[i] for i in indices)

    return centroids, topic_counts, node_to_indices


def dominant_topic(counts: Counter[str]) -> str:
    if not counts:
        return "n/a"
    return counts.most_common(1)[0][0]


def build_concept_labels(n_nodes: int, topics: List[str]) -> List[str]:
    unique_topics = list(dict.fromkeys(topics))
    labels = ["root truth anchor"]
    if not unique_topics:
        labels.extend(f"concept {i}" for i in range(1, n_nodes))
        return labels
    idx = 0
    while len(labels) < n_nodes:
        topic = unique_topics[idx % len(unique_topics)]
        labels.append(f"{topic} concept")
        idx += 1
    return labels


def analyze_top_edges(
    graph: re.ResonanceGraph,
    centroids: dict[int, np.ndarray],
    topic_counts: dict[int, Counter[str]],
    node_to_indices: dict[int, List[int]],
    sentences: List[str],
    top_n: int = 20,
) -> None:
    top_edges = sorted(graph.edges, key=lambda e: e.tension, reverse=True)[:top_n]
    print("Top edges by tension:")
    for rank, edge in enumerate(top_edges, start=1):
        src_vec = graph.nodes[edge.src].vector
        dst_vec = graph.nodes[edge.dst].vector
        hd_sim = float(np.dot(src_vec, dst_vec) / graph.dim)
        src_centroid = centroids.get(edge.src)
        dst_centroid = centroids.get(edge.dst)
        sem_sim = 0.0
        if src_centroid is not None and dst_centroid is not None:
            sem_sim = cosine_similarity(src_centroid, dst_centroid)

        src_topic = dominant_topic(topic_counts.get(edge.src, Counter()))
        dst_topic = dominant_topic(topic_counts.get(edge.dst, Counter()))
        print(
            f"{rank:02d}. {edge.src}->{edge.dst} "
            f"tension={edge.tension:.4f} "
            f"hd_sim={hd_sim:.3f} sem_sim={sem_sim:.3f} topics={src_topic}/{dst_topic}"
        )

        src_examples = node_to_indices.get(edge.src, [])[:1]
        dst_examples = node_to_indices.get(edge.dst, [])[:1]
        if src_examples:
            print(f"    src example: {sentences[src_examples[0]]}")
        if dst_examples:
            print(f"    dst example: {sentences[dst_examples[0]]}")


def sanity_check_hd_sim(
    graph: re.ResonanceGraph,
    topic_counts: dict[int, Counter[str]],
    sample_pairs: int = 25,
    seed: int = 123,
) -> None:
    node_topics = {
        node_idx: dominant_topic(counts) for node_idx, counts in topic_counts.items()
    }
    topic_nodes: dict[str, List[int]] = defaultdict(list)
    for node_idx, topic in node_topics.items():
        if topic == "n/a":
            continue
        topic_nodes[topic].append(node_idx)

    vec_cache: dict[int, np.ndarray] = {}

    def hd_sim(a: int, b: int) -> float:
        if a not in vec_cache:
            vec_cache[a] = graph.nodes[a].vector
        if b not in vec_cache:
            vec_cache[b] = graph.nodes[b].vector
        return float(np.dot(vec_cache[a], vec_cache[b]) / graph.dim)

    print("Sanity check hd_sim:")
    same_topic_pair = None
    for topic, nodes in topic_nodes.items():
        if len(nodes) >= 2:
            same_topic_pair = (topic, nodes[0], nodes[1])
            break
    if same_topic_pair:
        topic, a, b = same_topic_pair
        print(f"  same_topic {topic} {a}->{b} hd_sim={hd_sim(a, b):.3f}")
    else:
        print("  same_topic n/a (not enough labeled nodes)")

    topics = [t for t, nodes in topic_nodes.items() if nodes]
    if len(topics) >= 2:
        a_topic = topics[0]
        b_topic = topics[1]
        a = topic_nodes[a_topic][0]
        b = topic_nodes[b_topic][0]
        print(f"  cross_topic {a_topic}/{b_topic} {a}->{b} hd_sim={hd_sim(a, b):.3f}")
    else:
        print("  cross_topic n/a (not enough labeled nodes)")

    rng = np.random.default_rng(seed)
    same_samples: List[float] = []
    cross_samples: List[float] = []
    same_topics = [t for t, nodes in topic_nodes.items() if len(nodes) >= 2]
    for _ in range(sample_pairs):
        if same_topics:
            topic = rng.choice(same_topics)
            nodes = topic_nodes[topic]
            a, b = rng.choice(nodes, size=2, replace=False)
            same_samples.append(hd_sim(int(a), int(b)))
        if len(topics) >= 2:
            t_a, t_b = rng.choice(topics, size=2, replace=False)
            a = rng.choice(topic_nodes[t_a])
            b = rng.choice(topic_nodes[t_b])
            cross_samples.append(hd_sim(int(a), int(b)))

    if same_samples and cross_samples:
        print(
            f"  sample_avg same_topic={float(np.mean(same_samples)):.3f} "
            f"cross_topic={float(np.mean(cross_samples)):.3f} "
            f"n={len(same_samples)}"
        )
    else:
        print("  sample_avg n/a")


def main() -> None:
    corpus = build_corpus(per_topic=20, seed=1)
    sentences = [text for text, _topic in corpus]
    topics = [topic for _text, topic in corpus]
    if not (100 <= len(sentences) <= 500):
        raise ValueError("Corpus size must be between 100 and 500 sentences.")

    embeddings = embed_sentences(sentences)
    n_nodes = 500
    hd_dim = 10_000
    projection = re.build_projection(hd_dim, embeddings.shape[1], seed=re.PROJECTION_SEED)
    token_vectors = random_projection_bipolar(
        embeddings,
        dim=hd_dim,
        projection=projection,
    )
    concept_labels = build_concept_labels(n_nodes, topics)

    np.random.seed(42)
    graph = re.ResonanceGraph(
        n_nodes=n_nodes,
        dim=hd_dim,
        sparsity=0.03,
        concept_labels=concept_labels,
        projection=projection,
    )

    entry_seed = 99
    entries = pick_entry_indices(len(token_vectors), graph.n_nodes, seed=entry_seed)
    np.random.seed(entry_seed)
    re.stream_train(graph, list(token_vectors), steps_per_token=50, settle_steps=20)

    centroids, topic_counts, node_to_indices = build_node_centroids(
        embeddings, entries, topics
    )
    analyze_top_edges(
        graph,
        centroids,
        topic_counts,
        node_to_indices,
        sentences,
        top_n=20,
    )
    sanity_check_hd_sim(graph, topic_counts)


if __name__ == "__main__":
    main()
