import unittest
from unittest.mock import patch

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import resonance_engine as re


class ResonanceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_node_vector_bipolar(self) -> None:
        projection = re.build_projection(10_000, 384, seed=7)
        nodes = []
        for i in range(10):
            embedding = np.random.standard_normal(384).astype(np.float32)
            nodes.append(re.Node.from_embedding(i, embedding, projection))
        for node in nodes:
            vec = node.vector
            self.assertEqual(vec.shape, (node.dim,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertTrue(np.all(np.isin(vec, (-1.0, 1.0))))

    def test_adjacency_population(self) -> None:
        graph = re.ResonanceGraph(n_nodes=6, dim=64, sparsity=1.0, top_k=3)
        for edge in graph.edges:
            self.assertIn(edge, graph.adj_in[edge.dst])
            self.assertIn(edge, graph.adj_out[edge.src])
            self.assertEqual(graph.adj_in[edge.dst].count(edge), 1)
            self.assertEqual(graph.adj_out[edge.src].count(edge), 1)

    def test_init_root(self) -> None:
        graph = re.ResonanceGraph(n_nodes=6, dim=64, sparsity=1.0, top_k=3)
        root = graph.nodes[0]
        self.assertTrue(root.is_root)
        self.assertEqual(root.activation, 1.0)
        self.assertTrue(np.all(root.vector == 1.0))
        for edge in graph.root_edges:
            self.assertEqual(edge.src, 0)
            self.assertTrue(edge.read_only)
            self.assertGreaterEqual(edge.tension, re.ROOT_TENSION_FLOOR)

    def test_local_learn_normalizes(self) -> None:
        graph = re.ResonanceGraph(n_nodes=8, dim=64, sparsity=0.8, top_k=3)
        for _ in range(10):
            activations = np.random.rand(graph.n_nodes).astype(np.float32)
            for idx, value in enumerate(activations):
                graph.nodes[idx].activation = float(value)
            graph.nodes[0].activation = 1.0
            graph.local_learn()

        for edge in graph.edges:
            self.assertTrue(np.isfinite(edge.tension))

        for node in graph.nodes[1:]:
            incoming_edges = graph.adj_in.get(node.idx, [])
            total = sum(edge.tension for edge in incoming_edges)
            if total <= re.EPS:
                continue
            self.assertAlmostEqual(total, 1.0, delta=0.05)

    def test_propagate_bounds(self) -> None:
        graph = re.ResonanceGraph(n_nodes=10, dim=64, sparsity=0.6, top_k=4)
        for node in graph.nodes:
            node.activation = 0.0
        graph.nodes[0].activation = 1.0
        graph.nodes[1].activation = 1.0

        for _ in range(100):
            graph.propagate()
            self.assertEqual(graph.nodes[0].activation, 1.0)

        activations = np.array([node.activation for node in graph.nodes], dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(activations)))
        self.assertGreaterEqual(float(np.min(activations)), 0.0)
        self.assertLessEqual(float(np.max(activations)), 1.0)

    def test_query_neighbors(self) -> None:
        graph = re.ResonanceGraph(n_nodes=12, dim=64, sparsity=0.6, top_k=4)
        for idx in range(graph.n_nodes):
            neighbors = graph._query_neighbors(idx)
            self.assertLessEqual(len(neighbors), graph.top_k)
            self.assertNotIn(idx, neighbors)
            for neighbor in neighbors:
                self.assertGreaterEqual(neighbor, 0)
                self.assertLess(neighbor, graph.n_nodes)

    def test_override_interceptor(self) -> None:
        graph = re.ResonanceGraph(n_nodes=6, dim=64, sparsity=0.6, top_k=3)
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        interceptor = re.OverrideInterceptor(pub, graph)
        data = re.sign_command(b"LOCK", priv)

        self.assertTrue(interceptor.check_input(data))
        self.assertTrue(graph.locked)
        for node in graph.nodes[1:]:
            self.assertEqual(node.activation, 0.0)
            self.assertEqual(node.ema, 0.0)

        self.assertFalse(interceptor.check_input(data))

        with patch("resonance_engine.time.time", return_value=1000):
            expired = re.sign_command(b"LOCK", priv)
        with patch("resonance_engine.time.time", return_value=1000 + re.OVERRIDE_WINDOW_S + 1):
            self.assertFalse(interceptor.check_input(expired))

    def test_stream_train_progress(self) -> None:
        graph = re.ResonanceGraph(n_nodes=6, dim=64, sparsity=0.6, top_k=3)
        tokens = [np.random.randn(graph.dim).astype(np.float32) for _ in range(5)]
        steps_per_token = 2
        settle_steps = 1
        expected_ticks = (steps_per_token + settle_steps) * len(tokens)

        re.stream_train(graph, tokens, steps_per_token, settle_steps)

        self.assertEqual(graph.tick, expected_ticks)
        activations = np.array([node.activation for node in graph.nodes], dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(activations)))
        for edge in graph.edges:
            self.assertTrue(np.isfinite(edge.tension))


if __name__ == "__main__":
    unittest.main()
