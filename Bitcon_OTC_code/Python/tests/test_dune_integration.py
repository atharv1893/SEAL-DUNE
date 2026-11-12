import sys
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse as ssp
import torch

_test_runner_args = sys.argv
sys.argv = [sys.argv[0]]
from seal_bitcoin import (
    SEALWeightRegressor,
    build_directed_weight_map,
    build_structural_graph,
    cmd_args,
    make_graph,
)
from DUNE_embedding import DUNE
from util_functions import node_label
sys.argv = _test_runner_args


class DuneSealIntegrationTests(unittest.TestCase):
    def test_node_label_handles_unreachable_nodes_without_cast_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            labels = node_label(ssp.csr_matrix((4, 4)))

        self.assertEqual(labels.tolist(), [1, 1, 0, 0])

    def test_dune_receives_training_only_directed_edges_and_trains(self):
        context_edges = pd.DataFrame(
            [
                {"src": 0, "dst": 2, "weight": 0.3},
                {"src": 2, "dst": 0, "weight": 0.1},
                {"src": 2, "dst": 1, "weight": -0.7},
                {"src": 1, "dst": 0, "weight": 0.1},
            ]
        )
        query = SimpleNamespace(src=0, dst=1, weight=0.4, rating=4.0)
        graph = make_graph(
            query,
            build_structural_graph(
                pd.concat([
                    context_edges,
                    pd.DataFrame([{"src": 0, "dst": 1, "weight": 0.4}]),
                ], ignore_index=True),
                3,
            ),
            hop=1,
            max_nodes_per_hop=None,
            pair_features=np.zeros(8, dtype=np.float32),
            directed_weight_map={
                **build_directed_weight_map(context_edges),
                (0, 1): 0.4,
            },
            node_information=np.zeros((3, 6), dtype=np.float32),
        )

        edges = graph.dune_edge_pairs.T.tolist()
        weights = dict(zip(map(tuple, edges), graph.dune_edge_weights.tolist()))
        self.assertEqual(set(weights), {(0, 2), (2, 0), (2, 1), (1, 0)})
        np.testing.assert_allclose(
            [weights[(0, 2)], weights[(2, 0)], weights[(2, 1)], weights[(1, 0)]],
            [0.3, 0.1, -0.7, 0.1],
        )
        self.assertNotIn((0, 1), weights)
        np.testing.assert_array_equal(
            graph.directed_distances,
            [[0, 2], [2, 0], [1, 1]],
        )

        empty_graph = make_graph(
            query,
            build_structural_graph(
                pd.DataFrame(columns=["src", "dst", "weight"]), 2
            ),
            hop=1,
            max_nodes_per_hop=None,
            pair_features=np.zeros(8, dtype=np.float32),
            directed_weight_map={},
            node_information=np.zeros((2, 6), dtype=np.float32),
        )
        graphs = [empty_graph, graph]
        max_distance = max(
            int(item.directed_distances[item.directed_distances >= 0].max())
            for item in graphs if np.any(item.directed_distances >= 0)
        )
        cmd_args.feat_dim = max_distance + 2
        model = SEALWeightRegressor(
            feat_dim=cmd_args.feat_dim,
            pair_dim=8,
            hidden=8,
            dropout=False,
            node_attr_dim=6,
        )
        self.assertIsInstance(model.gnn, DUNE)
        self.assertEqual(model.h1.in_features, model.gnn.embedding_dim + 8)
        self.assertEqual(model.gnn.embedding_dim, 8 * model.gnn.hidden_dim)

        predictions, _, loss = model(graphs)
        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertTrue(torch.all(predictions.abs() < 10))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))
        self.assertEqual(graph.label, 4.0)

    def test_directed_hop_extraction_includes_outgoing_neighbors(self):
        query = SimpleNamespace(src=0, dst=1, weight=0.4, rating=4.0)
        edge = pd.DataFrame([{"src": 0, "dst": 2, "weight": -0.5}])
        graph = make_graph(
            query,
            build_structural_graph(edge, 3),
            hop=1,
            max_nodes_per_hop=None,
            directed_weight_map=build_directed_weight_map(edge),
        )

        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.dune_edge_pairs.T.tolist(), [[0, 2]])
        self.assertEqual(graph.dune_edge_weights.tolist(), [-0.5])


if __name__ == "__main__":
    unittest.main()
