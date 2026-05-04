"""
Standalone checks for TV4 logic filtering.

Run:
    python -m modules.bayes_logic.test_logic_filter
"""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from modules.bayes_logic.logic_filter import filter_graph


class LogicFilterTest(unittest.TestCase):
    def test_filters_edges_by_if_then_rules(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(
            [
                (1, {"x": 0.0, "y": 0.0}),
                (2, {"x": 1.0, "y": 0.0}),
                (3, {"x": 2.0, "y": 0.0}),
            ]
        )
        graph.add_edge(1, 2, key=0, length=100, access="yes")
        graph.add_edge(2, 3, key=0, length=80, access="private")
        graph.add_edge(3, 1, key=0, length=50, maxweight="2.0 t")
        graph.add_edge(1, 3, key=0, length=40, highway="construction")
        graph.add_edge(2, 1, key=0)
        graph.add_edge(3, 2, key=0, length=0)

        edges = pd.DataFrame(
            [
                {"u": 1, "v": 2, "key": 0, "length": 100, "highway": "residential"},
                {"u": 2, "v": 3, "key": 0, "length": 80, "highway": "residential"},
                {"u": 3, "v": 1, "key": 0, "length": 50, "highway": "residential"},
                {"u": 1, "v": 3, "key": 0, "length": 40, "highway": "construction"},
                {"u": 2, "v": 1, "key": 0, "length": 30, "highway": "residential"},
                {"u": 3, "v": 2, "key": 0, "length": 0, "highway": "residential"},
            ]
        )

        cleaned = filter_graph({"G": graph, "edges": edges}, vehicle_weight=2.5)

        self.assertEqual(sorted(cleaned.edges(keys=True)), [(1, 2, 0), (2, 1, 0)])
        self.assertEqual(graph.number_of_edges(), 6)
        self.assertEqual(cleaned[2][1][0]["length"], 30)

        stats = cleaned.graph["logic_filter_stats"]
        self.assertEqual(stats["original_edges"], 6)
        self.assertEqual(stats["removed_invalid_length"], 1)
        self.assertEqual(stats["removed_restricted_access"], 1)
        self.assertEqual(stats["removed_weight_limit"], 1)
        self.assertEqual(stats["removed_closed_or_construction"], 1)
        self.assertEqual(stats["removed_total"], 4)
        self.assertEqual(stats["remaining_edges"], 2)
        self.assertEqual(stats["metadata_edges_synced"], 6)


if __name__ == "__main__":
    unittest.main()
