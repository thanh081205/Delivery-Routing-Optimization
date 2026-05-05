"""
Standalone checks for TV4 real-time Bayes congestion output.

Run:
    python -m modules.bayes_logic.test_bayes_model
"""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from modules.bayes_logic.bayes_model import BayesCongestionModel, compute_congestion_matrix


class BayesRealtimeTest(unittest.TestCase):
    def _graph_data(self) -> dict:
        graph = nx.MultiDiGraph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edge(1, 2, key=0, length=120, maxspeed=40, highway="residential")
        graph.add_edge(1, 2, key=1, length=220, maxspeed=30, highway="secondary")
        graph.add_edge(2, 3, key=0)

        edges = pd.DataFrame(
            [
                {"u": 1, "v": 2, "key": 0, "length": 120, "maxspeed": 40, "highway": "residential"},
                {"u": 1, "v": 2, "key": 1, "length": 220, "maxspeed": 30, "highway": "secondary"},
                {"u": 2, "v": 3, "key": 0, "length": 300, "maxspeed": 50, "highway": "tertiary"},
                {"u": 3, "v": 1, "key": 0, "length": 90, "maxspeed": 40, "highway": "service"},
            ]
        )
        return {"G": graph, "edges": edges}

    def test_congestion_matrix_covers_current_graph(self) -> None:
        matrix = compute_congestion_matrix(
            self._graph_data(),
            weather="rain",
            time_of_day="peak",
            observed_at="2026-05-04T00:00:00+00:00",
        )

        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix.attrs["weather"], "rain")
        self.assertEqual(matrix.attrs["time_of_day"], "peak")
        self.assertEqual(matrix.attrs["observed_at"], "2026-05-04T00:00:00+00:00")

        edge_probabilities = matrix.attrs["edge_probabilities"]
        self.assertEqual(len(edge_probabilities), 3)
        self.assertNotIn((3, 1), set(zip(edge_probabilities["u"], edge_probabilities["v"])))

        p_parallel_max = edge_probabilities[edge_probabilities["u"].eq(1) & edge_probabilities["v"].eq(2)][
            "p_congestion"
        ].max()
        self.assertEqual(matrix.loc[1, 2], p_parallel_max)
        self.assertGreater(matrix.loc[2, 3], 0.0)

    def test_realtime_wrapper_returns_tv3_feature_frame(self) -> None:
        graph_data = self._graph_data()
        clear_matrix = compute_congestion_matrix(graph_data, weather="clear", time_of_day="normal")
        rainy_matrix = compute_congestion_matrix(graph_data, weather="rain", time_of_day="peak")

        self.assertGreater(rainy_matrix.loc[1, 2], clear_matrix.loc[1, 2])

        model = BayesCongestionModel(source=graph_data)
        model.update_realtime(
            weather="rain",
            time_of_day="peak",
            observed_at="2026-05-04T01:00:00+00:00",
        )
        features = model.as_feature_frame()

        self.assertEqual(list(features.columns), ["u", "v", "key", "p_congestion", "weather", "time_of_day", "observed_at"])
        self.assertTrue(features["p_congestion"].between(0.0, 1.0).all())
        self.assertTrue(features["weather"].eq("rain").all())
        self.assertTrue(features["time_of_day"].eq("peak").all())


if __name__ == "__main__":
    unittest.main()
