"""
Simple integration test for TV3 module.

Usage:
    python -m modules.ml.test_integration
"""

from __future__ import annotations

from modules.bayes_logic.bayes_model import compute_congestion
from modules.ml.predictor import predict_travel_time as predict_one_edge
from modules.ml.sample_data import demo_edges
from modules.ml.travel_time_predictor import predict_travel_time as predict_batch


def run_demo() -> None:
    edges = demo_edges()
    congestion_df = compute_congestion(
        edges=edges,
        weather="rain",
        time_of_day="peak",
    )
    congestion_df["weather"] = "rain"
    congestion_df["time_of_day"] = "peak"

    print("=== TV3 Integration Demo ===")
    print("\nInput edges:")
    print(edges.to_string(index=False))

    print("\nCongestion from TV4:")
    print(congestion_df.to_string(index=False))

    weighted_edges = predict_batch(edges=edges, congestion_df=congestion_df)
    print("\nPredicted weighted edges for TV2:")
    print(weighted_edges.to_string(index=False))

    first_edge = edges.iloc[0].to_dict()
    single_prediction = predict_one_edge(
        edge=first_edge,
        weather="rain",
        is_peak_hour=1,
        congestion_prob=float(congestion_df.iloc[0]["p_congestion"]),
    )
    print(
        "\nSingle-edge API check:"
        f" u={int(first_edge['u'])} -> v={int(first_edge['v'])},"
        f" travel_time_min={single_prediction:.4f}"
    )


if __name__ == "__main__":
    run_demo()
