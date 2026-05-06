"""
Batch prediction interface for TV3 integration contract.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from modules.ml.predictor import predict_travel_time as predict_one_edge
from modules.ml.preprocess import require_columns, to_peak_hour_flag


def predict_travel_time(edges: pd.DataFrame, congestion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict travel time for each edge using congestion probability from Bayes module.

    Args:
        edges: DataFrame with columns [u, v, length, maxspeed, highway] and
            optional edge key column [key].
        congestion_df: DataFrame with columns [u, v, p_congestion] and optional
            edge key/context columns [key, weather, time_of_day].

    Returns:
        DataFrame with columns [u, v, travel_time_min] and [key] when present
        on the input edges.
    """
    require_columns(edges, {"u", "v", "length", "maxspeed", "highway"}, "edges")
    require_columns(congestion_df, {"u", "v", "p_congestion"}, "congestion_df")

    join_columns = ["u", "v"]
    include_key = "key" in edges.columns
    if include_key and "key" in congestion_df.columns:
        join_columns.append("key")

    congestion_clean = congestion_df.copy()
    congestion_clean["p_congestion"] = pd.to_numeric(
        congestion_clean["p_congestion"],
        errors="coerce",
    ).fillna(0.2)
    congestion_clean = congestion_clean.drop_duplicates(subset=join_columns, keep="first")

    merged = edges.merge(congestion_clean, on=join_columns, how="left")
    merged["p_congestion"] = pd.to_numeric(merged["p_congestion"], errors="coerce").fillna(0.2)
    merged["weather"] = merged.get("weather", "clear")
    merged["time_of_day"] = merged.get("time_of_day", "normal")
    merged["weather"] = merged["weather"].fillna("clear")
    merged["time_of_day"] = merged["time_of_day"].fillna("normal")

    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        edge = {
            "u": row["u"],
            "v": row["v"],
            "length": row["length"],
            "maxspeed": row["maxspeed"],
            "highway": row["highway"],
        }
        travel_time_min = predict_one_edge(
            edge=edge,
            weather=str(row["weather"]),
            is_peak_hour=to_peak_hour_flag(row["time_of_day"]),
            congestion_prob=float(row["p_congestion"]),
        )
        records.append(
            {
                "u": int(row["u"]),
                "v": int(row["v"]),
                **({"key": int(row["key"])} if include_key else {}),
                "travel_time_min": round(float(travel_time_min), 4),
            }
        )

    output_columns = ["u", "v", "travel_time_min"]
    if include_key:
        output_columns = ["u", "v", "key", "travel_time_min"]

    return pd.DataFrame(records, columns=output_columns)
