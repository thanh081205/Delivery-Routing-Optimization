"""
Synthetic edge data utilities used when map/network data is unavailable.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ROAD_TYPES: tuple[str, ...] = (
    "motorway",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
)


def generate_edge_catalog(num_edges: int = 300, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a reproducible synthetic edge catalog that matches the TV3 interface.
    """
    rng = np.random.default_rng(random_state)

    road_type_values = rng.choice(
        ROAD_TYPES,
        size=num_edges,
        p=[0.05, 0.13, 0.19, 0.23, 0.30, 0.10],
    )
    base_speed_by_type = {
        "motorway": 70,
        "primary": 55,
        "secondary": 45,
        "tertiary": 40,
        "residential": 30,
        "service": 25,
    }

    lengths = rng.uniform(80.0, 2200.0, size=num_edges)
    speed_noise = rng.integers(-5, 6, size=num_edges)
    speeds = [
        max(20, base_speed_by_type[str(road_type)] + int(noise))
        for road_type, noise in zip(road_type_values, speed_noise)
    ]

    u_nodes = np.arange(1000, 1000 + num_edges)
    v_offset = rng.integers(1, 60, size=num_edges)
    v_nodes = u_nodes + v_offset

    return pd.DataFrame(
        {
            "u": u_nodes.astype(int),
            "v": v_nodes.astype(int),
            "key": np.zeros(num_edges, dtype=int),
            "length": lengths.astype(float),
            "maxspeed": speeds,
            "highway": road_type_values.astype(str),
        }
    )


def demo_edges() -> pd.DataFrame:
    """Small deterministic sample for quick demos and integration tests."""
    return pd.DataFrame(
        [
            {"u": 1, "v": 2, "key": 0, "length": 180.0, "maxspeed": 30, "highway": "residential"},
            {"u": 2, "v": 3, "key": 0, "length": 420.0, "maxspeed": 40, "highway": "tertiary"},
            {"u": 3, "v": 4, "key": 0, "length": 900.0, "maxspeed": 50, "highway": "secondary"},
            {"u": 4, "v": 5, "key": 0, "length": 1600.0, "maxspeed": 55, "highway": "primary"},
        ]
    )


def describe_edge_source(source_name: str, edge_count: int) -> str:
    """Human-readable summary used by the training script output."""
    return f"{source_name} ({edge_count} edges)"


__all__: Iterable[str] = [
    "ROAD_TYPES",
    "demo_edges",
    "describe_edge_source",
    "generate_edge_catalog",
]
