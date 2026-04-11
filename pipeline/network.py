"""
Step 3 — Network
Builds a weighted edge list from whitened embeddings using cosine similarity.
Uses only numpy and scipy — no networkx.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import numpy as np


def cosine_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot product == cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def build_edge_list(
    embeddings: np.ndarray,
    top_k: int = 20,
    min_similarity: float = 0.0,
    progress_callback=None,
) -> list[tuple[int, int, float]]:
    """Compute top-k cosine-similar neighbors for each paper.

    Returns a deduplicated edge list: [(i, j, weight), ...] where i < j.
    Only edges with weight >= min_similarity are kept.
    """
    normed = cosine_normalize(embeddings)
    n = len(normed)
    # Use a sparse matrix to accumulate edges and avoid duplicates
    edge_map: dict[tuple[int, int], float] = {}

    batch_size = 256

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normed[start:end]  # (B, D)

        # Similarities between batch rows and all rows: (B, N)
        sims = batch @ normed.T

        for local_idx in range(end - start):
            global_idx = start + local_idx
            row = sims[local_idx].copy()
            row[global_idx] = -1.0  # exclude self

            # top_k indices
            if top_k < n - 1:
                # argpartition is faster than full argsort
                top_indices = np.argpartition(row, -(top_k))[-top_k:]
            else:
                top_indices = np.arange(n)

            for j in top_indices:
                if j == global_idx:
                    continue
                w = float(row[j])
                if w < min_similarity:
                    continue
                i, j = (global_idx, int(j)) if global_idx < int(j) else (int(j), global_idx)
                key = (i, j)
                edge_map[key] = edge_map.get(key, 0.0) + w

        if progress_callback:
            progress_callback(end, n)

    return [(i, j, w) for (i, j), w in sorted(edge_map.items())]


def save_edge_list(
    edges: list[tuple[int, int, float]],
    out_dir: Union[str, Path],
    name: str = "network",
) -> Path:
    """Save the edge list to a CSV file with columns: source, target, weight."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        writer.writerows(edges)

    return out_path
