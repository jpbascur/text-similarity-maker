"""
UMAP dimensionality reduction.
Reduces embeddings to 2D for visualization.
"""

from __future__ import annotations

import numpy as np


def umap_reduce(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP with cosine metric.

    Returns a float32 array of shape (N, 2).
    """
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32)
