"""
HDBSCAN clustering on 2D UMAP coordinates.
"""

from __future__ import annotations

import numpy as np


def hdbscan_cluster(
    coords: np.ndarray,
    min_cluster_size: int | None = None,
) -> np.ndarray:
    """Cluster 2D coordinates with HDBSCAN.

    Returns an int array of shape (N,) with cluster labels.
    Label -1 means noise.
    """
    import hdbscan

    if min_cluster_size is None:
        min_cluster_size = max(10, len(coords) // 100)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(coords)
    return labels.astype(np.int32)
