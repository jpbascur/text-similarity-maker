"""
Step 4 — VOSviewer Export
Converts the edge list + paper metadata into VOSviewer map and network files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union


def load_meta(meta_csv: Union[str, Path]) -> list[dict]:
    """Load paper metadata CSV produced by embed.py (columns: id, title)."""
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_edge_list(network_csv: Union[str, Path]) -> list[tuple[int, int, float]]:
    """Load edge list CSV produced by network.py (columns: source, target, weight)."""
    edges = []
    with open(network_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((int(row["source"]), int(row["target"]), float(row["weight"])))
    return edges


def compute_node_weights(
    papers: list[dict],
    edges: list[tuple[int, int, float]],
) -> dict[int, float]:
    """Compute per-node weight as the sum of incident edge weights."""
    weights: dict[int, float] = {i: 0.0 for i in range(len(papers))}
    for i, j, w in edges:
        weights[i] += w
        weights[j] += w
    return weights


def export_vosviewer(
    papers: list[dict],
    edges: list[tuple[int, int, float]],
    out_dir: Union[str, Path],
    name: str = "vosviewer",
) -> tuple[Path, Path]:
    """Write VOSviewer map and network files.

    Map file (tab-separated):   id  label  weight<scores>
    Network file (tab-separated): id1  id2  weight

    VOSviewer uses 1-based integer IDs.

    Returns (map_path, network_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_weights = compute_node_weights(papers, edges)

    # --- Map file ---
    map_path = out_dir / f"{name}_map.txt"
    with open(map_path, "w", encoding="utf-8", newline="") as f:
        f.write("id\tlabel\tweight<links>\n")
        for idx, paper in enumerate(papers):
            vos_id = idx + 1  # 1-based
            label = paper.get("title", paper.get("id", str(vos_id)))
            weight = round(node_weights[idx], 6)
            f.write(f"{vos_id}\t{label}\t{weight}\n")

    # --- Network file ---
    net_path = out_dir / f"{name}_network.txt"
    with open(net_path, "w", encoding="utf-8", newline="") as f:
        for i, j, w in edges:
            f.write(f"{i + 1}\t{j + 1}\t{round(w, 6)}\n")

    return map_path, net_path
