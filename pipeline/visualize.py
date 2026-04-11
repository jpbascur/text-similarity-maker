"""
Interactive 2D scatter plot using Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Distinct colors for clusters; noise (-1) gets gray
_NOISE_COLOR = "#aaaaaa"
_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
]

_MAX_LABELS = 80  # max title annotations shown across the whole plot


def build_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    papers: list[dict],
) -> go.Figure:
    """Build an interactive Plotly scatter plot.

    Args:
        coords:  (N, 2) float array — UMAP x/y positions
        labels:  (N,) int array — cluster labels (-1 = noise)
        papers:  list of dicts with at least 'id' and 'title';
                 'year' is used in tooltips if present

    Returns a Plotly Figure.
    """
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels,
        "id":    [p.get("id", str(i)) for i, p in enumerate(papers)],
        "title": [p.get("title", "") for p in papers],
        "year":  [p.get("year", "") for p in papers],
    })

    fig = go.Figure()
    unique_labels = sorted(df["cluster"].unique())

    # ── marker traces ────────────────────────────────────────────────────────
    for label in unique_labels:
        mask = df["cluster"] == label
        subset = df[mask]

        if label == -1:
            name = "Noise"
            color = _NOISE_COLOR
            visible = "legendonly"
        else:
            name = f"Cluster {label}"
            color = _PALETTE[label % len(_PALETTE)]
            visible = True

        hover = []
        for _, row in subset.iterrows():
            parts = [f"<b>{row['title']}</b>"]
            if row["year"]:
                parts.append(f"Year: {row['year']}")
            hover.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers",
            name=name,
            visible=visible,
            marker=dict(color=color, size=6, opacity=0.75),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            customdata=subset["id"],
        ))

    # ── title label annotations (sampled, non-noise only) ───────────────────
    labeled = df[df["cluster"] != -1].copy()
    if len(labeled) > 0:
        n_labels = min(_MAX_LABELS, len(labeled))
        # sample spatially: divide x-range into a grid and pick one point per cell
        rng = np.random.default_rng(42)
        sample = labeled.sample(n=n_labels, random_state=42) if len(labeled) >= n_labels else labeled

        # Truncate long titles for readability
        def _short(t: str, maxlen: int = 40) -> str:
            return t if len(t) <= maxlen else t[:maxlen - 1] + "…"

        fig.add_trace(go.Scatter(
            x=sample["x"],
            y=sample["y"],
            mode="text",
            text=[_short(t) for t in sample["title"]],
            textfont=dict(size=8, color="rgba(40,40,40,0.55)"),
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
    )
    return fig
