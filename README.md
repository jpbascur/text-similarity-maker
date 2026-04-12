---
title: Text Similarity Maker
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TSM — Text Similarity Maker

A local Streamlit pipeline that turns a list of scientific papers into a
**VOSviewer-compatible** network. Each step can be saved and reloaded
independently — you never have to re-run earlier steps if the outputs already
exist.

## Pipeline

| Step | Input | Output | Key parameters |
|------|-------|--------|----------------|
| 1 · Embeddings | CSV/JSON of papers | `embeddings.npy` + `embeddings_meta.csv` | — |
| 2 · Whitening | One or more `.npy` files | `whitened.npy` | — |
| 3 · Network | `whitened.npy` | `network.csv` (edge list) | top-k, min similarity |
| 4 · VOSviewer Export | edge list + metadata | `vosviewer_map.txt`, `vosviewer_network.txt` | — |

## Input format

**CSV** — columns: `id, title, abstract`

**JSON** — list of `{"id": "...", "title": "...", "abstract": "..."}`

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `adapters` can be large. Install PyTorch first following
> the [official instructions](https://pytorch.org/get-started/locally/) for your
> platform and CUDA version, then install the rest.

## Running

```bash
streamlit run streamlit_app.py
```

## Loading into VOSviewer

1. Open VOSviewer → **Create** → **Create a map based on network data**
2. Select `vosviewer_network.txt` as the network file
3. Select `vosviewer_map.txt` as the map file
4. VOSviewer handles visualization and community detection

## Tech stack

- [Streamlit](https://streamlit.io) — UI
- [SPECTER2](https://huggingface.co/allenai/specter2_base) + proximity adapter — paper embeddings
- NumPy + SciPy — ZCA whitening and cosine-similarity edge list
