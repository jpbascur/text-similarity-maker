---
title: Text Similarity Maker
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TSM — Text Similarity Maker

Upload scientific papers and get a [VOSviewer](https://www.vosviewer.com) science map based on text similarity.

Use the **One click map** for the simplest experience, or the step-by-step sections for more control and speed. If you get stuck, reload.

## Pipeline

| Step | Input | Output |
|------|-------|--------|
| 1 · Parse | Reference export file | Papers CSV (id, title, abstract) |
| 2 · Embed | Papers CSV | Embeddings CSV (SPECTER2 vectors) |
| 3 · Network | Embeddings CSV | Network CSV (cosine similarity edge list) |
| 3 · UMAP | Embeddings CSV | Coordinates CSV (2D layout) |
| 4 · Export | Papers + Network/Coordinates | VOSviewer JSON |

Each file can be saved and re-uploaded independently — you never have to re-run earlier steps if the outputs already exist.

## Supported input formats

- **RIS** (.ris) — Scopus, Web of Science, Zotero, Mendeley, EndNote
- **BibTeX** (.bib) — Google Scholar, Zotero, most reference managers
- **PubMed** (.txt, .nbib) — from the PubMed website
- **Excel** (.xlsx) — manually built spreadsheets with columns `id`, `title`, `abstract`

## Tech stack

- [Streamlit](https://streamlit.io) — UI
- [SPECTER2](https://huggingface.co/allenai/specter2_base) + proximity adapter — paper embeddings
- [UMAP](https://umap-learn.readthedocs.io) — 2D dimensionality reduction
- NumPy + pandas — similarity computation and data handling

## Deployment

The app ships as two Docker images:

- **Base image** (`Dockerfile.base`) — installs dependencies and pre-downloads the SPECTER2 model
- **App image** (`Dockerfile`) — copies app code on top of the base image

```bash
# Build base image (once, when deps or model change)
docker build -f Dockerfile.base -t tsm-base .

# Build and run app
docker build -t tsm .
docker run -p 7860:7860 tsm
```

Then open [http://localhost:7860](http://localhost:7860).
