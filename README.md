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
| 4 · Export | Papers + Network / Coordinates | VOSviewer JSON |

Each file can be saved and re-uploaded independently — you never have to re-run earlier steps if the outputs already exist.

All pipeline logic lives in `app/pipeline.py`. Each of the six main functions is self-contained: it takes file bytes as input and returns file bytes as output, with no dependency on Streamlit or the web app. You can copy any function directly into your own script or notebook — just bring the helper functions defined above it (noted in the section header comments).

## Supported input formats

- **RIS** (.ris) — Scopus, Web of Science, Zotero, Mendeley, EndNote
- **BibTeX** (.bib) — Google Scholar, Zotero, most reference managers
- **PubMed** (.txt, .nbib) — from the PubMed website
- **Excel** (.xlsx) — manually built spreadsheets with columns `id`, `title`, `abstract`

## Repo structure

```
Dockerfile            # App image — builds on top of the base image
Dockerfile.base       # Base image — installs dependencies and pre-downloads SPECTER2
DOCKER_prefetch.py    # Run at build time to cache the SPECTER2 model on disk
requirements.txt
app/
  streamlit_app.py    # UI — all pipeline logic imported from pipeline.py
  pipeline.py         # Six pure functions, one per output file
  sample_papers.ris   # Demo file for the download button
  static/             # Session JSON files served to VOSviewer Online (auto-cleaned after 24 h)
```

## Tech stack

- [Streamlit](https://streamlit.io) — UI
- [SPECTER2](https://huggingface.co/allenai/specter2_base) + proximity adapter — paper embeddings
- [UMAP](https://umap-learn.readthedocs.io) — 2D dimensionality reduction
- NumPy + pandas — similarity computation and data handling

## Deployment

The app uses a two-stage Docker build to keep deploys fast. The base image bakes in all dependencies and the SPECTER2 model so the app image only needs to copy the code.

```bash
# Build and push base image (only needed when dependencies or model change)
docker build -f Dockerfile.base -t ghcr.io/jpbascur/tsm-base:latest .
docker push ghcr.io/jpbascur/tsm-base:latest

# Build and run app image
docker build -t tsm .
docker run -p 7860:7860 tsm
```

Then open [http://localhost:7860](http://localhost:7860).
