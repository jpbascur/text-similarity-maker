---
title: Text Similarity Maker
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TSM — Text Similarity Maker

Turn a list of scientific papers into a [VOSviewer](https://www.vosviewer.com) science map based on text similarity.

**Live app:** [huggingface.co/spaces/juanbascur/text-similarity-maker](https://huggingface.co/spaces/juanbascur/text-similarity-maker)

Use the **One click map** for the simplest experience, or the step-by-step sections below for more control. If you get stuck, reload.

## Supported input formats

- **RIS** (.ris) — Scopus, Web of Science, Zotero, Mendeley, EndNote
- **BibTeX** (.bib) — Google Scholar, Zotero, most reference managers
- **PubMed** (.txt, .nbib) — from the PubMed website
- **Excel** (.xlsx) — manually built spreadsheets with columns `id`, `title`, `abstract`

## How it works

1. **Parse** — reads your reference export and extracts title and abstract for each paper
2. **Embed** — encodes each paper into a vector using SPECTER2 (a scientific language model)
3. **Network** — connects each paper to its most similar neighbours using cosine similarity
4. **UMAP** *(optional)* — projects papers into 2D for a coordinate map
5. **Export** — builds a VOSviewer JSON you can open directly in VOSviewer Online

Each intermediate file (papers CSV, embeddings CSV, network CSV) can be saved and re-uploaded — you never have to redo earlier steps if the outputs already exist.

## For developers

All pipeline logic lives in `app/pipeline.py` as six self-contained functions. Each takes file bytes as input and returns file bytes as output, with no dependency on Streamlit. You can copy any function into your own script or notebook — the section header comments note which shared helpers to bring along.

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

Important: this repository's GitHub branch is `master`, but the Hugging Face Space is served from its `main` branch. Pushing to the Space remote's `master` branch can succeed without changing the live app. Deploy app code to Hugging Face with:

```bash
git push hf master:main
```

After deploying, verify the served Space source, not only the git push output:

```bash
curl https://huggingface.co/spaces/juanbascur/text-similarity-maker/raw/main/app/streamlit_app.py
```

If dependency or model-loading files changed, rebuild the base image first through the GitHub Actions workflow, then push the app code to `hf/main`.

```bash
# Build and push base image (only needed when dependencies or model change)
docker build -f Dockerfile.base -t ghcr.io/jpbascur/tsm-base:latest .
docker push ghcr.io/jpbascur/tsm-base:latest

# Build and run app image
docker build -t tsm .
docker run -p 7860:7860 tsm
```

Then open [http://localhost:7860](http://localhost:7860).
