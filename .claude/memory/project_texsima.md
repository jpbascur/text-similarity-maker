---
name: Project TexSiMa
description: Streamlit pipeline for turning scientific papers into VOSviewer-compatible networks
type: project
---

TexSiMa (Text Similarity Maker) is a local Streamlit app with 4 pipeline steps:
1. Embed papers via SPECTER2 + proximity adapter → .npy + metadata CSV
2. ZCA-whiten one or more .npy files together → whitened .npy
3. Build cosine-similarity edge list (top-k, min threshold, pure numpy/scipy) → network CSV
4. Export VOSviewer map + network tab-separated files

**Why:** VOSviewer handles visualization and community detection; the app only produces the inputs it needs.

**How to apply:** No networkx. Each step saves independently. User can re-enter at any step by uploading a prior output.

Stack: Streamlit, transformers + adapters (SPECTER2), numpy, scipy, pandas.
