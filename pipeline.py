"""
TSM pipeline — six functions, one per output file.

Each function owns its single-use helpers, defined directly above it.
Two parsers are shared across multiple functions and live at the top:
  - parse_papers_csv    (used by embed_papers, build_vos_network_json, build_vos_coords_json)
  - parse_embeddings_csv (used by build_edge_list, umap_reduce)

Data flow
---------
reference export  ──► parse_reference_file  ──► papers CSV
papers CSV        ──► embed_papers           ──► embeddings CSV
embeddings CSV    ──► build_edge_list        ──► network CSV
network CSV       ──► build_vos_network_json ──► VOSviewer network JSON
embeddings CSV    ──► umap_reduce            ──► coordinates CSV
coordinates CSV   ──► build_vos_coords_json  ──► VOSviewer coordinate JSON
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd


# ── shared parsers (used by more than one pipeline function) ──────────────────

def parse_papers_csv(data: bytes) -> list[dict]:
    """Parse a papers CSV (columns: id, title, abstract) into a list of dicts.

    Shared by: embed_papers, build_vos_network_json, build_vos_coords_json.
    """
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The papers file is empty.") from exc
    missing = [c for c in ["id", "title", "abstract"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. The file must have columns: id, title, abstract."
        )
    return df.to_dict("records")


def parse_embeddings_csv(data: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse an embeddings CSV (no header, first column = id) into (array, ids).

    Shared by: build_edge_list, umap_reduce.
    """
    try:
        df = pd.read_csv(io.BytesIO(data), header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The embeddings file is empty.") from exc
    if df.empty:
        raise ValueError("The embeddings file is empty.")
    try:
        ids = df.iloc[:, 0].astype(str).tolist()
        arr = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "The embeddings file appears to have a header row. "
            "The first row should be data: paper ID followed by embedding values."
        ) from exc
    if arr.ndim != 2 or arr.shape[1] != 768:
        n_dims = arr.shape[1] if arr.ndim == 2 else 0
        raise ValueError(
            f"Expected 768 embedding dimensions per row, got {n_dims}. "
            "This file does not look like a SPECTER2 embeddings file."
        )
    return arr, ids


# ── 1. parse_reference_file ───────────────────────────────────────────────────
# Input:  reference export bytes (.ris / .bib / .txt / .nbib / .xlsx)
# Output: papers CSV bytes

def _parse_pubmed(text: str) -> list[dict]:
    import re
    records, current, current_tag = [], {}, None
    for line in text.splitlines():
        if re.match(r'^ER\s*-', line):
            if current:
                records.append(current)
            current, current_tag = {}, None
            continue
        m = re.match(r'^([A-Z]+)\s*-\s+(.*)', line)
        if m:
            current_tag = m.group(1)
            val = m.group(2).strip()
            current[current_tag] = (current[current_tag] + " " + val) if current_tag in current else val
        elif line.startswith("      ") and current_tag:
            current[current_tag] += " " + line.strip()
        elif not line.strip():
            if current:
                records.append(current)
            current, current_tag = {}, None
    if current:
        records.append(current)
    return [
        {"id": r["PMID"].strip(), "title": r["TI"].strip(), "abstract": r.get("AB", "").strip()}
        for r in records if "PMID" in r and "TI" in r
    ]


def _parse_ris(text: str) -> list[dict]:
    import re
    records, current, current_tag = [], {}, None
    for line in text.splitlines():
        if re.match(r'^ER\s*-', line):
            if current:
                records.append(current)
            current, current_tag = {}, None
            continue
        m = re.match(r'^([A-Z][A-Z0-9])\s+-\s+(.*)', line)
        if m:
            current_tag = m.group(1)
            val = m.group(2).strip()
            current[current_tag] = (current[current_tag] + " " + val) if current_tag in current else val
        elif line.startswith("  ") and current_tag:
            current[current_tag] += " " + line.strip()
        elif not line.strip():
            if current:
                records.append(current)
            current, current_tag = {}, None
    if current:
        records.append(current)
    result = []
    for i, r in enumerate(records):
        id_ = r.get("ID") or r.get("AN") or r.get("UT") or r.get("DO") or str(i + 1)
        title = r.get("TI") or r.get("T1", "")
        abstract = r.get("AB") or r.get("N2", "")
        if title:
            result.append({"id": id_.strip(), "title": title.strip(), "abstract": abstract.strip()})
    return result


def _parse_bibtex(text: str) -> list[dict]:
    import re
    result = []
    for entry in re.split(r'(?=@\w+\{)', text):
        key_m = re.match(r'@\w+\{([^,\n]+),', entry)
        if not key_m:
            continue
        key = key_m.group(1).strip()
        def _field(name):
            m = re.search(rf'\b{name}\s*=\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}', entry, re.IGNORECASE)
            if not m:
                m = re.search(rf'\b{name}\s*=\s*"([^"]*)"', entry, re.IGNORECASE)
            if m:
                return re.sub(r'\{([^{}]*)\}', r'\1', m.group(1)).strip()
            return ""
        title = _field("title")
        abstract = _field("abstract")
        if title:
            result.append({"id": key, "title": title, "abstract": abstract})
    return result


def _parse_excel(data: bytes) -> list[dict]:
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not read Excel file: {exc}") from exc
    df = df.dropna(how="all")
    col_lower = {c.strip().lower(): c for c in df.columns}
    def _find(*names):
        for n in names:
            if n in col_lower:
                return col_lower[n]
        return None
    title_col    = _find("title", "paper title", "article title")
    abstract_col = _find("abstract", "summary")
    id_col       = _find("id")
    missing = [name for name, col in [("id", id_col), ("title", title_col), ("abstract", abstract_col)] if col is None]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Columns found in file: {list(df.columns)}. "
            "The spreadsheet must have columns named 'id', 'title', and 'abstract'."
        )
    assert id_col is not None and title_col is not None and abstract_col is not None
    result = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        title    = row[title_col].strip()
        abstract = row[abstract_col].strip()
        id_      = row[id_col].strip()
        if not id_:
            raise ValueError(f"Row {i} has an empty id. Every row must have an id value.")
        if title:
            result.append({"id": id_, "title": title, "abstract": abstract})
    return result


def parse_reference_file(data: bytes, filename: str, ignore_incomplete: bool = False) -> list[dict]:
    """Parse a reference export file into a list of paper dicts.

    Supported formats: RIS (.ris), BibTeX (.bib), PubMed (.txt, .nbib), Excel (.xlsx).

    Args:
        data:              Raw file bytes.
        filename:          Original filename — used to detect the format from the extension.
        ignore_incomplete: If True, papers without both a title and an abstract are dropped.
                           Pass False to embed everything and let the user own the results.

    Returns a list of dicts with keys: id, title, abstract.
    """
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    if ext == ".xlsx":
        papers = _parse_excel(data)
    elif ext == ".bib":
        papers = _parse_bibtex(data.decode("utf-8", errors="replace"))
    elif ext == ".ris":
        papers = _parse_ris(data.decode("utf-8", errors="replace"))
    else:
        papers = _parse_pubmed(data.decode("utf-8", errors="replace"))
    if ignore_incomplete:
        papers = [p for p in papers if p.get("title", "").strip() and p.get("abstract", "").strip()]
    return papers


# ── 2. embed_papers ───────────────────────────────────────────────────────────
# Input:  papers CSV bytes
# Output: embeddings CSV bytes
# Note:   embed_papers calls parse_papers_csv (defined above). If you copy embed_papers, copy that function too.

def _embeddings_to_csv_bytes(embeddings: np.ndarray, ids: list[str]) -> bytes:
    buf = io.StringIO()
    for row_id, row in zip(ids, embeddings):
        buf.write(row_id + "," + ",".join(f"{v:.8f}" for v in row) + "\n")
    return buf.getvalue().encode()


def _check_embedding_dependencies():
    from importlib.metadata import PackageNotFoundError, version
    try:
        transformers_version = version("transformers")
        adapters_version     = version("adapters")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing embedding dependency. Install the pinned requirements with "
            "`pip install -r requirements.txt`."
        ) from exc
    major_minor = tuple(int(p) for p in transformers_version.split(".")[:2])
    if major_minor < (4, 40) or major_minor >= (4, 52):
        raise RuntimeError(
            f"Incompatible transformers {transformers_version} / adapters {adapters_version}. "
            "Install pinned requirements: transformers>=4.40,<4.52."
        )


loaded_model: dict = {}  # keys: tokenizer, model, loaded_at — managed by the app


def _load_model():
    if loaded_model:
        return loaded_model["tokenizer"], loaded_model["model"]
    _check_embedding_dependencies()
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.eval()
    loaded_model["tokenizer"] = tokenizer
    loaded_model["model"]     = model
    return tokenizer, model


def embed_papers(papers_csv: bytes, batch_size: int = 64, progress_callback=None) -> bytes:
    """Encode papers with SPECTER2 + proximity adapter; return embeddings CSV bytes.

    Args:
        papers_csv:        Papers CSV bytes (columns: id, title, abstract).
        batch_size:        Number of papers to encode per forward pass (default 64).
        progress_callback: Optional function(current, total) called after each batch.
                           Use this to update a progress bar in your UI. Leave as None
                           if you don't need progress updates (e.g. scripts, notebooks).

    Returns embeddings CSV bytes (no header; first column = paper id).
    """
    import torch

    papers = parse_papers_csv(papers_csv)
    tokenizer, model = _load_model()
    all_embeddings, ids = [], []

    with torch.no_grad():
        for start in range(0, len(papers), batch_size):
            batch = papers[start: start + batch_size]
            texts = [p["title"] + tokenizer.sep_token + p.get("abstract", "") for p in batch]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            output = model(**encoded)
            all_embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy())
            ids.extend(p["id"] for p in batch)
            if progress_callback:
                progress_callback(min(start + batch_size, len(papers)), len(papers))

    if not all_embeddings:
        raise ValueError("No papers to embed — the papers CSV is empty.")
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return _embeddings_to_csv_bytes(embeddings, ids)


# ── 3. build_edge_list ────────────────────────────────────────────────────────
# Input:  embeddings CSV bytes
# Output: network CSV bytes
# Note:   build_edge_list calls parse_embeddings_csv (defined above). If you copy build_edge_list, copy that function too.

def _edges_to_csv_bytes(edges: list[tuple[int, int, float]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["source", "target", "weight"])
    w.writerows(edges)
    return buf.getvalue().encode()


def build_edge_list(
    embeddings_csv: bytes,
    top_k: int = 20,
    min_similarity: float = 0.0,
    progress_callback=None,
) -> bytes:
    """Compute top-k cosine-similar neighbours for each paper; return network CSV bytes.

    Args:
        embeddings_csv:    Embeddings CSV bytes (no header; first column = paper id).
        top_k:             Number of nearest neighbours per paper.
        min_similarity:    Edges below this cosine similarity are dropped.
        progress_callback: Optional function(current, total) called after each batch.
                           Use this to update a progress bar in your UI. Leave as None
                           if you don't need progress updates (e.g. scripts, notebooks).

    Returns network CSV bytes (columns: source, target, weight).
    """
    embeddings, _ = parse_embeddings_csv(embeddings_csv)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit_embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    n_papers = len(unit_embeddings)
    edges: dict[tuple[int, int], float] = {}
    batch_size = 256

    for batch_start in range(0, n_papers, batch_size):
        batch_end = min(batch_start + batch_size, n_papers)
        batch = unit_embeddings[batch_start:batch_end]
        similarities = batch @ unit_embeddings.T  # (batch, n_papers)

        for batch_offset in range(batch_end - batch_start):
            paper_idx = batch_start + batch_offset
            paper_sims = similarities[batch_offset].copy()
            paper_sims[paper_idx] = -1.0  # exclude self

            neighbors = (
                np.argpartition(paper_sims, -top_k)[-top_k:]
                if top_k < n_papers - 1 else np.arange(n_papers)
            )
            for neighbor_idx in neighbors:
                if neighbor_idx == paper_idx:
                    continue
                similarity = float(paper_sims[neighbor_idx])
                if similarity < min_similarity:
                    continue
                a, b = (paper_idx, int(neighbor_idx)) if paper_idx < int(neighbor_idx) else (int(neighbor_idx), paper_idx)
                edges[(a, b)] = edges.get((a, b), 0.0) + similarity

        if progress_callback:
            progress_callback(batch_end, n_papers)

    return _edges_to_csv_bytes([(a, b, w) for (a, b), w in sorted(edges.items())])


# ── 4. build_vos_network_json ─────────────────────────────────────────────────
# Input:  papers CSV bytes + network CSV bytes
# Output: VOSviewer network JSON dict
# Note:   build_vos_network_json calls parse_papers_csv (defined above). If you copy build_vos_network_json, copy that function too.

def parse_edge_csv(data: bytes) -> list[tuple[int, int, float]]:
    """Parse a network CSV (columns: source, target, weight) into an edge list."""
    try:
        df = pd.read_csv(io.BytesIO(data), usecols=["source", "target", "weight"])
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The network file is empty.") from exc
    except ValueError as exc:
        raise ValueError("The file must have columns: source, target, weight.") from exc
    return list(zip(
        df["source"].to_numpy(dtype=np.int64).tolist(),
        df["target"].to_numpy(dtype=np.int64).tolist(),
        df["weight"].to_numpy(dtype=np.float64).tolist(),
    ))


def build_vos_network_json(papers_csv: bytes, edges_csv: bytes, clustering: str = "auto") -> dict:
    """Build a VOSviewer network JSON from papers and edges file bytes.

    Args:
        papers_csv: Papers CSV bytes (columns: id, title, abstract).
        edges_csv:  Network CSV bytes (columns: source, target, weight).
        clustering: ``"auto"`` omits the cluster field so VOSviewer runs its own
                    algorithm; ``"none"`` pins all items to cluster 1 (same colour).

    Returns a ``{"network": {"items": [...], "links": [...]}}`` dict.
    """
    papers = parse_papers_csv(papers_csv)
    edges  = parse_edge_csv(edges_csv)
    items  = [
        {"id": str(idx + 1), "label": p.get("title", p.get("id", str(idx + 1))),
         "description": str(p.get("id", ""))}
        for idx, p in enumerate(papers)
    ]
    if clustering == "none":
        items = [{**it, "cluster": 1} for it in items]
    links = [
        {"source_id": str(i + 1), "target_id": str(j + 1), "strength": round(w, 6)}
        for i, j, w in edges
    ]
    return {"network": {"items": items, "links": links}}


# ── 5. umap_reduce ────────────────────────────────────────────────────────────
# Input:  embeddings CSV bytes
# Output: coordinates CSV bytes
# Note:   umap_reduce calls parse_embeddings_csv (defined above). If you copy umap_reduce, copy that function too.

def _coords_to_csv_bytes(ids: list[str], coords: np.ndarray) -> bytes:
    buf = io.StringIO()
    buf.write("id,x,y\n")
    for pid, (x, y) in zip(ids, coords):
        buf.write(f"{pid},{x:.6f},{y:.6f}\n")
    return buf.getvalue().encode()


def umap_reduce(
    embeddings_csv: bytes,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> bytes:
    """Reduce embeddings to 2D with UMAP (cosine metric); return coordinates CSV bytes.

    Args:
        embeddings_csv: Embeddings CSV bytes (no header; first column = paper id).
        n_neighbors:    Controls local vs global structure. Lower = more local clusters.
        min_dist:       Minimum distance between points. Lower = tighter clusters.
        random_state:   Random seed for reproducibility.

    Returns coordinates CSV bytes (columns: id, x, y).
    """
    from umap import UMAP  # type: ignore[import-untyped]

    embeddings, ids = parse_embeddings_csv(embeddings_csv)
    reducer = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    return _coords_to_csv_bytes(ids, coords)


# ── 6. build_vos_coords_json ──────────────────────────────────────────────────
# Input:  papers CSV bytes + coordinates CSV bytes
# Output: VOSviewer coordinate JSON dict
# Note:   build_vos_coords_json calls parse_papers_csv (defined above). If you copy build_vos_coords_json, copy that function too.

def parse_coords_csv(data: bytes) -> tuple[list[str], np.ndarray]:
    """Parse a coordinates CSV (columns: id, x, y) into (ids, array)."""
    try:
        df = pd.read_csv(io.BytesIO(data), usecols=["id", "x", "y"])
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The coordinates file is empty.") from exc
    except ValueError as exc:
        raise ValueError("The file must have columns: id, x, y.") from exc
    return df["id"].astype(str).tolist(), df[["x", "y"]].to_numpy(dtype=np.float32)


def build_vos_coords_json(papers_csv: bytes, coords_csv: bytes) -> dict:
    """Build a VOSviewer coordinate JSON from papers and coordinates file bytes.

    Args:
        papers_csv: Papers CSV bytes (columns: id, title, abstract).
        coords_csv: Coordinates CSV bytes (columns: id, x, y).

    Returns a ``{"network": {"items": [...], "links": []}}`` dict where every
    item carries ``x``, ``y``, and ``cluster: 1``.
    """
    papers = parse_papers_csv(papers_csv)
    ids, coords = parse_coords_csv(coords_csv)
    coord_lookup = {pid: (float(x), float(y)) for pid, (x, y) in zip(ids, coords)}
    items = []
    for idx, paper in enumerate(papers):
        pid = str(paper.get("id", idx + 1))
        x, y = coord_lookup.get(pid, (0.0, 0.0))
        items.append({
            "id": str(idx + 1), "label": paper.get("title", pid),
            "description": pid, "x": round(x, 6), "y": round(y, 6), "cluster": 1,
        })
    return {"network": {"items": items, "links": []}}
