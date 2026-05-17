"""
TSM - Text Similarity Maker
Streamlit pipeline for building VOSviewer-compatible paper networks.

Architecture principles
-----------------------
1. Functions receive file bytes, return data structures or file bytes.
   Pipeline functions never read from st.session_state; every input is an
   explicit argument.  This means swapping the bytes changes the output
   automatically — no hidden state can drift out of sync with the UI.

2. Session state stores file bytes under ``_dl_*`` keys plus a small set of
   computed in-memory objects (numpy arrays, edge lists) needed for display.
   The bytes are the source of truth; the in-memory objects are caches only.

3. Hyperparameters (top_k, min_similarity, n_neighbors, clustering, …) are
   explicit function arguments.  OCM hard-codes them; the Step-3 UI reads
   them from widgets and passes them explicitly.  No function reads a widget
   value internally.

4. OCM is the same pipeline with hard-coded arguments, not a separate
   implementation.  If you add a pipeline function, OCM calls it too —
   just pass its fixed value instead of a widget value.
"""
import csv
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

# ── memory (read before any st calls so variables are available anywhere) ─────
def _read_cgroup_mem():
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        usage = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        return usage, limit
    except Exception:
        pass
    limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
    usage = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
    return usage, limit

try:
    _mem_used, _mem_limit = _read_cgroup_mem()
    _mem_used_gb  = _mem_used  / 1024**3
    _mem_total_gb = _mem_limit / 1024**3
    _mem_free_gb  = (_mem_limit - _mem_used) / 1024**3
    _mem_pct      = _mem_used / _mem_limit * 100
except Exception:
    _mem_used_gb = _mem_total_gb = _mem_free_gb = _mem_pct = None

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSM Text Similarity Maker",
    page_icon=None,
    layout="wide",
)
st.title("TSM - Text Similarity Maker")
st.caption(
    "Built by [Juan Pablo Bascur](https://jpbascur.com) - "
    "Problems? Contact [juanpablobascurcifuentes@gmail.com](mailto:juanpablobascurcifuentes@gmail.com) - "
    "Source: [github.com/jpbascur/text-similarity-maker](https://github.com/jpbascur/text-similarity-maker)"
)
if _mem_pct is not None:
    with st.expander("Server memory", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Used", f"{_mem_used_gb:.1f} GB")
        col_m2.metric("Free", f"{_mem_free_gb:.1f} GB")
        col_m3.metric("Total", f"{_mem_total_gb:.1f} GB")
        st.progress(_mem_pct / 100)
        st.caption("Memory is shared across all users. If memory is low, please wait for it to be released by another user.")
st.markdown(
    "Upload your papers and get a text similarity science map. "
    "Use the One click map for the simplest experience, "
    "or the step-by-step sections below for more control."
)
st.markdown(
    "**Supported input formats:**\n\n"
    "- **RIS** (.ris) — exported by Scopus, Web of Science, Zotero, Mendeley, EndNote\n"
    "- **BibTeX** (.bib) — exported by Google Scholar, Zotero, and most reference managers\n"
    "- **PubMed** (.txt, .nbib) — from the PubMed website\n"
    "- **Excel** (.xlsx) — for manually built spreadsheets; must have columns named *id*, *title*, and *abstract*"
)
with st.expander("How to export with abstracts included", expanded=False):
    st.markdown(
        "This tool needs title and abstract for each paper. "
        "Papers without an abstract are skipped at the embedding step. "
        "Follow the instructions below for your source to make sure abstracts are included.\n\n"

        "---\n\n"

        "**PubMed** — abstracts always included\n\n"
        "For a `.txt` file: search on PubMed, click **Send to** → **File**, "
        "set Format to **PubMed**, click **Create file**.\n\n"
        "For a `.nbib` file: same search, click **Send to** → **Citation Manager**, "
        "click **Create file**.\n\n"

        "---\n\n"

        "**Scopus** — you must manually select the abstract field\n\n"
        "1. Run your search on Scopus and select the papers you want.\n"
        "2. Click **Export**.\n"
        "3. Choose **RIS format**.\n"
        "4. Under *Information to export*, check **Abstract**. "
        "It is not selected by default.\n"
        "5. Click **Export**. Downloads as `.ris`.\n\n"

        "---\n\n"

        "**Web of Science** — use Full Record to include abstracts\n\n"
        "1. Run your search and select the papers you want.\n"
        "2. Click **Export** → **RIS File**.\n"
        "3. Set *Record Content* to **Full Record**. "
        "The default Author, Title, Source option does not include abstracts.\n"
        "4. Click **Export**. Downloads as `.ris`.\n\n"

        "---\n\n"

        "**Zotero** — abstracts exported if stored in your library\n\n"
        "1. Select the papers in your Zotero library.\n"
        "2. Right-click → **Export Items**, or go to **File** → **Export Items**.\n"
        "3. Choose **RIS** or **BibTeX** as format.\n"
        "4. Click **OK**.\n\n"
        "Abstracts are included automatically if Zotero has them. "
        "When you add papers using the Zotero browser connector, abstracts are usually captured. "
        "If a paper was added manually or imported without an abstract, it will not have one.\n\n"

        "---\n\n"

        "**Mendeley** — abstracts exported if stored in your library\n\n"
        "1. Select the papers in Mendeley.\n"
        "2. Go to **File** → **Export**.\n"
        "3. Choose **RIS** format.\n\n"
        "Same caveat as Zotero: abstracts are only exported if they are stored in your Mendeley library.\n\n"

        "---\n\n"

        "**EndNote** — requires an output style that includes abstracts\n\n"
        "1. Select the references in EndNote.\n"
        "2. Go to **File** → **Export**.\n"
        "3. Set *Output style* to **RefMan (RIS) Export**. "
        "This style includes the abstract field. Other styles may not.\n"
        "4. Save as a `.ris` file.\n\n"

        "---\n\n"

        "**Google Scholar** — abstracts are NOT included in the export\n\n"
        "Google Scholar's BibTeX export only contains basic metadata "
        "(title, authors, year, venue) and does not include abstracts. "
        "Papers imported this way will have no abstract and will be skipped at the embedding step. "
        "If your papers are on Google Scholar, the best approach is to import them into Zotero "
        "using the Zotero browser connector (which captures the abstract), "
        "and then export from Zotero.\n\n"

        "---\n\n"

        "**Excel** (.xlsx) — for a spreadsheet you are building manually\n\n"
        "If you are assembling a paper list yourself (e.g. by copy-pasting titles and abstracts), "
        "create a spreadsheet with three columns: **id**, **title**, and **abstract**. "
        "Column names are not case-sensitive. Every row must have a value in all three columns. "
        "Save the file as **.xlsx** (Excel Workbook) before uploading."
    )

# ── helpers ───────────────────────────────────────────────────────────────────

def array_to_csv_bytes(arr: np.ndarray, ids: list[str] | None = None) -> bytes:
    buf = io.StringIO()
    if ids is not None:
        for row_id, row in zip(ids, arr):
            buf.write(row_id + "," + ",".join(f"{v:.8f}" for v in row) + "\n")
    else:
        np.savetxt(buf, arr, delimiter=",", fmt="%.8f")
    return buf.getvalue().encode()

def parse_embeddings_csv(data: bytes) -> tuple[np.ndarray, list[str]]:
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

def parse_edge_csv(data: bytes) -> list[tuple[int, int, float]]:
    try:
        df = pd.read_csv(io.BytesIO(data), usecols=["source", "target", "weight"])
    except ValueError as exc:
        raise ValueError("The file must have columns: source, target, weight.") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The network file is empty.") from exc
    return list(zip(
        df["source"].to_numpy(dtype=np.int64).tolist(),
        df["target"].to_numpy(dtype=np.int64).tolist(),
        df["weight"].to_numpy(dtype=np.float64).tolist(),
    ))

def parse_coords_csv(data: bytes) -> tuple[list[str], np.ndarray]:
    try:
        df = pd.read_csv(io.BytesIO(data), usecols=["id", "x", "y"])
    except ValueError as exc:
        raise ValueError("The file must have columns: id, x, y.") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The coordinates file is empty.") from exc
    return df["id"].astype(str).tolist(), df[["x", "y"]].to_numpy(dtype=np.float32)

def parse_papers_csv(data: bytes) -> list[dict]:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The papers file is empty.") from exc
    missing_cols = [c for c in ["id", "title", "abstract"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. The file must have columns: id, title, abstract.")
    return df.to_dict("records")

def uploaded_file_bytes(upload_key: str, file_obj) -> tuple[bytes, str] | None:
    data = file_obj.getvalue()
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()
    signature = f"{file_obj.name}:{len(data)}:{digest}"
    processed_key = f"_processed_{upload_key}"
    if st.session_state.get(processed_key) == signature:
        return None
    st.session_state[processed_key] = signature
    return data, signature

def handle_panel_upload(upload_key: str, slot_key: str):
    """Streamlit on_change callback — validate an uploaded file and store its bytes.

    Reads the uploaded file once per unique (name, size, content) combination,
    validates it by calling the appropriate parser, stores raw bytes under the
    canonical ``_dl_*`` key, and clears any stale downstream state.  On error
    the bytes are not stored and the error message is written to session state
    so the UI can display it without raising.
    """
    file_obj = st.session_state.get(upload_key)
    if file_obj is None:
        return
    uploaded = uploaded_file_bytes(upload_key, file_obj)
    if uploaded is None:
        return
    raw, _signature = uploaded
    error_key = f"_upload_error_{upload_key}"
    try:
        if slot_key == "ocm_raw":
            st.session_state["ocm_raw_file"] = (file_obj.name, raw)
            st.session_state.pop("ocm_raw_parsed", None)
            st.session_state.pop("ocm_raw_parsed_key", None)
        elif slot_key == "raw":
            st.session_state["raw_file"] = (file_obj.name, raw)
            st.session_state.pop("_raw_parsed", None)
            st.session_state.pop("_raw_parsed_key", None)
        elif slot_key == "papers":
            parse_papers_csv(raw)  # validate
            _clear_downstream_papers()
            st.session_state["_dl_papers"] = raw
        elif slot_key == "embeddings":
            arr, ids = parse_embeddings_csv(raw)
            st.session_state["step1_embeddings"] = arr
            st.session_state["step1_embed_ids"] = ids
            _clear_downstream_embeddings()
            st.session_state["_dl_embeddings"] = raw
        elif slot_key == "edges":
            st.session_state["step2_edges"] = parse_edge_csv(raw)
            _clear_downstream_edges()
            st.session_state["_dl_edges"] = raw
            if "_dl_papers" in st.session_state:
                _clustering_val = "none" if st.session_state.get("net_clustering", "").startswith("None") else "auto"
                sid = st.session_state["session_id"]
                vos_data = _build_vos_network_json(st.session_state["_dl_papers"], raw, clustering=_clustering_val)
                st.session_state["vos_json"]     = json.dumps(vos_data, indent=2)
                st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", vos_data)
        elif slot_key == "network_vos":
            json_str = raw.decode("utf-8")
            data = json.loads(json_str)
            sid = st.session_state["session_id"]
            st.session_state["vos_json"] = json_str
            st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", data)
        elif slot_key == "coords":
            ids, coords = parse_coords_csv(raw)
            st.session_state["viz_ids"]    = ids
            st.session_state["viz_coords"] = coords
            _clear_downstream_coords()
            st.session_state["_dl_coords"] = raw
            if "_dl_papers" in st.session_state:
                sid = st.session_state["session_id"]
                vos_data = _build_vos_coords_json(st.session_state["_dl_papers"], raw)
                st.session_state["viz_vos_json"]     = json.dumps(vos_data, indent=2)
                st.session_state["viz_vos_json_url"] = _write_static_json(f"{sid}_umap.json", vos_data)
        elif slot_key == "coords_vos":
            json_str = raw.decode("utf-8")
            data = json.loads(json_str)
            sid = st.session_state["session_id"]
            st.session_state["viz_vos_json"] = json_str
            st.session_state["viz_vos_json_url"] = _write_static_json(f"{sid}_umap.json", data)
        st.session_state.pop(error_key, None)
    except Exception as exc:
        st.session_state[error_key] = str(exc)

def parse_pubmed_export(text: str) -> list[dict]:
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

def parse_ris_export(text: str) -> list[dict]:
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

def parse_bibtex_export(text: str) -> list[dict]:
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

def parse_excel_export(data: bytes) -> list[dict]:
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
    result = []
    for i, row in df.iterrows():
        title    = row[title_col].strip()
        abstract = row[abstract_col].strip()
        id_      = row[id_col].strip()
        if not id_:
            raise ValueError(f"Row {i + 1} has an empty id. Every row must have an id value.")
        if title:
            result.append({"id": id_, "title": title, "abstract": abstract})
    return result

def parse_reference_file(data: bytes, filename: str, ignore_incomplete: bool = False) -> list[dict]:
    """Route *data* to the correct format parser and optionally drop incomplete papers.

    Centralises format dispatch + filtering so every call site (Step-1 UI,
    OCM button) uses identical logic.  OCM always passes ``ignore_incomplete=False``
    — it embeds everything and lets the user own the results.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".xlsx":
        papers = parse_excel_export(data)
    elif ext == ".bib":
        papers = parse_bibtex_export(data.decode("utf-8", errors="replace"))
    elif ext == ".ris":
        papers = parse_ris_export(data.decode("utf-8", errors="replace"))
    else:
        papers = parse_pubmed_export(data.decode("utf-8", errors="replace"))
    if ignore_incomplete:
        papers = [p for p in papers if p.get("title", "").strip() and p.get("abstract", "").strip()]
    return papers

def _papers_to_csv_bytes(papers: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["id", "title", "abstract"])
    w.writeheader()
    w.writerows(papers)
    return buf.getvalue().encode()

def _edges_to_csv_bytes(edges: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["source", "target", "weight"])
    w.writerows(edges)
    return buf.getvalue().encode()

def _coords_to_csv_bytes(ids: list, coords: np.ndarray) -> bytes:
    buf = io.StringIO()
    buf.write("id,x,y\n")
    for pid, (x, y) in zip(ids, coords):
        buf.write(f"{pid},{x:.6f},{y:.6f}\n")
    return buf.getvalue().encode()

def _clear_downstream_papers():
    for key in ["step1_embeddings", "step1_embed_ids", "step2_edges",
                "viz_coords", "viz_ids", "vos_json", "vos_json_url",
                "viz_vos_json", "viz_vos_json_url",
                "_dl_embeddings", "_dl_edges", "_dl_coords"]:
        st.session_state.pop(key, None)

def _clear_downstream_embeddings():
    for key in ["step2_edges", "viz_coords", "viz_ids",
                "vos_json", "vos_json_url",
                "viz_vos_json", "viz_vos_json_url",
                "_dl_embeddings", "_dl_edges", "_dl_coords"]:
        st.session_state.pop(key, None)

def _clear_downstream_edges():
    for key in ["vos_json", "vos_json_url", "_dl_edges"]:
        st.session_state.pop(key, None)

def _clear_downstream_coords():
    for key in ["viz_vos_json", "viz_vos_json_url", "_dl_coords"]:
        st.session_state.pop(key, None)

# ── session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
if "running" not in st.session_state:
    st.session_state["running"] = False

_is_running = st.session_state["running"]
if _is_running:
    st.warning("A job is already running in this session. Please wait for it to finish.")


_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

def _public_app_base_url() -> str | None:
    for env_name in ("PUBLIC_BASE_URL", "APP_BASE_URL", "STREAMLIT_PUBLIC_URL"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value.rstrip("/")
    space_host = os.environ.get("SPACE_HOST", "").strip()
    if space_host:
        return f"https://{space_host}".rstrip("/")
    space_id = os.environ.get("SPACE_ID", "").strip()
    if space_id:
        slug = space_id.replace("/", "-").lower()
        return f"https://{slug}.hf.space"
    return None

def _write_static_json(filename: str, data: dict) -> str | None:
    path = _STATIC_DIR / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    base_url = _public_app_base_url()
    if not base_url:
        return None
    return f"{base_url}/app/static/{filename}"

def _vosviewer_url(json_url: str | None, **params) -> str | None:
    if not json_url:
        return None
    query = urlencode({"json": json_url, **params})
    return f"https://app.vosviewer.com/?{query}"

def _build_vos_network_json(papers_csv: bytes, edges_csv: bytes, clustering: str) -> dict:
    """Build a VOSviewer network JSON from file bytes.

    Args:
        papers_csv: Clean paper list CSV bytes (columns: id, title, abstract).
        edges_csv:  Network CSV bytes (columns: source, target, weight).
        clustering: ``"auto"`` omits the cluster field so VOSviewer runs its
                    own algorithm; ``"none"`` pins every item to cluster 1.

    Returns a ``{"network": {"items": [...], "links": [...]}}`` dict ready for
    ``json.dumps``.  Always takes bytes so the caller controls which version of
    the files is used.
    """
    papers = parse_papers_csv(papers_csv)
    edges = parse_edge_csv(edges_csv)
    items = [
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

def _build_vos_coords_json(papers_csv: bytes, coords_csv: bytes) -> dict:
    """Build a VOSviewer coordinate JSON from file bytes.

    Args:
        papers_csv: Clean paper list CSV bytes (columns: id, title, abstract).
        coords_csv: Coordinate CSV bytes (columns: id, x, y).

    Returns a ``{"network": {"items": [...], "links": []}}`` dict where every
    item carries ``x``, ``y``, and ``cluster: 1`` (coordinate maps have no
    links and no clustering).  Always takes bytes — same rationale as
    ``_build_vos_network_json``.
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONE CLICK MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "_demo_bytes" not in st.session_state:
    st.session_state["_demo_bytes"] = (Path(__file__).parent / "sample_papers.ris").read_bytes()
st.download_button(
    "No file yet? Download 50 sample papers to try.",
    st.session_state["_demo_bytes"],
    "sample_papers.ris", mime="text/plain", key="dl_demo",
)

with st.expander("One click map", expanded=True):
    st.caption("Upload your papers and get a VOSviewer map in one click.")

    st.file_uploader(
        "Reference export",
        type=["ris", "bib", "txt", "nbib", "xlsx"],
        key="ocm_raw_upload",
        on_change=handle_panel_upload,
        args=("ocm_raw_upload", "ocm_raw"),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if st.session_state.get("_upload_error_ocm_raw_upload"):
        st.error(st.session_state["_upload_error_ocm_raw_upload"])

    if "ocm_raw_file" in st.session_state:
        _ocm_fname, _ocm_bytes = st.session_state["ocm_raw_file"]
        _ocm_cache_key = ("ocm_raw_parsed", _ocm_fname, len(_ocm_bytes))
        if st.session_state.get("ocm_raw_parsed_key") != _ocm_cache_key:
            try:
                _ocm_parsed = parse_reference_file(_ocm_bytes, _ocm_fname, ignore_incomplete=False)
                st.session_state["ocm_raw_parsed"] = _ocm_parsed
                st.session_state["ocm_raw_parsed_key"] = _ocm_cache_key
            except Exception as _ocm_exc:
                st.error(f"Could not parse file: {_ocm_exc}")
                st.session_state.pop("ocm_raw_parsed", None)
                st.session_state.pop("ocm_raw_parsed_key", None)

        if "ocm_raw_parsed" in st.session_state:
            _ocm_refs = st.session_state["ocm_raw_parsed"]
            _ocm_n = len(_ocm_refs)
            _ocm_n_abs = sum(1 for p in _ocm_refs if p["abstract"])
            st.caption(f"{_ocm_n} papers found - {_ocm_n_abs} with abstracts, {_ocm_n - _ocm_n_abs} without.")
            if _ocm_n > 0 and st.button(
                "Create map", key="ocm_create", type="primary",
                disabled=_is_running, use_container_width=True,
            ):
                from pipeline.embed import embed_papers
                from pipeline.network import build_edge_list
                _ocm_papers = parse_reference_file(_ocm_bytes, _ocm_fname, ignore_incomplete=False)
                if not _ocm_papers:
                    st.error("No papers with both title and abstract found.")
                else:
                    st.session_state["running"] = True
                    try:
                        _ocm_dl = _papers_to_csv_bytes(_ocm_papers)
                        _clear_downstream_papers()
                        st.session_state["_dl_papers"] = _ocm_dl

                        _ocm_prog = st.progress(0, text="Loading SPECTER2 model...")
                        def _ocm_cb(cur, tot):
                            _ocm_prog.progress(cur / tot, text=f"Encoding {cur}/{tot}...")
                        _ocm_emb = embed_papers(parse_papers_csv(_ocm_dl), progress_callback=_ocm_cb)
                        _ocm_prog.progress(1.0, text="Embeddings done.")
                        _ocm_ids = [p["id"] for p in parse_papers_csv(_ocm_dl)]
                        _clear_downstream_embeddings()
                        st.session_state["step1_embeddings"] = _ocm_emb
                        st.session_state["step1_embed_ids"] = _ocm_ids
                        st.session_state["_dl_embeddings"] = array_to_csv_bytes(_ocm_emb, ids=_ocm_ids)

                        _ocm_prog2 = st.progress(0, text="Building network...")
                        def _ocm_cb_net(cur, tot):
                            _ocm_prog2.progress(cur / tot, text=f"Processing {cur}/{tot} papers...")
                        with st.spinner("Computing similarities..."):
                            _ocm_edges = build_edge_list(
                                _ocm_emb, top_k=20, min_similarity=0.0, progress_callback=_ocm_cb_net,
                            )
                        _ocm_prog2.progress(1.0, text="Network done.")
                        _clear_downstream_edges()
                        st.session_state["step2_edges"] = _ocm_edges
                        _ocm_edges_csv = _edges_to_csv_bytes(_ocm_edges)
                        st.session_state["_dl_edges"] = _ocm_edges_csv

                        _ocm_sid = st.session_state["session_id"]
                        _ocm_vos = _build_vos_network_json(_ocm_dl, _ocm_edges_csv, clustering="auto")
                        st.session_state["vos_json"]     = json.dumps(_ocm_vos, indent=2)
                        st.session_state["vos_json_url"] = _write_static_json(f"{_ocm_sid}_network.json", _ocm_vos)
                    except Exception as _ocm_err:
                        st.error(f"Error creating map: {_ocm_err}")
                    finally:
                        st.session_state["running"] = False
                    st.rerun()

    if "vos_json" in st.session_state:
        st.success(
            f"Map ready - {len(parse_papers_csv(st.session_state['_dl_papers']))} papers, "
            f"{len(st.session_state['step2_edges'])} connections."
        )
        _ocm_vos_url = _vosviewer_url(st.session_state.get("vos_json_url"), max_n_links=0)
        if _ocm_vos_url:
            st.link_button("Open in VOSviewer Online", _ocm_vos_url, type="primary", use_container_width=True)
        else:
            st.caption("Download the VOSviewer network file from the Files panel below and open it in VOSviewer Online manually.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 - PAPER INPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("1. Paper Input", expanded=False):
    st.caption(
        "Upload a reference export file. "
        "Supported formats: RIS (.ris), BibTeX (.bib), PubMed (.txt, .nbib), Excel (.xlsx)."
    )
    st.file_uploader(
        "Reference export",
        type=["ris", "bib", "txt", "nbib", "xlsx"],
        key="s1_raw_upload",
        on_change=handle_panel_upload,
        args=("s1_raw_upload", "raw"),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if st.session_state.get("_upload_error_s1_raw_upload"):
        st.error(st.session_state["_upload_error_s1_raw_upload"])

    ignore = st.checkbox("Ignore papers without title or abstract", value=True, key="s1_ignore_incomplete")

    if "raw_file" in st.session_state:
        raw_fname, raw_bytes = st.session_state["raw_file"]
        cache_key = ("raw_parsed", raw_fname, len(raw_bytes))
        if st.session_state.get("_raw_parsed_key") != cache_key:
            try:
                parsed = parse_reference_file(raw_bytes, raw_fname, ignore_incomplete=False)
                st.session_state["_raw_parsed"] = parsed
                st.session_state["_raw_parsed_key"] = cache_key
            except Exception as e:
                st.error(f"Could not parse file: {e}")
                st.session_state.pop("_raw_parsed", None)
                st.session_state.pop("_raw_parsed_key", None)

        if "_raw_parsed" in st.session_state:
            papers_ref = st.session_state["_raw_parsed"]
            n_total = len(papers_ref)
            n_abstract = sum(1 for p in papers_ref if p["abstract"])
            st.caption(f"{n_total} papers found - {n_abstract} with abstracts, {n_total - n_abstract} without.")
            if n_total > 0 and st.button("Use these papers", key="load_ref", type="primary"):
                papers_to_use = parse_reference_file(raw_bytes, raw_fname, ignore_incomplete=ignore)
                _dl = _papers_to_csv_bytes(papers_to_use)
                _clear_downstream_papers()
                st.session_state["_dl_papers"] = _dl
                st.rerun()

    st.divider()
    st.markdown("**Or upload a pre-formatted paper list**")
    st.caption("CSV with columns: id, title, abstract. Skips the format step entirely.")
    st.file_uploader(
        "Clean paper list",
        type=["csv"],
        key="ul_adv_papers",
        on_change=handle_panel_upload,
        args=("ul_adv_papers", "papers"),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if st.session_state.get("_upload_error_ul_adv_papers"):
        st.error(st.session_state["_upload_error_ul_adv_papers"])

    if "_dl_papers" in st.session_state:
        st.success(f"{len(parse_papers_csv(st.session_state['_dl_papers']))} papers ready.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 - EMBEDDINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("2. Embeddings", expanded=False):
    col_srv, col_colab = st.columns(2)

    with col_srv:
        st.markdown("**Generate on this server**")
        st.caption("Runs using SPECTER2. Papers without title or abstract are skipped.")
        if st.button("Generate embeddings", key="run_embed", disabled=_is_running, use_container_width=True):
            if "_dl_papers" not in st.session_state:
                st.error("No papers loaded. Upload a file in the Paper Input section first.")
            else:
                from pipeline.embed import embed_papers
                papers = parse_papers_csv(st.session_state["_dl_papers"])
                papers = [p for p in papers if p.get("title", "").strip() and p.get("abstract", "").strip()]
                st.session_state["running"] = True
                prog = st.progress(0, text="Loading SPECTER2 model...")
                def _cb(cur, tot):
                    prog.progress(cur / tot, text=f"Encoding {cur}/{tot}...")
                try:
                    embeddings = embed_papers(papers, progress_callback=_cb)
                    prog.progress(1.0, text="Done.")
                    ids = [p["id"] for p in papers]
                    st.session_state["step1_embeddings"] = embeddings
                    st.session_state["step1_embed_ids"] = ids
                finally:
                    st.session_state["running"] = False
                _clear_downstream_embeddings()
                st.session_state["_dl_embeddings"] = array_to_csv_bytes(embeddings, ids=ids)
                st.rerun()
        st.caption("Or upload an embeddings CSV:")
        st.file_uploader(
            "Embeddings CSV",
            type=["csv"],
            key="ul_adv_embed",
            on_change=handle_panel_upload,
            args=("ul_adv_embed", "embeddings"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_ul_adv_embed"):
            st.error(st.session_state["_upload_error_ul_adv_embed"])

    with col_colab:
        st.markdown("**Generate on Colab** (faster, requires a Google account)")
        st.caption("Runs on a free GPU. You need to move files manually.")
        _papers_dl_bytes = st.session_state.get("_dl_papers", b"")
        st.download_button(
            "1. Download papers CSV", _papers_dl_bytes, "papers.csv",
            mime="text/csv", key="dl_papers_for_colab",
            disabled=not bool(_papers_dl_bytes), use_container_width=True,
        )
        st.link_button(
            "2. Open Colab notebook",
            "https://colab.research.google.com/github/jpbascur/snipets/blob/main/generate_embeddings.ipynb",
            use_container_width=True,
        )
        st.caption("3. Upload the embeddings file you get from Colab:")
        st.file_uploader(
            "Embeddings CSV from Colab",
            type=["csv"],
            key="s2_embed_upload",
            on_change=handle_panel_upload,
            args=("s2_embed_upload", "embeddings"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_s2_embed_upload"):
            st.error(st.session_state["_upload_error_s2_embed_upload"])

    if "step1_embeddings" in st.session_state:
        emb = st.session_state["step1_embeddings"]
        st.success(f"{emb.shape[0]} papers embedded ({emb.shape[1]} dimensions).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 - CREATE MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("3. Create Map", expanded=False):
    st.markdown("**Network map**")
    st.caption("Builds a cosine similarity network - each paper connects to its most similar papers.")
    col_net_gen, col_net_up = st.columns(2)

    with col_net_gen:
        st.markdown("**Build from embeddings**")
        _net_top_k = st.slider(
            "Connections per paper (top_k)", min_value=1, max_value=50, value=20, step=1,
            key="net_top_k",
            help="How many nearest neighbours each paper is connected to.",
        )
        _net_min_sim = st.slider(
            "Minimum similarity", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            key="net_min_sim",
            help="Edges below this cosine similarity are dropped.",
        )
        _net_clustering = st.radio(
            "Clustering in VOSviewer",
            options=["Auto (VOSviewer detects clusters)", "None (all papers same color)"],
            key="net_clustering",
            help="Auto lets VOSviewer run its own clustering algorithm. None puts all papers in one group.",
        )
        if st.button("Build network map", key="run_network", disabled=_is_running, use_container_width=True):
            if "step1_embeddings" not in st.session_state:
                st.error("No embeddings loaded. Generate or upload embeddings in the Embeddings section first.")
            else:
                embed_src = st.session_state["step1_embeddings"]
                from pipeline.network import build_edge_list
                st.session_state["running"] = True
                prog = st.progress(0, text="Building network...")
                def _cb_net(cur, tot):
                    prog.progress(cur / tot, text=f"Processing {cur}/{tot} papers...")
                try:
                    with st.spinner("Computing similarities..."):
                        edges = build_edge_list(
                            embed_src, top_k=_net_top_k, min_similarity=_net_min_sim, progress_callback=_cb_net,
                        )
                    prog.progress(1.0, text="Done.")
                    _clear_downstream_edges()
                    st.session_state["step2_edges"] = edges
                    edges_csv = _edges_to_csv_bytes(edges)
                    st.session_state["_dl_edges"] = edges_csv
                    _clustering_val = "none" if _net_clustering.startswith("None") else "auto"
                    sid = st.session_state["session_id"]
                    vos_data = _build_vos_network_json(st.session_state["_dl_papers"], edges_csv, clustering=_clustering_val)
                    st.session_state["vos_json"]     = json.dumps(vos_data, indent=2)
                    st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", vos_data)
                finally:
                    st.session_state["running"] = False
                st.rerun()

    with col_net_up:
        st.markdown("**Upload pre-built files**")
        st.caption("Edges CSV (columns: source, target, weight):")
        st.file_uploader(
            "Edges CSV",
            type=["csv"],
            key="ul_adv_edges",
            on_change=handle_panel_upload,
            args=("ul_adv_edges", "edges"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_ul_adv_edges"):
            st.error(st.session_state["_upload_error_ul_adv_edges"])
        st.caption("Or a VOSviewer network JSON:")
        st.file_uploader(
            "VOSviewer network JSON",
            type=["json"],
            key="ul_adv_network_vos",
            on_change=handle_panel_upload,
            args=("ul_adv_network_vos", "network_vos"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_ul_adv_network_vos"):
            st.error(st.session_state["_upload_error_ul_adv_network_vos"])

    if "vos_json" in st.session_state:
        st.success(f"Network map ready - {len(st.session_state['step2_edges'])} connections.")
        vos_url = _vosviewer_url(st.session_state.get("vos_json_url"), max_n_links=0)
        if vos_url:
            st.link_button("Open in VOSviewer Online", vos_url, type="primary", use_container_width=True)
        else:
            st.caption("Download the VOSviewer network file from the Files panel below and open it in VOSviewer Online manually.")

    st.divider()
    st.markdown("**Coordinate map**")
    st.caption(
        "Projects papers into 2D space using UMAP - similar papers end up close together. "
        "Good for seeing the overall semantic landscape."
    )
    col_coord_gen, col_coord_up = st.columns(2)

    with col_coord_gen:
        st.markdown("**Build from embeddings**")
        _umap_n_neighbors = st.slider(
            "n_neighbors", min_value=2, max_value=100, value=15, step=1,
            key="umap_n_neighbors",
            help="Controls how much local vs global structure UMAP preserves. Lower = more local clusters.",
        )
        _umap_min_dist = st.slider(
            "min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
            key="umap_min_dist",
            help="Minimum distance between points in the 2D layout. Lower = tighter clusters.",
        )
        if st.button("Build coordinate map", key="run_umap", disabled=_is_running, use_container_width=True):
            if "step1_embeddings" not in st.session_state:
                st.error("No embeddings loaded. Generate or upload embeddings in the Embeddings section first.")
            else:
                from pipeline.reduce import umap_reduce
                _adv_embed_src = st.session_state["step1_embeddings"]
                n = len(_adv_embed_src)
                estimate = "a few seconds" if n < 500 else "~30 seconds" if n < 2000 else "a few minutes"
                viz_embed_ids = (st.session_state.get("step1_embed_ids")
                                 or [p["id"] for p in parse_papers_csv(st.session_state["_dl_papers"])])
                st.session_state["running"] = True
                try:
                    with st.spinner(f"Running UMAP on {n} papers... ({estimate})"):
                        coords = umap_reduce(_adv_embed_src, n_neighbors=_umap_n_neighbors, min_dist=_umap_min_dist)
                    _clear_downstream_coords()
                    st.session_state["viz_coords"] = coords
                    st.session_state["viz_ids"]    = viz_embed_ids
                    coords_csv = _coords_to_csv_bytes(viz_embed_ids, coords)
                    st.session_state["_dl_coords"] = coords_csv
                    sid = st.session_state["session_id"]
                    vos_data = _build_vos_coords_json(st.session_state["_dl_papers"], coords_csv)
                    st.session_state["viz_vos_json"]     = json.dumps(vos_data, indent=2)
                    st.session_state["viz_vos_json_url"] = _write_static_json(f"{sid}_umap.json", vos_data)
                finally:
                    st.session_state["running"] = False
                st.rerun()

    with col_coord_up:
        st.markdown("**Upload pre-built files**")
        st.caption("Coordinates CSV (columns: id, x, y):")
        st.file_uploader(
            "Coordinates CSV",
            type=["csv"],
            key="ul_adv_coords",
            on_change=handle_panel_upload,
            args=("ul_adv_coords", "coords"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_ul_adv_coords"):
            st.error(st.session_state["_upload_error_ul_adv_coords"])
        st.caption("Or a VOSviewer coordinates JSON:")
        st.file_uploader(
            "VOSviewer coordinates JSON",
            type=["json"],
            key="ul_adv_coords_vos",
            on_change=handle_panel_upload,
            args=("ul_adv_coords_vos", "coords_vos"),
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if st.session_state.get("_upload_error_ul_adv_coords_vos"):
            st.error(st.session_state["_upload_error_ul_adv_coords_vos"])

    if "viz_vos_json" in st.session_state:
        st.success(f"Coordinate map ready - {len(st.session_state['viz_coords'])} papers.")
        viz_vos_url = _vosviewer_url(st.session_state.get("viz_vos_json_url"))
        if viz_vos_url:
            st.link_button("Open in VOSviewer Online", viz_vos_url, type="primary", use_container_width=True)
        else:
            st.caption("Download the VOSviewer coordinates file from the Files panel below and open it in VOSviewer Online manually.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILES PANEL (advanced)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("Files tracker", expanded=False):
    st.caption("All files in your current session. Upload or download any file individually.")

    def _file_slot_state(slot_key: str) -> tuple[bool, str, bytes, str, str]:
        if slot_key == "raw":
            loaded = "raw_file" in st.session_state
            filename = st.session_state["raw_file"][0] if loaded else "export"
            data = st.session_state["raw_file"][1] if loaded else b""
            return loaded, filename if loaded else "-", data, filename, "application/octet-stream"
        if slot_key == "papers":
            loaded = "_dl_papers" in st.session_state and bool(st.session_state["_dl_papers"])
            status = f"{len(parse_papers_csv(st.session_state['_dl_papers']))} papers" if loaded else "-"
            return loaded, status, st.session_state.get("_dl_papers", b""), "papers.csv", "text/csv"
        if slot_key == "embeddings":
            loaded = "step1_embeddings" in st.session_state
            status = (
                f"{st.session_state['step1_embeddings'].shape[0]} x {st.session_state['step1_embeddings'].shape[1]}"
                if loaded else "-"
            )
            return loaded, status, st.session_state.get("_dl_embeddings", b""), "embeddings.csv", "text/csv"
        if slot_key == "edges":
            loaded = "step2_edges" in st.session_state
            status = f"{len(st.session_state['step2_edges'])} edges" if loaded else "-"
            return loaded, status, st.session_state.get("_dl_edges", b""), "network.csv", "text/csv"
        if slot_key == "network_vos":
            loaded = "vos_json" in st.session_state
            data = st.session_state["vos_json"].encode() if loaded else b""
            return loaded, "Ready" if loaded else "-", data, "vosviewer_network.json", "application/json"
        if slot_key == "coords":
            loaded = "viz_coords" in st.session_state
            status = f"{len(st.session_state['viz_coords'])} points" if loaded else "-"
            return loaded, status, st.session_state.get("_dl_coords", b""), "coords.csv", "text/csv"
        if slot_key == "coords_vos":
            loaded = "viz_vos_json" in st.session_state
            data = st.session_state["viz_vos_json"].encode() if loaded else b""
            return loaded, "Ready" if loaded else "-", data, "vosviewer_umap.json", "application/json"
        raise ValueError(f"Unknown file slot: {slot_key}")

    _FILE_SLOTS = [
        ("raw",               "Reference export",                 "dl_panel_raw",           "ul_panel_raw",           ["txt", "nbib", "ris", "bib", "xlsx"]),
        ("papers",            "Clean paper list",                 "dl_panel_papers",         "ul_panel_papers",        ["csv"]),
        ("embeddings",        "Embeddings",                       "dl_panel_embed",          "ul_panel_embed",         ["csv"]),
        ("edges",             "Paper network",                    "dl_panel_edges",          "ul_panel_edges",         ["csv"]),
        ("network_vos", "VOSviewer network map", "dl_panel_net_vos", "ul_panel_net_vos", ["json"]),
        ("coords",            "Paper coordinates",                "dl_panel_coords",         "ul_panel_coords",        ["csv"]),
        ("coords_vos",        "VOSviewer coordinate map",         "dl_panel_coords_vos",     "ul_panel_coords_vos",    ["json"]),
    ]

    with st.container(border=True):
        for i, (slot_key, label, dl_key, ul_key, ul_types) in enumerate(_FILE_SLOTS):
            loaded, status, dl_data, dl_fn, dl_mime = _file_slot_state(slot_key)
            col_label, col_dl, col_ul = st.columns([4.5, 1.2, 2.6])
            with col_label:
                st.markdown(f"**{label}**")
                st.caption(status)
            with col_dl:
                st.download_button(
                    "Download", dl_data, dl_fn, mime=dl_mime,
                    key=dl_key, disabled=not loaded, use_container_width=True,
                )
            with col_ul:
                if ul_key is not None:
                    st.file_uploader(
                        "Upload", type=ul_types, key=ul_key,
                        accept_multiple_files=False,
                        on_change=handle_panel_upload,
                        args=(ul_key, slot_key),
                        label_visibility="collapsed",
                    )
                    upload_error = st.session_state.get(f"_upload_error_{ul_key}")
                    if upload_error:
                        st.error(upload_error)
            if i < len(_FILE_SLOTS) - 1:
                st.divider()
