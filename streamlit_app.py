"""
TSM - Text Similarity Maker
Streamlit pipeline for building VOSviewer-compatible paper networks.
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
st.markdown(
    "Upload your papers and get a text similarity science map. "
    "Use the One click map for the simplest experience, "
    "or the step-by-step sections below for more control."
)
st.markdown(
    "**Supported input formats:**\n\n"
    "- **RIS** (.ris) — exported by Scopus, Web of Science, Zotero, Mendeley, EndNote\n"
    "- **BibTeX** (.bib) — exported by Google Scholar, Zotero, and most reference managers\n"
    "- **PubMed** (.txt) — from PubMed: click *Send to* → *File* → Format: *PubMed*\n"
    "- **PubMed Citation Manager** (.nbib) — from PubMed: click *Send to* → *Citation Manager*\n"
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
    file_obj = st.session_state.get(upload_key)
    if file_obj is None:
        return
    uploaded = uploaded_file_bytes(upload_key, file_obj)
    if uploaded is None:
        return
    raw, _signature = uploaded
    error_key = f"_upload_error_{upload_key}"
    try:
        if slot_key == "raw":
            st.session_state["raw_file"] = (file_obj.name, raw)
            st.session_state.pop("_raw_parsed", None)
            st.session_state.pop("_raw_parsed_key", None)
        elif slot_key == "papers":
            parsed = parse_papers_csv(raw)
            st.session_state["step1_papers"] = parsed
            st.session_state["s1_file"] = (file_obj.name, raw)
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
        elif slot_key == "network_vos":
            json_str = raw.decode("utf-8")
            data = json.loads(json_str)
            sid = st.session_state["session_id"]
            st.session_state["vos_json_auto"] = json_str
            st.session_state["vos_json_fixed"] = json_str
            st.session_state["vos_json_url_auto"] = _write_static_json(f"{sid}_network_auto.json", data)
            st.session_state["vos_json_url_fixed"] = _write_static_json(f"{sid}_network_fixed.json", data)
        elif slot_key == "coords":
            st.session_state["viz_ids"], st.session_state["viz_coords"] = parse_coords_csv(raw)
            _clear_downstream_coords()
            st.session_state["_dl_coords"] = raw
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
                "viz_coords", "viz_ids", "vos_json_auto", "vos_json_fixed",
                "vos_json_url_auto", "vos_json_url_fixed",
                "viz_vos_json", "viz_vos_json_url",
                "_dl_embeddings", "_dl_edges", "_dl_coords"]:
        st.session_state.pop(key, None)

def _clear_downstream_embeddings():
    for key in ["step2_edges", "viz_coords", "viz_ids",
                "vos_json_auto", "vos_json_fixed",
                "vos_json_url_auto", "vos_json_url_fixed",
                "viz_vos_json", "viz_vos_json_url",
                "_dl_embeddings", "_dl_edges", "_dl_coords"]:
        st.session_state.pop(key, None)

def _clear_downstream_edges():
    for key in ["vos_json_auto", "vos_json_fixed",
                "vos_json_url_auto", "vos_json_url_fixed",
                "_dl_edges"]:
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

try:
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
    _mem_used, _mem_limit = _read_cgroup_mem()
    _mem_used_gb  = _mem_used  / 1024**3
    _mem_total_gb = _mem_limit / 1024**3
    _mem_free_gb  = (_mem_limit - _mem_used) / 1024**3
    _mem_pct      = _mem_used / _mem_limit * 100
    _mem_caption  = (
        f"Container memory: {_mem_used_gb:.1f} GB used / {_mem_total_gb:.1f} GB total - "
        f"{_mem_free_gb:.1f} GB free ({100 - _mem_pct:.0f}% available). "
        "This tool has limited memory shared across all users. "
        "If memory is low, please wait for it to be released by another user."
    )
except Exception:
    _mem_caption = None

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
        args=("ocm_raw_upload", "raw"),
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if st.session_state.get("_upload_error_ocm_raw_upload"):
        st.error(st.session_state["_upload_error_ocm_raw_upload"])

    if "raw_file" in st.session_state:
        _ocm_fname, _ocm_bytes = st.session_state["raw_file"]
        _ocm_cache_key = ("raw_parsed", _ocm_fname, len(_ocm_bytes))
        if st.session_state.get("_raw_parsed_key") != _ocm_cache_key:
            try:
                _ocm_ext = Path(_ocm_fname).suffix.lower()
                if _ocm_ext == ".xlsx":
                    _ocm_parsed = parse_excel_export(_ocm_bytes)
                elif _ocm_ext == ".bib":
                    _ocm_parsed = parse_bibtex_export(_ocm_bytes.decode("utf-8", errors="replace"))
                elif _ocm_ext == ".ris":
                    _ocm_parsed = parse_ris_export(_ocm_bytes.decode("utf-8", errors="replace"))
                else:
                    _ocm_parsed = parse_pubmed_export(_ocm_bytes.decode("utf-8", errors="replace"))
                st.session_state["_raw_parsed"] = _ocm_parsed
                st.session_state["_raw_parsed_key"] = _ocm_cache_key
            except Exception as _ocm_exc:
                st.error(f"Could not parse file: {_ocm_exc}")
                st.session_state.pop("_raw_parsed", None)
                st.session_state.pop("_raw_parsed_key", None)

        if "_raw_parsed" in st.session_state:
            _ocm_refs = st.session_state["_raw_parsed"]
            _ocm_n = len(_ocm_refs)
            _ocm_n_abs = sum(1 for p in _ocm_refs if p["abstract"])
            st.caption(f"{_ocm_n} papers found - {_ocm_n_abs} with abstracts, {_ocm_n - _ocm_n_abs} without.")
            if _ocm_n > 0 and st.button(
                "Create map", key="ocm_create", type="primary",
                disabled=_is_running, use_container_width=True,
            ):
                from pipeline.embed import embed_papers
                from pipeline.network import build_edge_list
                _ocm_papers = [
                    p for p in _ocm_refs
                    if p.get("title", "").strip() and p.get("abstract", "").strip()
                ]
                if not _ocm_papers:
                    st.error("No papers with both title and abstract found.")
                else:
                    st.session_state["running"] = True
                    try:
                        _ocm_dl = _papers_to_csv_bytes(_ocm_papers)
                        st.session_state["step1_papers"] = _ocm_papers
                        st.session_state["s1_file"] = ("papers.csv", _ocm_dl)
                        _clear_downstream_papers()
                        st.session_state["_dl_papers"] = _ocm_dl

                        _ocm_prog = st.progress(0, text="Loading SPECTER2 model...")
                        def _ocm_cb(cur, tot):
                            _ocm_prog.progress(cur / tot, text=f"Encoding {cur}/{tot}...")
                        _ocm_emb = embed_papers(_ocm_papers, progress_callback=_ocm_cb)
                        _ocm_prog.progress(1.0, text="Embeddings done.")
                        _ocm_ids = [p["id"] for p in _ocm_papers]
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
                        st.session_state["_dl_edges"] = _edges_to_csv_bytes(_ocm_edges)

                        _ocm_items = [
                            {"id": str(idx + 1), "label": p.get("title", p.get("id", str(idx + 1))),
                             "description": str(p.get("id", ""))}
                            for idx, p in enumerate(_ocm_papers)
                        ]
                        _ocm_links = [
                            {"source_id": str(i + 1), "target_id": str(j + 1), "strength": round(w, 6)}
                            for i, j, w in _ocm_edges
                        ]
                        _ocm_sid = st.session_state["session_id"]
                        _ocm_vos_auto  = {"network": {"items": _ocm_items, "links": _ocm_links}}
                        _ocm_vos_fixed = {"network": {"items": [{**_it, "cluster": 1} for _it in _ocm_items], "links": _ocm_links}}
                        st.session_state["vos_json_auto"]      = json.dumps(_ocm_vos_auto,  indent=2)
                        st.session_state["vos_json_fixed"]     = json.dumps(_ocm_vos_fixed, indent=2)
                        st.session_state["vos_json_url_auto"]  = _write_static_json(f"{_ocm_sid}_network_auto.json",  _ocm_vos_auto)
                        st.session_state["vos_json_url_fixed"] = _write_static_json(f"{_ocm_sid}_network_fixed.json", _ocm_vos_fixed)
                    except Exception as _ocm_err:
                        st.error(f"Error creating map: {_ocm_err}")
                    finally:
                        st.session_state["running"] = False
                    st.rerun()

    if "vos_json_auto" in st.session_state:
        st.success(
            f"Map ready - {len(st.session_state['step1_papers'])} papers, "
            f"{len(st.session_state['step2_edges'])} connections."
        )
        _ocm_vos_url = _vosviewer_url(st.session_state.get("vos_json_url_auto"), max_n_links=0)
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

    if "raw_file" in st.session_state:
        raw_fname, raw_bytes = st.session_state["raw_file"]
        cache_key = ("raw_parsed", raw_fname, len(raw_bytes))
        if st.session_state.get("_raw_parsed_key") != cache_key:
            try:
                ext = Path(raw_fname).suffix.lower()
                if ext == ".xlsx":
                    parsed = parse_excel_export(raw_bytes)
                elif ext == ".bib":
                    parsed = parse_bibtex_export(raw_bytes.decode("utf-8", errors="replace"))
                elif ext == ".ris":
                    parsed = parse_ris_export(raw_bytes.decode("utf-8", errors="replace"))
                else:
                    parsed = parse_pubmed_export(raw_bytes.decode("utf-8", errors="replace"))
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
            ignore = st.checkbox("Ignore papers without title or abstract", value=True, key="s1_ignore_incomplete")
            if n_total > 0 and st.button("Use these papers", key="load_ref", type="primary"):
                papers_to_use = papers_ref
                if ignore:
                    papers_to_use = [p for p in papers_ref
                                     if p.get("title", "").strip() and p.get("abstract", "").strip()]
                _dl = _papers_to_csv_bytes(papers_to_use)
                st.session_state["step1_papers"] = papers_to_use
                st.session_state["s1_file"] = ("papers.csv", _dl)
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

    if "step1_papers" in st.session_state:
        st.success(f"{len(st.session_state['step1_papers'])} papers ready.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 - EMBEDDINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("2. Embeddings", expanded=False):
    if "step1_papers" not in st.session_state:
        st.info("No papers in the tool yet. Upload a file above.")
    else:
        st.caption("Runs on this server using SPECTER2. Papers without title or abstract are skipped.")
        if _mem_caption:
            st.caption(_mem_caption)
        if st.button("Generate embeddings", key="run_embed", disabled=_is_running, use_container_width=True):
            from pipeline.embed import embed_papers
            papers = [p for p in st.session_state["step1_papers"]
                      if p.get("title", "").strip() and p.get("abstract", "").strip()]
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

    st.divider()
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

    st.divider()
    st.markdown("**Upload embeddings directly**")
    st.caption("Skip generation entirely by uploading your own embeddings CSV.")
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

    if "step1_embeddings" in st.session_state:
        emb = st.session_state["step1_embeddings"]
        st.success(f"{emb.shape[0]} papers embedded ({emb.shape[1]} dimensions).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 - CREATE MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("3. Create Map", expanded=False):
    if "step1_embeddings" not in st.session_state:
        st.info("No embeddings in the tool yet. Generate or upload them above.")
    else:
        embed_src = st.session_state["step1_embeddings"]
        papers_src = st.session_state["step1_papers"]

        st.caption("Builds a cosine similarity network - each paper connects to its most similar papers.")
        if st.button("Build network map", key="run_network", disabled=_is_running, use_container_width=True):
            from pipeline.network import build_edge_list
            st.session_state["running"] = True
            prog = st.progress(0, text="Building network...")
            def _cb_net(cur, tot):
                prog.progress(cur / tot, text=f"Processing {cur}/{tot} papers...")
            try:
                with st.spinner("Computing similarities..."):
                    edges = build_edge_list(
                        embed_src, top_k=20, min_similarity=0.0, progress_callback=_cb_net,
                    )
                prog.progress(1.0, text="Done.")
                _clear_downstream_edges()
                st.session_state["step2_edges"] = edges
                st.session_state["_dl_edges"] = _edges_to_csv_bytes(edges)
                items = [
                    {"id": str(idx + 1), "label": p.get("title", p.get("id", str(idx + 1))),
                     "description": str(p.get("id", ""))}
                    for idx, p in enumerate(papers_src)
                ]
                links = [
                    {"source_id": str(i + 1), "target_id": str(j + 1), "strength": round(w, 6)}
                    for i, j, w in edges
                ]
                sid = st.session_state["session_id"]
                vos_data_auto  = {"network": {"items": items, "links": links}}
                vos_data_fixed = {"network": {"items": [{**item, "cluster": 1} for item in items], "links": links}}
                st.session_state["vos_json_auto"]      = json.dumps(vos_data_auto,  indent=2)
                st.session_state["vos_json_fixed"]     = json.dumps(vos_data_fixed, indent=2)
                st.session_state["vos_json_url_auto"]  = _write_static_json(f"{sid}_network_auto.json",  vos_data_auto)
                st.session_state["vos_json_url_fixed"] = _write_static_json(f"{sid}_network_fixed.json", vos_data_fixed)
            finally:
                st.session_state["running"] = False
            st.rerun()

        if "vos_json_auto" in st.session_state:
            st.success(f"Network map ready - {len(st.session_state['step2_edges'])} connections.")
            vos_url = _vosviewer_url(st.session_state.get("vos_json_url_auto"), max_n_links=0)
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
    if "step1_embeddings" not in st.session_state:
        st.info("No embeddings in the tool yet.")
    else:
        _adv_embed_src = st.session_state["step1_embeddings"]
        _adv_papers_src = st.session_state["step1_papers"]
        n = len(_adv_embed_src)
        estimate = "a few seconds" if n < 500 else "~30 seconds" if n < 2000 else "a few minutes"
        st.caption(f"Expected time: {estimate} for {n} papers.")
        if st.button("Build coordinate map", key="run_umap", disabled=_is_running, use_container_width=True):
            from pipeline.reduce import umap_reduce
            viz_embed_ids = (st.session_state.get("step1_embed_ids")
                             or [p["id"] for p in _adv_papers_src])
            st.session_state["running"] = True
            try:
                with st.spinner(f"Running UMAP on {n} papers... ({estimate})"):
                    coords = umap_reduce(_adv_embed_src, n_neighbors=15, min_dist=0.1)
                _clear_downstream_coords()
                st.session_state["viz_coords"] = coords
                st.session_state["viz_ids"]    = viz_embed_ids
                st.session_state["_dl_coords"] = _coords_to_csv_bytes(viz_embed_ids, coords)
                coord_lookup = {pid: (float(x), float(y)) for pid, (x, y) in zip(viz_embed_ids, coords)}
                items = []
                for idx, paper in enumerate(_adv_papers_src):
                    pid = str(paper.get("id", idx + 1))
                    x, y = coord_lookup.get(pid, (0.0, 0.0))
                    items.append({
                        "id": str(idx + 1), "label": paper.get("title", pid),
                        "description": pid, "x": round(x, 6), "y": round(y, 6), "cluster": 1,
                    })
                vos_data = {"network": {"items": items, "links": []}}
                sid = st.session_state["session_id"]
                st.session_state["viz_vos_json"]     = json.dumps(vos_data, indent=2)
                st.session_state["viz_vos_json_url"] = _write_static_json(f"{sid}_umap.json", vos_data)
            finally:
                st.session_state["running"] = False
            st.rerun()

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
            loaded = "step1_papers" in st.session_state
            status = f"{len(st.session_state['step1_papers'])} papers" if loaded else "-"
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
            loaded = "vos_json_auto" in st.session_state
            data = st.session_state["vos_json_auto"].encode() if loaded else b""
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
        ("raw",         "Reference export",          "dl_panel_raw",       "ul_panel_raw",       ["txt", "nbib", "ris", "bib", "xlsx"]),
        ("papers",      "Clean paper list",          "dl_panel_papers",    "ul_panel_papers",    ["csv"]),
        ("embeddings",  "Embeddings",                "dl_panel_embed",     "ul_panel_embed",     ["csv"]),
        ("edges",       "Paper network",             "dl_panel_edges",     "ul_panel_edges",     ["csv"]),
        ("network_vos", "VOSviewer network map",     "dl_panel_net_vos",   "ul_panel_net_vos",   ["json"]),
        ("coords",      "Paper coordinates",         "dl_panel_coords",    "ul_panel_coords",    ["csv"]),
        ("coords_vos",  "VOSviewer coordinate map",  "dl_panel_coords_vos","ul_panel_coords_vos",["json"]),
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
