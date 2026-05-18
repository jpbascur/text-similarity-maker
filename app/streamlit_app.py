"""
TSM - Text Similarity Maker
Streamlit UI — all pipeline logic lives in pipeline.py.

Session state model
-------------------
All pipeline state is stored as file bytes under ``_dl_*`` keys.
There are no in-memory arrays or parsed lists in session state —
any count or shape shown in the UI is derived by parsing the bytes on demand.

``_dl_papers``     — papers CSV bytes (columns: id, title, abstract)
``_dl_embeddings`` — embeddings CSV bytes (no header; first column = paper id)
``_dl_edges``      — network CSV bytes (columns: source, target, weight)
``_dl_coords``     — coordinates CSV bytes (columns: id, x, y)
``vos_json``       — VOSviewer network JSON string (for download)
``vos_json_url``   — public URL of the hosted network JSON (for VOSviewer Online)
``viz_vos_json``   — VOSviewer coordinate JSON string (for download)
``viz_vos_json_url`` — public URL of the hosted coordinate JSON

Display caches (keyed by file identity, never fed into the pipeline):
``raw_file``        — (filename, bytes) of the last reference export upload
``ocm_raw_file``    — same, for the OCM uploader
``_raw_parsed``     — parsed paper list for Step-1 preview
``ocm_raw_parsed``  — parsed paper list for OCM preview
"""
import csv
import hashlib
import io
import json
import os
import time as _time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from pipeline import (
    build_edge_list,
    build_vos_coords_json,
    build_vos_network_json,
    embed_papers,
    load_model,
    parse_coords_csv,
    parse_edge_csv,
    parse_embeddings_csv,
    parse_papers_csv,
    parse_reference_file,
    umap_reduce,
)


_MODEL_TTL = 3600

@st.cache_resource
def _model_container() -> dict:
    return {"tokenizer": None, "model": None, "loaded_at": None}

def _evict_model_if_stale():
    c = _model_container()
    if c["loaded_at"] is not None and _time.time() - c["loaded_at"] >= _MODEL_TTL:
        c["tokenizer"] = None
        c["model"] = None
        c["loaded_at"] = None

def _get_model():
    _evict_model_if_stale()
    c = _model_container()
    if c["model"] is None:
        c["tokenizer"], c["model"] = load_model()
        c["loaded_at"] = _time.time()
    return c["tokenizer"], c["model"]


def papers_to_csv_bytes(papers: list[dict]) -> bytes:
    """Serialise a list of paper dicts to CSV bytes (columns: id, title, abstract)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["id", "title", "abstract"])
    w.writeheader()
    w.writerows(papers)
    return buf.getvalue().encode()

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
    with st.expander("Server memory", expanded=True):
        _evict_model_if_stale()
        _specter_is_loaded = _model_container()["model"] is not None
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("RAM used", f"{_mem_used_gb:.1f} GB")
        col_m2.metric("RAM free", f"{_mem_free_gb:.1f} GB")
        col_m3.metric("RAM total", f"{_mem_total_gb:.1f} GB")
        col_m4.metric("SPECTER2", "Loaded" if _specter_is_loaded else "Not loaded")
        st.progress(_mem_pct / 100)
        st.caption("Memory is shared across all users. If memory is low, please wait for it to be released by another user.")
st.markdown(
    "Upload your papers and get a text similarity science map. "
    "Use the One click map for the simplest experience, "
    "or the step-by-step sections below for more control and speed. "
    "If you get stuck, reload."
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
        "Abstracts improve embedding quality — papers with only a title produce weaker vectors. "
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
        "Papers imported this way will have no abstract, which reduces embedding quality. "
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

# ── UI helpers ────────────────────────────────────────────────────────────────

def _clear_downstream_papers():
    for key in ["_dl_embeddings", "_dl_edges", "_dl_coords",
                "vos_json", "vos_json_url", "viz_vos_json", "viz_vos_json_url"]:
        st.session_state.pop(key, None)

def _clear_downstream_embeddings():
    # Called before replacing embeddings — clears old embeddings and everything derived from them.
    for key in ["_dl_embeddings", "_dl_edges", "_dl_coords",
                "vos_json", "vos_json_url", "viz_vos_json", "viz_vos_json_url"]:
        st.session_state.pop(key, None)

def _clear_downstream_edges():
    for key in ["_dl_edges", "vos_json", "vos_json_url"]:
        st.session_state.pop(key, None)

def _clear_downstream_coords():
    for key in ["_dl_coords", "viz_vos_json", "viz_vos_json_url"]:
        st.session_state.pop(key, None)

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
            parse_embeddings_csv(raw)  # validate
            _clear_downstream_embeddings()
            st.session_state["_dl_embeddings"] = raw
        elif slot_key == "edges":
            parse_edge_csv(raw)  # validate
            _clear_downstream_edges()
            st.session_state["_dl_edges"] = raw
            if "_dl_papers" in st.session_state:
                _clustering_val = "none" if st.session_state.get("net_clustering", "").startswith("None") else "auto"
                sid = st.session_state["session_id"]
                vos_data = build_vos_network_json(st.session_state["_dl_papers"], raw, clustering=_clustering_val)
                st.session_state["vos_json"]     = json.dumps(vos_data, indent=2)
                st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", vos_data)
        elif slot_key == "network_vos":
            json_str = raw.decode("utf-8")
            data = json.loads(json_str)
            sid = st.session_state["session_id"]
            st.session_state["vos_json"] = json_str
            st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", data)
        elif slot_key == "coords":
            parse_coords_csv(raw)  # validate
            _clear_downstream_coords()
            st.session_state["_dl_coords"] = raw
            if "_dl_papers" in st.session_state:
                sid = st.session_state["session_id"]
                vos_data = build_vos_coords_json(st.session_state["_dl_papers"], raw)
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

# ── session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
if "running" not in st.session_state:
    st.session_state["running"] = False

_is_running = st.session_state["running"]
if _is_running:
    st.warning("A job is already running in this session. Please wait for it to finish.")
if "_job_error" in st.session_state:
    st.error(st.session_state.pop("_job_error"))

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

# Housekeeping on every page load.
_now = _time.time()
for _f in _STATIC_DIR.glob("*.json"):      # static JSON files older than 24 h
    if _now - _f.stat().st_mtime > 86400:
        _f.unlink(missing_ok=True)


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
                _ocm_papers = parse_reference_file(_ocm_bytes, _ocm_fname, ignore_incomplete=False)
                if not _ocm_papers:
                    st.error("No papers found in file.")
                else:
                    st.session_state["running"] = True
                    try:
                        _ocm_dl = papers_to_csv_bytes(_ocm_papers)
                        _clear_downstream_papers()
                        st.session_state["_dl_papers"] = _ocm_dl

                        _evict_model_if_stale()
                        _ocm_init_text = "Encoding papers..." if _model_container()["model"] is not None else "Loading SPECTER2 model — this happens only once per session and takes about 30 seconds..."
                        _ocm_prog = st.progress(0, text=_ocm_init_text)
                        _ocm_tokenizer, _ocm_model = _get_model()
                        def _ocm_cb(cur, tot):
                            _ocm_prog.progress(cur / tot, text=f"Encoding {cur}/{tot}...")
                        _ocm_emb_bytes = embed_papers(_ocm_dl, _ocm_tokenizer, _ocm_model, progress_callback=_ocm_cb)
                        _ocm_prog.progress(1.0, text="Embeddings done.")
                        _clear_downstream_embeddings()
                        st.session_state["_dl_embeddings"] = _ocm_emb_bytes

                        _ocm_prog2 = st.progress(0, text="Building network...")
                        def _ocm_cb_net(cur, tot):
                            _ocm_prog2.progress(cur / tot, text=f"Processing {cur}/{tot} papers...")
                        with st.spinner("Computing similarities..."):
                            _ocm_edges_bytes = build_edge_list(
                                _ocm_emb_bytes, top_k=20, min_similarity=0.0, progress_callback=_ocm_cb_net,
                            )
                        _ocm_prog2.progress(1.0, text="Network done.")
                        _clear_downstream_edges()
                        st.session_state["_dl_edges"] = _ocm_edges_bytes

                        _ocm_sid = st.session_state["session_id"]
                        _ocm_vos = build_vos_network_json(_ocm_dl, _ocm_edges_bytes, clustering="auto")
                        st.session_state["vos_json"]     = json.dumps(_ocm_vos, indent=2)
                        st.session_state["vos_json_url"] = _write_static_json(f"{_ocm_sid}_network.json", _ocm_vos)
                    except Exception as _ocm_err:
                        st.session_state["_job_error"] = f"Error creating map: {_ocm_err}"
                    finally:
                        st.session_state["running"] = False
                    st.rerun()

    if "vos_json" in st.session_state:
        _n_papers = len(parse_papers_csv(st.session_state["_dl_papers"])) if "_dl_papers" in st.session_state else 0
        _n_edges  = len(parse_edge_csv(st.session_state["_dl_edges"])) if "_dl_edges" in st.session_state else None
        _ocm_summary = f"{_n_papers} papers" + (f", {_n_edges} connections" if _n_edges is not None else "")
        st.success(f"Map ready — {_ocm_summary}.")
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
                _clear_downstream_papers()
                st.session_state["_dl_papers"] = papers_to_csv_bytes(papers_to_use)
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
        st.caption("Runs using SPECTER2. All papers are embedded, including those without an abstract.")
        if st.button("Generate embeddings", key="run_embed", disabled=_is_running, use_container_width=True):
            if "_dl_papers" not in st.session_state:
                st.error("No papers loaded. Upload a file in the Paper Input section first.")
            else:
                st.session_state["running"] = True
                _evict_model_if_stale()
                _init_text = "Encoding papers..." if _model_container()["model"] is not None else "Loading SPECTER2 model — this happens only once per session and takes about 30 seconds..."
                prog = st.progress(0, text=_init_text)
                def _cb(cur, tot):
                    prog.progress(cur / tot, text=f"Encoding {cur}/{tot}...")
                try:
                    _tokenizer, _model = _get_model()
                    _dl_emb = embed_papers(st.session_state["_dl_papers"], _tokenizer, _model, progress_callback=_cb)
                    prog.progress(1.0, text="Done.")
                    _clear_downstream_embeddings()
                    st.session_state["_dl_embeddings"] = _dl_emb
                except Exception as _e:
                    st.session_state["_job_error"] = f"Embedding failed: {_e}"
                finally:
                    st.session_state["running"] = False
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

    if "_dl_embeddings" in st.session_state:
        _emb_arr, _ = parse_embeddings_csv(st.session_state["_dl_embeddings"])
        st.success(f"{_emb_arr.shape[0]} papers embedded ({_emb_arr.shape[1]} dimensions).")


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
            if "_dl_embeddings" not in st.session_state:
                st.error("No embeddings loaded. Generate or upload embeddings in the Embeddings section first.")
            else:
                st.session_state["running"] = True
                prog = st.progress(0, text="Building network...")
                def _cb_net(cur, tot):
                    prog.progress(cur / tot, text=f"Processing {cur}/{tot} papers...")
                try:
                    with st.spinner("Computing similarities..."):
                        _edges_bytes = build_edge_list(
                            st.session_state["_dl_embeddings"],
                            top_k=_net_top_k, min_similarity=_net_min_sim, progress_callback=_cb_net,
                        )
                    prog.progress(1.0, text="Done.")
                    _clear_downstream_edges()
                    st.session_state["_dl_edges"] = _edges_bytes
                    _clustering_val = "none" if _net_clustering.startswith("None") else "auto"
                    sid = st.session_state["session_id"]
                    vos_data = build_vos_network_json(st.session_state["_dl_papers"], _edges_bytes, clustering=_clustering_val)
                    st.session_state["vos_json"]     = json.dumps(vos_data, indent=2)
                    st.session_state["vos_json_url"] = _write_static_json(f"{sid}_network.json", vos_data)
                except Exception as _e:
                    st.session_state["_job_error"] = f"Network build failed: {_e}"
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
        _edge_count = len(parse_edge_csv(st.session_state["_dl_edges"])) if "_dl_edges" in st.session_state else None
        _net_summary = f"{_edge_count} connections" if _edge_count is not None else "ready"
        st.success(f"Network map {_net_summary}.")
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
            if "_dl_embeddings" not in st.session_state:
                st.error("No embeddings loaded. Generate or upload embeddings in the Embeddings section first.")
            else:
                _emb_arr, _ = parse_embeddings_csv(st.session_state["_dl_embeddings"])
                n = len(_emb_arr)
                estimate = "a few seconds" if n < 500 else "~30 seconds" if n < 2000 else "a few minutes"
                st.session_state["running"] = True
                try:
                    with st.spinner(f"Running UMAP on {n} papers... ({estimate})"):
                        _coords_bytes = umap_reduce(
                            st.session_state["_dl_embeddings"],
                            n_neighbors=_umap_n_neighbors, min_dist=_umap_min_dist,
                        )
                    _clear_downstream_coords()
                    st.session_state["_dl_coords"] = _coords_bytes
                    sid = st.session_state["session_id"]
                    vos_data = build_vos_coords_json(st.session_state["_dl_papers"], _coords_bytes)
                    st.session_state["viz_vos_json"]     = json.dumps(vos_data, indent=2)
                    st.session_state["viz_vos_json_url"] = _write_static_json(f"{sid}_umap.json", vos_data)
                except Exception as _e:
                    st.session_state["_job_error"] = f"UMAP failed: {_e}"
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
        _n_coords, _ = parse_coords_csv(st.session_state["_dl_coords"])
        st.success(f"Coordinate map ready - {len(_n_coords)} papers.")
        viz_vos_url = _vosviewer_url(st.session_state.get("viz_vos_json_url"))
        if viz_vos_url:
            st.link_button("Open in VOSviewer Online", viz_vos_url, type="primary", use_container_width=True)
        else:
            st.caption("Download the VOSviewer coordinates file from the Files panel below and open it in VOSviewer Online manually.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILES PANEL
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
            loaded = "_dl_embeddings" in st.session_state and bool(st.session_state["_dl_embeddings"])
            if loaded:
                _arr, _ = parse_embeddings_csv(st.session_state["_dl_embeddings"])
                status = f"{_arr.shape[0]} x {_arr.shape[1]}"
            else:
                status = "-"
            return loaded, status, st.session_state.get("_dl_embeddings", b""), "embeddings.csv", "text/csv"
        if slot_key == "edges":
            loaded = "_dl_edges" in st.session_state and bool(st.session_state["_dl_edges"])
            status = f"{len(parse_edge_csv(st.session_state['_dl_edges']))} edges" if loaded else "-"
            return loaded, status, st.session_state.get("_dl_edges", b""), "network.csv", "text/csv"
        if slot_key == "network_vos":
            loaded = "vos_json" in st.session_state
            data = st.session_state["vos_json"].encode() if loaded else b""
            return loaded, "Ready" if loaded else "-", data, "vosviewer_network.json", "application/json"
        if slot_key == "coords":
            loaded = "_dl_coords" in st.session_state and bool(st.session_state["_dl_coords"])
            status = f"{len(parse_coords_csv(st.session_state['_dl_coords'])[0])} points" if loaded else "-"
            return loaded, status, st.session_state.get("_dl_coords", b""), "coords.csv", "text/csv"
        if slot_key == "coords_vos":
            loaded = "viz_vos_json" in st.session_state
            data = st.session_state["viz_vos_json"].encode() if loaded else b""
            return loaded, "Ready" if loaded else "-", data, "vosviewer_umap.json", "application/json"
        raise ValueError(f"Unknown file slot: {slot_key}")

    _FILE_SLOTS = [
        ("raw",         "Reference export",        "dl_panel_raw",       "ul_panel_raw",       ["txt", "nbib", "ris", "bib", "xlsx"]),
        ("papers",      "Clean paper list",         "dl_panel_papers",    "ul_panel_papers",    ["csv"]),
        ("embeddings",  "Embeddings",               "dl_panel_embed",     "ul_panel_embed",     ["csv"]),
        ("edges",       "Paper network",            "dl_panel_edges",     "ul_panel_edges",     ["csv"]),
        ("network_vos", "VOSviewer network map",    "dl_panel_net_vos",   "ul_panel_net_vos",   ["json"]),
        ("coords",      "Paper coordinates",        "dl_panel_coords",    "ul_panel_coords",    ["csv"]),
        ("coords_vos",  "VOSviewer coordinate map", "dl_panel_coords_vos","ul_panel_coords_vos",["json"]),
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
