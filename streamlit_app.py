"""
TSM — Text Similarity Maker
Streamlit pipeline for building VOSviewer-compatible paper networks.
"""

import csv
import io
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

# Skip GCP services (Firestore lock, GCS file hosting) in Colab and HF Spaces
_COLAB_MODE = os.environ.get("TSM_COLAB") == "1" or "SPACE_ID" in os.environ

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSM Text Similarity Maker",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 TSM — Text Similarity Maker")
st.header("Create a text similarity science map")
st.write("Generate embeddings of your documents' titles and abstracts — a numerical representation of their semantic content. Then use those embeddings to build a science map. Two map types are available: a text similarity network map, and an embedding space reduction map.")
st.markdown("Built by [Juan Pablo Bascur](https://jpbascur.com)")

# ── helpers ──────────────────────────────────────────────────────────────────

def array_to_csv_bytes(arr: np.ndarray, ids: list[str] | None = None) -> bytes:
    buf = io.StringIO()
    if ids is not None:
        for row_id, row in zip(ids, arr):
            buf.write(row_id + "," + ",".join(f"{v:.8f}" for v in row) + "\n")
    else:
        np.savetxt(buf, arr, delimiter=",", fmt="%.8f")
    return buf.getvalue().encode()

def csv_bytes_to_array(data: bytes) -> np.ndarray:
    """Parse an embeddings CSV. First column is always the paper ID and is dropped.
    No header row is expected or supported."""
    lines = [l for l in data.decode("utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError("The embeddings file is empty.")
    first_cells = lines[0].split(",")
    try:
        float(first_cells[1])
    except (ValueError, IndexError):
        raise ValueError(
            "The embeddings file appears to have a header row. "
            "This file must not have a header — the first row should be data."
        )
    rows = []
    for line in lines:
        cells = line.split(",")
        rows.append([float(c) for c in cells[1:]])
    arr = np.array(rows, dtype=np.float32)
    if arr.shape[1] != 768:
        raise ValueError(
            f"Expected 768 embedding dimensions per row, got {arr.shape[1]}. "
            "This file does not look like a SPECTER2 embeddings file."
        )
    return arr

def show_array_info(arr: np.ndarray, label: str = "Array"):
    st.caption(f"{label}: {arr.shape[0]} papers × {arr.shape[1]} dimensions")

def save_upload(file_obj, state_key: str):
    if file_obj is not None:
        st.session_state[state_key] = (file_obj.name, file_obj.read())

# Session-unique ID for static file naming (avoids collisions between users)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

if "running" not in st.session_state:
    st.session_state["running"] = False


_is_running = st.session_state["running"]
if _is_running:
    st.warning("A job is already running in this session. Please wait for it to finish.")

try:
    def _read_cgroup_mem():
        """Read container memory limits from cgroup (accurate inside Docker)."""
        # Try cgroups v2 first
        try:
            limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
            usage = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            return usage, limit
        except Exception:
            pass
        # Fall back to cgroups v1
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        usage = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
        return usage, limit

    _mem_used, _mem_limit = _read_cgroup_mem()
    _mem_used_gb  = _mem_used  / 1024**3
    _mem_total_gb = _mem_limit / 1024**3
    _mem_free_gb  = (_mem_limit - _mem_used) / 1024**3
    _mem_pct      = _mem_used / _mem_limit * 100
    _mem_caption  = (
        f"Container memory: {_mem_used_gb:.1f} GB used / {_mem_total_gb:.1f} GB total — "
        f"{_mem_free_gb:.1f} GB free ({100 - _mem_pct:.0f}% available). "
        "This tool has limited memory shared across all users. "
        "If memory is low, please wait for it to be released by another user."
    )
except Exception:
    _mem_caption = None

_STATIC_DIR = Path(__file__).parent / "static"
_GCS_BUCKET  = "tsm-app-static"
_GCS_BASE    = f"https://storage.googleapis.com/{_GCS_BUCKET}"

@st.cache_resource
def _gcs_client():
    from google.cloud import storage
    return storage.Client(project="text-similarity-maker")

def _upload_to_gcs(name: str, content: str) -> str:
    """Upload a text file to GCS and return its public URL."""
    client = _gcs_client()
    bucket = client.bucket(_GCS_BUCKET)
    blob   = bucket.blob(name)
    blob.upload_from_string(content, content_type="text/plain")
    return f"{_GCS_BASE}/{name}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROOT — EMBEDDINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("Text to Embeddings", expanded=True):
    st.subheader("Text to Embeddings")
    st.caption("Encodes each paper into a numerical vector that captures its semantic meaning. Uses [SPECTER2](https://github.com/allenai/SPECTER2) with the proximity adapter, a transformer model optimized to generate embeddings of paper titles and abstracts such that semantically similar papers end up with similar embeddings.")

    st.caption("Requirements: CSV columns: id, title, abstract. Avoid including papers without a title or abstract — they will produce poor-quality embeddings.")
    save_upload(
        st.file_uploader("Upload papers file", type=["csv"], key="s1_upload"),
        "s1_file",
    )

    if "s1_file" in st.session_state and "step1_papers" not in st.session_state:
        from pipeline.embed import load_papers
        fname, raw = st.session_state["s1_file"]
        suffix = Path(fname).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            st.session_state["step1_papers"] = load_papers(tmp_path)
        except Exception as e:
            st.error(f"Could not load papers: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    if "step1_papers" in st.session_state:
        st.caption(f"{len(st.session_state['step1_papers'])} papers loaded.")

    s1_run = st.button("Run Embeddings", key="run_embed",
                       disabled=_is_running or "step1_papers" not in st.session_state)
    if _mem_caption:
        st.caption(_mem_caption)

    if s1_run and "step1_papers" in st.session_state:
        from pipeline.embed import embed_papers
        papers = st.session_state["step1_papers"]
        st.session_state["running"] = True
        prog = st.progress(0, text="Loading SPECTER2 model…")
        def _cb(cur, tot):
            prog.progress(cur / tot, text=f"Encoding {cur}/{tot}…")
        try:
            embeddings = embed_papers(papers, progress_callback=_cb)
            prog.progress(1.0, text="Done.")
            st.session_state["step1_embeddings"] = embeddings
        finally:
            st.session_state["running"] = False
        st.rerun()

    if "step1_embeddings" in st.session_state:
        embeddings = st.session_state["step1_embeddings"]
        papers     = st.session_state["step1_papers"]
        show_array_info(embeddings, "Embeddings ready")
        st.download_button(
            "Download embeddings (.csv)",
            array_to_csv_bytes(embeddings, ids=[p["id"] for p in papers]),
            "embeddings.csv", mime="text/csv", key="dl_embed_csv",
        )

    with st.expander("Don't have data? Try the demo data to start", expanded=False):
        st.caption("500 sample papers to try the tool without your own data.")
        demo_path = Path(__file__).parent / "sample_papers.csv"
        demo_bytes = demo_path.read_bytes()
        col_a, col_b = st.columns(2)
        if col_a.button("Load demo data", key="load_demo"):
            st.session_state["s1_file"] = ("sample_papers.csv", demo_bytes)
            st.session_state.pop("step1_papers", None)
            st.session_state.pop("step1_embeddings", None)
            st.rerun()
        col_b.download_button(
            "Download demo data", demo_bytes,
            "sample_papers.csv", mime="text/csv", key="dl_demo",
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCH A — NETWORK MAP (VOSviewer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("Text Similarity Network Map", expanded=False):
    st.header("Text Similarity Network Map")
    st.caption("Traditional science map. Builds a cosine similarity network and exports it to VOSviewer — emulating the citation-based approach used in bibliometrics.")
    st.caption("Each document is connected to its most similar documents based on cosine similarity of their embeddings. The resulting network can be visualized and explored in VOSviewer.")

    # ── Create Network ──
    with st.container(border=True):
        st.subheader("Create Network")

        embed_src = None
        has_embeddings = "step1_embeddings" in st.session_state
        use_above_net = st.checkbox("Use embeddings from Text to Embeddings", value=has_embeddings, key="use_step1_net", disabled=not has_embeddings)
        if use_above_net and has_embeddings:
            embed_src = st.session_state["step1_embeddings"]

        st.caption("Requirements: CSV with no header. First column: paper ID. Remaining 768 columns: embedding values. Generated by the Text to Embeddings step.")
        save_upload(
            st.file_uploader("Or upload an embeddings file (.csv)", type=["csv"], key="s2_upload"),
            "s2_file",
        )
        if not use_above_net and "s2_file" in st.session_state:
            try:
                embed_src = csv_bytes_to_array(st.session_state["s2_file"][1])
            except Exception as e:
                st.error(f"Could not load file: {e}")

        n_papers = len(embed_src) if embed_src is not None else None
        c1, c2 = st.columns(2)
        c1.caption("A document connects to its most similar documents. Higher values produce a denser network.")
        top_k = c1.number_input(
            "Maximum number of connections per document", min_value=1, max_value=n_papers - 1 if n_papers else None,
            value=20, step=1, key="top_k",
        )
        c2.caption("Minimum similarity required to keep a connection. Higher values produce a sparser network.")
        min_sim = c2.slider("Min similarity", 0.0, 1.0, 0.0, 0.01, key="min_sim")

        s2_run = st.button("Build Network", key="run_network", disabled=_is_running or embed_src is None)

        if s2_run and embed_src is not None:
            from pipeline.network import build_edge_list
            st.session_state["running"] = True
            prog = st.progress(0, text="Building edge list…")
            def _cb(cur, tot):
                prog.progress(cur / tot, text=f"Processing {cur}/{tot} papers…")
            try:
                with st.spinner("Computing cosine similarities…"):
                    edges = build_edge_list(embed_src, top_k=int(top_k), min_similarity=float(min_sim), progress_callback=_cb)
                prog.progress(1.0, text="Done.")
                st.session_state["step2_edges"] = edges
            finally:
                st.session_state["running"] = False
            st.rerun()

        if "step2_edges" in st.session_state:
            edges = st.session_state["step2_edges"]
            st.caption(f"{len(edges)} edges ready.")
            edge_buf = io.StringIO()
            ew = csv.writer(edge_buf)
            ew.writerow(["source", "target", "weight"])
            ew.writerows(edges)
            st.download_button(
                "Download edge list (.csv)", edge_buf.getvalue().encode(),
                "network.csv", mime="text/csv", key="dl_network",
            )

    # ── VOSviewer Export ──
    with st.container(border=True):
        st.subheader("Visualize Network with VOSviewer")

        edges_src = None
        has_edges = "step2_edges" in st.session_state
        use_above_edges = st.checkbox("Use edge list from Create Network", value=has_edges, key="use_step2_edges", disabled=not has_edges)
        if use_above_edges and has_edges:
            edges_src = st.session_state["step2_edges"]

        st.caption("Requirements: CSV columns: source, target, weight. Generated by the Create Network step.")
        save_upload(
            st.file_uploader("Or upload an edge list CSV", type=["csv"], key="s3_edges_upload"),
            "s3_edges_file",
        )
        if not use_above_edges and "s3_edges_file" in st.session_state:
            _, raw = st.session_state["s3_edges_file"]
            edges_src = [
                (int(r["source"]), int(r["target"]), float(r["weight"]))
                for r in csv.DictReader(io.StringIO(raw.decode()))
            ]

        papers_src = None
        has_papers = "step1_papers" in st.session_state
        use_above_meta = st.checkbox("Use papers from Text to Embeddings", value=has_papers, key="use_step1_meta", disabled=not has_papers)
        if use_above_meta and has_papers:
            papers_src = st.session_state["step1_papers"]

        st.caption("Requirements: CSV columns: id, title. Your original papers CSV works here — it already has these columns.")
        save_upload(
            st.file_uploader("Or upload a papers CSV (id, title)", type=["csv"], key="s3_meta_upload"),
            "s3_meta_file",
        )
        if not use_above_meta and "s3_meta_file" in st.session_state:
            _, raw = st.session_state["s3_meta_file"]
            papers_src = list(csv.DictReader(io.StringIO(raw.decode())))

        s3_run = st.button("Generate VOSviewer Files", key="run_vos",
                           disabled=(edges_src is None or papers_src is None))

        if s3_run and edges_src is not None and papers_src is not None:
            with st.spinner("Generating VOSviewer files…"):
                map_lines = ["id\tlabel\tdescription"]
                for idx, paper in enumerate(papers_src):
                    label = paper.get("title", paper.get("id", str(idx + 1)))
                    description = paper.get("id", "")
                    map_lines.append(f"{idx + 1}\t{label}\t{description}")
                net_lines = []
                for i, j, w in edges_src:
                    net_lines.append(f"{i + 1}\t{j + 1}\t{round(w, 6)}")
                vos_map_str = "\n".join(map_lines)
                vos_net_str = "\n".join(net_lines)
                if not _COLAB_MODE:
                    sid = st.session_state["session_id"]
                    map_url = _upload_to_gcs(f"{sid}_net_map.txt", vos_map_str)
                    net_url = _upload_to_gcs(f"{sid}_net_network.txt", vos_net_str)
                    st.session_state["vos_map_url"] = map_url
                    st.session_state["vos_net_url"] = net_url
            st.session_state["vos_map"] = vos_map_str
            st.session_state["vos_net"] = vos_net_str
            st.rerun()

        if "vos_map" in st.session_state and "vos_net" in st.session_state:
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download Map file", st.session_state["vos_map"].encode(),
                "vosviewer_map.txt", mime="text/plain", key="dl_vos_map",
            )
            c2.download_button(
                "Download Network file", st.session_state["vos_net"].encode(),
                "vosviewer_network.txt", mime="text/plain", key="dl_vos_net",
            )
            if "vos_map_url" in st.session_state:
                vos_url = f"https://app.vosviewer.com/?map={st.session_state['vos_map_url']}&network={st.session_state['vos_net_url']}"
                st.link_button("Open in VOSviewer Online", vos_url)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRANCH B — EMBEDDING SPACE MAP (UMAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("Embedding Space Reduction Map", expanded=False):
    st.header("Embedding Space Reduction Map")
    st.caption("Alternative science map. Projects embeddings directly into 2D space with UMAP — more faithful to the semantic structure of the embeddings.")
    st.caption("UMAP finds a 2D layout that preserves the high-dimensional relationships between documents as faithfully as possible. Similar documents end up close together; dissimilar ones far apart.")

    # ── UMAP ──
    with st.container(border=True):
        st.subheader("Generate 2D Coordinates")

        viz_embed_src = None
        viz_embed_ids = []
        has_embeddings_umap = "step1_embeddings" in st.session_state
        use_above_umap = st.checkbox("Use embeddings from Text to Embeddings", value=has_embeddings_umap, key="use_step1_umap", disabled=not has_embeddings_umap)
        if use_above_umap and has_embeddings_umap:
            viz_embed_src = st.session_state["step1_embeddings"]
            viz_embed_ids = [p["id"] for p in st.session_state["step1_papers"]]

        st.caption("Requirements: CSV with no header. First column: paper ID. Remaining 768 columns: embedding values. Generated by the Text to Embeddings step.")
        save_upload(
            st.file_uploader("Or upload an embeddings file (.csv)", type=["csv"], key="viz_embed_upload"),
            "viz_embed_file",
        )
        if not use_above_umap and "viz_embed_file" in st.session_state:
            _, raw = st.session_state["viz_embed_file"]
            try:
                viz_embed_src = csv_bytes_to_array(raw)
                viz_embed_ids = [l.split(",")[0] for l in raw.decode("utf-8").splitlines() if l.strip()]
            except Exception as e:
                st.error(f"Could not load file: {e}")

        c1, c2 = st.columns(2)
        c1.caption("Number of documents to consider at a time when preserving the embedding space structure. Low values keep local structure, high values keep global structure.")
        umap_n_neighbors = c1.number_input(
            "n_neighbors", min_value=2, value=15, step=1, key="umap_n_neighbors",
        )
        c2.caption("Minimum distance between documents in the 2D projection. Low values place similar documents into tight clumps, high values spread them more uniformly.")
        umap_min_dist = c2.slider(
            "min_dist", 0.0, 1.0, 0.1, 0.01, key="umap_min_dist",
        )

        if viz_embed_src is not None:
            n = len(viz_embed_src)
            estimate = "a few seconds" if n < 500 else "~30 seconds" if n < 2000 else "a few minutes"
            st.caption(f"Expected time: {estimate} ({n} papers)")

        sa_run = st.button("Run UMAP", key="run_umap", disabled=_is_running or viz_embed_src is None)

        if sa_run and viz_embed_src is not None:
            from pipeline.reduce import umap_reduce
            st.session_state["running"] = True
            try:
                with st.spinner(f"Running UMAP on {n} papers… ({estimate})"):
                    coords = umap_reduce(viz_embed_src, n_neighbors=int(umap_n_neighbors), min_dist=float(umap_min_dist))
                st.session_state["viz_coords"] = coords
                st.session_state["viz_ids"]    = viz_embed_ids
            finally:
                st.session_state["running"] = False
            st.rerun()

        if "viz_coords" in st.session_state:
            st.caption(f"{len(st.session_state['viz_coords'])} points projected.")
            coords_buf = io.StringIO()
            coords_buf.write("id,x,y\n")
            for pid, (x, y) in zip(st.session_state["viz_ids"], st.session_state["viz_coords"]):
                coords_buf.write(f"{pid},{x:.6f},{y:.6f}\n")
            st.download_button(
                "Download coords (.csv)", coords_buf.getvalue().encode(),
                "coords.csv", mime="text/csv", key="dl_coords",
            )

    # ── VOSviewer Map Export ──
    with st.container(border=True):
        st.subheader("Visualize with VOSviewer")
        st.caption("Generates a VOSviewer map file with UMAP coordinates, so VOSviewer positions nodes according to the projection.")

        vos_coords = None
        vos_coord_ids = []
        has_coords = "viz_coords" in st.session_state
        use_above_vos_coords = st.checkbox("Use coordinates from Generate 2D Coordinates", value=has_coords, key="use_viz_coords_vos", disabled=not has_coords)
        if use_above_vos_coords and has_coords:
            vos_coords    = st.session_state["viz_coords"]
            vos_coord_ids = st.session_state["viz_ids"]

        st.caption("Requirements: CSV columns: id, x, y. Generated by the Generate 2D Coordinates step.")
        save_upload(
            st.file_uploader("Or upload a coordinates CSV (id, x, y)", type=["csv"], key="vos_coords_upload"),
            "vos_coords_file",
        )
        if not use_above_vos_coords and "vos_coords_file" in st.session_state:
            _, raw = st.session_state["vos_coords_file"]
            rows = list(csv.DictReader(io.StringIO(raw.decode())))
            vos_coord_ids = [r["id"] for r in rows]
            vos_coords    = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float32)

        vos_map_papers = None
        has_papers_vos = "step1_papers" in st.session_state
        use_above_vos_meta = st.checkbox("Use papers from Text to Embeddings", value=has_papers_vos, key="use_step1_vos_map_meta", disabled=not has_papers_vos)
        if use_above_vos_meta and has_papers_vos:
            vos_map_papers = st.session_state["step1_papers"]

        st.caption("Requirements: CSV columns: id, title. Your original papers CSV works here — it already has these columns.")
        save_upload(
            st.file_uploader("Or upload a papers CSV (id, title)", type=["csv"], key="vos_map_meta_upload"),
            "vos_map_meta_file",
        )
        if not use_above_vos_meta and "vos_map_meta_file" in st.session_state:
            _, raw = st.session_state["vos_map_meta_file"]
            vos_map_papers = list(csv.DictReader(io.StringIO(raw.decode())))

        sc_run = st.button("Generate VOSviewer Map", key="run_vos_map",
                           disabled=(vos_coords is None or vos_map_papers is None))

        if sc_run and vos_coords is not None and vos_map_papers is not None:
            with st.spinner("Generating VOSviewer map…"):
                coord_lookup = {pid: (float(x), float(y)) for pid, (x, y) in zip(vos_coord_ids, vos_coords)}
                map_lines = ["id\tlabel\tdescription\tx\ty"]
                for idx, paper in enumerate(vos_map_papers):
                    pid   = str(paper.get("id", idx + 1))
                    label = paper.get("title", pid)
                    x, y  = coord_lookup.get(pid, (0.0, 0.0))
                    map_lines.append(f"{idx + 1}\t{label}\t{pid}\t{x:.6f}\t{y:.6f}")
                viz_map_str = "\n".join(map_lines)
                if not _COLAB_MODE:
                    sid = st.session_state["session_id"]
                    umap_map_url = _upload_to_gcs(f"{sid}_umap_map.txt", viz_map_str)
                    st.session_state["viz_vos_map_url"] = umap_map_url
            st.session_state["viz_vos_map"] = viz_map_str
            st.rerun()

        if "viz_vos_map" in st.session_state:
            st.download_button(
                "Download Map file", st.session_state["viz_vos_map"].encode(),
                "vosviewer_map.txt", mime="text/plain", key="dl_viz_vos_map",
            )
            if "viz_vos_map_url" in st.session_state:
                vos_url = f"https://app.vosviewer.com/?map={st.session_state['viz_vos_map_url']}"
                st.link_button("Open in VOSviewer Online", vos_url)
