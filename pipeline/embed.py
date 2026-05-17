"""
Step 1 — Embeddings
Encodes paper titles and abstracts using SPECTER2 with the proximity adapter.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Union

import numpy as np


def _check_embedding_dependencies():
    """Fail early if the installed SPECTER2 adapter stack is incompatible."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        transformers_version = version("transformers")
        adapters_version = version("adapters")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Missing embedding dependency. Install the pinned requirements with "
            "`pip install -r requirements.txt`."
        ) from exc

    major_minor = tuple(int(part) for part in transformers_version.split(".")[:2])
    if major_minor < (4, 40) or major_minor >= (4, 52):
        raise RuntimeError(
            "Incompatible embedding dependencies: "
            f"transformers {transformers_version} with adapters {adapters_version}. "
            "Install the pinned requirements with `pip install -r requirements.txt` "
            "so transformers stays in the supported >=4.40,<4.52 range."
        )


def load_papers(path: Union[str, Path]) -> list[dict]:
    """Load papers from a CSV or JSON file.

    Expected fields: id, title, abstract
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            papers = list(reader)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Use .csv or .json")

    required = {"id", "title", "abstract"}
    for i, paper in enumerate(papers):
        missing = required - set(paper.keys())
        if missing:
            raise ValueError(f"Paper at index {i} is missing fields: {missing}")

    return papers


_model_cache: dict = {}

def _load_model():
    """Load and cache the SPECTER2 model and tokenizer (once per process)."""
    if _model_cache:
        return _model_cache["tokenizer"], _model_cache["model"]
    _check_embedding_dependencies()
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    model_name = "allenai/specter2_base"
    adapter_name = "allenai/specter2"
    adapter_load_name = "proximity"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.load_adapter(adapter_name, source="hf", load_as=adapter_load_name, set_active=True)
    model.eval()
    _model_cache["tokenizer"] = tokenizer
    _model_cache["model"] = model
    return tokenizer, model


def embed_papers(
    papers: list[dict],
    progress_callback=None,
) -> np.ndarray:
    """Encode papers with SPECTER2 + proximity adapter.

    Returns a float32 array of shape (N, D).
    progress_callback(current, total) is called after each batch if provided.
    """
    tokenizer, model = _load_model()

    batch_size = 64
    all_embeddings = []
    total = len(papers)

    import torch

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = papers[start : start + batch_size]
            texts = [
                p["title"] + tokenizer.sep_token + p.get("abstract", "")
                for p in batch
            ]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = model(**encoded)
            # CLS token representation
            embeddings = output.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu().numpy())

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

    return np.vstack(all_embeddings).astype(np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    papers: list[dict],
    out_dir: Union[str, Path],
    name: str = "embeddings",
) -> tuple[Path, Path]:
    """Save embeddings array (.npy) and metadata CSV.

    Returns (npy_path, meta_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{name}.npy"
    meta_path = out_dir / f"{name}_meta.csv"

    np.save(npy_path, embeddings)

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title"])
        writer.writeheader()
        for paper in papers:
            writer.writerow({"id": paper["id"], "title": paper["title"]})

    return npy_path, meta_path
