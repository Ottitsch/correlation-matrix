#!/usr/bin/env python3
"""
Ablation of baseline: English-first text format instead of German-first.

Everything else is identical to baseline/:
  - Model:          paraphrase-multilingual-mpnet-base-v2
  - Neighbour count: top-9
  - Symmetrization: code1's perspective
  - Value scale:    9 → 1 off-diagonal

The only change: text format from "{nameDe} {nameEn}" to "{nameEn} / {nameDe}".

The purpose is to isolate whether the input text ordering and separator
character affect embedding quality enough to change the hit-count results.
This is expected to have minimal impact since paraphrase-multilingual-mpnet-base-v2
produces language-agnostic representations, but it is worth verifying.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_texts(fields: list[dict]) -> list[str]:
    # ← only change from baseline: English first, slash separator
    return [f"{field['nameEn']} / {field['nameDe']}" for field in fields]


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    included: set[tuple[int, int]] = set()
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        for j in np.argsort(row)[::-1][:9]:
            included.add((min(i, j), max(i, j)))

    groups: dict[int, list[int]] = {i: [] for i in range(n)}
    for (i, j) in included:
        groups[i].append(j)

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})

        neighbors = sorted(groups[i], key=lambda j: -sim[i, j])[:9]
        for rank_k, j in enumerate(neighbors):
            entries.append({"code1": ids[i], "code2": ids[j], "value": 9 - rank_k})

    return entries


def main():
    fields = load_fields("../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]
    texts = build_texts(fields)

    print(f"Loaded {len(fields)} work fields.")
    print("Downloading / loading model (first run ~420 MB)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    print("Embedding fields...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Computing cosine similarity...")
    sim = cosine_similarity(embeddings)

    print("Building matrix entries (English-first text)...")
    entries = build_entries(ids, sim)

    output_path = "correlation_matrix.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_diag = len(fields)
    n_off = len(entries) - n_diag
    print(
        f"Done. {len(entries)} entries written to {output_path} "
        f"({n_diag} diagonal + {n_off} off-diagonal pairs)."
    )


if __name__ == "__main__":
    main()
