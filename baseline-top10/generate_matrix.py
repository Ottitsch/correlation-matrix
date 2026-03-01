#!/usr/bin/env python3
"""
Ablation of baseline: top-10 neighbours instead of top-9.

Everything else is identical to baseline/:
  - Model:          paraphrase-multilingual-mpnet-base-v2
  - Text format:    "{nameDe} {nameEn}"  (German first, space-separated)
  - Symmetrization: value taken from code1's own perspective

The only change: the union of included pairs is built from each field's
top-10 most similar neighbours (instead of top-9), and values run
10 (most similar) → 1 (10th most similar).

NOTE: value 10 is now ambiguous — both the diagonal self-pair and
the single most-similar neighbour receive value 10.  This matches
the new-repo's SBERT approach exactly on this axis.  The purpose of
this folder is to isolate whether the wider net (top-10 vs top-9)
alone explains the hit-count difference.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_texts(fields: list[dict]) -> list[str]:
    return [f"{field['nameDe']} {field['nameEn']}" for field in fields]


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)
    TOP_K = 10  # ← only change from baseline (was 9)

    included: set[tuple[int, int]] = set()
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        for j in np.argsort(row)[::-1][:TOP_K]:
            included.add((min(i, j), max(i, j)))

    groups: dict[int, list[int]] = {i: [] for i in range(n)}
    for (i, j) in included:
        groups[i].append(j)

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})

        neighbors = sorted(groups[i], key=lambda j: -sim[i, j])[:TOP_K]
        for rank_k, j in enumerate(neighbors):
            entries.append({"code1": ids[i], "code2": ids[j], "value": TOP_K - rank_k})

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

    print("Building matrix entries (top-10)...")
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
