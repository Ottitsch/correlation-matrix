#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: multilingual sentence embeddings (paraphrase-multilingual-mpnet-base-v2)
+ cosine similarity. See README.md for full explanation.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_texts(fields: list[dict]) -> list[str]:
    """Concatenate German and English names for richer bilingual signal."""
    return [f"{field['nameDe']} {field['nameEn']}" for field in fields]


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity, shape (n, n)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    # For each pair (lo, hi) nominated by either field's top-9, store the
    # maximum rank value from either direction.  This captures asymmetric
    # relationships: if B ranks A as its #2 neighbour but A only ranks B
    # as its #8, the stored value is 8 (from B's perspective) not 2.
    pair_value: dict[tuple[int, int], int] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self-similarity
        for rank_k, j in enumerate(np.argsort(row)[::-1][:9]):
            lo, hi = min(i, j), max(i, j)
            val = 9 - rank_k
            pair_value[(lo, hi)] = max(pair_value.get((lo, hi), 0), val)

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})
    for (lo, hi), val in pair_value.items():
        entries.append({"code1": ids[lo], "code2": ids[hi], "value": val})

    return entries


def main():
    fields = load_fields("../../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]
    texts = build_texts(fields)

    print(f"Loaded {len(fields)} work fields.")
    print("Downloading / loading model (first run ~420 MB)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    print("Embedding fields...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Computing cosine similarity...")
    sim = cosine_similarity(embeddings)

    print("Building matrix entries...")
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
