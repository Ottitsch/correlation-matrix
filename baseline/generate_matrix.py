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

    # Determine which upper-triangle pairs to include: union of each field's top-9.
    # A pair (i, j) is included if j is in i's top-9 OR i is in j's top-9.
    included: set[tuple[int, int]] = set()
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self-similarity
        for j in np.argsort(row)[::-1][:9]:
            included.add((min(i, j), max(i, j)))

    # Group included pairs by code1 (lower index)
    groups: dict[int, list[int]] = {i: [] for i in range(n)}
    for (i, j) in included:
        groups[i].append(j)

    entries: list[dict] = []

    for i in range(n):
        # Diagonal: self-correlation, always value 10
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})

        # Off-diagonal: rank neighbors by sim[i][j] from code1's own perspective.
        # Values run 9 (most similar) → 1 (least similar), keeping 10 reserved
        # exclusively for the diagonal (identity). Top 9 neighbors per field.
        neighbors = sorted(groups[i], key=lambda j: -sim[i, j])[:9]
        for rank_k, j in enumerate(neighbors):
            entries.append(
                {
                    "code1": ids[i],
                    "code2": ids[j],
                    "value": 9 - rank_k,
                }
            )

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
