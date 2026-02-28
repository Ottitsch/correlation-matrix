#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: multilingual-e5-large-instruct embeddings + bilingual
LLM-generated descriptions + cosine similarity. (AlexU-NLP inspired)
See README.md for full explanation.

Run generate_descriptions.py first to produce descriptions.json.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_descriptions(path: str) -> dict[str, dict]:
    """Load descriptions.json, keyed by correlationMatrixId."""
    try:
        with open(path, encoding="utf-8") as f:
            return {e["correlationMatrixId"]: e for e in json.load(f)}
    except FileNotFoundError:
        return {}


def build_texts(fields: list[dict], descriptions: dict[str, dict]) -> list[str]:
    """
    Build one text per field: bilingual name + bilingual description (if available).
    Falls back to name-only for any field missing a description.
    """
    texts = []
    for field in fields:
        cid = field["correlationMatrixId"]
        base = f"{field['nameDe']} {field['nameEn']}"
        desc = descriptions.get(cid)
        if desc and desc.get("descriptionEn") and desc.get("descriptionDe"):
            texts.append(
                f"{base}\n{desc['descriptionEn']}\n{desc['descriptionDe']}"
            )
        else:
            texts.append(base)
    return texts


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

    descriptions = load_descriptions("descriptions.json")
    n_with_desc = sum(1 for f in fields if f["correlationMatrixId"] in descriptions)
    print(f"Loaded {len(fields)} work fields, {n_with_desc} with descriptions.")

    texts = build_texts(fields, descriptions)

    # multilingual-e5-large-instruct: instruction prefix is recommended for
    # symmetric similarity tasks to get the most out of the instruct tuning.
    prompt = "Instruct: Given a work field, retrieve semantically similar work fields\nQuery: "

    print("Downloading / loading model (first run ~560 MB)...")
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    print("Embedding fields...")
    embeddings = model.encode(
        texts, prompt=prompt, show_progress_bar=True, convert_to_numpy=True
    )

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
