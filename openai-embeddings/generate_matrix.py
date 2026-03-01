#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: OpenAI text-embedding-3-small via standard OpenAI API.
All 180 fields are encoded in a single batched API call.
Cosine similarity is computed from normalised vectors via dot product.
See README.md for full explanation.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv("../.env")


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    # Union of each field's top-9 neighbours (upper triangle only)
    included: set[tuple[int, int]] = set()
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        for j in np.argsort(row)[::-1][:9]:
            included.add((min(i, j), max(i, j)))

    groups: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in included:
        groups[i].append(j)

    entries: list[dict] = []
    for i in range(n):
        # Diagonal: self-correlation, always value 10
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})

        # Off-diagonal: rank by sim[i][j], values 9 (most similar) → 1
        neighbors = sorted(groups[i], key=lambda j: -sim[i, j])[:9]
        for rank_k, j in enumerate(neighbors):
            entries.append({"code1": ids[i], "code2": ids[j], "value": 9 - rank_k})

    return entries


def main():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to ../.env or environment.")

    fields = load_fields("../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]
    texts = [f"{f['nameEn']} / {f['nameDe']}" for f in fields]

    print(f"Loaded {len(fields)} work fields.")
    print("Calling OpenAI embeddings API (text-embedding-3-small)...")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)

    # response.data is guaranteed to be in input order
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    print("Computing cosine similarity...")
    sim = np.dot(vectors, vectors.T)

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
