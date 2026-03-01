#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: JobBERT-v3 embeddings over LLM-generated concrete job titles.
JobBERT-v3 was trained on real job ad titles paired with ESCO skill annotations,
so it works best on specific titles ("Netzwerkingenieur") rather than abstract
category names ("Telecommunication"). For each field, an LLM generates 5
representative titles in both German and English; each title is embedded
separately and the field's final embedding is the mean of all 10 title vectors.

Run generate_titles.py first to produce titles.json.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_titles(path: str) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return {e["correlationMatrixId"]: e for e in json.load(f)}
    except FileNotFoundError:
        return {}


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_field_embeddings(
    fields: list[dict], titles: dict[str, dict], model: SentenceTransformer
) -> np.ndarray:
    """
    For each field, embed all available job titles (DE + EN separately) and
    average them into one field-level embedding.
    Falls back to the field name if no titles are available.
    """
    field_embeddings = []

    for field in fields:
        cid = field["correlationMatrixId"]
        entry = titles.get(cid)

        if entry:
            title_texts = entry.get("titlesEn", []) + entry.get("titlesDe", [])
            title_texts = [t for t in title_texts if t.strip()]
        else:
            title_texts = []

        if not title_texts:
            # Fallback: use the field name itself (same as original techwolf-inspired)
            title_texts = [field["nameEn"], field["nameDe"]]

        vecs = model.encode(title_texts, show_progress_bar=False, convert_to_numpy=True)
        field_embeddings.append(vecs.mean(axis=0))

    return np.array(field_embeddings)


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    pair_value: dict[tuple[int, int], int] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
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
    fields = load_fields("../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]

    titles = load_titles("titles.json")
    n_with_titles = sum(1 for f in fields if f["correlationMatrixId"] in titles)
    print(f"Loaded {len(fields)} work fields, {n_with_titles} with generated job titles.")

    print("Downloading / loading model (first run ~280 MB)...")
    model = SentenceTransformer("TechWolf/JobBERT-v3")

    print("Embedding fields via generated job titles...")
    embeddings = build_field_embeddings(fields, titles, model)

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
