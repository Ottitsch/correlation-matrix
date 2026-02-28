#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: multilingual-e5-large-instruct embeddings enriched with
LLM-generated required skills and education per field. Two fields that require
similar skills or educational backgrounds will be close in embedding space,
grounding similarity in job market requirements rather than name text alone.

Run generate_enrichment.py first to produce enrichment.json.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_enrichment(path: str) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return {e["correlationMatrixId"]: e for e in json.load(f)}
    except FileNotFoundError:
        return {}


def build_texts(fields: list[dict], enrichment: dict[str, dict]) -> list[str]:
    texts = []
    for field in fields:
        cid = field["correlationMatrixId"]
        base = f"{field['nameDe']} {field['nameEn']}"
        enc = enrichment.get(cid)
        if enc and enc.get("skills") and enc.get("education"):
            texts.append(
                f"{base}\nSkills: {enc['skills']}\nEducation: {enc['education']}"
            )
        else:
            texts.append(base)
    return texts


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
    for i, j in included:
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

    enrichment = load_enrichment("enrichment.json")
    n_enriched = sum(1 for f in fields if f["correlationMatrixId"] in enrichment)
    print(f"Loaded {len(fields)} work fields, {n_enriched} with skills/education data.")

    texts = build_texts(fields, enrichment)

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
