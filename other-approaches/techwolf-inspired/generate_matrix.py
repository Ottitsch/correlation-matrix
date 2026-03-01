#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: TechWolf/JobBERT-v3 embeddings + cosine similarity. (TechWolf inspired)
JobBERT-v3 is trained via contrastive learning on job title / ESCO skill pairs,
giving it a skill-grounded semantic space rather than pure text similarity.
See README.md for full explanation.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity, shape (n, n)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def bilingual_embeddings(
    fields: list[dict], model: SentenceTransformer
) -> np.ndarray:
    """
    Embed German and English names separately, then average.
    JobBERT-v3 is trained on monolingual job titles, so mixed-language
    concatenation degrades quality. Averaging two clean monolingual
    embeddings gives a fairer bilingual representation.
    """
    de_texts = [f["nameDe"] for f in fields]
    en_texts = [f["nameEn"] for f in fields]

    de_emb = model.encode(de_texts, show_progress_bar=False, convert_to_numpy=True)
    en_emb = model.encode(en_texts, show_progress_bar=False, convert_to_numpy=True)

    return (de_emb + en_emb) / 2


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    pair_maxrank: dict[tuple[int, int], int] = {}
    pair_score: dict[tuple[int, int], float] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self-similarity
        for rank_k, j in enumerate(np.argsort(row)[::-1][:9]):
            lo, hi = min(i, j), max(i, j)
            val = 9 - rank_k
            pair_maxrank[(lo, hi)] = max(pair_maxrank.get((lo, hi), 0), val)
            if (lo, hi) not in pair_score:
                pair_score[(lo, hi)] = float(sim[lo, hi])

    groups: dict[int, list[tuple[int, int, float]]] = {i: [] for i in range(n)}
    for (lo, hi), maxrank in pair_maxrank.items():
        groups[lo].append((hi, maxrank, pair_score[(lo, hi)]))

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})
        best_per_tier: dict[int, tuple[int, float]] = {}
        for j, maxrank, score in groups[i]:
            if maxrank not in best_per_tier or score > best_per_tier[maxrank][1]:
                best_per_tier[maxrank] = (j, score)
        sorted_tiers = sorted(best_per_tier.items(), key=lambda x: -x[0])
        for rank_k, (_maxrank, (j, score)) in enumerate(sorted_tiers):
            entries.append({
                "code1": ids[i],
                "code2": ids[j],
                "value": 9 - rank_k,
                "score": round(score, 6),
            })

    return entries


def main():
    fields = load_fields("../../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]

    print(f"Loaded {len(fields)} work fields.")
    print("Downloading / loading model (first run ~280 MB)...")
    model = SentenceTransformer("TechWolf/JobBERT-v3")

    print("Embedding fields (DE + EN separately, then averaged)...")
    embeddings = bilingual_embeddings(fields, model)

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
