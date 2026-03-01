#!/usr/bin/env python3
"""
Ablation of baseline: max symmetrization instead of code1-perspective.

Everything else is identical to baseline/:
  - Model:          paraphrase-multilingual-mpnet-base-v2
  - Text format:    "{nameDe} {nameEn}"  (German first, space-separated)
  - Neighbour count: top-9

The only change: when a pair (A, B) is nominated by both A's top-9
and B's top-9, baseline stores the value from A's perspective (A has
the lower index).  This folder instead stores the MAXIMUM value from
either direction — so a pair gets the best rank that either field
assigned to the other.

The purpose is to isolate whether the max-symmetry rule (not the
wider top-10 net) is what drives the hit-count difference.
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

    # Collect (lo, hi) → max value from either direction  ← only change from baseline
    pair_maxrank: dict[tuple[int, int], int] = {}
    pair_score: dict[tuple[int, int], float] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
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
    texts = build_texts(fields)

    print(f"Loaded {len(fields)} work fields.")
    print("Downloading / loading model (first run ~420 MB)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    print("Embedding fields...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Computing cosine similarity...")
    sim = cosine_similarity(embeddings)

    print("Building matrix entries (max symmetrization)...")
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
