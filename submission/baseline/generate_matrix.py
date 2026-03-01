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

    # Phase 1 (unchanged): for each pair (lo, hi) nominated by either field's
    # top-9, record the max rank from either direction AND the raw cosine score.
    # Keeping the max-rank drives *which* pairs are selected (coverage), while
    # the raw cosine score is used only to break ties within the same max-rank.
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

    # Phase 2: group by code1 (lo index).
    groups: dict[int, list[tuple[int, int, float]]] = {i: [] for i in range(n)}
    for (lo, hi), maxrank in pair_maxrank.items():
        groups[lo].append((hi, maxrank, pair_score[(lo, hi)]))

    # Phase 3: for each code1 field, resolve ties *within each max-rank tier*
    # by keeping only the highest-cosine-sim pair per tier, then re-assign
    # consecutive unique ranks 9→1.  This mirrors the original selection logic
    # (max-rank from either direction determines importance) while guaranteeing
    # no duplicate value within any code1's neighbour list.
    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})

        # One representative per max-rank level: best cosine sim in each tier.
        best_per_tier: dict[int, tuple[int, float]] = {}  # maxrank → (j, score)
        for j, maxrank, score in groups[i]:
            if maxrank not in best_per_tier or score > best_per_tier[maxrank][1]:
                best_per_tier[maxrank] = (j, score)

        # Sort tiers high → low, re-assign consecutive ranks 9→1.
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
