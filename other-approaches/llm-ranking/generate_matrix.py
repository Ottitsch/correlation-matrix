#!/usr/bin/env python3
"""
Generate a correlation matrix from LLM-produced rankings.

Reads rankings.json (produced by generate_rankings.py) and converts
per-field ranked lists into the standard upper-triangle + diagonal format.
No embeddings — similarity is inferred entirely from LLM world knowledge.
"""

import json

TOP_K = 9


def build_entries(ids: list[str], rankings: dict[str, list[str]]) -> list[dict]:
    n = len(ids)
    idx = {cid: i for i, cid in enumerate(ids)}

    # score[i][j] = rank score from i's perspective: rank 1 → 9, rank 2 → 8, …, rank 9 → 1, absent → 0
    scores = [[0] * n for _ in range(n)]
    for cid, neighbors in rankings.items():
        i = idx[cid]
        for rank_k, neighbor in enumerate(neighbors[:TOP_K]):
            if neighbor in idx:
                scores[i][idx[neighbor]] = TOP_K - rank_k

    # Nominate pairs: union of top-K from each direction
    pair_maxrank: dict[tuple[int, int], int] = {}
    for cid, neighbors in rankings.items():
        i = idx[cid]
        for rank_k, neighbor in enumerate(neighbors[:TOP_K]):
            if neighbor in idx:
                j = idx[neighbor]
                lo, hi = min(i, j), max(i, j)
                val = TOP_K - rank_k
                pair_maxrank[(lo, hi)] = max(pair_maxrank.get((lo, hi), 0), val)

    # Tiebreaker within a tier: sum of both directions' rank scores
    groups: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(n)}
    for (lo, hi), maxrank in pair_maxrank.items():
        sum_score = scores[lo][hi] + scores[hi][lo]
        groups[lo].append((hi, maxrank, sum_score))

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})
        best_per_tier: dict[int, tuple[int, int]] = {}  # maxrank → (j, sum_score)
        for j, maxrank, sum_score in groups[i]:
            if maxrank not in best_per_tier or sum_score > best_per_tier[maxrank][1]:
                best_per_tier[maxrank] = (j, sum_score)
        sorted_tiers = sorted(best_per_tier.items(), key=lambda x: -x[0])
        for rank_k, (_maxrank, (j, _)) in enumerate(sorted_tiers):
            entries.append({
                "code1": ids[i],
                "code2": ids[j],
                "value": 9 - rank_k,
            })

    return entries


def main():
    with open("../../work_fields.json", encoding="utf-8") as f:
        fields = json.load(f)
    ids = [f["correlationMatrixId"] for f in fields]

    with open("rankings.json", encoding="utf-8") as f:
        rankings = json.load(f)
    print(f"Loaded rankings for {len(rankings)}/{len(fields)} fields.")

    print("Building matrix entries...")
    entries = build_entries(ids, rankings)

    output_path = "correlation_matrix.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_diag = len(fields)
    n_off = len(entries) - n_diag
    print(f"Done. {len(entries)} entries written to {output_path} ({n_diag} diagonal + {n_off} off-diagonal).")


if __name__ == "__main__":
    main()
