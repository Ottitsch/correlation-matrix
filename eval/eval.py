#!/usr/bin/env python3
"""
Compare each approach's correlation_matrix.json against ground_truth.json.

Metrics per approach:
  - Hits:       how many ground truth pairs appear in the predicted matrix at all
  - Rank delta: average |predicted_value - ground_truth_value| for hits
  - Score:      hits * (1 - avg_rank_delta / 9)  - rewards both coverage and rank accuracy
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

APPROACHES = {
    # Submission
    "baseline":              ROOT / "submission/baseline/correlation_matrix.json",
    # Other approaches
    "techwolf-inspired":     ROOT / "other-approaches/techwolf-inspired/correlation_matrix.json",
    "alexU-inspired":        ROOT / "other-approaches/alexU-inspired/correlation_matrix.json",
    "llm-ranking":           ROOT / "other-approaches/llm-ranking/correlation_matrix.json",
    "skills-enriched":       ROOT / "other-approaches/skills-enriched/correlation_matrix.json",
    "techwolf-jobtitles":    ROOT / "other-approaches/techwolf-jobtitles/correlation_matrix.json",
    "hybrid":                ROOT / "other-approaches/hybrid/correlation_matrix.json",
    # Baseline ablations (each isolates one variable vs baseline)
    "baseline-top10":        ROOT / "other-approaches/baseline-top10/correlation_matrix.json",
    "baseline-max-sym":      ROOT / "other-approaches/baseline-max-sym/correlation_matrix.json",
    "baseline-en-first":     ROOT / "other-approaches/baseline-en-first/correlation_matrix.json",
}

GROUND_TRUTH = HERE / "ground_truth.json"


def load_matrix(path: str) -> dict[frozenset, int]:
    """Index a matrix as {frozenset({code1, code2}): value} for order-agnostic lookup."""
    with open(path) as f:
        data = json.load(f)
    return {frozenset({e["code1"], e["code2"]}): e["value"] for e in data}


def main():
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    with open(ROOT / "work_fields.json") as f:
        fields = json.load(f)
    id_to_name = {f["correlationMatrixId"]: f["nameEn"] for f in fields}

    print(f"Ground truth: {len(gt)} pairs for '{id_to_name.get(gt[0]['code1'], gt[0]['code1'])}'\n")

    results = {}

    for name, path in APPROACHES.items():
        matrix = load_matrix(path)

        hits = 0
        rank_deltas = []
        rows = []

        for entry in gt:
            key = frozenset({entry["code1"], entry["code2"]})
            gt_value = entry["value"]
            pred_value = matrix.get(key)

            if pred_value is not None:
                hits += 1
                delta = abs(pred_value - gt_value)
                rank_deltas.append(delta)
                rows.append((entry["code2"], gt_value, pred_value, delta))
            else:
                rows.append((entry["code2"], gt_value, None, None))

        avg_delta = sum(rank_deltas) / len(rank_deltas) if rank_deltas else 9
        score = hits * (1 - avg_delta / 9)
        results[name] = score

        print(f"-- {name} --")
        print(f"  Hits: {hits}/{len(gt)}   Avg rank delta: {avg_delta:.2f}   Score: {score:.2f}")
        print(f"  {'Field':<35} {'GT':>4} {'Pred':>6} {'D':>4}")
        for code, gt_val, pred_val, delta in rows:
            pred_str  = str(pred_val) if pred_val is not None else "miss"
            delta_str = str(delta)    if delta    is not None else "-"
            print(f"  {id_to_name.get(code, code):<35} {gt_val:>4} {pred_str:>6} {delta_str:>4}")
        print()

    winner = max(results, key=results.get)
    print(f"Winner: {winner}  (score {results[winner]:.2f})")


if __name__ == "__main__":
    main()
