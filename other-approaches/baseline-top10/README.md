# Ablation: Baseline + Top-10 Neighbours

**Isolates:** does using top-10 instead of top-9 neighbours explain the hit-count gap?

## What changed vs baseline

| Setting | baseline | this folder |
|---------|----------|-------------|
| Model | paraphrase-multilingual-mpnet-base-v2 | same |
| Text format | `"{nameDe} {nameEn}"` | same |
| Symmetrization | code1's perspective | same |
| Neighbour count | **9** | **10** |
| Value scale | 9 → 1 off-diagonal | 10 → 1 off-diagonal |

## Side-effect

With top-10, value `10` is no longer reserved exclusively for the diagonal  -  the single most-similar neighbour also receives value `10`. This ambiguity is intentional here: it matches the new-repo SBERT approach exactly so the comparison is clean.

## Run

```bash
python generate_matrix.py
```
