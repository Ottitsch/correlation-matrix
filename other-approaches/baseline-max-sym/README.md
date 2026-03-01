# Ablation: Baseline + Max Symmetrization

**Isolates:** does taking the max rank from either direction (vs always using code1's perspective) explain the hit-count gap?

## What changed vs baseline

| Setting | baseline | this folder |
|---------|----------|-------------|
| Model | paraphrase-multilingual-mpnet-base-v2 | same |
| Text format | `"{nameDe} {nameEn}"` | same |
| Neighbour count | top-9 | same |
| Value scale | 9 → 1 off-diagonal | same |
| **Symmetrization** | **code1's rank of code2** | **max(rank_A→B, rank_B→A)** |

## Why this matters

In baseline, when pair `(A, B)` is stored (A has lower index), the value reflects how A ranks B  -  even if B ranks A much more highly. Example: if Mechanical Engineering ranks Civil Engineering as its 8th neighbour (value 2) but Civil Engineering ranks Mechanical Engineering as its 2nd neighbour (value 8), baseline stores `value = 2`. This folder stores `value = 8`.

This can affect both **hit count** (if the pair is only in the matrix because B nominated A, baseline still gives it a low value from A's perspective  -  but it's still a hit) and **rank delta** (the stored value is closer to the ground truth rank).

## Run

```bash
python generate_matrix.py
```
