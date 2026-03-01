# Ablation: Baseline + English-First Text Format

**Isolates:** does the text ordering and separator character affect the embeddings enough to change results?

## What changed vs baseline

| Setting | baseline | this folder |
|---------|----------|-------------|
| Model | paraphrase-multilingual-mpnet-base-v2 | same |
| Neighbour count | top-9 | same |
| Symmetrization | code1's perspective | same |
| Value scale | 9 → 1 off-diagonal | same |
| **Text format** | **`"{nameDe} {nameEn}"`** | **`"{nameEn} / {nameDe}"`** |

## Why this matters (or probably doesn't)

`paraphrase-multilingual-mpnet-base-v2` produces language-agnostic sentence embeddings trained to be invariant to surface form  -  word order and punctuation should have minimal effect. This ablation verifies that assumption. If results are identical to baseline, the text format can be ruled out as a factor in the new-repo SBERT gap.

## Run

```bash
python generate_matrix.py
```
