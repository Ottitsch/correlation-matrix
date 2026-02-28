# Baseline Approach

**Model:** `paraphrase-multilingual-mpnet-base-v2`

## Method

Each work field is represented as the concatenation of its German and English names (e.g. `"Telekommunikation Telecommunication"`). These short texts are embedded using a general-purpose multilingual sentence transformer, and pairwise cosine similarity is computed over the resulting vectors.

No descriptions, no fine-tuning, no external data.

## Run

```bash
python generate_matrix.py
```

## Notes

This is the simplest possible approach — a direct embedding of the field names. It serves as the lower bound for comparison against the other approaches.
