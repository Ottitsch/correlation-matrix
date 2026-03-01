# OpenAI Embeddings

**Model:** `text-embedding-3-small` via OpenAI API

## Method

Each work field is encoded as a dense vector using OpenAI's `text-embedding-3-small` embedding model. The input text concatenates both language labels:

```
"Mechanical Engineering / Maschinenbau"
```

All 180 fields are encoded in a single batched API call. Vectors are normalised and pairwise cosine similarity is computed as a dot product. For each field the top-9 most similar other fields are ranked 9 (best) → 1.

`text-embedding-3-small` was trained on a broad multilingual corpus and performs strongly across languages including German, at substantially lower cost than the large variant.

**External dependency:** OpenAI API (`OPENAI_API_KEY` in `../.env`)

## Run

```bash
python generate_matrix.py
```

## Notes

Unlike `paraphrase-multilingual-mpnet-base-v2` (offline after first download), this approach requires an internet connection and incurs a small API cost (~$0.02 for 180 fields). Results are deterministic — the same input always returns the same embedding vector.

This approach differs from the AlexU-inspired and skills-enriched variants in that it uses a proprietary closed-source model without any description enrichment: embeddings are computed from the raw field names alone.
