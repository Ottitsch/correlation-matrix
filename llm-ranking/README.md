# LLM Ranking

**Model:** `gpt-5.2-chat` via Azure OpenAI

## Method

No embeddings. For each of the 180 work fields, the complete field list is passed to the LLM with the prompt:

> *"From the list below, identify the 9 most similar work fields, ordered from most similar to least similar. Consider skill overlap, domain proximity, and typical career paths. Return ONLY a JSON array of 9 correlationMatrixIds."*

Similarity is inferred entirely from the LLM's world knowledge about occupational domains. The ranked lists are cached in `rankings.json` (resume-safe) and converted to the standard matrix format by `generate_matrix.py`.

Score assignment mirrors the embedding approaches: rank 1 → value 9, rank 9 → value 1. For pairs nominated by both directions, the code1 field's own score determines the final value.

## Run

```bash
python generate_rankings.py   # 180 API calls, saves rankings.json
python generate_matrix.py
```

## Results (sample: Mechanical Engineering neighbors)

| Value | Field | GT |
|------:|-------|:--:|
| 9 | Plant Engineering | 9 |
| 8 | Civil Engineering | — |
| 7 | Automotive | 7 ✓ |
| 6 | Building Architecture | — |
| 5 | Fabrication | — |

**Score: 2.44** (3 hits, avg rank delta 1.67) — best across all six approaches.

## Notes

The LLM ranks Plant Engineering as value 8 (GT: 9) and Automotive at exactly value 7. The higher precision compared to embedding approaches likely comes from the LLM reasoning about occupational domain structure directly rather than approximating it through cosine distance in a generic vector space. The main limitation is non-determinism — different runs may produce different rankings for ambiguous pairs.
