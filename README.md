# Work Field Correlation Matrix

Generates a pairwise correlation matrix over 180 occupational work fields from `work_fields.json`, producing `correlation_matrix.json`.

## Reproduction

```bash
pip install -r requirements.txt
python generate_matrix.py
```

On first run the model (~420 MB) is downloaded automatically and cached by `sentence-transformers`. Subsequent runs are fast (< 30 s on CPU).

## Methodology

### Approach considered

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Multilingual sentence embeddings** | Fully local, fast, captures semantic meaning, uses both language labels | May miss purely operational links (e.g. Import ↔ Export) | **Chosen** |
| LLM-scored pairs | Nuanced, reasoning-based | ~16 K API calls for upper triangle; expensive, slow, non-reproducible | Rejected |
| Ontology mapping (ISCO/O\*NET) | Principled, hierarchical | Requires manual mapping of all 180 fields; many won't map cleanly | Rejected |
| Co-occurrence from job postings | Grounded in real HR data | Requires external dataset not available here | Rejected |

### Chosen: multilingual sentence embeddings

**Model:** [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

A sentence-transformer model trained on paraphrase data across 50+ languages, well-suited for short professional text. It maps each field name to a dense 768-dimensional vector in a shared multilingual semantic space.

**Text representation:** Both German and English names are concatenated into a single string per field (e.g. `"Telekommunikation Telecommunication"`). This gives the model twice the signal compared to using either language alone, and the model handles bilingual input natively.

**Similarity:** Pairwise cosine similarity over the normalized embedding matrix — symmetric by construction, O(n²) but trivial for n = 180.

### Matrix construction

1. For each of the 180 fields, find the 10 most similar other fields by cosine similarity.
2. A pair (A, B) is included in the output if B is in A's top-10 **or** A is in B's top-10 (union). This is intentionally inclusive: for candidate matching, it's better to surface a plausible related field than to miss it.
3. When both directions contribute a rank-value, the **maximum** is used — reflecting the stronger of the two relationships.
4. Rank values: the k-th nearest neighbor receives value `11 - k`, so the closest neighbor gets 10 and the 10th closest gets 1.
5. Every field includes its own diagonal entry with value 10.
6. The raw cosine similarity score is stored alongside the integer rank in a `score` field for downstream use.

### Threshold

No hard similarity floor is applied beyond the top-10 filter. For very niche fields, a forced low-similarity pairing is arguably worse than a sparse neighbourhood; the top-10 constraint already prevents noise accumulation.

## Output schema

```json
[
  { "code1": "w_tele", "code2": "w_tele", "value": 10 },
  { "code1": "w_tele", "code2": "w_info", "value": 9, "score": 0.8732 }
]
```

- `code1`, `code2`: `correlationMatrixId` values; `code1 ≤ code2` (upper triangle + diagonal)
- `value`: integer rank 1–10 (10 = most similar)
- `score`: raw cosine similarity (off-diagonal entries only)
