# TechWolf + Job Titles

**Model:** `TechWolf/JobBERT-v3`

**Paper:** *Multilingual JobBERT for Cross-Lingual Job Title Matching*  -  Decorte, De Lange, Van Hautte (TechWolf, CLEF 2025)

## Method

Addresses the root cause of `techwolf-inspired`'s underperformance. JobBERT-v3 was trained on real job advertisement titles like `"Maschinenbauingenieur"` or `"Mechanical Engineer"` paired with ESCO skill annotations. Feeding it abstract category names like `"Mechanical Engineering"` produces embeddings that don't land in any meaningful skill cluster.

Fix: use `gpt-5.2-chat` to generate 5 representative concrete job titles in both German and English for each field. All 10 titles are embedded separately with JobBERT-v3 and averaged into a single field-level vector. This puts the input firmly back into the distribution the model was trained on.

## Run

```bash
python generate_titles.py    # generates titles.json, resumes if interrupted
python generate_matrix.py
```

## Results (sample: Mechanical Engineering neighbors)

| Value | Field | GT |
|------:|-------|:--:|
| 9 | Plant Engineering | 9 ✓ |
| 8 | Electrical Engineering |  -  |
| 7 | Civil Engineering |  -  |
| 6 | Fabrication |  -  |
| 5 | Automotive |  -  |

**Score: 1.78** (2 hits, avg rank delta 1.00)  -  improves over plain TechWolf (1.33).

## Notes

Plant Engineering is ranked exactly (pred 9, GT 9)  -  the only approach besides LLM-ranking to hit this. Civil Engineering is also in the top 9 (pred 7, GT 5, delta 2). However, Automotive and Fabrication are missed entirely, suggesting 5 titles per field is still too sparse to cover all relevant skill clusters within the field. Increasing to 10–15 titles, or choosing titles that explicitly span the field's sub-domains, would likely improve coverage.
