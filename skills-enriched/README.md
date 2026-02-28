# Skills-Enriched

**Model:** `intfloat/multilingual-e5-large-instruct`

## Method

Extension of the AlexU approach with structured rather than free-form LLM output. For each field, `gpt-5.2-chat` generates:

- **SKILLS:** 5–8 key professional skills (comma-separated)
- **EDUCATION:** typical educational paths or qualifications (comma-separated)

Each field is then embedded as:

```
{nameDe} {nameEn}
Skills: {skills}
Education: {education}
```

using `multilingual-e5-large-instruct` with the standard symmetric similarity instruction prefix. The hypothesis: two fields that require the same skills and educational background are similar regardless of how their names sound — grounding similarity in actual job market requirements rather than surface text.

## Run

```bash
python generate_enrichment.py   # generates enrichment.json, resumes if interrupted
python generate_matrix.py
```

## Results (sample: Mechanical Engineering neighbors)

| Value | Field | GT |
|------:|-------|:--:|
| 9 | Automotive | — |
| 8 | Plant Engineering | — |
| 7 | Civil Engineering | — |
| 6 | Electrical Engineering | — |
| 5 | Fabrication | — |

**Score: 2.22** (3 hits, avg rank delta 2.33) — ties with AlexU-inspired.

## Notes

Structured skills/education data performs on par with AlexU's free-form descriptions (both score 2.22). The added structure does not appear to meaningfully change what information the embedding model receives. Automotive is over-ranked (pred 9, GT 7) and Plant Engineering is under-ranked (pred 8 → effectively 6 relative to GT 9). Whether longer skill lists or finer-grained educational breakdowns would help is an open question.
