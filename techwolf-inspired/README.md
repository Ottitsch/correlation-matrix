# TechWolf-Inspired Approach

**Model:** `TechWolf/JobBERT-v3`

**Paper:** *Multilingual JobBERT for Cross-Lingual Job Title Matching* — Decorte, De Lange, Van Hautte (TechWolf, CLEF 2025)

## Method

Drop-in replacement of the baseline model with JobBERT-v3, a domain-specific model trained via contrastive learning on over 21 million job titles paired with ESCO skill annotations. Rather than learning similarity from text overlap, JobBERT learns that two titles are similar if they share the same skill profile — giving it a skill-grounded semantic space.

Input text is identical to the baseline: `"{nameDe} {nameEn}"`.

## Run

```bash
python generate_matrix.py
```

## Notes

JobBERT-v3 was designed for specific job titles from real job ads. It performs noticeably worse on our abstract work field categories (e.g. "Telekommunikation") compared to the baseline — the skill anchors it learned don't map cleanly to high-level field names.
