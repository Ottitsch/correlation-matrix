# TechWolf-Inspired Approach

**Model:** `TechWolf/JobBERT-v3`

**Paper:** *Multilingual JobBERT for Cross-Lingual Job Title Matching* — Decorte, De Lange, Van Hautte (TechWolf, CLEF 2025)

## Method

Drop-in replacement of the baseline model with JobBERT-v3, a domain-specific model trained via contrastive learning on over 21 million job titles paired with ESCO skill annotations. Rather than learning similarity from text overlap, JobBERT learns that two titles are similar if they share the same skill profile — giving it a skill-grounded semantic space.

Each field's German and English names are embedded **separately** and then averaged. JobBERT-v3 was trained on monolingual job titles, so mixed-language concatenation (e.g. `"Telekommunikation Telecommunication"`) degrades quality — it falls into no known skill cluster. Averaging two clean monolingual embeddings is a fairer use of the model.

## Run

```bash
python generate_matrix.py
```

## Results (sample: Telecommunication neighbors)

| Value | Field |
|------:|-------|
| 9 | Construction |
| 8 | Electrical Engineering |
| 7 | Network Administration |
| 6 | Corporate Communication |
| 5 | Marketing |

## Notes

Despite the bilingual averaging fix, results are weaker than the baseline. The root cause: JobBERT's skill-grounded space was learned from specific job ad titles (`"Netzwerkingenieur"`, `"Network Engineer"`), not abstract domain categories (`"Telecommunication"`). Abstract field names don't anchor to any meaningful skill cluster, so embeddings land in unpredictable regions of the space. The approach would likely perform much better if our input were actual job titles rather than field category names.
