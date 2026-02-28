# Work Field Correlation Matrix

Generates a pairwise correlation matrix over 180 occupational work fields from `work_fields.json`.

Three approaches were implemented and evaluated against a ground truth sample from the Romagnolo platform. The best-performing output is in `alexU-inspired/correlation_matrix.json`.

---

## Reproduction

```bash
pip install -r requirements.txt

# Baseline
cd baseline && python generate_matrix.py

# TechWolf-inspired
cd techwolf-inspired && python generate_matrix.py

# AlexU-inspired (requires Azure OpenAI key in ../.env as KEY=...)
cd alexU-inspired
python generate_descriptions.py   # generates descriptions.json, resumes if interrupted
python generate_matrix.py

# LLM ranking (requires Azure OpenAI key in ../.env as KEY=...)
cd llm-ranking
python generate_rankings.py       # generates rankings.json, resumes if interrupted
python generate_matrix.py

# Skills-enriched (requires Azure OpenAI key in ../.env as KEY=...)
cd skills-enriched
python generate_enrichment.py     # generates enrichment.json, resumes if interrupted
python generate_matrix.py

# Evaluate all approaches against ground truth
python eval.py
```

---

## Output Schema

All three approaches produce the same format:

```json
[
  { "code1": "w_tele", "code2": "w_tele", "value": 10 },
  { "code1": "w_tele", "code2": "w_sale", "value": 9 }
]
```

- `code1`, `code2`: `correlationMatrixId` values; `code1` always has a lower index in `work_fields.json` (upper triangle + diagonal)
- `value`: integer rank — `10` reserved for self-correlation (diagonal), `9` (most similar) → `1` (9th most similar) for off-diagonal entries

---

## Approaches

Five approaches were implemented, each in its own folder with a dedicated README.

### 1. Baseline — `paraphrase-multilingual-mpnet-base-v2`

The simplest approach. Both language names are concatenated (`"Telekommunikation Telecommunication"`) and embedded with a general-purpose multilingual sentence transformer. No external data, no descriptions.

**Model size:** ~420 MB

### 2. TechWolf-inspired — `TechWolf/JobBERT-v3`

Inspired by *Multilingual JobBERT for Cross-Lingual Job Title Matching* (Decorte, De Lange, Van Hautte — TechWolf, CLEF 2025).

JobBERT-v3 is trained via contrastive learning on 21 million job titles paired with ESCO skill annotations, giving it a skill-grounded semantic space. German and English names are embedded **separately** and averaged rather than concatenated, since the model was trained on monolingual job titles.

**Model size:** ~280 MB

### 3. AlexU-inspired — `intfloat/multilingual-e5-large-instruct` + LLM descriptions

Inspired by *AlexU-NLP at TalentCLEF 2025* (Barakat, Mokhtar, Torki, Elmakky — Alexandria University, CLEF 2025), specifically their inference-time description enrichment ablation (Table 4): adding descriptions to corpus entries improves mAP by ~2.6% even on their fine-tuned model, suggesting the signal is robust and likely carries over to zero-shot use.

`gpt-5.2-chat` via Azure OpenAI generates a bilingual description (EN + DE, 2–3 sentences each) for every work field. Each field is then embedded as `"{nameDe} {nameEn}\n{descriptionEn}\n{descriptionDe}"` using `multilingual-e5-large-instruct` with a symmetric similarity instruction prefix.

**Model size:** ~560 MB
**External dependency:** Azure OpenAI (`KEY` in `.env`)

### 4. LLM ranking — direct GPT ranking

No embeddings at all. For each of the 180 fields, the full field list is passed to the LLM and it is asked to rank the top 9 most similar fields by skill overlap, domain proximity, and career path. Similarity is inferred entirely from the model's world knowledge about occupational domains.

**External dependency:** Azure OpenAI (`KEY` in `.env`)

### 5. Skills-enriched — `multilingual-e5-large-instruct` + structured requirements

Extension of the AlexU approach with more structured LLM output. Instead of a free-form description, the LLM generates a comma-separated list of required skills and typical educational paths for each field. The embedding text becomes `"{nameDe} {nameEn}\nSkills: ...\nEducation: ..."`. The hypothesis: two fields that require the same skills and education are similar regardless of how their names sound, grounding similarity in actual job market requirements rather than surface text.

**Model size:** ~560 MB
**External dependency:** Azure OpenAI (`KEY` in `.env`)

---

## Evaluation

Evaluated against a single ground truth sample (5 pairs for *Mechanical Engineering*) sourced from the Romagnolo platform.

| Approach | Hits (out of 5) | Avg rank delta | Score |
|---|:---:|:---:|:---:|
| baseline | 2 | 2.00 | 1.56 |
| techwolf-inspired | 2 | 3.00 | 1.33 |
| **alexU-inspired** | **3** | **2.33** | **2.22** |

Score = `hits × (1 − avg_rank_delta / 9)` — rewards both coverage and rank accuracy.

### Per-field breakdown (Mechanical Engineering)

| Field | GT | Baseline | TechWolf | AlexU |
|---|:---:|:---:|:---:|:---:|
| Plant Engineering | 9 | miss | 6 | 4 |
| Building Craft | 8 | miss | miss | miss |
| Automotive | 7 | 4 | miss | **7** ✓ |
| Fabrication | 6 | 5 | miss | 8 |
| Civil Engineering | 5 | miss | 8 | miss |

AlexU gets the most hits (3/5) and is the only approach to nail an exact rank match (Automotive, value 7). Building Craft is missed by all three approaches.

### Key findings

- **AlexU wins**: LLM-generated descriptions give the embedding model enough context to surface domain-relevant neighbors that name-only approaches miss entirely.
- **TechWolf underperforms**: JobBERT-v3's skill-grounded space was learned from specific job ad titles, not abstract field category names. The model degrades on our input type despite the bilingual averaging fix.
- **Baseline is competitive**: Despite its simplicity, the baseline beats TechWolf — a strong reminder that domain fit matters more than model sophistication.

---

## Design Decisions

**Value scale:** `10` is reserved exclusively for the diagonal (self-correlation). Off-diagonal values run `9` (most similar) → `1` (9th most similar), ensuring no ambiguity between "identical field" and "most similar other field".

**Sparsity:** Top-9 neighbors per field, union across both directions. No hard similarity floor beyond that — for niche fields, forcing low-similarity pairings is worse than a sparse neighbourhood.

**Symmetry:** Cosine similarity is symmetric by construction. For pairs nominated by both directions, the value is based on the code1 field's own ranking of its neighbors, ensuring unique and score-consistent ranks within each field's group.

**Upper triangle:** `code1` always corresponds to the field with the lower index in `work_fields.json`, matching the ordering in the task example.

---

## Outlook

With more time, interesting directions would include:

- **Larger eval**: one ground truth sample is not enough to draw strong conclusions. Interviewing domain experts or judging more job titles manually would give more reliable signal.
- **Ensemble**: blend similarity scores from multiple approaches (e.g. AlexU + baseline) before ranking.
- **ESCO skill mapping**: map the 180 fields to ESCO skill sets and use skill overlap as a complementary signal — the TechWolf-inspired Option B that was not pursued here.
- **Fine-tuning**: given labeled pairs, curriculum learning (AlexU) or GISTEmbed (pjmathematician) would likely push results significantly higher.
- **GDPR-compliant scaling**: Azure AI or Google Vertex AI for larger models while keeping data within EU boundaries.

---

## Citations

```
@inproceedings{gasco2025overview,
  title={{Overview of the TalentCLEF 2025}},
  author={Gasco et al.},
  booktitle={{ECIR}},
  year={2025}
}

@inproceedings{alexunlp2025,
  title={{AlexU-NLP at TalentCLEF 2025: Curriculum-Driven Hybrid Retrieval for Multilingual Job Title Matching}},
  author={Rana Barakat, Omar Mokhtar, Marwan Torki and Nagwa Elmakky},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}

@inproceedings{pjmathematician2025,
  title={{pjmathematician at TalentCLEF 2025: Enhancing Job Title and Skill Matching with GISTEmbed and LLM-Augmented Data}},
  author={Poojan Vachharajani},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}

@inproceedings{techwolf2025,
  title={{Multilingual JobBERT for Cross-Lingual Job Title Matching}},
  author={Jens-Joris Decorte, Matthias De Lange and Jeroen Van Hautte},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}
```
