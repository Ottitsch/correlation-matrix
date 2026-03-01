# Work Field Correlation Matrix

A semantic correlation matrix over the 180 occupational work fields used by danube.ai, generated with multiple approaches and evaluated against ground truth.

## Background

The matrix is used in the **Romagnolo** HR platform for:
1. **Candidate matching** — correlated fields compensate for incomplete job ads by giving bonus scores to candidates with related fields.
2. **Smart search completions** — searching for a work field surfaces highly correlated ones as suggestions.

---

## Reproduction

!!! IF YOU DONT WANT TO RE-RUN GENAI ONLY RUN generate_matrix.py NO API KEY REQUIRED !!!

```bash
pip install -r requirements.txt

# Baseline
cd baseline && python generate_matrix.py

# TechWolf-inspired
cd techwolf-inspired && python generate_matrix.py

# Hybrid (SBERT + domain cluster boost)
cd hybrid && python generate_matrix.py

# AlexU-inspired (requires Azure OpenAI key in .env as KEY=...)
cd alexU-inspired
python generate_descriptions.py   # generates descriptions.json, resumes if interrupted
python generate_matrix.py

# LLM ranking (requires Azure OpenAI key in .env as KEY=...)
cd llm-ranking
python generate_rankings.py       # generates rankings.json, resumes if interrupted
python generate_matrix.py

# Skills-enriched (requires Azure OpenAI key in .env as KEY=...)
cd skills-enriched
python generate_enrichment.py     # generates enrichment.json, resumes if interrupted
python generate_matrix.py

# TechWolf + job titles (requires Azure OpenAI key in .env as KEY=...)
cd techwolf-jobtitles
python generate_titles.py         # generates titles.json, resumes if interrupted
python generate_matrix.py

# OpenAI Embeddings (requires OPENAI_API_KEY in .env)
cd openai-embeddings && python generate_matrix.py

# Baseline ablations (all free, no API key required)
cd baseline-top10 && python generate_matrix.py
cd baseline-max-sym && python generate_matrix.py
cd baseline-en-first && python generate_matrix.py

# Evaluate all approaches against ground truth
python eval.py
```

---

## Output Schema

All approaches produce the same format:

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

Eight approaches were implemented, each in its own folder with a dedicated README. The best-performing output is in `llm-ranking/correlation_matrix.json`.

### Overview

| # | Approach | Mechanism | API Required | Hits (5 GT pairs) | Score |
|---|----------|-----------|:---:|:---:|:---:|
| 1 | **Baseline** | Local SBERT embeddings (paraphrase-multilingual-mpnet-base-v2) | — | 2/5 | 1.56 |
| 2 | **TechWolf-inspired** | JobBERT-v3 embeddings, names embedded separately and averaged | — | 2/5 | 1.33 |
| 3 | **Hybrid** | SBERT embeddings + manual domain-cluster similarity boost (+0.15) | — | — | — |
| 4 | **AlexU-inspired** | multilingual-e5-large-instruct + LLM-generated bilingual descriptions | Azure OpenAI | 3/5 | 2.22 |
| 5 | **LLM ranking** | Direct GPT ranking, no embeddings | Azure OpenAI | 3/5 | **2.44** |
| 6 | **Skills-enriched** | multilingual-e5-large-instruct + structured skills/education data | Azure OpenAI | 3/5 | 2.22 |
| 7 | **TechWolf + job titles** | JobBERT-v3 + LLM-generated concrete job titles | Azure OpenAI | 2/5 | 1.78 |
| 8 | **OpenAI Embeddings** | text-embedding-3-small via OpenAI API | OpenAI API | — | — |
| — | **baseline-top10** *(ablation)* | Baseline but top-10 neighbours (wider net) | — | 2/5 | 1.78 |
| — | **baseline-max-sym** *(ablation)* | Baseline but max rank from either direction | — | **5/5** | **4.00** |
| — | **baseline-en-first** *(ablation)* | Baseline but `"nameEn / nameDe"` text format | — | 2/5 | 1.22 |

Score = `hits × (1 − avg_rank_delta / 9)`. Approaches 3 and 8 have not yet been run. See **Ablation Studies** section for findings.

---

### 1. Baseline — `paraphrase-multilingual-mpnet-base-v2`

The simplest approach. Both language names are concatenated (`"Telekommunikation Telecommunication"`) and embedded with a general-purpose multilingual sentence transformer. No external data, no descriptions.

**Pros:** Free, reproducible, fast (~30 s on CPU), no API key needed.
**Cons:** Rankings reflect distributional word similarity — it cannot reason about occupational career paths or the physical nature of work.

**Model size:** ~420 MB

---

### 2. TechWolf-inspired — `TechWolf/JobBERT-v3`

Inspired by *Multilingual JobBERT for Cross-Lingual Job Title Matching* (Decorte, De Lange, Van Hautte — TechWolf, CLEF 2025).

JobBERT-v3 is trained via contrastive learning on 21 million job titles paired with ESCO skill annotations, giving it a skill-grounded semantic space. German and English names are embedded **separately** and averaged rather than concatenated, since the model was trained on monolingual job titles.

**Pros:** Skill-grounded representations, no API needed.
**Cons:** Abstract category names like "Telecommunication" are far from the concrete job-ad titles the model was trained on, hurting retrieval quality.

**Model size:** ~280 MB

---

### 3. Hybrid — `paraphrase-multilingual-mpnet-base-v2` + domain cluster boost

Uses the SBERT similarity matrix as a base and adds +0.15 to all pairs that share the same manually assigned domain cluster. Fifteen clusters were defined (`IT_Software`, `Engineering`, `Manufacturing`, `Finance`, `Sales_Marketing`, `HR_Education`, `Healthcare`, `Management`, `Science`, `Legal`, `Admin_Office`, `Creative_Media`, `Logistics`, `Services`, `Real_Estate`) covering all 180 fields.

The boost magnitude (+0.15) approximates the typical cosine similarity gap between related and unrelated fields, enough to surface intra-cluster pairs in the top-9 without completely overriding the embedding signal.

**Pros:** Adds interpretable domain knowledge, deterministic, no API costs.
**Cons:** Single-membership clustering loses cross-domain signal for fields that span multiple domains (e.g., Medical Technology spans Healthcare and Engineering).

**Model size:** ~420 MB

---

### 4. AlexU-inspired — `intfloat/multilingual-e5-large-instruct` + LLM descriptions

Inspired by *AlexU-NLP at TalentCLEF 2025* (Barakat, Mokhtar, Torki, Elmakky — Alexandria University, CLEF 2025), specifically their inference-time description enrichment ablation (Table 4): adding descriptions to corpus entries improves mAP by ~2.6% even on their fine-tuned model, suggesting the signal is robust and likely carries over to zero-shot use.

`gpt-5.2-chat` via Azure OpenAI generates a bilingual description (EN + DE, 2–3 sentences each) for every work field. Each field is then embedded as `"{nameDe} {nameEn}\n{descriptionEn}\n{descriptionDe}"` using `multilingual-e5-large-instruct` with a symmetric similarity instruction prefix.

**Pros:** Richer semantic signal from descriptions, strong multilingual model.
**Cons:** Requires API access, descriptions add latency on first run.

**Model size:** ~560 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 5. LLM ranking — direct GPT ranking

No embeddings at all. For each of the 180 fields, the full field list is passed to the LLM and it is asked to rank the top 9 most similar fields by skill overlap, domain proximity, and career path. Similarity is inferred entirely from the model's world knowledge about occupational domains.

**Pros:** Genuine occupational domain reasoning, not just lexical similarity. Best score across all evaluated approaches.
**Cons:** Non-deterministic — re-running may produce slightly different rankings. Slower than embedding approaches (~4 min for 180 fields).

**External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 6. Skills-enriched — `multilingual-e5-large-instruct` + structured requirements

Extension of the AlexU approach with more structured LLM output. Instead of a free-form description, the LLM generates a comma-separated list of required skills and typical educational paths for each field. The embedding text becomes `"{nameDe} {nameEn}\nSkills: ...\nEducation: ..."`. The hypothesis: two fields that require the same skills and education are similar regardless of how their names sound, grounding similarity in actual job market requirements rather than surface text.

**Pros:** Structured skill/education data may generalise better across naming conventions.
**Cons:** Structured data is no more informative than free-form descriptions for this task at this scale (tied with AlexU on the eval sample).

**Model size:** ~560 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 7. TechWolf + job titles — `TechWolf/JobBERT-v3` + LLM-generated concrete titles

Addresses the root cause of TechWolf-inspired's underperformance: JobBERT-v3 was trained on concrete job ad titles, not abstract category names. An LLM generates 5 representative job titles in both German and English for each field; all 10 titles are embedded separately with JobBERT-v3 and averaged into one field-level vector. This puts the input back into the distribution the model was trained on.

**Pros:** Inputs align with the model's training distribution. Plant Engineering is now ranked exactly (pred 9, GT 9).
**Cons:** The 5-title sample per field is too sparse to fully cover all relevant skill clusters. Remaining misses (Automotive, Fabrication) suggest more titles or a different sampling strategy would help.

**Model size:** ~280 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 8. OpenAI Embeddings — `text-embedding-3-small`

Each work field is encoded as a dense vector using OpenAI's `text-embedding-3-small` model. The input text concatenates both language labels (`"Mechanical Engineering / Maschinenbau"`). All 180 fields are encoded in a single batched API call.

`text-embedding-3-small` was trained on a broad multilingual corpus and performs strongly across languages including German, at substantially lower cost than the large variant. Unlike the AlexU and skills-enriched approaches, no description enrichment is applied — embeddings are computed from raw field names alone.

**Pros:** Strong closed-source multilingual embeddings, single API call, deterministic.
**Cons:** Requires internet access and incurs API cost (~$0.02 for 180 fields). No description enrichment; raw names only.

**External dependency:** OpenAI API (`OPENAI_API_KEY` in `.env`)

---

## Ablation Studies

The new-repo SBERT implementation (identical model to `baseline`) reported 5/5 hits vs baseline's 2/5. Three things differ between them — the folders below each change exactly one variable to identify which factor is responsible.

| Folder | Change vs baseline | Hits | Avg Δ | Score | Verdict |
|--------|--------------------|:----:|:-----:|:-----:|---------|
| `baseline` *(reference)* | — | 2/5 | 2.00 | 1.56 | — |
| `baseline-top10/` | Neighbour count: 9 → 10 | 2/5 | 1.00 | 1.78 | **No help on hit count.** The 3 missed pairs are outside Mechanical Engineering's top-10 — they're only found when looking from *their* direction. |
| `baseline-max-sym/` | Symmetrization: code1's rank → max from either direction | **5/5** | 1.80 | **4.00** | **This was the key.** All 5 GT pairs are found by at least one direction; taking the max surfaces them. |
| `baseline-en-first/` | Text format: `"{nameDe} {nameEn}"` → `"{nameEn} / {nameDe}"` | 2/5 | 3.50 | 1.22 | **Slightly hurts.** German-first is marginally better for this model on this task. |

**Finding:** The entire 2→5 hit gap between `baseline` and the new-repo SBERT implementation comes from **max symmetrization**, not the wider top-10 net. Plant Engineering, Building Craft, and Civil Engineering all rank Mechanical Engineering highly in their own top-9, but Mechanical Engineering does not rank them in its top-9 — the relationship is asymmetric, and only max-sym captures it.

---

## Evaluation

Evaluated against a single ground truth sample (5 pairs for *Mechanical Engineering*) sourced from the Romagnolo platform.

| Approach | Hits (out of 5) | Avg rank delta | Score |
|---|:---:|:---:|:---:|
| baseline | 2 | 2.00 | 1.56 |
| techwolf-inspired | 2 | 3.00 | 1.33 |
| alexU-inspired | 3 | 2.33 | 2.22 |
| **llm-ranking** | **3** | **1.67** | **2.44** |
| skills-enriched | 3 | 2.33 | 2.22 |
| techwolf-jobtitles | 2 | 1.00 | 1.78 |

Score = `hits × (1 − avg_rank_delta / 9)` — rewards both coverage and rank accuracy.

Approaches **hybrid** and **openai-embeddings** have not yet been evaluated in this repo's format.

### Per-field breakdown (Mechanical Engineering)

| Field | GT | Baseline | TechWolf | AlexU | LLM-ranking | Skills-enriched | TechWolf+titles |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Plant Engineering | 9 | miss | 6 | 4 | 8 | 6 | **9** ✓ |
| Building Craft | 8 | miss | miss | miss | miss | miss | miss |
| Automotive | 7 | 4 | miss | **7** ✓ | **7** ✓ | 9 | miss |
| Fabrication | 6 | 5 | miss | 8 | 2 | 4 | miss |
| Civil Engineering | 5 | miss | 8 | miss | miss | 7 | 7 |

### Key findings

- **LLM-ranking wins** (score 2.44): Asking the LLM to rank directly beats all embedding approaches. The LLM reasons about occupational domain structure without needing to map it through a vector space, and its rank estimates for Plant Engineering (pred 8, GT 9) and Automotive (pred 7, GT 7 exact) are sharper.
- **AlexU and Skills-enriched tie** (2.22 each): Both LLM-enriched embedding approaches match on hits and avg delta. Structured skills/education data is no more informative than free-form descriptions for this task at this scale.
- **TechWolf + job titles improves over plain TechWolf** (1.78 vs 1.33): Generating concrete titles (e.g. "Mechanical Engineer", "Maschinenbauingenieur") puts the input back into JobBERT's training distribution. Plant Engineering is now ranked exactly (pred 9, GT 9). The remaining misses suggest the 5-title sample per field is still too sparse.
- **Building Craft is missed by every approach**: Its ground truth similarity to Mechanical Engineering (GT 8) appears to rely on domain knowledge that neither embeddings nor the LLM surface reliably — likely a candidate for manual overrides or ESCO skill-overlap signals.

---

## Design Decisions

**Value scale:** `10` is reserved exclusively for the diagonal (self-correlation). Off-diagonal values run `9` (most similar) → `1` (9th most similar), ensuring no ambiguity between "identical field" and "most similar other field".

**Sparsity:** Top-9 neighbors per field, union across both directions. No hard similarity floor beyond that — for niche fields, forcing low-similarity pairings is worse than a sparse neighbourhood.

**Symmetry:** Cosine similarity is symmetric by construction. For pairs nominated by both directions, the value is based on the code1 field's own ranking of its neighbors, ensuring unique and score-consistent ranks within each field's group.

**Upper triangle:** `code1` always corresponds to the field with the lower index in `work_fields.json`, matching the ordering in the task example.

---

## Limitations

- **Narrow ground truth:** Only 5 pairs for one field (`w_mash`). A broader ground truth across diverse fields would give more reliable comparisons.
- **Rank calibration:** Embedding approaches correctly identify related fields but may misjudge their ordering relative to each other, since cosine distance reflects distributional similarity rather than domain hierarchy.
- **LLM non-determinism:** Re-running the GPT approaches may produce slightly different rankings. The committed `correlation_matrix.json` files capture a specific run.
- **Cluster boundaries:** The manual taxonomy in the Hybrid approach is a single-membership scheme — fields that span multiple domains may lose cross-cluster signal.

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
