# Work Field Correlation Matrix

A semantic correlation matrix over the 180 occupational work fields used by danube.ai, generated with multiple approaches and evaluated against ground truth.

## Background

The matrix is used in the **Romagnolo** HR platform for:
1. **Candidate matching** - correlated fields compensate for incomplete job ads by giving bonus scores to candidates with related fields.
2. **Smart search completions** - searching for a work field surfaces highly correlated ones as suggestions.

---

## Reproduction

!!! IF YOU DONT WANT TO RE-RUN GENAI ONLY RUN generate_matrix.py NO API KEY REQUIRED !!!

```bash
pip install -r requirements.txt

# Baseline (submission)
cd submission/baseline && python generate_matrix.py

# TechWolf-inspired
cd other-approaches/techwolf-inspired && python generate_matrix.py

# Hybrid (SBERT + domain cluster boost)
cd other-approaches/hybrid && python generate_matrix.py

# AlexU-inspired (requires Azure OpenAI key in .env as KEY=...)
cd other-approaches/alexU-inspired
python generate_descriptions.py   # generates descriptions.json, resumes if interrupted
python generate_matrix.py

# LLM ranking (requires Azure OpenAI key in .env as KEY=...)
cd other-approaches/llm-ranking
python generate_rankings.py       # generates rankings.json, resumes if interrupted
python generate_matrix.py

# Skills-enriched (requires Azure OpenAI key in .env as KEY=...)
cd other-approaches/skills-enriched
python generate_enrichment.py     # generates enrichment.json, resumes if interrupted
python generate_matrix.py

# TechWolf + job titles (requires Azure OpenAI key in .env as KEY=...)
cd other-approaches/techwolf-jobtitles
python generate_titles.py         # generates titles.json, resumes if interrupted
python generate_matrix.py

# Baseline ablations (all free, no API key required)
cd other-approaches/baseline-top10 && python generate_matrix.py
cd other-approaches/baseline-max-sym && python generate_matrix.py
cd other-approaches/baseline-en-first && python generate_matrix.py

# Evaluate all approaches against ground truth
python eval/eval.py
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
- `value`: integer rank. `10` reserved for self-correlation (diagonal), `9` (most similar) to `1` (9th most similar) for off-diagonal entries

---

## Approaches

Seven approaches were implemented, each in its own folder with a dedicated README. The best-performing output is in `submission/baseline/correlation_matrix.json` (or equivalently `other-approaches/techwolf-jobtitles/correlation_matrix.json` for a job-title-grounded alternative).

### Overview

| # | Approach | Mechanism | API Required | Hits (5 GT pairs) | Score |
|---|----------|-----------|:---:|:---:|:---:|
| 1 | **Baseline** | Local SBERT embeddings (paraphrase-multilingual-mpnet-base-v2) | - | **5/5** | **4.00** |
| 2 | **TechWolf-inspired** | JobBERT-v3 embeddings, names embedded separately and averaged | - | 3/5 | 2.22 |
| 3 | **Hybrid** | SBERT embeddings + manual domain-cluster similarity boost (+0.15) | - | 4/5 | 3.44 |
| 4 | **AlexU-inspired** | multilingual-e5-large-instruct + LLM-generated bilingual descriptions | Azure OpenAI | 3/5 | 2.44 |
| 5 | **LLM ranking** | Direct GPT ranking, no embeddings | Azure OpenAI | 3/5 | 2.78 |
| 6 | **Skills-enriched** | multilingual-e5-large-instruct + structured skills/education data | Azure OpenAI | 3/5 | 2.78 |
| 7 | **TechWolf + job titles** | JobBERT-v3 + LLM-generated concrete job titles | Azure OpenAI | 4/5 | 3.56 |
| - | **baseline-top10** *(ablation)* | Baseline but top-10 neighbours (wider net) | - | 2/5 | 1.78 |
| - | **baseline-max-sym** *(ablation)* | Baseline but max rank from either direction | - | **5/5** | **4.00** |
| - | **baseline-en-first** *(ablation)* | Baseline but `"nameEn / nameDe"` text format | - | 2/5 | 1.22 |

Score = `hits × (1 − avg_rank_delta / 9)`. All approaches now use max symmetrization (finding from ablation study). See **Ablation Studies** section for findings.

---

### 1. Baseline: `paraphrase-multilingual-mpnet-base-v2`

The simplest approach. Both language names are concatenated (`"Telekommunikation Telecommunication"`) and embedded with a general-purpose multilingual sentence transformer. No external data, no descriptions.

**Pros:** Free, reproducible, fast (~30 s on CPU), no API key needed.
**Cons:** Rankings reflect distributional word similarity: it cannot reason about occupational career paths or the physical nature of work.

**Model size:** ~420 MB

---

### 2. TechWolf-inspired: `TechWolf/JobBERT-v3`

Inspired by *Multilingual JobBERT for Cross-Lingual Job Title Matching* (Decorte, De Lange, Van Hautte, TechWolf, CLEF 2025).

JobBERT-v3 is trained via contrastive learning on 21 million job titles paired with ESCO skill annotations, giving it a skill-grounded semantic space. German and English names are embedded **separately** and averaged rather than concatenated, since the model was trained on monolingual job titles.

**Pros:** Skill-grounded representations, no API needed.
**Cons:** Abstract category names like "Telecommunication" are far from the concrete job-ad titles the model was trained on, hurting retrieval quality.

**Model size:** ~280 MB

---

### 3. Hybrid: `paraphrase-multilingual-mpnet-base-v2` + domain cluster boost

Uses the SBERT similarity matrix as a base and adds +0.15 to all pairs that share the same manually assigned domain cluster. Fifteen clusters were defined (`IT_Software`, `Engineering`, `Manufacturing`, `Finance`, `Sales_Marketing`, `HR_Education`, `Healthcare`, `Management`, `Science`, `Legal`, `Admin_Office`, `Creative_Media`, `Logistics`, `Services`, `Real_Estate`) covering all 180 fields.

The boost magnitude (+0.15) approximates the typical cosine similarity gap between related and unrelated fields, enough to surface intra-cluster pairs in the top-9 without completely overriding the embedding signal.

**Pros:** Adds interpretable domain knowledge, deterministic, no API costs.
**Cons:** Single-membership clustering loses cross-domain signal for fields that span multiple domains (e.g., Medical Technology spans Healthcare and Engineering).

**Model size:** ~420 MB

---

### 4. AlexU-inspired: `intfloat/multilingual-e5-large-instruct` + LLM descriptions

Inspired by *AlexU-NLP at TalentCLEF 2025* (Barakat, Mokhtar, Torki, Elmakky, Alexandria University, CLEF 2025), specifically their inference-time description enrichment ablation (Table 4): adding descriptions to corpus entries improves mAP by ~2.6% even on their fine-tuned model, suggesting the signal is robust and likely carries over to zero-shot use.

`gpt-5.2-chat` via Azure OpenAI generates a bilingual description (EN + DE, 2–3 sentences each) for every work field. Each field is then embedded as `"{nameDe} {nameEn}\n{descriptionEn}\n{descriptionDe}"` using `multilingual-e5-large-instruct` with a symmetric similarity instruction prefix.

**Pros:** Richer semantic signal from descriptions, strong multilingual model.
**Cons:** Requires API access, descriptions add latency on first run.

**Model size:** ~560 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 5. LLM ranking: direct GPT ranking

No embeddings at all. For each of the 180 fields, the full field list is passed to the LLM and it is asked to rank the top 9 most similar fields by skill overlap, domain proximity, and career path. Similarity is inferred entirely from the model's world knowledge about occupational domains.

**Pros:** Genuine occupational domain reasoning, not just lexical similarity. Sharpest rank accuracy (avg delta 0.67) among all approaches.
**Cons:** Non-deterministic: re-running may produce slightly different rankings. Slower than embedding approaches (~4 min for 180 fields).

**External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 6. Skills-enriched: `multilingual-e5-large-instruct` + structured requirements

Extension of the AlexU approach with more structured LLM output. Instead of a free-form description, the LLM generates a comma-separated list of required skills and typical educational paths for each field. The embedding text becomes `"{nameDe} {nameEn}\nSkills: ...\nEducation: ..."`. The hypothesis: two fields that require the same skills and education are similar regardless of how their names sound, grounding similarity in actual job market requirements rather than surface text.

**Pros:** Structured skill/education data may generalise better across naming conventions.
**Cons:** Structured data is no more informative than free-form descriptions for this task at this scale (tied with AlexU on the eval sample).

**Model size:** ~560 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

### 7. TechWolf + job titles: `TechWolf/JobBERT-v3` + LLM-generated concrete titles

Addresses the root cause of TechWolf-inspired's underperformance: JobBERT-v3 was trained on concrete job ad titles, not abstract category names. An LLM generates 5 representative job titles in both German and English for each field; all 10 titles are embedded separately with JobBERT-v3 and averaged into one field-level vector. This puts the input back into the distribution the model was trained on.

**Pros:** Inputs align with the model's training distribution. Plant Engineering is ranked exactly (pred 9, GT 9); Automotive, Fabrication, and Civil Engineering all hit.
**Cons:** The 5-title sample per field is too sparse to fully cover all relevant skill clusters. Building Craft is still missed: it may require more craft/manual-trade titles to appear in the field's neighbourhood.

**Model size:** ~280 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

## Ablation Studies

The new-repo SBERT implementation (identical model to `baseline`) reported 5/5 hits vs baseline's 2/5. Three things differ between them: the folders below each change exactly one variable to identify which factor is responsible.

| Folder | Change vs reference | Hits | Avg Δ | Score | Verdict |
|--------|--------------------|:----:|:-----:|:-----:|---------|
| *(reference: code1's-perspective baseline)* | - | 2/5 | 2.00 | 1.56 | - |
| `baseline-top10/` | Neighbour count: 9 → 10 | 2/5 | 1.00 | 1.78 | **No help on hit count.** The 3 missed pairs are outside Mechanical Engineering's top-10: they're only found when looking from *their* direction. |
| `baseline-max-sym/` | Symmetrization: code1's rank → max from either direction | **5/5** | 1.80 | **4.00** | **This was the key.** All 5 GT pairs are found by at least one direction; taking the max surfaces them. |
| `baseline-en-first/` | Text format: `"{nameDe} {nameEn}"` → `"{nameEn} / {nameDe}"` | 2/5 | 3.50 | 1.22 | **Slightly hurts.** German-first is marginally better for this model on this task. |

**Finding:** The entire 2→5 hit gap between `baseline` and the new-repo SBERT implementation comes from **max symmetrization**, not the wider top-10 net. Plant Engineering, Building Craft, and Civil Engineering all rank Mechanical Engineering highly in their own top-9, but Mechanical Engineering does not rank them in its top-9: the relationship is asymmetric, and only max-sym captures it.

---

## Evaluation

Evaluated against a single ground truth sample (5 pairs for *Mechanical Engineering*) sourced from the Romagnolo platform.

| Approach | Hits (out of 5) | Avg rank delta | Score |
|---|:---:|:---:|:---:|
| **baseline** | **5** | 1.80 | **4.00** |
| **techwolf-jobtitles** | **4** | 1.00 | 3.56 |
| hybrid | 4 | 1.25 | 3.44 |
| llm-ranking | 3 | 0.67 | 2.78 |
| skills-enriched | 3 | 0.67 | 2.78 |
| alexU-inspired | 3 | 1.67 | 2.44 |
| techwolf-inspired | 3 | 2.33 | 2.22 |

Score = `hits × (1 − avg_rank_delta / 9)`: rewards both coverage and rank accuracy. All approaches use max symmetrization.

### Per-field breakdown (Mechanical Engineering)

| Field | GT | Baseline | TechWolf | AlexU | LLM-ranking | Skills-enriched | TechWolf+titles | Hybrid |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Plant Engineering | 9 | 5 ✓ | 6 ✓ | 6 ✓ | **9** ✓ | **9** ✓ | **9** ✓ | 6 ✓ |
| Building Craft | 8 | 5 ✓ | miss | miss | miss | miss | miss | miss |
| Automotive | 7 | 8 ✓ | 6 ✓ | 8 ✓ | 8 ✓ | 9 ✓ | 6 ✓ | **9** ✓ |
| Fabrication | 6 | **6** ✓ | miss | 7 ✓ | 7 ✓ | **6** ✓ | **6** ✓ | **6** ✓ |
| Civil Engineering | 5 | 4 ✓ | 8 ✓ | miss | miss | miss | 8 ✓ | 5 ✓ |

### Key findings

- **Baseline is first** (score 4.00, 5/5 hits): After applying max symmetrization, the simple SBERT baseline matches the ablation result. Plant Engineering, Building Craft, and Civil Engineering all rank Mechanical Engineering highly in their own top-9 even though Mechanical Engineering doesn't return the favour: max-sym captures this asymmetry and surfaces all 5 GT pairs.
- **TechWolf + job titles is second** (score 3.56, 4/5 hits): Generating concrete job titles puts inputs back into JobBERT's training distribution. Plant Engineering is ranked exactly (pred 9, GT 9), Automotive and Fabrication both hit: only Building Craft is still missed.
- **Hybrid is third** (score 3.44, 4/5 hits): The domain cluster boost recovers Civil Engineering exactly (pred 5, GT 5) and pushes Automotive to 9, but cannot help Building Craft: it sits in the Manufacturing cluster while Mechanical Engineering is in Engineering, so the two fields receive no boost and the pair is missed.
- **LLM-ranking and Skills-enriched tie for fourth** (score 2.78, 3/5 hits): LLM ranking offers the sharpest rank accuracy (avg delta 0.67) among the 3-hit group, confirming strong domain reasoning. Both miss Building Craft and Civil Engineering.
- **AlexU-inspired is sixth** (score 2.44, 3/5 hits): LLM-generated descriptions give a mild edge over TechWolf-inspired (avg delta 1.67 vs 2.33) but don't improve hit count.
- **TechWolf-inspired is last** (score 2.22, 3/5 hits): Abstract category names fall far outside JobBERT's training distribution of concrete job-ad titles.
- **Building Craft is the hardest GT pair** (missed by 6/7 approaches): Only the baseline surfaces it via max-sym: Building Craft places Mechanical Engineering in its own top-9, so the pair is captured from that direction. All other approaches miss it entirely.

---

## Design Decisions

**Value scale:** `10` is reserved exclusively for the diagonal (self-correlation). Off-diagonal values run `9` (most similar) → `1` (9th most similar), ensuring no ambiguity between "identical field" and "most similar other field".

**Sparsity:** Top-9 neighbors per field, union across both directions. No hard similarity floor beyond that: for niche fields, forcing low-similarity pairings is worse than a sparse neighbourhood.

**Symmetry:** A pair `(i, j)` is included if `j` is in `i`'s top-9 *or* `i` is in `j`'s top-9. Its value is the **maximum** rank score from either direction: so if Civil Engineering ranks Mechanical Engineering 4th but Mechanical Engineering doesn't rank Civil Engineering at all, the pair still gets a value of 6 (= 9 − 3). This max symmetrization was identified as the key factor enabling 5/5 ground truth recall; see the Ablation Studies section.

**Upper triangle:** `code1` always corresponds to the field with the lower index in `work_fields.json`, matching the ordering in the task example.

---

## Limitations

- **Narrow ground truth:** Only 5 pairs for one field (`w_mash`). A broader ground truth across diverse fields would give more reliable comparisons.
- **Rank calibration:** Embedding approaches correctly identify related fields but may misjudge their ordering relative to each other, since cosine distance reflects distributional similarity rather than domain hierarchy.
- **LLM non-determinism:** Re-running the GPT approaches may produce slightly different rankings. The committed `correlation_matrix.json` files capture a specific run.
- **Cluster boundaries:** The manual taxonomy in the Hybrid approach is a single-membership scheme: fields that span multiple domains may lose cross-cluster signal.

---

## Outlook

With more time, interesting directions would include:

- **Larger eval**: one ground truth sample is not enough to draw strong conclusions. Interviewing domain experts or judging more job titles manually would give more reliable signal.
- **Ensemble**: blend similarity scores from multiple approaches (e.g. AlexU + baseline) before ranking.
- **ESCO skill mapping**: map the 180 fields to ESCO skill sets and use skill overlap as a complementary signal: the TechWolf-inspired Option B that was not pursued here.
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
