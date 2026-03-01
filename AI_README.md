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

Seven approaches were implemented, each in its own folder with a dedicated README. Three approaches are tied for best: `submission/baseline/`, `other-approaches/baseline-max-sym/`, and `other-approaches/alexU-inspired/` all score **2.44** (3/5 GT hits, avg rank delta 1.67).

### Overview

| # | Approach | Mechanism | API Required | Hits (5 GT pairs) | Score |
|---|----------|-----------|:---:|:---:|:---:|
| 1 | **Baseline** | Local SBERT embeddings (paraphrase-multilingual-mpnet-base-v2) | - | **3/5** | **2.44** |
| 2 | **AlexU-inspired** | multilingual-e5-large-instruct + LLM-generated bilingual descriptions | Azure OpenAI | **3/5** | **2.44** |
| 3 | **TechWolf + job titles** | JobBERT-v3 + LLM-generated concrete job titles | Azure OpenAI | 2/5 | 2.00 |
| 4 | **Skills-enriched** | multilingual-e5-large-instruct + structured skills/education data | Azure OpenAI | 2/5 | 1.78 |
| 5 | **Hybrid** | SBERT embeddings + manual domain-cluster similarity boost (+0.15) | - | 2/5 | 1.67 |
| 6 | **TechWolf-inspired** | JobBERT-v3 embeddings, names embedded separately and averaged | - | 2/5 | 1.33 |
| 7 | **LLM ranking** | Direct GPT ranking, no embeddings | Azure OpenAI | 1/5 | 0.89 |
| - | **baseline-max-sym** *(ablation)* | Baseline but max rank from either direction | - | **3/5** | **2.44** |
| - | **baseline-top10** *(ablation)* | Baseline but top-10 neighbours (wider net) | - | 2/5 | 1.56 |
| - | **baseline-en-first** *(ablation)* | Baseline but `"nameEn / nameDe"` text format | - | 2/5 | 1.22 |

Score = `hits × (1 − avg_rank_delta / 9)`. All approaches use max-rank symmetrization with per-tier deduplication to guarantee unique ranks per field. See **Ablation Studies** section for findings.

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

**Pros:** Genuine occupational domain reasoning, not just lexical similarity.
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

**Pros:** Inputs align with the model's training distribution. Plant Engineering is ranked exactly (pred 9, GT 9) and Fabrication hits exactly (pred 6, GT 6): perfect rank accuracy on both hits (avg delta 0.00).
**Cons:** The 5-title sample per field is too sparse to fully cover all relevant skill clusters. Only 2/5 GT pairs are surfaced; Automotive, Building Craft, and Civil Engineering are all missed.

**Model size:** ~280 MB | **External dependency:** Azure OpenAI (`KEY` in `.env`)

---

## Ablation Studies

The new-repo SBERT implementation (identical model to `baseline`) reported 5/5 hits vs baseline's 2/5. Three things differ between them: the folders below each change exactly one variable to identify which factor is responsible.

| Folder | Change vs reference | Hits | Avg Δ | Score | Verdict |
|--------|--------------------|:----:|:-----:|:-----:|---------|
| *(reference: code1's-perspective baseline, pre-fix)* | - | 2/5 | 2.00 | 1.56 | - |
| `baseline-top10/` | Neighbour count: 9 → 10 | 2/5 | 2.00 | 1.56 | **No help on hit count.** Missed pairs are only found from the *other* field's direction; a wider net per code1 doesn't surface them. |
| `baseline-max-sym/` | Symmetrization: code1's rank → max from either direction | **3/5** | 1.67 | **2.44** | **Key insight confirmed.** Max-sym surfaces pairs that are asymmetric — but with the top-9-per-code1 constraint correctly enforced, some GT pairs are still displaced by stronger neighbors within the same tier. |
| `baseline-en-first/` | Text format: `"{nameDe} {nameEn}"` → `"{nameEn} / {nameDe}"` | 2/5 | 3.50 | 1.22 | **Slightly hurts.** German-first is marginally better for this model on this task. |

**Finding:** Max symmetrization was the key insight — Plant Engineering, Building Craft, and Civil Engineering all rank Mechanical Engineering highly in their own top-9, but the reverse is not true. The pre-fix approaches reached 5/5 by emitting all nominated pairs without a per-code1 cap (outputting 31 entries for w_mash, violating the top-10 constraint). After enforcing the constraint correctly, the ceiling drops to 3/5: Automotive and Building Craft are displaced from w_mash's top-9 by other nominally higher-ranked pairs.

---

## Evaluation

Evaluated against a single ground truth sample (5 pairs for *Mechanical Engineering*) sourced from the Romagnolo platform.

| Approach | Hits (out of 5) | Avg rank delta | Score |
|---|:---:|:---:|:---:|
| **baseline** | **3** | 1.67 | **2.44** |
| **alexU-inspired** | **3** | 1.67 | **2.44** |
| **baseline-max-sym** *(ablation)* | **3** | 1.67 | **2.44** |
| techwolf-jobtitles | 2 | 0.00 | 2.00 |
| skills-enriched | 2 | 1.00 | 1.78 |
| hybrid | 2 | 1.50 | 1.67 |
| baseline-top10 *(ablation)* | 2 | 2.00 | 1.56 |
| techwolf-inspired | 2 | 3.00 | 1.33 |
| baseline-en-first *(ablation)* | 2 | 3.50 | 1.22 |
| llm-ranking | 1 | 1.00 | 0.89 |

Score = `hits × (1 − avg_rank_delta / 9)`: rewards both coverage and rank accuracy. All approaches use max-rank symmetrization with per-tier deduplication.

### Per-field breakdown (Mechanical Engineering)

| Field | GT | Baseline | AlexU | TechWolf+titles | Skills-enriched | Hybrid | TechWolf-inspired | LLM-ranking |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Plant Engineering | 9 | 5 ✓ | 6 ✓ | **9** ✓ | miss | 6 ✓ | 6 ✓ | miss |
| Building Craft | 8 | miss | miss | miss | miss | miss | miss | miss |
| Automotive | 7 | miss | 8 ✓ | miss | **9** ✓ | miss | miss | 8 ✓ |
| Fabrication | 6 | **6** ✓ | 7 ✓ | **6** ✓ | **6** ✓ | miss | miss | miss |
| Civil Engineering | 5 | 4 ✓ | miss | miss | miss | **5** ✓ | 8 ✓ | miss |

### Key findings

- **Baseline, AlexU, and baseline-max-sym tie at first** (score 2.44, 3/5 hits): With the top-9 sparsity constraint correctly enforced, these three approaches share the ceiling. Baseline and baseline-max-sym are now algorithmically identical (both use max-rank symmetrization with per-tier dedup); AlexU hits a different subset of 3 pairs (Plant, Automotive, Fabrication vs. Baseline's Plant, Fabrication, Civil).
- **TechWolf + job titles is fourth** (score 2.00, 2/5 hits): Despite only 2 hits, it scores 2.00 because it nails both predictions exactly (Plant: pred 9 = GT 9, Fabrication: pred 6 = GT 6, avg delta 0.00). The job-title enrichment puts inputs into JobBERT's training distribution — but the 5-title sample is too sparse to cover all relevant skill clusters.
- **Skills-enriched** (score 1.78, 2/5 hits): Hits Automotive and Fabrication; misses Plant, Building Craft, and Civil. The structured skills/education embedding shifts the model's focus toward skill overlap rather than domain names.
- **Hybrid** (score 1.67, 2/5 hits): The +0.15 cluster boost recovers Civil Engineering exactly (pred 5 = GT 5) but misses Fabrication. Building Craft (Manufacturing cluster) receives no boost against Mechanical Engineering (Engineering cluster), so the hardest GT pair remains missed.
- **LLM-ranking scores last among named approaches** (score 0.89, 1/5 hits): Without embeddings, the ranking only finds Automotive. Strong rank accuracy on that single hit (pred 8, GT 7, delta 1), but coverage is poor.
- **Building Craft is missed by every approach**: With the top-9 constraint enforced, no approach surfaces the (w_mash, w_craft) pair. The pre-fix approaches reached it by emitting 31+ entries for w_mash without a per-code1 cap — a constraint violation that has since been corrected.

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
