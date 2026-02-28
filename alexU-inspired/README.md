# AlexU-NLP-Inspired Approach

**Model:** `intfloat/multilingual-e5-large-instruct`

**Paper:** *AlexU-NLP at TalentCLEF 2025: Curriculum-Driven Hybrid Retrieval for Multilingual Job Title Matching* — Barakat, Mokhtar, Torki, Elmakky (Alexandria University, CLEF 2025)

## Method

Two-step pipeline replicating ALEXU's inference-time description enrichment (their Table 4 ablation) without any fine-tuning:

1. **Description generation** (`generate_descriptions.py`): A local `qwen2.5:14b` model (via Ollama) generates a bilingual description (2–3 sentences in English + 2–3 sentences in German) for each of the 180 work fields.

2. **Embedding** (`generate_matrix.py`): Each field is represented as `"{nameDe} {nameEn}\n{descriptionEn}\n{descriptionDe}"` and embedded with `multilingual-e5-large-instruct` using a symmetric similarity instruction prefix.

The rationale: short 2–3 word field names give embeddings very little signal. Adding a descriptive paragraph disambiguates semantically similar but lexically distinct fields. ALEXU showed this alone improves mAP by ~3% zero-shot.

## Run

```bash
# Step 1 — generate descriptions (resumes automatically if interrupted)
python generate_descriptions.py

# Step 2 — generate matrix
python generate_matrix.py
```

## Notes

`generate_descriptions.py` saves progress after every field to `descriptions.json`. If interrupted, re-running it picks up from where it left off.
