# AlexU-NLP-Inspired Approach

**Model:** `intfloat/multilingual-e5-large-instruct`

**Paper:** *AlexU-NLP at TalentCLEF 2025: Curriculum-Driven Hybrid Retrieval for Multilingual Job Title Matching*  -  Barakat, Mokhtar, Torki, Elmakky (Alexandria University, CLEF 2025)

## Method

Two-step pipeline replicating ALEXU's inference-time description enrichment (their Table 4 ablation) without any fine-tuning:

1. **Description generation** (`generate_descriptions.py`): `gpt-5.2-chat` via Azure OpenAI generates a bilingual description (2–3 sentences in English + 2–3 sentences in German) for each of the 180 work fields.

2. **Embedding** (`generate_matrix.py`): Each field is represented as `"{nameDe} {nameEn}\n{descriptionEn}\n{descriptionDe}"` and embedded with `multilingual-e5-large-instruct` using a symmetric similarity instruction prefix.

The rationale: short 2–3 word field names give embeddings very little signal. Adding a descriptive paragraph disambiguates semantically similar but lexically distinct fields. AlexU's Table 4 ablation shows descriptions improve mAP by ~2.6% on their fine-tuned model; the signal is strong enough that it plausibly carries over to zero-shot use.

## Run

```bash
# Step 1  -  generate descriptions (resumes automatically if interrupted)
python generate_descriptions.py

# Step 2  -  generate matrix
python generate_matrix.py
```

## Results (sample: Telecommunication neighbors)

| Value | Field |
|------:|-------|
| 9 | Network Administration |
| 8 | Network Development |
| 7 | System Engineering |
| 6 | System Administration |
| 5 | Construction |

Strongest results across all three approaches. The descriptions allow the model to disambiguate fields that share short names but differ in meaning, and to surface domain-relevant neighbors (Network Admin, Network Development, System Engineering) that name-only approaches miss.

## Notes

`generate_descriptions.py` saves progress after every field to `descriptions.json`. If interrupted, re-running it picks up from where it left off. Requires a `.env` file in the parent directory with a `KEY` variable set to your Azure OpenAI API key.
