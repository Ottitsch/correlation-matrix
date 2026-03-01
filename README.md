# METHODOLOGY AND THOUGHTS:

In Total I spent around 3 hours actually coding.
Research Time was mainly done at the gym or in the shower in the back of my head, so I did not count it.

----------------

First of all, I decided to optimize towards eval/GROUND_TURTH.png, as I believe we will be evaluated based upon the existing implementation.

I decided against simply web-scraping the whole romagnolo.ai, as it feels like this would go against the spirit of the competition.
However this means, that my ground_truth.json is just one sample and could easily just be a measurement of noise.

IF I was less confident, that we would be evaluated based on the current romagnolo.ai implementation I would spend more time into making a better eval by manually judging swe jobs and their correlations myself and if I had infinite time by interviewing domain experts.

----------------

Afterwards I decided to implement as many different approaches as I could within the time limit and measure which worked best.

In general the approaches can be categorized into 2 categories:
GENAI and NO-GENAI approaches, as it turns one of the NO-GENAI approaches worked the best, which leads me to believe that
romagnolo.ai most likely avoided GENAI in their Product as it also fits the whole company idea of danube.ai more
(Data Accountability and doing more with less).

----------------

A Major Benefit of the NO-GENAI approaches is that they are deterministic and don't require API costs and aren't reliant on external providers.

----------------

Sidenote on the papers folder:
They all rely on fine-tuning their models for a slightly different task (job titles instead of job fields), so while I was able to draw some inspiration from them often their models failed to outperform the baseline.

I believe to improve the baseline fine-tuning would be benefitial, but due to time constraints I avoided this.
The off the shelf model from Techwolf (JobBERT-V3) required me to use GENAI to create job title's for it to work properly.
But it still didn't outperform the baseline.

----------------


# FOLDER STRUCTURE:

```
correlation-matrix/
├── work_fields.json              # 180 work fields with German/English names and IDs
├── requirements.txt
├── README.md                     # this file
├── AI_README.md                  # detailed technical write-up (generated with Claude)
├── ideas_before_coding.md        # my raw notes and approach decisions before coding
│
├── submission/
│   └── baseline/                 # winning submission
│       ├── generate_matrix.py
│       └── correlation_matrix.json
│
├── other-approaches/             # all alternative approaches I tried
│   ├── techwolf-inspired/        # JobBERT-v3 on raw field names
│   ├── techwolf-jobtitles/       # JobBERT-v3 + LLM-generated job titles (requires Azure OpenAI)
│   ├── alexU-inspired/           # multilingual-e5 + LLM descriptions (requires Azure OpenAI)
│   ├── skills-enriched/          # multilingual-e5 + LLM skill/education data (requires Azure OpenAI)
│   ├── llm-ranking/              # pure GPT ranking, no embeddings (requires Azure OpenAI)
│   ├── hybrid/                   # SBERT + manual domain cluster boost
│   ├── baseline-top10/           # ablation: top-10 neighbours instead of top-9
│   ├── baseline-max-sym/         # ablation: max symmetrization (key finding)
│   └── baseline-en-first/        # ablation: English-first text format
│
├── eval/
│   ├── eval.py                   # compares all approaches against ground truth
│   ├── ground_truth.json         # 5 GT pairs for Mechanical Engineering
│   └── GROUND_TRUTH.png          # screenshot from romagnolo.ai used to derive GT
│
└── papers/                       # TalentCLEF 2025 papers I drew inspiration from
    ├── paper_364-AlexU-NLP.*
    ├── paper_367-TechWolf.*
    └── paper_375-pjmathematician.*
```

# REPRODUCTION:

```bash
pip install -r requirements.txt

# Run the winning approach (no API key needed, ~30s on CPU)
cd submission/baseline && python generate_matrix.py

# Run all other no-API approaches
cd other-approaches/hybrid && python generate_matrix.py
cd other-approaches/techwolf-inspired && python generate_matrix.py
cd other-approaches/baseline-top10 && python generate_matrix.py
cd other-approaches/baseline-max-sym && python generate_matrix.py
cd other-approaches/baseline-en-first && python generate_matrix.py

# Azure OpenAI approaches (KEY=... in a .env file required)
# The intermediate JSON files (descriptions/rankings/titles/enrichment) are
# already committed, so generate_matrix.py can be re-run without the API key.
cd other-approaches/alexU-inspired && python generate_matrix.py
cd other-approaches/llm-ranking && python generate_matrix.py
cd other-approaches/skills-enriched && python generate_matrix.py
cd other-approaches/techwolf-jobtitles && python generate_matrix.py

# Evaluate all approaches
python eval/eval.py
```

### .env STRUCTURE:
```bash
KEY=INSERT YOUR KEY
DEPLOYMENT = "gpt-5.2-chat"
ENDPOINT = "INSERT YOUR AZURE ENDPOINT"
API_VERSION = "2025-04-01-preview"
```