# Baseline Approach

**Model:** `paraphrase-multilingual-mpnet-base-v2`

## Method

Each work field is represented as the concatenation of its German and English names (e.g. `"Telekommunikation Telecommunication"`). These short texts are embedded using a general-purpose multilingual sentence transformer, and pairwise cosine similarity is computed over the resulting vectors.

No descriptions, no fine-tuning, no external data.

## Run

```bash
python generate_matrix.py
```

## Results (sample: Telecommunication neighbors)

| Value | Field |
|------:|-------|
| 9 | Corporate Communication |
| 8 | Service Technic |
| 7 | Network Administration |
| 6 | Network Development |
| 5 | Delivery Services |

Gets the domain right (Network Admin, Network Development) but also pulls in unrelated fields (Delivery Services, Live-Saving Service) due to shallow text overlap. Serves as the lower bound for comparison.
