# Hybrid Approach

**Base model:** `paraphrase-multilingual-mpnet-base-v2`
**Augmentation:** manual domain-cluster similarity boost (+0.15)

## Method

Uses the SBERT multilingual similarity matrix as a base and adds +0.15 to the cosine similarity of all pairs that share the same manually assigned domain cluster. Fifteen clusters cover all 180 work fields:

`IT_Software`, `Engineering`, `Manufacturing`, `Finance`, `Sales_Marketing`, `HR_Education`, `Healthcare`, `Management`, `Science`, `Legal`, `Admin_Office`, `Creative_Media`, `Logistics`, `Services`, `Real_Estate`

The boost magnitude (+0.15) approximates the typical cosine similarity gap between related and unrelated fields  -  large enough to surface intra-cluster pairs in the top-9 without completely overriding the embedding signal.

**Motivation:** The baseline SBERT approach occasionally fails to recognise domain proximity when surface names are misleading. For instance, "Building Craft" and "Mechanical Engineering" share a physical-work domain but their names are not lexically similar. The cluster boost compensates for this by encoding human-curated domain knowledge.

## Run

```bash
python generate_matrix.py
```

## Notes

**Limitation:** Manual cluster assignments use single membership. Fields that genuinely span multiple domains (e.g., Medical Technology spans Healthcare and Engineering) lose cross-cluster signal. The boost missed the `w_mash` ↔ `w_craft` ground truth pair because Building Craft was assigned to Manufacturing while Mechanical Engineering was placed in Engineering  -  meaning the two fields were in different clusters and received no boost.
