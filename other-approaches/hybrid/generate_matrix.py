#!/usr/bin/env python3
"""
Generate a correlation matrix over 180 occupational work fields.

Methodology: SBERT multilingual embeddings (paraphrase-multilingual-mpnet-base-v2)
as the base similarity matrix, with a +0.15 cosine similarity boost applied to
all pairs that share the same manually assigned domain cluster. Fifteen clusters
cover all 180 work fields. See README.md for full explanation.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer


DOMAIN_CLUSTERS: dict[str, list[str]] = {
    "IT_Software": [
        "w_seng", "w_swe", "w_ar", "w_itpm", "w_itcon", "w_itqm", "w_itman",
        "w_maint", "w_aana", "w_aadm", "w_sysa", "w_sadm", "w_nwa", "w_nwde",
        "w_dba", "w_dbde", "w_dwh", "w_hw", "w_emb", "w_tele", "w_info",
        "w_cc", "w_web",
    ],
    "Engineering": [
        "w_mash", "w_plant", "w_auto", "w_elec", "w_civil", "w_archi",
        "w_const", "w_syen", "w_faben", "w_energ", "w_synth", "w_space",
        "w_upstream", "w_statics", "w_siteman", "w_safety", "w_cad",
        "w_street", "w_mate", "w_heet", "w_medte", "w_qm",
    ],
    "Manufacturing": [
        "w_fabr", "w_craft", "w_metal", "w_wood", "w_monta", "w_tool",
        "w_stone", "w_food", "w_textil", "w_sete", "w_autom", "w_optic",
    ],
    "Finance": [
        "w_book", "w_acco", "w_tax", "w_audit", "w_bank", "w_insu",
        "w_insum", "w_inves", "w_asset", "w_sectr", "w_cont", "w_riskm",
        "w_payr", "w_rev",
    ],
    "Sales_Marketing": [
        "w_sale", "w_sain", "w_safo", "w_sama", "w_keyacc", "w_retai",
        "w_whole", "w_mark", "w_onlma", "w_adve", "w_mare", "w_pr",
        "w_comm", "w_ebus", "w_presa", "w_cust", "w_expo", "w_impo",
        "w_whkus",
    ],
    "HR_Education": [
        "w_pers", "w_hrdev", "w_recru", "w_pera", "w_train", "w_educ",
        "w_whuni", "w_whkig", "w_soci",
    ],
    "Healthcare": [
        "w_medic", "w_medas", "w_nurse", "w_ther", "w_doc", "w_sani",
        "w_phar", "w_klifo", "w_dent", "w_veter", "w_lab", "w_hospital",
    ],
    "Management": [
        "w_pm", "w_proce", "w_org", "w_top", "w_con", "w_busan",
        "w_prod", "w_scm", "w_munic", "w_ana",
    ],
    "Science": [
        "w_natur", "w_bio", "w_chem", "w_phys", "w_math", "w_scie",
        "w_envir", "w_rese",
    ],
    "Legal": [
        "w_judi", "w_advoc", "w_notar",
    ],
    "Admin_Office": [
        "w_admin", "w_offi", "w_secr", "w_ass", "w_busi", "w_arch", "w_rece",
    ],
    "Creative_Media": [
        "w_graph", "w_desig", "w_film", "w_phot", "w_music", "w_stage",
        "w_edito", "w_jour", "w_publ", "w_trans", "w_tred", "w_illumi",
        "w_modes", "w_hand", "w_muse", "w_room",
    ],
    "Logistics": [
        "w_frei", "w_ware", "w_deli", "w_propur", "w_purch", "w_techpur",
    ],
    "Services": [
        "w_hote", "w_beaut", "w_clean", "w_sport", "w_tour", "w_home",
        "w_relig", "w_event", "w_secu", "w_defe", "w_anim", "w_bestatt",
        "w_agri", "w_waste", "w_bibl", "w_hist",
    ],
    "Real_Estate": [
        "w_immo", "w_house", "w_faci",
    ],
}

CLUSTER_BOOST = 0.15


def load_fields(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def apply_cluster_boost(sim: np.ndarray, ids: list[str]) -> np.ndarray:
    id_to_cluster: dict[str, str] = {}
    for cluster, codes in DOMAIN_CLUSTERS.items():
        for code in codes:
            id_to_cluster[code] = cluster

    unclustered = [fid for fid in ids if fid not in id_to_cluster]
    if unclustered:
        print(f"  Warning: {len(unclustered)} fields have no cluster: {unclustered}")

    n = len(ids)
    boosted = sim.copy()
    for i in range(n):
        c1 = id_to_cluster.get(ids[i])
        for j in range(i + 1, n):
            c2 = id_to_cluster.get(ids[j])
            if c1 and c2 and c1 == c2:
                boosted[i][j] += CLUSTER_BOOST
                boosted[j][i] += CLUSTER_BOOST

    return np.clip(boosted, -1.0, 1.0)


def build_entries(ids: list[str], sim: np.ndarray) -> list[dict]:
    n = len(ids)

    pair_value: dict[tuple[int, int], int] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        for rank_k, j in enumerate(np.argsort(row)[::-1][:9]):
            lo, hi = min(i, j), max(i, j)
            val = 9 - rank_k
            pair_value[(lo, hi)] = max(pair_value.get((lo, hi), 0), val)

    entries: list[dict] = []
    for i in range(n):
        entries.append({"code1": ids[i], "code2": ids[i], "value": 10})
    for (lo, hi), val in pair_value.items():
        entries.append({"code1": ids[lo], "code2": ids[hi], "value": val})

    return entries


def main():
    fields = load_fields("../../work_fields.json")
    ids = [f["correlationMatrixId"] for f in fields]
    texts = [f"{field['nameDe']} {field['nameEn']}" for field in fields]

    print(f"Loaded {len(fields)} work fields.")
    print("Downloading / loading model (first run ~420 MB)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    print("Embedding fields...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Computing cosine similarity...")
    base_sim = cosine_similarity(embeddings)

    print(f"Applying cluster boost (+{CLUSTER_BOOST}) to same-cluster pairs...")
    sim = apply_cluster_boost(base_sim, ids)

    print("Building matrix entries...")
    entries = build_entries(ids, sim)

    output_path = "correlation_matrix.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_diag = len(fields)
    n_off = len(entries) - n_diag
    print(
        f"Done. {len(entries)} entries written to {output_path} "
        f"({n_diag} diagonal + {n_off} off-diagonal pairs)."
    )


if __name__ == "__main__":
    main()
