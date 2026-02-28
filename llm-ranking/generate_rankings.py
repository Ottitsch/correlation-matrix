#!/usr/bin/env python3
"""
For each work field, ask an LLM to directly rank the 9 most similar fields
from the full list of 180. Saves output to rankings.json.

Run this first, then generate_matrix.py to build the correlation matrix.
"""

import json
import os
import re
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv("../.env")

DEPLOYMENT = "gpt-5.2-chat"
ENDPOINT = "https://e1202-ml1273va-swedencentral.cognitiveservices.azure.com/"
API_VERSION = "2025-04-01-preview"
OUTPUT_PATH = "rankings.json"
TOP_K = 9

PROMPT_TEMPLATE = """\
You are an expert in occupational classification and labor market analysis.

Work field: "{name_de}" / "{name_en}"

From the list below, identify the {top_k} most similar work fields, ordered from most similar (rank 1) to least similar (rank {top_k}). Consider skill overlap, domain proximity, and typical career paths.

Return ONLY a JSON array of {top_k} correlationMatrixId strings, most similar first. No explanation, no extra text.

Fields:
{field_list}"""


def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["KEY"],
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
    )


def build_field_list(fields: list[dict]) -> str:
    return "\n".join(
        f"  {f['correlationMatrixId']}: {f['nameDe']} / {f['nameEn']}"
        for f in fields
    )


def call_api(client: AzureOpenAI, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1} after error: {e}")
                time.sleep(2)
            else:
                raise


def parse_ranking(text: str, valid_ids: set[str]) -> list[str]:
    """Extract a JSON array of correlationMatrixIds from LLM response."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            ids = json.loads(match.group())
            return [i for i in ids if i in valid_ids]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def main():
    with open("../work_fields.json", encoding="utf-8") as f:
        fields = json.load(f)

    valid_ids = {f["correlationMatrixId"] for f in fields}
    field_list_str = build_field_list(fields)

    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            rankings = json.load(f)
        print(f"Resuming — {len(rankings)} rankings already done.")
    except FileNotFoundError:
        rankings = {}

    client = build_client()

    for i, field in enumerate(fields):
        cid = field["correlationMatrixId"]
        if cid in rankings:
            continue

        prompt = PROMPT_TEMPLATE.format(
            name_de=field["nameDe"],
            name_en=field["nameEn"],
            top_k=TOP_K,
            field_list=field_list_str,
        )

        print(f"[{i + 1}/{len(fields)}] {field['nameEn']}...", end=" ", flush=True)
        raw = call_api(client, prompt)
        ranked = parse_ranking(raw, valid_ids)
        ranked = [r for r in ranked if r != cid][:TOP_K]
        print(f"{len(ranked)} neighbors")

        rankings[cid] = ranked

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(rankings)} rankings saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
