#!/usr/bin/env python3
"""
Generate bilingual (EN + DE) descriptions for all 180 work fields via Azure OpenAI.
Saves output to descriptions.json, which generate_matrix.py reads.
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
OUTPUT_PATH = "descriptions.json"

PROMPT_TEMPLATE = """\
Write a brief professional description of the work field "{name_en}" (German: "{name_de}").

Respond with exactly this format and nothing else:
EN: [2-3 sentences in English describing what this field involves, typical roles, and key skills]
DE: [2-3 Sätze auf Deutsch, die beschreiben, was dieses Berufsfeld umfasst, typische Rollen und wichtige Fähigkeiten]"""


def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["KEY"],
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
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


def parse_response(text: str) -> tuple[str, str]:
    """Extract EN and DE descriptions from the model response."""
    en = de = ""
    en_match = re.search(r"EN:\s*(.+?)(?=\nDE:|\Z)", text, re.DOTALL)
    de_match = re.search(r"DE:\s*(.+?)(?=\nEN:|\Z)", text, re.DOTALL)
    if en_match:
        en = en_match.group(1).strip()
    if de_match:
        de = de_match.group(1).strip()
    return en, de


def main():
    with open("../work_fields.json", encoding="utf-8") as f:
        fields = json.load(f)

    # Resume from existing output if present
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = {e["correlationMatrixId"]: e for e in json.load(f)}
        print(f"Resuming — {len(existing)} descriptions already done.")
    except FileNotFoundError:
        existing = {}

    results = dict(existing)
    client = build_client()

    for i, field in enumerate(fields):
        cid = field["correlationMatrixId"]
        if cid in results:
            continue

        prompt = PROMPT_TEMPLATE.format(
            name_en=field["nameEn"], name_de=field["nameDe"]
        )

        print(f"[{i + 1}/{len(fields)}] {field['nameEn']}...", end=" ", flush=True)
        raw = call_api(client, prompt)
        en, de = parse_response(raw)
        print("done")

        results[cid] = {
            "correlationMatrixId": cid,
            "descriptionEn": en,
            "descriptionDe": de,
        }

        # Save after every field so progress isn't lost on crash
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} descriptions saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
