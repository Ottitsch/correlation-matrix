#!/usr/bin/env python3
"""
For each work field, ask an LLM to generate 5 representative concrete job titles
in both German and English. Saves output to titles.json.

JobBERT-v3 was trained on real job ad titles like "Netzwerkingenieur" or
"Software Developer", not abstract category names like "Telecommunication".
Generating concrete titles bridges that gap and lets the model use its
skill-grounded semantic space as intended.

Run this first, then generate_matrix.py.
"""

import json
import os
import re
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv("../.env")

DEPLOYMENT = os.environ["DEPLOYMENT"]
ENDPOINT = os.environ["ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]
OUTPUT_PATH = "titles.json"

PROMPT_TEMPLATE = """\
List 5 concrete, realistic job titles for someone working in the field of "{name_en}" (German: "{name_de}").

Respond with exactly this format and nothing else:
EN: [title1, title2, title3, title4, title5]
DE: [Titel1, Titel2, Titel3, Titel4, Titel5]

Use short, realistic job ad titles (e.g. "Software Engineer", "Network Administrator"). No descriptions."""


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


def parse_response(text: str) -> tuple[list[str], list[str]]:
    en_titles = de_titles = []
    en_match = re.search(r"EN:\s*\[(.+?)\]", text, re.DOTALL)
    de_match = re.search(r"DE:\s*\[(.+?)\]", text, re.DOTALL)
    if en_match:
        en_titles = [t.strip().strip('"') for t in en_match.group(1).split(",")]
    if de_match:
        de_titles = [t.strip().strip('"') for t in de_match.group(1).split(",")]
    return en_titles, de_titles


def main():
    with open("../work_fields.json", encoding="utf-8") as f:
        fields = json.load(f)

    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = {e["correlationMatrixId"]: e for e in json.load(f)}
        print(f"Resuming — {len(existing)} fields already done.")
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
        en_titles, de_titles = parse_response(raw)
        print(f"{len(en_titles)} EN, {len(de_titles)} DE titles")

        results[cid] = {
            "correlationMatrixId": cid,
            "titlesEn": en_titles,
            "titlesDe": de_titles,
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} fields saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
