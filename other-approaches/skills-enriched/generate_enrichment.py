#!/usr/bin/env python3
"""
For each work field, ask an LLM to generate the typical required skills and
education. Saves output to enrichment.json, which generate_matrix.py reads.

Run this first, then generate_matrix.py to build the correlation matrix.
"""

import json
import os
import re
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv("../../.env")

DEPLOYMENT = os.environ["DEPLOYMENT"]
ENDPOINT = os.environ["ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]
OUTPUT_PATH = "enrichment.json"

PROMPT_TEMPLATE = """\
For the work field "{name_en}" (German: "{name_de}"), list the typical professional requirements.

Respond with exactly this format and nothing else:
SKILLS: [5–8 key professional skills, comma-separated]
EDUCATION: [typical educational paths or qualifications, comma-separated]"""


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
    skills = education = ""
    skills_match = re.search(r"SKILLS:\s*(.+?)(?=\nEDUCATION:|\Z)", text, re.DOTALL)
    education_match = re.search(r"EDUCATION:\s*(.+?)(?=\nSKILLS:|\Z)", text, re.DOTALL)
    if skills_match:
        skills = skills_match.group(1).strip()
    if education_match:
        education = education_match.group(1).strip()
    return skills, education


def main():
    with open("../../work_fields.json", encoding="utf-8") as f:
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
        skills, education = parse_response(raw)
        print("done")

        results[cid] = {
            "correlationMatrixId": cid,
            "skills": skills,
            "education": education,
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} fields saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
