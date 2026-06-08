"""Étape 1: générer et persister les réponses des chatbots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from evaluation_common import (
    append_jsonl,
    call_chat_api,
    load_config,
    load_jsonl,
    resolve_path,
    stable_hash,
)


def generate(config_path: str, selected_system: str | None, limit: int | None) -> None:
    config, base_dir = load_config(config_path)
    dataset_path = resolve_path(base_dir, config["dataset"])
    output_dir = resolve_path(base_dir, config["output_dir"]) / "responses"
    prompt_path = resolve_path(base_dir, config["chatbot_prompt_file"])
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    if limit:
        dataset = dataset[:limit]

    selected = [
        system
        for system in config["systems"]
        if not selected_system or system["name"] == selected_system
    ]
    if not selected:
        raise ValueError(f"Système absent de la configuration: {selected_system}")

    for system in selected:
        output_path = output_dir / f"{system['name']}.jsonl"
        cached_hashes = {
            record["request_hash"] for record in load_jsonl(output_path)
        }
        for index, row in enumerate(dataset, start=1):
            user_prompt = (
                f"Public concerné: {row['age']}\n"
                f"Langue: {row['langue']}\n"
                f"Question: {row['question_patient']}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            request_hash = stable_hash(
                {
                    "system": system,
                    "messages": messages,
                    "temperature": config.get("temperature", 0),
                    "max_tokens": config.get("max_tokens", 500),
                }
            )
            if request_hash in cached_hashes:
                print(f"[{system['name']}] cache {index}/{len(dataset)} {row['id']}")
                continue

            if system.get("type") == "reference":
                generation = {
                    "text": row["réponse_attendue"],
                    "latency_seconds": 0.0,
                    "usage": {},
                    "raw_api_response": None,
                }
                source = "local"
            else:
                generation = call_chat_api(
                    system,
                    messages,
                    temperature=config.get("temperature", 0),
                    max_tokens=config.get("max_tokens", 500),
                )
                source = "api"

            append_jsonl(
                output_path,
                {
                    "request_hash": request_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "system_name": system["name"],
                    "model": system.get("model", system.get("type")),
                    "question_id": row["id"],
                    "question": row["question_patient"],
                    "age": row["age"],
                    "langue": row["langue"],
                    "theme": row["theme"],
                    "niveau_risque": row["niveau_risque"],
                    "response": generation["text"],
                    "latency_seconds": generation["latency_seconds"],
                    "usage": generation["usage"],
                    "raw_api_response": generation["raw_api_response"],
                },
            )
            cached_hashes.add(request_hash)
            print(f"[{system['name']}] {source} {index}/{len(dataset)} {row['id']}")
        print(f"Réponses persistées: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.example.json")),
    )
    parser.add_argument("--system", help="Nom d'un seul système")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    generate(args.config, args.system, args.limit)


if __name__ == "__main__":
    main()
