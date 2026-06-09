"""Étape 1: générer et persister les réponses des chatbots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation_common import (
    append_jsonl,
    call_chat_api,
    load_config,
    load_jsonl,
    resolve_path,
    stable_hash,
)

LEGACY_BASELINE_PROMPT = """\
Tu es un assistant conversationnel. Réponds clairement en français.
Ne pose pas de diagnostic et recommande de consulter un professionnel de santé
si la situation peut être urgente."""
LEGACY_BASELINE_PROMPT_HASH = stable_hash(LEGACY_BASELINE_PROMPT)


ProgressCallback = Callable[[dict[str, Any]], None]


def generate(
    config_path: str,
    selected_system: str | None,
    limit: int | None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
) -> None:
    config, base_dir = load_config(config_path)
    dataset_path = resolve_path(base_dir, config["dataset"])
    output_dir = resolve_path(base_dir, config["output_dir"]) / "responses"

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
        prompt_file = system.get("prompt_file")
        if not prompt_file and system.get("type") != "reference":
            raise ValueError(f"prompt_file absent pour le système {system['name']}")
        prompt_path = (
            resolve_path(base_dir, prompt_file) if prompt_file else None
        )
        system_prompt = (
            prompt_path.read_text(encoding="utf-8").strip() if prompt_path else ""
        )
        prompt_hash = stable_hash(system_prompt)
        output_path = output_dir / f"{system['name']}.jsonl"
        existing_records = load_jsonl(output_path)
        cached_hashes = {record["request_hash"] for record in existing_records}
        latest_by_question = {
            record["question_id"]: record for record in existing_records
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
                if verbose:
                    print(
                        f"[{system['name']}] cache "
                        f"{index}/{len(dataset)} {row['id']}"
                    )
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "generation",
                            "system": system["name"],
                            "question_id": row["id"],
                            "source": "cache",
                        }
                    )
                continue
            legacy_record = latest_by_question.get(row["id"])
            same_semantic_request = (
                legacy_record
                and legacy_record.get("prompt_hash") == prompt_hash
                and legacy_record.get("model")
                == system.get("model", system.get("type"))
                and legacy_record.get("question") == row["question_patient"]
                and legacy_record.get("age") == row["age"]
                and legacy_record.get("langue") == row["langue"]
            )
            if same_semantic_request:
                if verbose:
                    print(
                        f"[{system['name']}] cache prompt "
                        f"{index}/{len(dataset)} {row['id']}"
                    )
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "generation",
                            "system": system["name"],
                            "question_id": row["id"],
                            "source": "cache",
                        }
                    )
                continue
            if (
                legacy_record
                and "prompt_hash" not in legacy_record
                and prompt_hash == LEGACY_BASELINE_PROMPT_HASH
                and legacy_record.get("model")
                == system.get("model", system.get("type"))
            ):
                if verbose:
                    print(
                        f"[{system['name']}] cache legacy "
                        f"{index}/{len(dataset)} {row['id']}"
                    )
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "generation",
                            "system": system["name"],
                            "question_id": row["id"],
                            "source": "cache",
                        }
                    )
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
                    "provider": system.get("provider"),
                    "model_group": system.get("model_group"),
                    "prompt_variant": system.get("prompt_variant"),
                    "prompt_file": prompt_file,
                    "prompt_hash": prompt_hash,
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
            latest_by_question[row["id"]] = {
                "request_hash": request_hash,
                "question_id": row["id"],
                "model": system.get("model", system.get("type")),
                "prompt_hash": prompt_hash,
            }
            if verbose:
                print(
                    f"[{system['name']}] {source} "
                    f"{index}/{len(dataset)} {row['id']}"
                )
            if progress_callback:
                progress_callback(
                    {
                        "stage": "generation",
                        "system": system["name"],
                        "question_id": row["id"],
                        "source": source,
                    }
                )
        if verbose:
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
