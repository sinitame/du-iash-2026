#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
translate_corpus.py

Traduit un corpus RAG généré par prepare_corpus.py vers l'anglais et le créole guadeloupéen.

Entrée attendue :
- corpus/documents.jsonl
- corpus/chunks.jsonl

Sorties par défaut :
- corpus_translated/chunks_en.jsonl
- corpus_translated/chunks_gcf.jsonl
- corpus_translated/documents_en.jsonl avec --translate-documents
- corpus_translated/documents_gcf.jsonl avec --translate-documents
- corpus_translated/translation_log.jsonl
- corpus_translated/items/<lang>/<id>.json

Fonctionnalités :
- traduction avec l'API OpenAI ;
- modèle par défaut : gpt-5-mini ;
- few-shot créole guadeloupéen via jhu-clsp/kreyol-mt ;
- reprise après interruption ;
- écritures atomiques ;
- parallélisation ;
- retries avec backoff ;
- suivi de progression avec tqdm.

Installation :
pip install -U openai tqdm datasets

Variable d'environnement requise :
export OPENAI_API_KEY="sk-..."
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


LANG_CONFIG = {
    "en": {
        "label": "anglais",
        "output_suffix": "en",
    },
    "gcf": {
        "label": "créole guadeloupéen",
        "output_suffix": "gcf",
    },
}

def supports_temperature(model: str) -> bool:
    """
    Certains modèles OpenAI récents, notamment les modèles de raisonnement,
    n'acceptent pas le paramètre temperature.
    On l'envoie uniquement pour les modèles qui le supportent.
    """
    model_lower = model.lower()
    no_temperature_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
    )
    return not model_lower.startswith(no_temperature_prefixes)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, data: dict | list) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in rows
    )
    atomic_write_text(path, content)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON invalide dans {path}, ligne {line_number}: {e}") from e

    return rows


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", value)
    value = value.strip("_")
    return value or "item"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_creole_examples(
    enabled: bool,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_examples: int,
) -> str:
    if not enabled:
        return ""

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Le package datasets est requis pour charger les exemples créoles. "
            "Installe-le avec : pip install -U datasets"
        ) from e

    print("Chargement dataset créole...")
    dataset = load_dataset(dataset_name, dataset_config)

    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{split}' introuvable. Splits disponibles : {available}")

    examples = []

    for i, row in enumerate(dataset[split]):
        if i >= max_examples:
            break

        translation = row.get("translation", {})

        # Dans jhu-clsp/kreyol-mt gcf-fra :
        # src_text = créole, tgt_text = français.
        creole = translation.get("src_text", "")
        french = translation.get("tgt_text", "")

        if creole and french:
            examples.append(
                f"Français : {french}\nCréole guadeloupéen : {creole}"
            )

    print(f"Dataset créole chargé : {len(examples)} exemple(s).")

    return "\n\n".join(examples)


def build_system_prompt(target_lang: str, creole_examples: str) -> str:
    if target_lang == "en":
        return (
            "Tu es un traducteur professionnel spécialisé dans les contenus médicaux "
            "et pédagogiques destinés aux enfants, adolescents et familles.\n"
            "Traduis le texte français vers l'anglais.\n"
            "Contraintes :\n"
            "- conserve le sens médical avec précision ;\n"
            "- garde un ton clair, rassurant et compréhensible ;\n"
            "- ne résume pas ;\n"
            "- ne rajoute pas d'informations ;\n"
            "- conserve les titres, listes et sauts de ligne autant que possible ;\n"
            "- retourne uniquement la traduction, sans commentaire."
        )

    if target_lang == "gcf":
        examples_block = ""
        if creole_examples.strip():
            examples_block = (
                "\n\nExemples authentiques de français vers créole guadeloupéen :\n"
                f"{creole_examples}\n"
            )

        return (
            "Tu es un traducteur expert en créole guadeloupéen.\n"
            "Traduis le texte français vers le créole guadeloupéen.\n"
            "Le corpus concerne des contenus médicaux et pédagogiques pour enfants, "
            "adolescents et familles.\n"
            "Contraintes :\n"
            "- conserve le sens médical avec précision ;\n"
            "- garde un ton clair, naturel et rassurant ;\n"
            "- ne résume pas ;\n"
            "- ne rajoute pas d'informations ;\n"
            "- conserve les titres, listes et sauts de ligne autant que possible ;\n"
            "- retourne uniquement la traduction, sans commentaire."
            f"{examples_block}"
        )

    raise ValueError(f"Langue cible non supportée : {target_lang}")


def build_user_prompt(text: str) -> str:
    return (
        "Traduis le texte suivant. Retourne uniquement la traduction.\n\n"
        "----- TEXTE À TRADUIRE -----\n"
        f"{text}"
    )


def extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    parts = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    return "\n".join(parts).strip()


def translate_with_openai(
    text: str,
    target_lang: str,
    model: str,
    creole_examples: str,
    temperature: float,
    max_retries: int,
    timeout: float,
) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    if not text or text.lower() == "nan":
        return ""

    client = OpenAI(timeout=timeout)

    system_prompt = build_system_prompt(target_lang, creole_examples)
    user_prompt = build_user_prompt(text)

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            request_kwargs = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            }

            if supports_temperature(model):
                request_kwargs["temperature"] = temperature

            response = client.responses.create(**request_kwargs)

            translated = extract_response_text(response)

            if not translated:
                raise RuntimeError("Réponse vide du modèle.")

            return translated

        except Exception as e:
            last_error = e

            if attempt >= max_retries:
                break

            sleep_s = min(60, (2 ** attempt) + random.uniform(0, 1.5))
            time.sleep(sleep_s)

    raise RuntimeError(f"Échec traduction après {max_retries + 1} tentative(s): {last_error}")


def translated_item_path(output_dir: Path, target_lang: str, item_id: str) -> Path:
    return output_dir / "items" / target_lang / f"{safe_id(item_id)}.json"


def is_item_done(path: Path, source_hash: str, target_lang: str) -> bool:
    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        data.get("status") == "done"
        and data.get("source_text_sha256") == source_hash
        and data.get("target_language") == target_lang
        and isinstance(data.get("text"), str)
        and len(data.get("text", "").strip()) > 0
    )


def load_done_item(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_translated_row(
    source_row: dict,
    translated_text: str,
    target_lang: str,
    model: str,
) -> dict:
    row = dict(source_row)

    source_text = str(source_row.get("text", ""))

    row["id"] = f"{source_row.get('id')}:{target_lang}"
    row["source_id"] = source_row.get("id")
    row["source_language"] = source_row.get("language", "fr")
    row["language"] = target_lang
    row["target_language"] = target_lang
    row["translation_model"] = model
    row["translated_at"] = now_iso()
    row["source_text_sha256"] = stable_hash(source_text)
    row["text"] = translated_text

    return row


def translate_one_item(task: dict) -> dict:
    source_row = task["source_row"]
    source_text = str(source_row.get("text", ""))
    source_hash = stable_hash(source_text)
    target_lang = task["target_lang"]
    output_path = Path(task["output_path"])

    if is_item_done(output_path, source_hash, target_lang):
        done = load_done_item(output_path)
        return done["row"]

    translated_text = translate_with_openai(
        text=source_text,
        target_lang=target_lang,
        model=task["model"],
        creole_examples=task["creole_examples"],
        temperature=task["temperature"],
        max_retries=task["max_retries"],
        timeout=task["timeout"],
    )

    row = build_translated_row(
        source_row=source_row,
        translated_text=translated_text,
        target_lang=target_lang,
        model=task["model"],
    )

    item_payload = {
        "status": "done",
        "source_id": source_row.get("id"),
        "target_language": target_lang,
        "source_text_sha256": source_hash,
        "translated_at": row["translated_at"],
        "model": task["model"],
        "text": translated_text,
        "row": row,
    }

    atomic_write_json(output_path, item_payload)

    return row


def translate_rows(
    rows: list[dict],
    target_lang: str,
    output_dir: Path,
    output_filename: str,
    model: str,
    creole_examples: str,
    workers: int,
    temperature: float,
    max_retries: int,
    timeout: float,
    log_path: Path,
    desc: str,
) -> list[dict]:
    output_path = output_dir / output_filename
    workers = max(1, workers)

    tasks = []
    already_done = 0

    for row in rows:
        source_text = str(row.get("text", ""))
        source_hash = stable_hash(source_text)
        item_path = translated_item_path(output_dir, target_lang, row.get("id", ""))

        if is_item_done(item_path, source_hash, target_lang):
            already_done += 1

        tasks.append({
            "source_row": row,
            "target_lang": target_lang,
            "model": model,
            "creole_examples": creole_examples if target_lang == "gcf" else "",
            "temperature": temperature,
            "max_retries": max_retries,
            "timeout": timeout,
            "output_path": str(item_path),
        })

    print(
        f"{desc} → {target_lang} : {len(rows)} item(s), "
        f"{already_done} déjà traduit(s), {workers} worker(s)."
    )

    translated_rows = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(translate_one_item, task): task
            for task in tasks
        }

        try:
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{desc} {target_lang}",
            ):
                task = futures[future]
                source_id = task["source_row"].get("id")

                try:
                    row = future.result()
                    translated_rows.append(row)

                    append_jsonl(
                        log_path,
                        {
                            "event": "done",
                            "time": now_iso(),
                            "kind": desc,
                            "target_lang": target_lang,
                            "source_id": source_id,
                        }
                    )

                except Exception as e:
                    append_jsonl(
                        log_path,
                        {
                            "event": "error",
                            "time": now_iso(),
                            "kind": desc,
                            "target_lang": target_lang,
                            "source_id": source_id,
                            "error": repr(e),
                        }
                    )
                    print(f"Erreur traduction {source_id} vers {target_lang} : {repr(e)}")

        except KeyboardInterrupt:
            print("\nInterruption détectée. Relance la même commande pour reprendre.")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    order = {row.get("id"): i for i, row in enumerate(rows)}
    translated_rows.sort(key=lambda r: order.get(r.get("source_id"), 10**12))

    write_jsonl(output_path, translated_rows)

    print(f"Écrit : {output_path}")

    return translated_rows


def main():
    parser = argparse.ArgumentParser(
        description="Traduit documents.jsonl et chunks.jsonl avec l'API OpenAI."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dossier contenant documents.jsonl et chunks.jsonl."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier de sortie des traductions."
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Modèle OpenAI à utiliser."
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "gcf"],
        choices=["en", "gcf"],
        help="Langues cibles : en et/ou gcf."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Nombre d'appels de traduction en parallèle."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Température du modèle. Ignorée automatiquement pour les modèles qui ne la supportent pas."
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Nombre de retries par item."
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout API par requête en secondes."
    )

    parser.add_argument(
        "--translate-documents",
        action="store_true",
        help="Traduit aussi documents.jsonl. Par défaut, seuls les chunks sont traduits."
    )

    parser.add_argument(
        "--creole-examples",
        type=int,
        default=10,
        help="Nombre d'exemples few-shot créoles à charger."
    )

    parser.add_argument(
        "--no-creole-examples",
        action="store_true",
        help="Désactive les exemples few-shot créoles."
    )

    parser.add_argument(
        "--creole-dataset-name",
        default="jhu-clsp/kreyol-mt",
        help="Dataset Hugging Face utilisé pour les exemples créoles."
    )

    parser.add_argument(
        "--creole-dataset-config",
        default="gcf-fra",
        help="Configuration du dataset créole."
    )

    parser.add_argument(
        "--creole-dataset-split",
        default="train",
        help="Split du dataset créole."
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY est absent. Configure-le avec : "
            "export OPENAI_API_KEY='sk-...'"
        )

    chunks_path = args.input_dir / "chunks.jsonl"
    documents_path = args.input_dir / "documents.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chunks_path}")

    if args.translate_documents and not documents_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {documents_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "translation_log.jsonl"

    append_jsonl(
        log_path,
        {
            "event": "start",
            "time": now_iso(),
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "languages": args.languages,
            "workers": args.workers,
            "translate_documents": args.translate_documents,
        }
    )

    creole_examples = ""

    if "gcf" in args.languages:
        creole_examples = load_creole_examples(
            enabled=not args.no_creole_examples,
            dataset_name=args.creole_dataset_name,
            dataset_config=args.creole_dataset_config,
            split=args.creole_dataset_split,
            max_examples=args.creole_examples,
        )

    chunks = load_jsonl(chunks_path)
    documents = load_jsonl(documents_path) if args.translate_documents else []

    print(f"Chunks chargés : {len(chunks)}")
    if args.translate_documents:
        print(f"Documents chargés : {len(documents)}")

    try:
        for lang in args.languages:
            suffix = LANG_CONFIG[lang]["output_suffix"]

            translate_rows(
                rows=chunks,
                target_lang=lang,
                output_dir=args.output_dir,
                output_filename=f"chunks_{suffix}.jsonl",
                model=args.model,
                creole_examples=creole_examples,
                workers=args.workers,
                temperature=args.temperature,
                max_retries=args.max_retries,
                timeout=args.timeout,
                log_path=log_path,
                desc="chunks",
            )

            if args.translate_documents:
                translate_rows(
                    rows=documents,
                    target_lang=lang,
                    output_dir=args.output_dir,
                    output_filename=f"documents_{suffix}.jsonl",
                    model=args.model,
                    creole_examples=creole_examples,
                    workers=args.workers,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    log_path=log_path,
                    desc="documents",
                )

        append_jsonl(
            log_path,
            {
                "event": "finish",
                "time": now_iso(),
            }
        )

        print("\nTraduction terminée.")
        print(f"Sorties : {args.output_dir}")
        print(f"Log : {log_path}")

    except KeyboardInterrupt:
        append_jsonl(
            log_path,
            {
                "event": "interrupted",
                "time": now_iso(),
                "message": "Relancer la même commande pour reprendre.",
            }
        )
        raise


if __name__ == "__main__":
    main()
