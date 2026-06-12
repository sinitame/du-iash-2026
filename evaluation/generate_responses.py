"""Étape 1: générer et persister les réponses des chatbots.

Version intégrée :
- conserve la version existante avec prompt_file par système ;
- conserve la génération concurrente via ThreadPoolExecutor ;
- ajoute les variantes RAG activables via la configuration ou la CLI.

evaluation/evaluation_common.py reste inchangé.

Dépendances RAG à installer uniquement si la RAG est utilisée :
pip install -U langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def normalize_language(value: str | None) -> str | None:
    """Normalise les langues du dataset vers les codes utilisés dans les métadonnées FAISS."""
    if not value:
        return None

    text = str(value).strip().lower()

    mapping = {
        "fr": "fr",
        "fra": "fr",
        "français": "fr",
        "francais": "fr",
        "french": "fr",
        "en": "en",
        "eng": "en",
        "anglais": "en",
        "english": "en",
        "gcf": "gcf",
        "créole": "gcf",
        "creole": "gcf",
        "créole guadeloupéen": "gcf",
        "creole guadeloupeen": "gcf",
        "guadeloupean creole": "gcf",
    }

    return mapping.get(text, text)


def clean_text_for_context(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n[...]"


def load_rag_db(
    rag_config: dict[str, Any],
    base_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    vectorstore_path = resolve_path(base_dir, rag_config["vectorstore_path"])
    embedding_model = rag_config.get(
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    device = rag_config.get("device", "cpu")
    normalize_embeddings = rag_config.get("normalize_embeddings", True)
    manifest_path = vectorstore_path / "manifest.json"

    if not vectorstore_path.is_dir():
        raise FileNotFoundError(
            f"Base vectorielle RAG introuvable: {vectorstore_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest RAG introuvable: {manifest_path}. "
            "Reconstruisez l'index avec rag/indexation.py."
        )

    import json

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Dépendances RAG absentes. Installez "
            "evaluation/requirements-rag.txt."
        ) from error

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed_model = manifest.get("embedding_model")
    if indexed_model and indexed_model != embedding_model:
        raise ValueError(
            "Le modèle d'embeddings configuré ne correspond pas à l'index: "
            f"{embedding_model} != {indexed_model}"
        )
    indexed_normalization = manifest.get("normalize_embeddings")
    if (
        indexed_normalization is not None
        and bool(indexed_normalization) != bool(normalize_embeddings)
    ):
        raise ValueError(
            "La normalisation des embeddings configurée ne correspond pas "
            "à celle utilisée pour construire l'index."
        )

    print(f"Chargement embeddings : {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )

    print(f"Chargement FAISS : {vectorstore_path}")
    db = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db, {
        "path": str(vectorstore_path),
        "manifest": manifest,
        "fingerprint": stable_hash(manifest),
    }


def retrieve_context(
    db,
    question: str,
    language: str | None,
    rag_config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    k = int(rag_config.get("k", 5))
    fetch_k = int(rag_config.get("fetch_k", max(k * 4, k)))
    max_context_chars = int(rag_config.get("max_context_chars", 6000))
    max_chunk_chars = int(rag_config.get("max_chunk_chars", 1200))
    filter_by_language = bool(rag_config.get("filter_by_language", True))

    normalized_lang = normalize_language(language)
    search_filter = (
        {"language": normalized_lang}
        if filter_by_language and normalized_lang
        else None
    )
    results = db.similarity_search_with_score(
        question,
        k=fetch_k,
        filter=search_filter,
    )

    selected = []
    for doc, score in results:
        metadata = dict(doc.metadata or {})
        doc_lang = normalize_language(metadata.get("language"))

        if (
            filter_by_language
            and normalized_lang
            and doc_lang != normalized_lang
        ):
            continue

        selected.append((doc, score))

        if len(selected) >= k:
            break

    blocks = []
    audit_records = []
    current_len = 0

    for rank, (doc, score) in enumerate(selected, start=1):
        metadata = dict(doc.metadata or {})
        chunk_text = clean_text_for_context(doc.page_content, max_chunk_chars)

        source_file = metadata.get("source_file", "source inconnue")
        modality = metadata.get("modality", "modalité inconnue")
        lang = metadata.get("language", "langue inconnue")
        chunk_index = metadata.get("chunk_index", "?")

        block = (
            f"[Source {rank}]\n"
            f"Fichier: {source_file}\n"
            f"Langue: {lang}\n"
            f"Type: {modality}\n"
            f"Chunk: {chunk_index}\n"
            f"Contenu:\n{chunk_text}"
        )

        if current_len + len(block) > max_context_chars and blocks:
            break

        blocks.append(block)
        current_len += len(block)

        audit_records.append(
            {
                "rank": rank,
                "score": float(score),
                "content": chunk_text,
                "metadata": metadata,
            }
        )

    return "\n\n---\n\n".join(blocks), audit_records


def build_baseline_user_prompt(row: dict[str, Any]) -> str:
    return (
        f"Public concerné: {row['age']}\n"
        f"Langue: {row['langue']}\n"
        f"Thème: {row['theme']}\n"
        f"Niveau de risque interne: {row['niveau_risque']}\n"
        f"Question: {row['question_patient']}"
    )


def build_rag_user_prompt(
    row: dict[str, Any],
    context_text: str,
    rag_config: dict[str, Any],
) -> str:
    no_answer_policy = rag_config.get(
        "no_answer_policy",
        (
            "Si le contexte ne contient pas l'information nécessaire, réponds prudemment "
            "en disant que tu ne disposes pas de suffisamment d'informations dans les documents fournis."
        ),
    )

    citation_policy = rag_config.get(
        "citation_policy",
        (
            "Utilise les informations du contexte en priorité. "
            "Ne cite pas les noms techniques des chunks dans la réponse finale, "
            "mais reste fidèle aux sources."
        ),
    )

    return (
        f"Public concerné: {row['age']}\n"
        f"Langue: {row['langue']}\n"
        f"Thème: {row['theme']}\n"
        f"Niveau de risque interne: {row['niveau_risque']}\n"
        f"Question: {row['question_patient']}\n\n"
        "Contexte documentaire récupéré par RAG :\n"
        "====================\n"
        f"{context_text}\n"
        "====================\n\n"
        "Consignes RAG :\n"
        f"- {citation_policy}\n"
        f"- {no_answer_policy}\n"
        "- Réponds dans la langue demandée.\n"
        "- Garde une réponse claire, adaptée à l'âge indiqué et médicalement prudente."
    )


def should_generate_rag_variants(
    config: dict[str, Any],
    cli_rag: bool,
    cli_no_rag: bool,
) -> bool:
    if cli_rag:
        return True
    if cli_no_rag:
        return False
    return bool(config.get("enable_rag", False))


def should_generate_baseline_variants(
    config: dict[str, Any],
    cli_only_rag: bool,
) -> bool:
    if cli_only_rag:
        return False
    return bool(config.get("enable_baseline", True))


def make_progress_callback_payload(
    *,
    output_system_name: str,
    question_id: str,
    source: str,
    variant: str,
) -> dict[str, Any]:
    return {
        "stage": "generation",
        "system": output_system_name,
        "question_id": question_id,
        "source": source,
        "variant": variant,
    }


def generate_one_variant(
    *,
    config: dict[str, Any],
    base_dir: Path,
    dataset: list[dict[str, Any]],
    selected_systems: list[dict[str, Any]],
    output_dir: Path,
    variant: str,
    rag_db=None,
    rag_index: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
    concurrency: int = 1,
) -> None:
    use_rag = variant == "rag"
    rag_config = config.get("rag", {})

    for system in selected_systems:
        if use_rag and system.get("type") == "reference" and not config.get("rag_reference", False):
            if verbose:
                print(f"[{system['name']}__rag] ignoré : type reference")
            continue

        provider_transport = config.get("provider_transport", {}).get(
            system.get("provider"),
            {},
        )
        api_system = {**system, **provider_transport}
        effective_concurrency = min(
            concurrency,
            int(provider_transport.get("max_concurrency", concurrency)),
        )

        system_name = system["name"]
        output_system_name = f"{system_name}__rag" if use_rag else system_name

        prompt_file = system.get("prompt_file")
        if not prompt_file and system.get("type") != "reference":
            raise ValueError(f"prompt_file absent pour le système {system['name']}")

        prompt_path = resolve_path(base_dir, prompt_file) if prompt_file else None
        system_prompt = (
            prompt_path.read_text(encoding="utf-8").strip() if prompt_path else ""
        )
        prompt_hash = stable_hash(system_prompt)

        max_tokens = system.get("max_tokens", config.get("max_tokens", 500))

        system_request_config = {
            key: value for key, value in system.items() if key != "prompt_file"
        }
        request_config = {
            "system": system_request_config,
            "temperature": config.get("temperature", 0),
            "max_tokens": max_tokens,
        }
        if use_rag:
            request_config.update(
                {
                    "variant": "rag",
                    "rag": rag_config,
                    "rag_index_fingerprint": (
                        rag_index.get("fingerprint") if rag_index else None
                    ),
                }
            )
        request_config_hash = stable_hash(request_config)

        output_path = output_dir / f"{output_system_name}.jsonl"
        existing_records = load_jsonl(output_path)
        cached_by_hash = {
            record["request_hash"]: record for record in existing_records
        }
        cached_hashes = set(cached_by_hash)
        latest_by_question = {
            record["question_id"]: record for record in existing_records
        }

        pending = []

        for index, row in enumerate(dataset, start=1):
            retrieved_context = []
            context_text = ""
            retrieval_latency = 0.0

            if use_rag:
                if rag_db is None:
                    raise RuntimeError("La base RAG n'a pas été chargée.")
                retrieval_started = time.perf_counter()
                context_text, retrieved_context = retrieve_context(
                    db=rag_db,
                    question=row["question_patient"],
                    language=row.get("langue"),
                    rag_config=rag_config,
                )
                retrieval_latency = round(
                    time.perf_counter() - retrieval_started,
                    4,
                )
                user_prompt = build_rag_user_prompt(row, context_text, rag_config)
            else:
                user_prompt = build_baseline_user_prompt(row)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            request = {
                "system": system_request_config,
                "prompt_hash": prompt_hash,
                "messages": messages,
                "temperature": config.get("temperature", 0),
                "max_tokens": max_tokens,
            }
            if use_rag:
                request.update(
                    {
                        "variant": "rag",
                        "rag": rag_config,
                        "rag_index_fingerprint": (
                            rag_index.get("fingerprint")
                            if rag_index
                            else None
                        ),
                    }
                )
            request_hash = stable_hash(request)

            if request_hash in cached_hashes:
                cached_record = cached_by_hash[request_hash]
                if cached_record.get("question_id") != row["id"]:
                    migrated_record = {
                        **cached_record,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "question_id": row["id"],
                        "question": row["question_patient"],
                        "age": row["age"],
                        "langue": row["langue"],
                        "theme": row["theme"],
                        "niveau_risque": row["niveau_risque"],
                        "cache_migrated_from_question_id": (
                            cached_record.get("question_id")
                        ),
                    }
                    append_jsonl(output_path, migrated_record)
                    cached_by_hash[request_hash] = migrated_record
                    latest_by_question[row["id"]] = migrated_record
                if verbose:
                    print(
                        f"[{output_system_name}] cache "
                        f"{index}/{len(dataset)} {row['id']}"
                    )
                if progress_callback:
                    progress_callback(
                        make_progress_callback_payload(
                            output_system_name=output_system_name,
                            question_id=row["id"],
                            source="cache",
                            variant=variant,
                        )
                    )
                continue

            if not use_rag:
                legacy_record = latest_by_question.get(row["id"])
                if (
                    legacy_record
                    and "request_config_hash" not in legacy_record
                    and legacy_record.get("prompt_hash") == prompt_hash
                    and legacy_record.get("model") == system.get("model", system.get("type"))
                    and legacy_record.get("question") == row["question_patient"]
                    and legacy_record.get("age") == row["age"]
                    and legacy_record.get("langue") == row["langue"]
                ):
                    if verbose:
                        print(
                            f"[{output_system_name}] cache migration "
                            f"{index}/{len(dataset)} {row['id']}"
                        )
                    if progress_callback:
                        progress_callback(
                            make_progress_callback_payload(
                                output_system_name=output_system_name,
                                question_id=row["id"],
                                source="cache",
                                variant=variant,
                            )
                        )
                    continue

                if (
                    legacy_record
                    and "prompt_hash" not in legacy_record
                    and prompt_hash == LEGACY_BASELINE_PROMPT_HASH
                    and legacy_record.get("model") == system.get("model", system.get("type"))
                ):
                    if verbose:
                        print(
                            f"[{output_system_name}] cache legacy "
                            f"{index}/{len(dataset)} {row['id']}"
                        )
                    if progress_callback:
                        progress_callback(
                            make_progress_callback_payload(
                                output_system_name=output_system_name,
                                question_id=row["id"],
                                source="cache",
                                variant=variant,
                            )
                        )
                    continue

            pending.append(
                {
                    "index": index,
                    "row": row,
                    "messages": messages,
                    "request_hash": request_hash,
                    "retrieved_context": retrieved_context,
                    "context_text": context_text,
                    "retrieval_latency_seconds": retrieval_latency,
                }
            )

        def generate_one(job: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
            row = job["row"]
            started = time.perf_counter()

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
                    api_system,
                    job["messages"],
                    temperature=config.get("temperature", 0),
                    max_tokens=max_tokens,
                )
                if not generation["text"].strip():
                    raise RuntimeError(
                        "Le modèle a renvoyé une réponse vide. "
                        f"Usage: {generation.get('usage', {})}"
                    )
                source = "api"

            generation_latency = round(time.perf_counter() - started, 4)
            return generation, source, generation_latency

        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {
                executor.submit(generate_one, job): job for job in pending
            }
            errors = []

            for future in as_completed(futures):
                job = futures[future]
                row = job["row"]

                try:
                    generation, source, generation_latency = future.result()
                except Exception as error:
                    errors.append((row["id"], error))
                    if verbose:
                        print(f"[{output_system_name}] erreur {row['id']}: {error}")
                    continue

                record = {
                    "request_hash": job["request_hash"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "system_name": output_system_name,
                    "model": system.get("model", system.get("type")),
                    "provider": system.get("provider"),
                    "model_group": system.get("model_group"),
                    "prompt_variant": (
                        "rag" if use_rag else system.get("prompt_variant")
                    ),
                    "prompt_file": prompt_file,
                    "prompt_hash": prompt_hash,
                    "request_config_hash": request_config_hash,
                    "question_id": row["id"],
                    "question": row["question_patient"],
                    "age": row["age"],
                    "langue": row["langue"],
                    "theme": row["theme"],
                    "niveau_risque": row["niveau_risque"],
                    "response": generation["text"],
                    "latency_seconds": generation["latency_seconds"],
                    "api_attempts": generation.get("api_attempts", 1),
                    "api_max_tokens": generation.get("api_max_tokens", max_tokens),
                    "usage": generation["usage"],
                    "raw_api_response": generation["raw_api_response"],
                    "variant": "rag" if use_rag else "baseline",
                }

                if use_rag:
                    record.update(
                        {
                            "base_system_name": system_name,
                            "retrieval_latency_seconds": job[
                                "retrieval_latency_seconds"
                            ],
                            "total_latency_seconds": round(
                                job["retrieval_latency_seconds"]
                                + generation_latency,
                                4,
                            ),
                            "rag": {
                                "k": rag_config.get("k", 5),
                                "fetch_k": rag_config.get("fetch_k"),
                                "filter_by_language": rag_config.get("filter_by_language", True),
                                "index": rag_index,
                                "retrieved_context": job["retrieved_context"],
                            },
                        }
                    )

                append_jsonl(output_path, record)
                cached_hashes.add(job["request_hash"])
                latest_by_question[row["id"]] = {
                    "request_hash": job["request_hash"],
                    "question_id": row["id"],
                    "model": system.get("model", system.get("type")),
                    "prompt_hash": prompt_hash,
                }

                if verbose:
                    print(
                        f"[{output_system_name}] {source} "
                        f"{job['index']}/{len(dataset)} {row['id']}"
                    )

                if progress_callback:
                    progress_callback(
                        make_progress_callback_payload(
                            output_system_name=output_system_name,
                            question_id=row["id"],
                            source=source,
                            variant=variant,
                        )
                    )

            if errors:
                question_id, error = errors[0]
                raise RuntimeError(
                    f"{len(errors)} génération(s) ont échoué pour "
                    f"{output_system_name}. Première erreur ({question_id}): {error}. "
                    "Les réponses réussies ont été persistées."
                ) from error

        if verbose:
            print(f"Réponses persistées: {output_path}")


def generate(
    config_path: str,
    selected_system: str | None,
    limit: int | None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
    concurrency: int = 1,
    rag: bool = False,
    no_rag: bool = False,
    only_rag: bool = False,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency doit être supérieur ou égal à 1")

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
        if (
            (
                selected_system
                and system["name"] == selected_system
            )
            or (
                not selected_system
                and not system.get("generated_variant", False)
            )
        )
    ]

    if not selected:
        raise ValueError(f"Système absent de la configuration: {selected_system}")

    generate_baseline = should_generate_baseline_variants(config, only_rag)
    generate_rag = should_generate_rag_variants(config, rag, no_rag)

    if generate_rag and "rag" not in config:
        raise ValueError(
            "La génération RAG est activée mais la config ne contient pas de section 'rag'."
        )

    rag_db = None
    rag_index = None
    if generate_rag:
        rag_db, rag_index = load_rag_db(config["rag"], base_dir)

    if generate_baseline and generate_rag:
        for system in selected:
            generate_one_variant(
                config=config,
                base_dir=base_dir,
                dataset=dataset,
                selected_systems=[system],
                output_dir=output_dir,
                variant="baseline",
                progress_callback=progress_callback,
                verbose=verbose,
                concurrency=concurrency,
            )
            generate_one_variant(
                config=config,
                base_dir=base_dir,
                dataset=dataset,
                selected_systems=[system],
                output_dir=output_dir,
                variant="rag",
                rag_db=rag_db,
                rag_index=rag_index,
                progress_callback=progress_callback,
                verbose=verbose,
                concurrency=concurrency,
            )
    else:
        if generate_baseline:
            generate_one_variant(
                config=config,
                base_dir=base_dir,
                dataset=dataset,
                selected_systems=selected,
                output_dir=output_dir,
                variant="baseline",
                progress_callback=progress_callback,
                verbose=verbose,
                concurrency=concurrency,
            )

        if generate_rag:
            generate_one_variant(
                config=config,
                base_dir=base_dir,
                dataset=dataset,
                selected_systems=selected,
                output_dir=output_dir,
                variant="rag",
                rag_db=rag_db,
                rag_index=rag_index,
                progress_callback=progress_callback,
                verbose=verbose,
                concurrency=concurrency,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.example.json")),
    )
    parser.add_argument("--system", help="Nom d'un seul système")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Active la génération des variantes RAG en plus de la baseline."
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Désactive la génération RAG même si enable_rag=true dans la config."
    )
    parser.add_argument(
        "--only-rag",
        action="store_true",
        help="Génère uniquement les variantes RAG, sans régénérer les baselines."
    )

    args = parser.parse_args()

    generate(
        args.config,
        args.system,
        args.limit,
        concurrency=args.concurrency,
        rag=args.rag,
        no_rag=args.no_rag,
        only_rag=args.only_rag,
    )


if __name__ == "__main__":
    main()
