"""Lancer génération, scoring et métriques avec suivi de progression."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compute_metrics import compute
from evaluation_common import (
    get_api_key,
    load_config,
    load_jsonl,
    resolve_path,
    stable_hash,
)
from generate_responses import generate
from score_responses import score


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_dataset_path(value: str, evaluation_dir: Path) -> Path:
    supplied = Path(value).expanduser()
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [Path.cwd() / supplied, evaluation_dir / supplied]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset CSV introuvable: {value}")


def count_questions(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def describe_dataset(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [row.get("id", "").strip() for row in rows]
    return {
        "questions": len(rows),
        "identifiants_vides": sum(not identifier for identifier in identifiers),
        "identifiants_dupliques": len(identifiers)
        - len(set(identifiers)),
        "themes": len({row["theme"] for row in rows}),
        "reponses_attendues_uniques": len(
            {row["réponse_attendue"] for row in rows}
        ),
    }


def select_systems(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "baseline":
        variants = {"baseline"}
    elif mode == "prompt":
        variants = {"baseline", "step_by_step"}
    elif mode in {"rag", "rag_selective"}:
        variants = {"baseline", "step_by_step"}
    else:
        raise ValueError(f"Mode inconnu: {mode}")

    systems = [
        system
        for system in config["systems"]
        if system.get("prompt_variant") in variants
    ]
    if not systems:
        raise ValueError(f"Aucun système configuré pour le mode {mode}")
    return systems


def build_rag_system(
    system: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    return {
        **system,
        "name": f"{system['name']}__{variant}",
        "prompt_variant": variant,
        "base_system_name": system["name"],
        "generated_variant": True,
    }


class ProgressTracker:
    def __init__(
        self,
        path: Path,
        dataset: Path,
        mode: str,
        systems: list[dict[str, Any]],
        question_count: int,
        concurrency: int,
        total_steps: int | None = None,
    ):
        self.path = path
        self.lock = threading.Lock()
        self.state = {
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "dataset": str(dataset),
            "mode": mode,
            "systems": [system["name"] for system in systems],
            "question_count": question_count,
            "concurrency": concurrency,
            "total_steps": (
                total_steps
                if total_steps is not None
                else question_count * len(systems) * 2 + 1
            ),
            "completed_steps": 0,
            "current_stage": "initialization",
            "current_system": None,
            "current_question_id": None,
            "cache_hits": 0,
            "api_calls": 0,
            "local_calls": 0,
            "error": None,
        }
        self._save()

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.path, self.state)

    def update(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.state["completed_steps"] += 1
            self.state["current_stage"] = event["stage"]
            self.state["current_system"] = event.get("system")
            self.state["current_question_id"] = event.get("question_id")
            source = event.get("source")
            if source == "cache":
                self.state["cache_hits"] += 1
            elif source == "api":
                self.state["api_calls"] += 1
            elif source == "local":
                self.state["local_calls"] += 1
            self._save()
            self._render()

    def metrics_done(self) -> None:
        self.update({"stage": "metrics", "source": "local"})

    def complete(self) -> None:
        with self.lock:
            self.state["status"] = "completed"
            self.state["completed_at"] = utc_now()
            self.state["current_stage"] = "completed"
            self.state["estimated_total_steps"] = self.state["total_steps"]
            self.state["total_steps"] = self.state["completed_steps"]
            self._save()
            self._render(final=True)

    def fail(self, error: Exception) -> None:
        with self.lock:
            self.state["status"] = "failed"
            self.state["failed_at"] = utc_now()
            self.state["error"] = str(error)
            self._save()
        print(file=sys.stderr)
        print(f"Échec. Progression sauvegardée dans {self.path}", file=sys.stderr)

    def _render(self, final: bool = False) -> None:
        completed = self.state["completed_steps"]
        total = self.state["total_steps"]
        ratio = completed / total if total else 1
        width = 28
        filled = min(width, round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"\r\033[2K[{bar}] {completed}/{total} ({ratio:.0%}) "
            f"{self.state['current_stage']} "
            f"{self.state.get('current_system') or ''} "
            f"{self.state.get('current_question_id') or ''}"
        )
        print(line.rstrip(), end="\n" if final else "", flush=True)


def build_run_config(
    base_config_path: Path,
    dataset_path: Path,
    mode: str,
    complete: bool = False,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    config, base_dir = load_config(base_config_path)
    is_rag_mode = mode in {"rag", "rag_selective"}
    non_rag_systems = select_systems(config, mode)
    rag_source_systems = [
        system
        for system in non_rag_systems
        if system.get("prompt_variant") == "baseline"
    ]
    if is_rag_mode and mode not in config:
        raise ValueError(
            f"Le mode {mode} nécessite une section '{mode}' dans la "
            "configuration."
        )
    systems = (
        non_rag_systems
        + [build_rag_system(system, mode) for system in rag_source_systems]
        if is_rag_mode
        else non_rag_systems
    )
    judges = [judge for judge in config["judges"] if judge.get("type") != "mock"]
    if len(judges) != 1:
        raise ValueError("La configuration doit contenir exactement un juge réel.")

    output_dir = resolve_path(base_dir, config["output_dir"])
    derived = {
        **config,
        "evaluation_mode": mode,
        "rag_generate_baseline": (
            True
            if is_rag_mode and complete
            else config.get("rag_generate_baseline", False)
        ),
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "judge_prompt_file": str(
            resolve_path(base_dir, config["judge_prompt_file"])
        ),
        "generation_systems": [
            system["name"] for system in non_rag_systems
        ],
        "rag_source_systems": [
            system["name"] for system in rag_source_systems
        ],
        "systems": [
            {
                **system,
                "prompt_file": str(resolve_path(base_dir, system["prompt_file"])),
            }
            for system in systems
        ],
        "judges": judges,
    }
    if is_rag_mode:
        derived[mode] = {
            **config[mode],
            "vectorstore_path": str(
                resolve_path(
                    base_dir,
                    config[mode]["vectorstore_path"],
                )
            ),
        }
    run_key = stable_hash(
        {
            "dataset": str(dataset_path),
            "mode": mode,
            "systems": derived["systems"],
            "judge": judges[0],
            "judge_prompt_file": derived["judge_prompt_file"],
            "judge_prompt_hash": stable_hash(
                Path(derived["judge_prompt_file"]).read_text(encoding="utf-8")
            ),
            "rag_config": derived.get(mode) if is_rag_mode else None,
            "rag_generate_baseline": derived["rag_generate_baseline"],
        }
    )[:12]
    config_path = output_dir / "run_configs" / f"{dataset_path.stem}_{mode}_{run_key}.json"
    atomic_write_json(config_path, derived)
    return config_path, derived, systems


def run(
    dataset_value: str,
    mode: str,
    base_config_value: str,
    concurrency: int = 1,
    partial_summary: bool = False,
    complete: bool = False,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency doit être supérieur ou égal à 1")
    if complete and partial_summary:
        raise ValueError(
            "--complete et --partial-summary ne peuvent pas être utilisés "
            "ensemble."
        )
    if concurrency > 10:
        print(
            f"Attention: concurrency={concurrency} est élevée. "
            "Les APIs peuvent renvoyer davantage d'erreurs transitoires; "
            "une valeur entre 4 et 8 est généralement plus stable.",
            file=sys.stderr,
        )

    evaluation_dir = Path(__file__).resolve().parent
    base_config_path = Path(base_config_value).resolve()
    dataset_path = resolve_dataset_path(dataset_value, evaluation_dir)
    run_config_path, run_config, systems = build_run_config(
        base_config_path,
        dataset_path,
        mode,
        complete=complete,
    )
    checked_credentials = set()
    for provider_config in [*systems, *run_config["judges"]]:
        if provider_config.get("type") in {"mock", "reference"}:
            continue
        api_key_env = provider_config.get("api_key_env", "LLM_API_KEY")
        if api_key_env in checked_credentials:
            continue
        get_api_key(provider_config)
        checked_credentials.add(api_key_env)
    active_providers = {system.get("provider") for system in systems}
    for provider, transport in run_config.get(
        "provider_transport",
        {},
    ).items():
        provider_limit = int(transport.get("max_concurrency", concurrency))
        if provider in active_providers and provider_limit < concurrency:
            print(
                f"Concurrence {provider}: {provider_limit} "
                f"(limite générale demandée: {concurrency}).",
                file=sys.stderr,
            )
    question_count = count_questions(dataset_path)
    dataset_scope = describe_dataset(dataset_path)
    if dataset_scope["questions"] < 30 or dataset_scope["themes"] < 3:
        print(
            "Attention: périmètre d'évaluation limité "
            f"({dataset_scope['questions']} questions, "
            f"{dataset_scope['themes']} thème(s), "
            f"{dataset_scope['reponses_attendues_uniques']} réponses attendues "
            "uniques). Les écarts entre systèmes peuvent être instables.",
            file=sys.stderr,
        )
    if dataset_scope["identifiants_vides"]:
        print(
            "Attention: le dataset contient "
            f"{dataset_scope['identifiants_vides']} identifiant(s) vide(s). "
            "Ces lignes ne peuvent pas être suivies fiablement dans le cache.",
            file=sys.stderr,
        )
    if dataset_scope["identifiants_dupliques"]:
        print(
            "Attention: le dataset contient "
            f"{dataset_scope['identifiants_dupliques']} identifiant(s) "
            "dupliqué(s).",
            file=sys.stderr,
        )
    if complete and (
        dataset_scope["identifiants_vides"]
        or dataset_scope["identifiants_dupliques"]
    ):
        raise ValueError(
            "Un run complet exige un identifiant non vide et unique pour "
            "chaque ligne du dataset."
        )
    output_dir = Path(run_config["output_dir"])
    progress_suffix = "_partial_summary" if partial_summary else ""
    progress_path = (
        output_dir
        / "progress"
        / f"{dataset_path.stem}_{mode}{progress_suffix}.json"
    )
    partial_response_count = None
    rag_total_steps = None
    if partial_summary:
        with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
            dataset = {row["id"]: row for row in csv.DictReader(handle)}
        partial_response_count = 0
        for system in systems:
            latest_responses = {}
            response_path = (
                output_dir / "responses" / f"{system['name']}.jsonl"
            )
            for response in load_jsonl(response_path):
                row = dataset.get(response["question_id"])
                if row and response.get("question") == row["question_patient"]:
                    latest_responses[response["question_id"]] = response
            partial_response_count += len(latest_responses)
    elif (
        mode in {"rag", "rag_selective"}
        and not run_config.get("rag_generate_baseline", False)
    ):
        with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
            dataset = {row["id"]: row for row in csv.DictReader(handle)}
        baseline_response_count = 0
        for system_name in run_config["generation_systems"]:
            latest_responses = {}
            response_path = (
                output_dir / "responses" / f"{system_name}.jsonl"
            )
            for response in load_jsonl(response_path):
                row = dataset.get(response["question_id"])
                if row and response.get("question") == row["question_patient"]:
                    latest_responses[response["question_id"]] = response
            baseline_response_count += len(latest_responses)
        rag_system_count = len(run_config["rag_source_systems"])
        rag_total_steps = (
            question_count * rag_system_count
            + question_count * rag_system_count
            + baseline_response_count
            + 1
        )
    tracker = ProgressTracker(
        progress_path,
        dataset_path,
        mode,
        systems,
        question_count,
        concurrency,
        total_steps=(
            partial_response_count + 1
            if partial_response_count is not None
            else rag_total_steps
        ),
    )

    try:
        if not partial_summary:
            if mode in {"rag", "rag_selective"}:
                if run_config.get("rag_generate_baseline", False):
                    for system_name in run_config["generation_systems"]:
                        generate(
                            str(run_config_path),
                            system_name,
                            None,
                            progress_callback=tracker.update,
                            verbose=False,
                            concurrency=concurrency,
                            no_rag=True,
                        )
                for system_name in run_config["rag_source_systems"]:
                    generate(
                        str(run_config_path),
                        system_name,
                        None,
                        progress_callback=tracker.update,
                        verbose=False,
                        concurrency=concurrency,
                        rag=True,
                        only_rag=True,
                        rag_variant=mode,
                    )
            else:
                for system in systems:
                    generate(
                        str(run_config_path),
                        system["name"],
                        None,
                        progress_callback=tracker.update,
                        verbose=False,
                        concurrency=concurrency,
                    )
        if complete:
            with dataset_path.open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                dataset = {
                    row["id"]: row for row in csv.DictReader(handle)
                }
            incomplete = []
            for system in systems:
                latest_responses = {}
                response_path = (
                    output_dir
                    / "responses"
                    / f"{system['name']}.jsonl"
                )
                for response in load_jsonl(response_path):
                    row = dataset.get(response["question_id"])
                    if (
                        row
                        and response.get("question")
                        == row["question_patient"]
                    ):
                        latest_responses[response["question_id"]] = response
                if len(latest_responses) != len(dataset):
                    incomplete.append(
                        f"{system['name']}="
                        f"{len(latest_responses)}/{len(dataset)}"
                    )
            if incomplete:
                raise RuntimeError(
                    "Couverture incomplète après génération: "
                    + ", ".join(incomplete)
                    + ". Relancez la même commande; les résultats réussis "
                    "sont déjà persistés."
                )
        judge_name = run_config["judges"][0]["name"]
        for system in systems:
            score(
                str(run_config_path),
                system["name"],
                judge_name,
                None,
                progress_callback=tracker.update,
                verbose=False,
                concurrency=concurrency,
            )
        if complete:
            judge_prompt_path = Path(run_config["judge_prompt_file"])
            judge_prompt_hash = stable_hash(
                judge_prompt_path.read_text(encoding="utf-8")
            )
            judge_version = (
                f"{judge_prompt_path.stem}_{judge_prompt_hash[:8]}"
            )
            judge_name = run_config["judges"][0]["name"]
            incomplete_scores = []
            for system in systems:
                active_responses = {}
                response_path = (
                    output_dir
                    / "responses"
                    / f"{system['name']}.jsonl"
                )
                for response in load_jsonl(response_path):
                    row = dataset.get(response["question_id"])
                    if (
                        row
                        and response.get("question")
                        == row["question_patient"]
                    ):
                        active_responses[
                            response["question_id"]
                        ] = response["request_hash"]
                scored_questions = set()
                score_path = (
                    output_dir
                    / judge_version
                    / "scores"
                    / system["name"]
                    / f"{judge_name}.jsonl"
                )
                for record in load_jsonl(score_path):
                    question_id = record.get("question_id")
                    if (
                        record.get("judge_prompt_hash")
                        == judge_prompt_hash
                        and active_responses.get(question_id)
                        == record.get("response_request_hash")
                    ):
                        scored_questions.add(question_id)
                if len(scored_questions) != len(dataset):
                    incomplete_scores.append(
                        f"{system['name']}="
                        f"{len(scored_questions)}/{len(dataset)}"
                    )
            if incomplete_scores:
                raise RuntimeError(
                    "Couverture de scoring incomplète: "
                    + ", ".join(incomplete_scores)
                    + ". Relancez la même commande; les scores réussis "
                    "sont déjà persistés."
                )
        compute(
            str(run_config_path),
            None,
            verbose=False,
            partial=partial_summary,
        )
        tracker.metrics_done()
        tracker.complete()
    except Exception as error:
        tracker.fail(error)
        raise

    print(f"Progression: {progress_path}")
    judge_prompt_path = Path(run_config["judge_prompt_file"])
    judge_prompt_hash = stable_hash(
        judge_prompt_path.read_text(encoding="utf-8")
    )
    judge_version = f"{judge_prompt_path.stem}_{judge_prompt_hash[:8]}"
    comparison_name = (
        f"comparison_{mode}"
        if mode in {"rag", "rag_selective"}
        else "comparison"
    )
    if partial_summary:
        comparison_name += "_partial"
    comparison_filename = f"{comparison_name}.json"
    print(
        "Comparaison: "
        f"{output_dir / judge_version / 'metrics' / comparison_filename}"
    )
    print(
        "Exécution: "
        f"{tracker.state['api_calls']} appels API, "
        f"{tracker.state['cache_hits']} résultats réutilisés, "
        f"{tracker.state['local_calls']} étape(s) locale(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lancer toute l'évaluation pour un CSV et un mode."
    )
    parser.add_argument("dataset", help="Chemin du fichier CSV")
    parser.add_argument(
        "mode",
        choices=("baseline", "prompt", "rag", "rag_selective"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.example.json")),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Nombre maximal d'appels API simultanés par système",
    )
    parser.add_argument(
        "--partial-summary",
        action="store_true",
        help=(
            "Sauter la génération, scorer les réponses disponibles et produire "
            "des métriques partielles séparées"
        ),
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help=(
            "Exiger une couverture complète; dans les modes RAG, générer "
            "aussi les baselines manquantes avant le scoring"
        ),
    )
    args = parser.parse_args()
    try:
        run(
            args.dataset,
            args.mode,
            args.config,
            args.concurrency,
            args.partial_summary,
            args.complete,
        )
    except NotImplementedError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
