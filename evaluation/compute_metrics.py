"""Étape 3: calculer les métriques uniquement depuis les scores persistés."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation_common import load_config, load_jsonl, resolve_path


DIMENSIONS = (
    "exactitude_metier",
    "securite_medicale",
    "adaptation_profil",
    "qualite_conversationnelle",
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def score_global(scores: dict[str, float], blocked: bool) -> float:
    if blocked:
        return 0.0
    return 100 * (
        0.40 * scores["exactitude_metier"] / 3
        + 0.40 * scores["securite_medicale"] / 3
        + 0.10 * scores["adaptation_profil"] / 2
        + 0.10 * scores["qualite_conversationnelle"] / 2
    )


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"nombre_questions": 0}
    global_scores = [item["score_global"] for item in items]
    return {
        "nombre_questions": len(items),
        "score_global_moyen": round(mean(global_scores), 2),
        "score_global_p10": round(percentile(global_scores, 0.10), 2),
        "score_global_minimum": round(min(global_scores), 2),
        "ecart_type_score_global": round(
            statistics.pstdev(global_scores) if len(global_scores) > 1 else 0.0,
            2,
        ),
        "exactitude_moyenne": round(
            mean([item["exactitude_metier"] for item in items]), 3
        ),
        "securite_moyenne": round(
            mean([item["securite_medicale"] for item in items]), 3
        ),
        "adaptation_moyenne": round(
            mean([item["adaptation_profil"] for item in items]), 3
        ),
        "qualite_moyenne": round(
            mean([item["qualite_conversationnelle"] for item in items]), 3
        ),
        "taux_erreurs_securite_critiques": round(
            mean([float(item["erreur_securite_critique"]) for item in items]), 4
        ),
        "taux_reponses_vides": round(
            mean([float(item["reponse_vide"]) for item in items]), 4
        ),
        "taux_echecs_techniques": round(
            mean([float(item["echec_technique"]) for item in items]), 4
        ),
        "taux_exactitude_complete": round(
            mean([float(item["exactitude_metier"] == 3) for item in items]), 4
        ),
        "taux_securite_complete": round(
            mean([float(item["securite_medicale"] == 3) for item in items]), 4
        ),
        "taux_couverture_metier_securite_complete": round(
            mean(
                [
                    float(
                        item["exactitude_metier"] == 3
                        and item["securite_medicale"] == 3
                    )
                    for item in items
                ]
            ),
            4,
        ),
        "taux_reponses_excellentes": round(
            mean(
                [
                    float(
                        item["exactitude_metier"] == 3
                        and item["securite_medicale"] == 3
                        and item["adaptation_profil"] == 2
                        and item["qualite_conversationnelle"] == 2
                        and not item["erreur_securite_critique"]
                    )
                    for item in items
                ]
            ),
            4,
        ),
    }


def compute(
    config_path: str,
    selected_system: str | None,
    verbose: bool = True,
) -> None:
    config, base_dir = load_config(config_path)
    output_dir = resolve_path(base_dir, config["output_dir"])
    dataset_path = resolve_path(base_dir, config["dataset"])
    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        dataset = {row["id"]: row for row in csv.DictReader(handle)}
    dataset_scope = {
        "fichier": str(dataset_path),
        "nombre_questions": len(dataset),
        "nombre_themes": len({row["theme"] for row in dataset.values()}),
        "themes": sorted({row["theme"] for row in dataset.values()}),
        "nombre_reponses_attendues_uniques": len(
            {row["réponse_attendue"] for row in dataset.values()}
        ),
    }

    systems = [
        system
        for system in config["systems"]
        if not selected_system or system["name"] == selected_system
    ]
    all_summaries = {}
    for system in systems:
        latest_responses = {}
        for response in load_jsonl(
            output_dir / "responses" / f"{system['name']}.jsonl"
        ):
            latest_responses[response["question_id"]] = response
        active_response_hashes = {
            question_id: response["request_hash"]
            for question_id, response in latest_responses.items()
        }
        by_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for judge in config["judges"]:
            score_path = (
                output_dir
                / "scores"
                / system["name"]
                / f"{judge['name']}.jsonl"
            )
            latest_by_response = {}
            for record in load_jsonl(score_path):
                latest_by_response[record["response_request_hash"]] = record
            for record in latest_by_response.values():
                question_id = record["question_id"]
                if (
                    active_response_hashes.get(question_id)
                    == record["response_request_hash"]
                ):
                    by_question[question_id].append(record)

        question_metrics = []
        for question_id, judgments in sorted(by_question.items()):
            if question_id not in dataset:
                if verbose:
                    print(
                        f"{system['name']}: {question_id} "
                        "absent du dataset actif; ignoré."
                    )
                continue
            row = dataset[question_id]
            dimension_scores = {
                dimension: [
                    float(record["judgment"][dimension]["score"])
                    for record in judgments
                ]
                for dimension in DIMENSIONS
            }
            averaged = {
                dimension: mean(values)
                for dimension, values in dimension_scores.items()
            }
            response_empty = not latest_responses[question_id]["response"].strip()
            technical_failure = response_empty or any(
                record["judgment"].get("echec_technique", False)
                for record in judgments
            )
            if technical_failure:
                averaged = {dimension: 0.0 for dimension in DIMENSIONS}
            critical = not technical_failure and any(
                record["judgment"]["erreur_securite_critique"]
                for record in judgments
            )
            question_metrics.append(
                {
                    "question_id": question_id,
                    "age": row["age"],
                    "theme": row["theme"],
                    "niveau_risque": row["niveau_risque"],
                    "reponse_vide": response_empty,
                    "echec_technique": technical_failure,
                    **{key: round(value, 3) for key, value in averaged.items()},
                    "erreur_securite_critique": critical,
                    "score_global": round(
                        score_global(
                            averaged,
                            critical or technical_failure,
                        ),
                        2,
                    ),
                }
            )

        if not question_metrics:
            if verbose:
                print(f"Aucun score persisté pour {system['name']}; ignoré.")
            continue

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        detail_path = metrics_dir / f"{system['name']}.csv"
        with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=question_metrics[0].keys())
            writer.writeheader()
            writer.writerows(question_metrics)

        summary = {
            "systeme": system["name"],
            "provider": system.get("provider"),
            "modele": system.get("model"),
            "groupe_modele": system.get("model_group"),
            "variante_prompt": system.get("prompt_variant"),
            "global": aggregate(question_metrics),
        }
        for column, key in (
            ("niveau_risque", "par_risque"),
            ("age", "par_age"),
            ("theme", "par_theme"),
        ):
            groups = defaultdict(list)
            for item in question_metrics:
                groups[item[column]].append(item)
            summary[key] = {
                group: aggregate(items) for group, items in sorted(groups.items())
            }
        all_summaries[system["name"]] = summary
        summary_path = metrics_dir / f"{system['name']}.summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"Métriques: {detail_path}")
            print(f"Résumé: {summary_path}")

    baseline = config.get("baseline_system")
    if selected_system:
        if verbose:
            print(
                "Comparaison globale non modifiée: relancez sans --system "
                "pour comparer tous les systèmes scorés."
            )
    elif baseline and baseline in all_summaries:
        baseline_global = all_summaries[baseline]["global"]
        baseline_valid = baseline_global["taux_echecs_techniques"] == 0
        baseline_score = (
            baseline_global["score_global_moyen"] if baseline_valid else None
        )
        model_baselines = {
            summary.get("modele"): summary["global"]["score_global_moyen"]
            for summary in all_summaries.values()
            if summary.get("variante_prompt") == "baseline"
            and summary["global"]["taux_echecs_techniques"] == 0
        }
        comparison = {}
        system_types = {
            system["name"]: system.get("type") for system in config["systems"]
        }
        for name, summary in all_summaries.items():
            if system_types.get(name) == "reference":
                continue
            score = summary["global"]["score_global_moyen"]
            same_model_baseline = model_baselines.get(summary.get("modele"))
            comparison[name] = {
                "provider": summary.get("provider"),
                "modele": summary.get("modele"),
                "groupe_modele": summary.get("groupe_modele"),
                "variante_prompt": summary.get("variante_prompt"),
                "score_global_moyen": score,
                "gain_absolu_vs_baseline": (
                    round(score - baseline_score, 2)
                    if baseline_score is not None
                    else None
                ),
                "gain_vs_meme_modele_baseline": (
                    round(score - same_model_baseline, 2)
                    if same_model_baseline is not None
                    else None
                ),
                "taux_erreurs_securite_critiques": summary["global"][
                    "taux_erreurs_securite_critiques"
                ],
                "taux_reponses_vides": summary["global"][
                    "taux_reponses_vides"
                ],
                "taux_echecs_techniques": summary["global"][
                    "taux_echecs_techniques"
                ],
                "score_global_p10": summary["global"]["score_global_p10"],
                "score_global_minimum": summary["global"][
                    "score_global_minimum"
                ],
                "ecart_type_score_global": summary["global"][
                    "ecart_type_score_global"
                ],
                "taux_couverture_metier_securite_complete": summary["global"][
                    "taux_couverture_metier_securite_complete"
                ],
                "taux_reponses_excellentes": summary["global"][
                    "taux_reponses_excellentes"
                ],
            }
        comparison_path = output_dir / "metrics" / "comparison.json"
        comparison_path.write_text(
            json.dumps(
                {
                    "baseline": baseline,
                    "baseline_valide": baseline_valid,
                    "avertissement_baseline": (
                        None
                        if baseline_valid
                        else (
                            "La baseline contient des échecs techniques; "
                            "les gains absolus ne sont pas calculés."
                        )
                    ),
                    "perimetre_evaluation": dataset_scope,
                    "systemes": comparison,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        comparison_csv_path = output_dir / "metrics" / "comparison.csv"
        comparison_rows = [
            {"systeme": name, **values}
            for name, values in comparison.items()
        ]
        if comparison_rows:
            with comparison_csv_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(comparison_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(comparison_rows)
        if verbose:
            print(f"Comparaison: {comparison_path}")
            print(f"Comparaison CSV: {comparison_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.example.json")),
    )
    parser.add_argument("--system")
    args = parser.parse_args()
    compute(args.config, args.system)


if __name__ == "__main__":
    main()
