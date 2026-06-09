"""Fonctions partagées par les trois étapes du pipeline d'évaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    return json.loads(config_path.read_text(encoding="utf-8")), config_path.parent


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def stable_hash(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSON invalide dans {path}:{line_number}") from error
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_template(template: str, values: dict[str, Any]) -> str:
    placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", template))
    missing = placeholders - set(values)
    if missing:
        raise ValueError(
            f"Variables absentes du template: {', '.join(sorted(missing))}"
        )
    rendered = template
    for key in placeholders:
        rendered = rendered.replace(f"{{{{{key}}}}}", str(values[key]))
    return rendered


def call_chat_api(
    provider: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    api_key_env = provider.get("api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Variable d'environnement absente: {api_key_env}")

    api_format = provider.get("api_format", "openai_chat")
    if api_format == "anthropic_messages":
        system_parts = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        api_messages = [
            message for message in messages if message["role"] in {"user", "assistant"}
        ]
        payload = {
            "model": provider["model"],
            "messages": api_messages,
            "max_tokens": max_tokens,
            **provider.get("request_options", {}),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if provider.get("use_temperature", True):
            payload["temperature"] = temperature
        endpoint = provider["base_url"].rstrip("/") + "/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": provider.get("anthropic_version", "2023-06-01"),
            "Content-Type": "application/json",
            **provider.get("headers", {}),
        }
    elif api_format == "openai_chat":
        payload = {
            "model": provider["model"],
            "messages": messages,
            provider.get("output_tokens_parameter", "max_tokens"): max_tokens,
            **provider.get("request_options", {}),
        }
        if provider.get("use_temperature", True):
            payload["temperature"] = temperature
        endpoint = provider["base_url"].rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **provider.get("headers", {}),
        }
    else:
        raise ValueError(f"Format API inconnu: {api_format}")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request, timeout=provider.get("timeout_seconds", 120)
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        try:
            error_body = json.loads(response_text)
            api_error = error_body.get("error", {})
            message = api_error.get("message", response_text)
            error_type = api_error.get("type")
            error_code = api_error.get("code")
        except json.JSONDecodeError:
            message = response_text or error.reason
            error_type = None
            error_code = None

        details = [f"HTTP {error.code}", str(message)]
        if error_type:
            details.append(f"type={error_type}")
        if error_code:
            details.append(f"code={error_code}")
        request_id = error.headers.get("x-request-id") or error.headers.get("request-id")
        if request_id:
            details.append(f"request_id={request_id}")
        if (
            api_format == "openai_chat"
            and error.code == 403
            and error_code == "model_not_found"
        ):
            details.append(
                "Vérifiez que la clé autorise Chat completions en écriture "
                "et que le projet a accès au modèle."
            )
        raise RuntimeError("Erreur API: " + " | ".join(details)) from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Connexion à l'API impossible: {error.reason}") from None
    if api_format == "anthropic_messages":
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        ).strip()
        usage = body.get("usage", {})
    else:
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
    if not text:
        raise RuntimeError("L'API a retourné une réponse sans contenu texte.")

    return {
        "text": text,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "usage": usage,
        "raw_api_response": body,
    }


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("La réponse du juge ne contient pas d'objet JSON.")
    return json.loads(match.group(0))
