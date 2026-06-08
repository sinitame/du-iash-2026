"""Fonctions partagées par les trois étapes du pipeline d'évaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
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

    payload = {
        "model": provider["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        provider["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **provider.get("headers", {}),
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(
        request, timeout=provider.get("timeout_seconds", 120)
    ) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {
        "text": body["choices"][0]["message"]["content"].strip(),
        "latency_seconds": round(time.perf_counter() - started, 4),
        "usage": body.get("usage", {}),
        "raw_api_response": body,
    }


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("La réponse du juge ne contient pas d'objet JSON.")
    return json.loads(match.group(0))
