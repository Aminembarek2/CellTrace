"""Public metadata verification. This command never performs inference."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from celltrace.providers.profiles import MODEL_PROFILES


def command_provider_check(arguments: argparse.Namespace) -> int:
    from celltrace.cli import EXIT_BLOCKED, CommandError

    output_directory = Path(arguments.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=20) as client:
            model_response = client.get("https://openrouter.ai/api/v1/models")
            model_response.raise_for_status()
            models = {model["id"]: model for model in model_response.json()["data"]}
            for model_id, profile in MODEL_PROFILES.items():
                source_url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
                response = client.get(source_url)
                record = {
                    "model_id": model_id,
                    "checked_at": datetime.now(UTC).isoformat(),
                    "http_status": response.status_code,
                    "source_url": source_url,
                    "reasoning": models.get(model_id, {}).get("reasoning", {}),
                    "response": response.json(),
                }
                output_path = output_directory / (model_id.replace("/", "_") + ".json")
                output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                endpoints = record["response"].get("data", {}).get("endpoints", [])
                print(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "intended_provider": profile.provider_name,
                            "provider_available": any(
                                endpoint["provider_name"] == profile.provider_name
                                for endpoint in endpoints
                            ),
                            "http_status": response.status_code,
                            "saved_to": str(output_path),
                            "inference_tested": False,
                        }
                    )
                )
    except (httpx.HTTPError, ValueError, KeyError) as failure:
        raise CommandError(
            f"public provider verification blocked: {type(failure).__name__}", EXIT_BLOCKED
        ) from failure
    return 0


def register(subparsers: Any) -> None:
    command = subparsers.add_parser("provider-check", help="save public endpoint capabilities")
    command.add_argument("--out", default="artifacts/public/environment/openrouter_metadata")
    command.set_defaults(handler=command_provider_check)
