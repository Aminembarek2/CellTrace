"""Endpoint evidence, charged protocol probes and uncertain outcomes without inference."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from celltrace.config import AppConfig, load_config
from celltrace.providers.openrouter_adapter import OpenRouterAdapter
from celltrace.providers.profiles import PRIMARY_MODEL_ID
from celltrace.providers.readiness import require_capability_probe, verify_endpoint_record
from celltrace.workflow.live_probe import run_live_probe
from tests.unit.test_openrouter_provider import completion_payload


def live_configuration(tmp_path: Path) -> AppConfig:
    checked_at = datetime.now(UTC).isoformat()
    source_url = f"https://openrouter.ai/api/v1/models/{PRIMARY_MODEL_ID}/endpoints"
    endpoint = {
        "checked_at": checked_at,
        "model_id": PRIMARY_MODEL_ID,
        "source_url": source_url,
        "http_status": 200,
        "reasoning": {"supported_efforts": ["high"]},
        "response": {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "DeepSeek",
                        "tag": "deepseek",
                        "supported_parameters": [
                            "tools",
                            "tool_choice",
                            "max_tokens",
                            "reasoning",
                            "reasoning_effort",
                        ],
                        "supports_tool_choice": {"auto": True},
                        "max_completion_tokens": 10000,
                        "pricing": {
                            "prompt": "0.0000003",
                            "completion": "0.0000012",
                            "input_cache_read": "0.000000006",
                        },
                    }
                ]
            }
        },
    }
    endpoint_path = tmp_path / "endpoint.json"
    endpoint_path.write_text(json.dumps(endpoint))
    rate_path = tmp_path / "rate.yaml"
    rate_path.write_text(
        yaml.safe_dump(
            {
                "model_id": PRIMARY_MODEL_ID,
                "provider_slug": "deepseek",
                "currency": "USD",
                "unit": "per_million_tokens",
                "input": 0.3,
                "output": 1.2,
                "cache_write_5m": 0,
                "cache_write_1h": 0,
                "cache_read": 0.006,
                "tool_use_system_prompt_tokens_auto": 0,
                "source_url": source_url,
                "checked_date": checked_at[:10],
                "cache_treatment": "subset",
                "extra_charges": "none",
            }
        )
    )
    return load_config(
        environment={},
        cli_overrides={
            ("runtime", "enable_live_calls"): True,
            ("runtime", "verified_rate_card"): rate_path,
            ("runtime", "verified_endpoint_record"): endpoint_path,
            ("runtime", "run_store"): tmp_path / "runs.sqlite3",
            ("runtime", "checkpoint_store"): tmp_path / "checkpoints.sqlite3",
        },
    )


def test_stale_or_unsupported_endpoint_is_rejected(tmp_path: Path) -> None:
    configuration = live_configuration(tmp_path)
    path = configuration.runtime.verified_endpoint_record
    assert path is not None
    record = json.loads(path.read_text())
    record["checked_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="past day"):
        verify_endpoint_record(configuration.runtime)
    record["checked_at"] = datetime.now(UTC).isoformat()
    record["response"]["data"]["endpoints"][0]["supported_parameters"].remove("tools")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="required parameters"):
        verify_endpoint_record(configuration.runtime)


@pytest.mark.parametrize("reasoning_returned", [True, False])
def test_both_role_probe_charges_continuation_and_terminal_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reasoning_returned: bool,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "local_mock_only")
    configuration = live_configuration(tmp_path)
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        instruction = body["messages"][0]["content"]
        payload = completion_payload()
        if not reasoning_returned:
            payload["choices"][0]["message"].pop("reasoning_details")
            payload["usage"]["completion_tokens_details"]["reasoning_tokens"] = 0
        function = payload["choices"][0]["message"]["tool_calls"][0]["function"]
        if "with exactly " in instruction:
            function["arguments"] = instruction.split("with exactly ", 1)[1]
            function["name"] = instruction.split("Call ", 1)[1].split(" with exactly ", 1)[0]
            if reasoning_returned:
                assert (
                    body["messages"][-3]["reasoning_details"][0]["data"] == "opaque_signature_123"
                )
            else:
                assert "reasoning_details" not in body["messages"][-3]
            assert body["messages"][-2]["role"] == "tool"
            assert body["messages"][-1]["role"] == "user"
            assert "Current ledger budget" in body["messages"][-1]["content"]
        return httpx.Response(200, json=payload)

    adapter = OpenRouterAdapter(
        PRIMARY_MODEL_ID, httpx.Client(transport=httpx.MockTransport(respond))
    )
    result = run_live_probe(PRIMARY_MODEL_ID, configuration, tmp_path / "probe", adapter)
    assert result["status"] == "passed"
    assert all(
        check["effective_settings"]["reasoning_observed"] == reasoning_returned
        and check["effective_settings"]["effective_effort"] is None
        for check in result["checks"].values()
    )
    assert len(requests) == 4
    assert result["usage"]["provider_attempts"] == 4
    assert result["usage"]["tool_requests"] == 2
    assert result["usage"]["unknown_usage_attempts"] == 0
    assert result["usage"]["estimated_usd"] == pytest.approx(0.0016)
    resolved = configuration.model_dump(mode="json")
    resolved["runtime"]["verified_capability_record"] = result["saved_to"]
    checked = AppConfig.model_validate(resolved)
    require_capability_probe(checked.runtime)
    saved_probe = Path(result["saved_to"])
    old_probe = json.loads(saved_probe.read_text())
    old_probe["identity"].pop("protocol_version")
    saved_probe.write_text(json.dumps(old_probe))
    with pytest.raises(ValueError, match="identity differs"):
        require_capability_probe(checked.runtime)
    saved_probe.write_text(json.dumps(result))
    resolved["runtime"]["maximum_output_tokens_per_attempt"] = 1024
    with pytest.raises(ValueError, match="identity differs"):
        require_capability_probe(AppConfig.model_validate(resolved).runtime)


@pytest.mark.parametrize("supported_efforts", [None, ["high", "low"]])
def test_gateway_effort_advertisement_accepts_null_and_explicit_high(
    tmp_path: Path, supported_efforts: list[str] | None
) -> None:
    configuration = live_configuration(tmp_path)
    path = configuration.runtime.verified_endpoint_record
    assert path is not None
    record = json.loads(path.read_text())
    record["reasoning"]["supported_efforts"] = supported_efforts
    path.write_text(json.dumps(record))
    verify_endpoint_record(configuration.runtime)
    record["reasoning"].pop("supported_efforts")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="effort selection"):
        verify_endpoint_record(configuration.runtime)
    record["reasoning"]["supported_efforts"] = ["low"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="effort high"):
        verify_endpoint_record(configuration.runtime)


def test_probe_timeout_stays_reserved_and_is_never_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "local_mock_only")
    configuration = live_configuration(tmp_path)
    requests = []

    def timeout(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("local timeout", request=request)

    adapter = OpenRouterAdapter(
        PRIMARY_MODEL_ID, httpx.Client(transport=httpx.MockTransport(timeout))
    )
    result = run_live_probe(PRIMARY_MODEL_ID, configuration, tmp_path / "probe", adapter)
    assert result["status"] == "failed"
    assert len(requests) == 1
    assert result["usage"]["provider_attempts"] == 1
    assert result["usage"]["unknown_usage_attempts"] == 1
    assert result["usage"]["estimated_usd"] > 0
