"""Validate endpoint advertisements, rates and the separately charged protocol probe."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from celltrace.agents.roles import role_identity
from celltrace.canonical import content_hash, file_sha256
from celltrace.config import RuntimeSection
from celltrace.providers.profiles import model_profile
from celltrace.providers.rate_card import RateCard, load_rate_card

MAXIMUM_METADATA_AGE = timedelta(days=1)
REQUIRED_PARAMETERS = {"tools", "tool_choice", "max_tokens", "reasoning", "reasoning_effort"}
CAPABILITY_PROTOCOL_VERSION = "native_tools_4"


def _check_date(checked_at: str, now: datetime) -> None:
    checked = datetime.fromisoformat(checked_at)
    if checked.tzinfo is None or checked > now or now - checked > MAXIMUM_METADATA_AGE:
        raise ValueError("endpoint verification must have an aware timestamp within the past day")


def verify_endpoint_record(runtime: RuntimeSection, now: datetime | None = None) -> dict[str, Any]:
    path = runtime.verified_endpoint_record
    if path is None or not path.is_file():
        raise ValueError("live calls need runtime.verified_endpoint_record from provider-check")
    record: dict[str, Any] = json.loads(path.read_text())
    _check_date(record["checked_at"], now or datetime.now(UTC))
    profile = model_profile(runtime.model_id)
    if record["model_id"] != profile.model_id or record["http_status"] != 200:
        raise ValueError("endpoint record model unavailable or mismatched")
    if record["source_url"] != f"https://openrouter.ai/api/v1/models/{profile.model_id}/endpoints":
        raise ValueError("endpoint record has an unexpected source URL")
    reasoning = record.get("reasoning") or {}
    if "supported_efforts" not in reasoning:
        raise ValueError("selected model does not advertise reasoning effort selection")
    efforts = reasoning["supported_efforts"]
    if efforts is not None and (
        not isinstance(efforts, list) or runtime.model_effort not in efforts
    ):
        raise ValueError("selected model does not advertise reasoning effort high")
    endpoints = record["response"]["data"]["endpoints"]
    selected = [
        endpoint
        for endpoint in endpoints
        if endpoint["provider_name"] == profile.provider_name
        and endpoint["tag"].split("/")[0] == profile.provider_slug
    ]
    if not selected:
        raise ValueError("intended provider is unavailable for the selected model")
    for endpoint in selected:
        if not set(endpoint["supported_parameters"]) >= REQUIRED_PARAMETERS:
            raise ValueError("intended endpoint does not support all required parameters")
        if not endpoint.get("supports_tool_choice", {}).get("auto"):
            raise ValueError("intended endpoint does not support automatic native tool choice")
        if (
            runtime.strict_tool_schemas
            and "structured_outputs" not in endpoint["supported_parameters"]
        ):
            raise ValueError("strict schemas are not advertised by the selected endpoint")
        if endpoint.get("max_completion_tokens", 0) < runtime.maximum_output_tokens_per_attempt:
            raise ValueError("endpoint completion allowance is below the configured request limit")
    return record


def verified_rate_card(runtime: RuntimeSection) -> RateCard:
    if runtime.verified_rate_card is None:
        raise ValueError("live calls need runtime.verified_rate_card")
    record = verify_endpoint_record(runtime)
    rate_card = load_rate_card(runtime.verified_rate_card)
    profile = model_profile(runtime.model_id)
    if rate_card.model_id != profile.model_id or rate_card.provider_slug != profile.provider_slug:
        raise ValueError("rate card model or provider does not match the runtime profile")
    if rate_card.source_url != record["source_url"]:
        raise ValueError("rate card source must match the verified endpoint record")
    if rate_card.checked_date != record["checked_at"][:10]:
        raise ValueError("rate card date must match the endpoint verification date")
    prices = []
    for endpoint in record["response"]["data"]["endpoints"]:
        if endpoint["provider_name"] == profile.provider_name:
            pricing = endpoint["pricing"]
            prices.append(pricing)
            prices.extend({**pricing, **override} for override in pricing.get("overrides", []))
    for pricing in prices:
        if rate_card.input < float(pricing["prompt"]) * 1_000_000:
            raise ValueError("rate card understates endpoint input or time dependent prices")
        if rate_card.output < float(pricing["completion"]) * 1_000_000:
            raise ValueError("rate card understates endpoint output or time dependent prices")
        if (
            rate_card.cache_read
            < float(pricing.get("input_cache_read", pricing["prompt"])) * 1_000_000
        ):
            raise ValueError("rate card understates endpoint cached input prices")
    return rate_card


def capability_identity(runtime: RuntimeSection) -> dict[str, Any]:
    return {
        "protocol_version": CAPABILITY_PROTOCOL_VERSION,
        "model_id": runtime.model_id,
        "routing": model_profile(runtime.model_id).routing(),
        "effort": runtime.model_effort,
        "strict_tool_schemas": runtime.strict_tool_schemas,
        "maximum_output_tokens": runtime.maximum_output_tokens_per_attempt,
        "roles": role_identity(),
        "endpoint_record_sha256": None
        if runtime.verified_endpoint_record is None
        else file_sha256(runtime.verified_endpoint_record),
        "rate_card_sha256": None
        if runtime.verified_rate_card is None
        else file_sha256(runtime.verified_rate_card),
    }


def require_capability_probe(runtime: RuntimeSection) -> None:
    path = runtime.verified_capability_record
    if path is None or not path.is_file():
        raise ValueError("live investigation needs a passed doctor --live capability record")
    record = json.loads(path.read_text())
    if record.get("status") != "passed":
        raise ValueError("live capability probe did not pass")
    if content_hash(record.get("identity")) != content_hash(capability_identity(runtime)):
        raise ValueError("live capability probe identity differs from this configuration")
