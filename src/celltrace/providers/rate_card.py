"""Verified provider prices used for estimated monetary budgets."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from celltrace.providers.base import ProviderUsage

TOKENS_PER_UNIT = 1_000_000


class RateCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_id: str
    provider_slug: str | None = None
    currency: str = Field(pattern="^USD$")
    unit: str = Field(pattern="^per_million_tokens$")
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_write_5m: float = Field(ge=0)
    cache_write_1h: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    tool_use_system_prompt_tokens_auto: int = Field(ge=0)
    source_url: str
    checked_date: str
    cache_treatment: str
    extra_charges: str

    def cost(self, usage: ProviderUsage) -> float:
        if usage.billed_usd is not None:
            return usage.billed_usd
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_input_tokens * self.cache_read
            + usage.cache_creation_input_tokens * self.cache_write_5m
        ) / TOKENS_PER_UNIT

    def reservation_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Upper estimate for admission control: all input treated as uncached."""
        return (input_tokens * self.input + output_tokens * self.output) / TOKENS_PER_UNIT


def load_rate_card(path: Path) -> RateCard:
    return RateCard.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
