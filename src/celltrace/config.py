"""Strict application configuration.

Precedence, highest first: explicit CLI override, documented environment
variable, supplied config file, validated defaults. Secrets are never stored
in this model, only whether a credential is present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from celltrace.providers.profiles import PRIMARY_MODEL_ID, model_profile

DEFAULT_CONFIG_PATH = Path("configs/project-defaults.yaml")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProjectSection(StrictModel):
    name: str = "celltrace-investigator"
    seed: int = 42
    execution: Literal["synchronous_serial"] = "synchronous_serial"


class DataSection(StrictModel):
    primary_source: Literal["raca_irish_5g"] = "raca_irish_5g"
    primary_cohort: Literal["file_download"] = "file_download"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    private_evaluation_dir: Path = Path("data/eval_private")
    preserve_raw: Literal[True] = True
    normalized_bin_ms: Literal[1000] = 1000
    maximum_forward_fill_ms: Literal[0] = 0
    split_manifest: Path | None = None


class DetectorSection(StrictModel):
    name: Literal["sustained_relative_drop"] = "sustained_relative_drop"
    reference_ms: PositiveInt = 30_000
    event_ms: PositiveInt = 5_000
    step_ms: PositiveInt = 1_000
    minimum_coverage: float = Field(0.8, gt=0, le=1)
    minimum_active_fraction: float = Field(0.8, gt=0, le=1)
    minimum_reference_mbps: PositiveFloat = 1.0
    minimum_absolute_drop_mbps: PositiveFloat = 0.5
    minimum_relative_drop: float = Field(0.6, gt=0, lt=1)
    merge_gap_ms: int = Field(1_000, ge=0)
    maximum_event_ms: PositiveInt = 30_000

    @model_validator(mode="after")
    def check_alignment(self) -> DetectorSection:
        for name in ("reference_ms", "event_ms", "step_ms"):
            if getattr(self, name) % 1000 != 0:
                raise ValueError(f"detector.{name} must be a whole number of 1000 ms bins")
        return self


class PredictionSection(StrictModel):
    horizon_ms: PositiveInt = 5_000
    maximum_history_ms: PositiveInt = 30_000
    require_complete_target_bins: Literal[True] = True
    baselines: tuple[Literal["persistence", "trailing_mean"], ...] = (
        "persistence",
        "trailing_mean",
    )
    model: Literal["hist_gradient_boosting"] = "hist_gradient_boosting"
    early_stopping: Literal[False] = False


class ObservationSection(StrictModel):
    rsrp_stability_tolerance_db: float = Field(3.0, ge=0, le=20)
    minimum_radio_coverage: float = Field(0.8, gt=0, le=1)


class AnalysisSection(StrictModel):
    detector: DetectorSection = DetectorSection()
    prediction: PredictionSection = PredictionSection()
    observation: ObservationSection = ObservationSection()


class RuntimeSection(StrictModel):
    provider: Literal["openrouter"] = "openrouter"
    enable_live_calls: bool = False
    model_id: str = PRIMARY_MODEL_ID
    model_effort: Literal["high"] = "high"
    strict_tool_schemas: bool = False
    run_store: Path = Path("artifacts/private/runs.sqlite3")
    checkpoint_store: Path = Path("artifacts/private/checkpoints.sqlite3")
    maximum_tool_requests: int = Field(12, ge=1, le=100)
    reviewer_tool_reserve: int = Field(0, ge=0, le=99)
    show_remaining_budget: bool = False
    require_review_measurement: bool = False
    review_round_tool_reserve: int = Field(0, ge=0, strict=True)
    review_round_provider_reserve: int = Field(0, ge=0, strict=True)
    review_round_input_reserve: int = Field(0, ge=0, strict=True)
    review_round_output_reserve: int = Field(0, ge=0, strict=True)
    maximum_provider_attempts: int = Field(12, ge=1, le=100)
    maximum_review_rounds: int = Field(2, ge=0, le=5)
    maximum_request_retries: int = Field(1, ge=0, le=3)
    maximum_active_seconds: PositiveInt = 300
    provider_timeout_seconds: PositiveFloat = 45
    maximum_input_tokens: PositiveInt = 40_000
    maximum_output_tokens: PositiveInt = 12_000
    maximum_output_tokens_per_attempt: PositiveInt = 2_048
    maximum_estimated_usd_per_run: PositiveFloat = 0.5
    maximum_estimated_usd_per_experiment: PositiveFloat = 5.0
    verified_rate_card: Path | None = None
    verified_endpoint_record: Path | None = None
    verified_capability_record: Path | None = None
    require_live_capability_probe: Literal[True] = True
    evidence_radius_before_ms: PositiveInt = 60_000
    evidence_radius_after_ms: PositiveInt = 30_000
    maximum_tool_window_ms: PositiveInt = 120_000
    maximum_result_bytes: PositiveInt = 16_000

    @model_validator(mode="after")
    def check_limits(self) -> RuntimeSection:
        model_profile(self.model_id)
        if self.reviewer_tool_reserve >= self.maximum_tool_requests:
            raise ValueError("reviewer tool reserve must be smaller than the total tool allowance")
        if self.maximum_output_tokens_per_attempt > self.maximum_output_tokens:
            raise ValueError("per attempt output tokens cannot exceed the run output allowance")
        for reserve, total in (
            (self.review_round_tool_reserve, self.maximum_tool_requests),
            (self.review_round_provider_reserve, self.maximum_provider_attempts),
            (self.review_round_input_reserve, self.maximum_input_tokens),
            (self.review_round_output_reserve, self.maximum_output_tokens),
        ):
            if reserve and reserve * self.maximum_review_rounds >= total:
                raise ValueError("review round protection must leave resources for investigation")
        if self.maximum_estimated_usd_per_run > self.maximum_estimated_usd_per_experiment:
            raise ValueError("run monetary ceiling cannot exceed the experiment ceiling")
        if self.enable_live_calls and self.verified_rate_card is None:
            raise ValueError("live calls need runtime.verified_rate_card for the monetary budget")
        return self


class EvaluationSection(StrictModel):
    proposed_heldout_base_windows: PositiveInt = 12
    variants_per_base_window: PositiveInt = 3
    primary_systems: tuple[str, ...] = (
        "investigator_raw",
        "investigator_gated",
        "investigator_reviewer_gated",
    )
    optional_systems: tuple[str, ...] = (
        "investigator_self_revision_gated",
        "deterministic_template",
    )
    repeat_stratified_cases: int = Field(6, ge=0)
    repeats_per_selected_case: int = Field(3, ge=1)
    bootstrap_unit: Literal["source_session"] = "source_session"
    include_failed_runs: Literal[True] = True


class ExtensionsSection(StrictModel):
    telecomts: Literal[False] = False
    free_form_sql: Literal[False] = False
    web_interface: Literal[False] = False
    local_llm: Literal[False] = False


class AppConfig(StrictModel):
    schema_version: Literal[1] = 1
    project: ProjectSection = ProjectSection()
    data: DataSection = DataSection()
    analysis: AnalysisSection = AnalysisSection()
    runtime: RuntimeSection = RuntimeSection()
    evaluation: EvaluationSection = EvaluationSection()
    extensions: ExtensionsSection = ExtensionsSection()

    def redacted_summary(self) -> dict[str, Any]:
        """Resolved non secret configuration, safe to log and persist."""
        summary = self.model_dump(mode="json")
        summary["credentials"] = {"openrouter_api_key_present": api_key_present()}
        return summary


ENVIRONMENT_OVERRIDES: dict[str, tuple[str, ...]] = {
    "CELLTRACE_MODEL": ("runtime", "model_id"),
    "CELLTRACE_DATA_DIR": ("data", "raw_dir"),
}


def api_key_present() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict[tuple[str, ...], Any] | None = None,
    environment: dict[str, str] | None = None,
) -> AppConfig:
    """Load YAML safely, apply environment then CLI overrides, then validate strictly."""
    environment = dict(os.environ) if environment is None else environment
    if config_path is None and environment.get("CELLTRACE_CONFIG"):
        config_path = Path(environment["CELLTRACE_CONFIG"])
    if config_path is None and DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH

    raw: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a mapping at the top level")
        raw = loaded or {}

    for variable, path in ENVIRONMENT_OVERRIDES.items():
        if environment.get(variable):
            value: Any = environment[variable]
            if variable == "CELLTRACE_DATA_DIR":
                value = str(Path(value) / "raw")
            _set_nested(raw, path, value)

    for path, value in (cli_overrides or {}).items():
        _set_nested(raw, path, value)

    return AppConfig.model_validate(raw)
