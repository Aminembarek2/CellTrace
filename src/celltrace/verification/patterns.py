"""Recompute active-rate and reported-RSRP patterns without private labels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from celltrace.config import DetectorSection, ObservationSection
from celltrace.contracts.common import AggregationBasis, DownloadState, MetricName, Statistic
from celltrace.contracts.evidence import RunContext
from celltrace.contracts.findings import (
    ActivityClaim,
    InvestigationFindings,
    MetricChangeClaim,
    ObservationPattern,
)
from celltrace.contracts.report import ObservationPatternCheck, VerificationCode
from celltrace.tools.repository import DropWindowSupport, MeasurementRepository

ACTIVE_DROP_CHECK_VERSION = "active_rate_drop_v1"
RADIO_CHECK_VERSION = "reported_rsrp_v2"


def complete_support(support: DropWindowSupport) -> bool:
    return support.expected_bins > 0 and all(
        count == support.expected_bins
        for count in (
            support.observed_bins,
            support.complete_bins,
            support.available_bins,
            support.same_segment_bins,
        )
    )


def check_active_rate_drop(
    context: RunContext,
    findings: InvestigationFindings,
    passed_claim_ids: set[str],
    repository: MeasurementRepository,
    configuration: DetectorSection,
    on_query: Callable[[str], None],
) -> ObservationPatternCheck:
    reference = context.candidate.representative_reference_window
    event = context.candidate.representative_event_window
    expected = {
        "check_version": ACTIVE_DROP_CHECK_VERSION,
        "reference_window": reference.model_dump(),
        "event_window": event.model_dump(),
        "basis": "bin_1s",
        "statistic": "median",
        "minimum_coverage": configuration.minimum_coverage,
        "minimum_active_fraction": configuration.minimum_active_fraction,
        "minimum_reference_mbps": configuration.minimum_reference_mbps,
        "minimum_absolute_drop_mbps": configuration.minimum_absolute_drop_mbps,
        "minimum_relative_drop": configuration.minimum_relative_drop,
    }

    def failure(
        message: str,
        actual: object = None,
        supporting_claim_ids: list[str] | None = None,
        code: VerificationCode = VerificationCode.observation_pattern_unsupported,
    ) -> ObservationPatternCheck:
        return ObservationPatternCheck(
            claim_id="pattern:active_rate_drop",
            pattern=ObservationPattern.active_rate_drop,
            passed=False,
            code=code,
            message=message,
            expected=expected,
            actual=actual,
            supporting_claim_ids=supporting_claim_ids or [],
        )

    if ObservationPattern.no_clear_symptom in findings.observation_patterns:
        return failure(
            "active_rate_drop conflicts with no_clear_symptom",
            code=VerificationCode.observation_pattern_conflict,
        )
    if (
        not context.allowed_window.contains(reference)
        or not context.allowed_window.contains(event)
        or reference.end_ms > event.start_ms
        or reference.length_ms != configuration.reference_ms
        or event.length_ms != configuration.event_ms
        or any(
            boundary % 1000
            for boundary in (reference.start_ms, reference.end_ms, event.start_ms, event.end_ms)
        )
    ):
        return failure(
            "representative windows must be in scope, aligned, disjoint and match "
            "the configured reference and event lengths"
        )
    verified_claims = [
        claim for claim in findings.factual_claims if claim.claim_id in passed_claim_ids
    ]
    rate_claims = [
        claim
        for claim in verified_claims
        if isinstance(claim, MetricChangeClaim)
        and claim.metric == MetricName.downlink_mbps
        and claim.basis == AggregationBasis.bin_1s
        and claim.statistic == Statistic.median
        and claim.reference_window == reference
        and claim.event_window == event
    ]
    activity_claims = [
        claim
        for claim in verified_claims
        if isinstance(claim, ActivityClaim)
        and claim.basis == AggregationBasis.native
        and claim.state == DownloadState.active
    ]
    reference_activity = next(
        (claim for claim in activity_claims if claim.window.contains(reference)), None
    )
    event_activity = next(
        (claim for claim in activity_claims if claim.window.contains(event)), None
    )
    if not rate_claims or reference_activity is None or event_activity is None:
        return failure(
            "active_rate_drop needs a verified median downlink change on the representative "
            "one second windows and verified native active state claims covering both windows"
        )
    supporting_ids = list(
        dict.fromkeys(
            (rate_claims[0].claim_id, reference_activity.claim_id, event_activity.claim_id)
        )
    )
    try:
        on_query("active rate drop reference support")
        reference_support = repository.drop_window(
            context.session_id, context.segment_id, reference, event.end_ms
        )
        on_query("active rate drop event support")
        event_support = repository.drop_window(
            context.session_id, context.segment_id, event, event.end_ms
        )
    except Exception as failure_reason:
        return failure(
            f"pattern recomputation failed: {type(failure_reason).__name__}",
            supporting_claim_ids=supporting_ids,
            code=VerificationCode.recomputation_failed,
        )
    actual: dict[str, object] = {
        "reference": asdict(reference_support),
        "event": asdict(event_support),
    }
    for window_support in (reference_support, event_support):
        if (
            not complete_support(window_support)
            or window_support.coverage is None
            or window_support.coverage < configuration.minimum_coverage
            or window_support.active_fraction is None
            or window_support.active_fraction < configuration.minimum_active_fraction
        ):
            return failure(
                "both windows need complete, available bins in this segment, sufficient "
                "rate coverage and active traffic",
                actual,
                supporting_ids,
            )
    reference_value = reference_support.median_mbps
    event_value = event_support.median_mbps
    if reference_value is None or reference_value < configuration.minimum_reference_mbps:
        return failure(
            "reference rate is missing or below the baseline minimum", actual, supporting_ids
        )
    if event_value is None:
        return failure("event rate is missing", actual, supporting_ids)
    absolute_drop = reference_value - event_value
    relative_drop = absolute_drop / reference_value
    actual.update(absolute_drop_mbps=absolute_drop, relative_drop=relative_drop)
    if (
        absolute_drop < configuration.minimum_absolute_drop_mbps
        or relative_drop < configuration.minimum_relative_drop
    ):
        return failure(
            "the recomputed rate change does not meet the drop thresholds", actual, supporting_ids
        )
    return ObservationPatternCheck(
        claim_id="pattern:active_rate_drop",
        pattern=ObservationPattern.active_rate_drop,
        passed=True,
        supporting_claim_ids=supporting_ids,
        expected=expected,
        actual=actual,
        message="active rate drop meets the configured observation rule, cause unresolved",
    )


def check_reported_radio(
    context: RunContext,
    findings: InvestigationFindings,
    pattern: ObservationPattern,
    passed_claim_ids: set[str],
    repository: MeasurementRepository,
    configuration: ObservationSection,
    on_query: Callable[[str], None],
) -> ObservationPatternCheck:
    """A bounded median RSRP observation, never a statement about unmeasured radio health."""
    reference = context.candidate.representative_reference_window
    event = context.candidate.representative_event_window
    expected = {
        "check_version": RADIO_CHECK_VERSION,
        "metric": "reported_rsrp_dbm",
        "statistic": "median",
        "basis": "from verified claim",
        "reference_window": reference.model_dump(),
        "event_window": event.model_dump(),
        **configuration.model_dump(),
    }

    def result(
        passed: bool,
        message: str,
        actual: object = None,
        ids: list[str] | None = None,
        code: VerificationCode = VerificationCode.observation_pattern_unsupported,
    ) -> ObservationPatternCheck:
        return ObservationPatternCheck(
            claim_id="pattern:" + pattern.value,
            pattern=pattern,
            passed=passed,
            message=message,
            expected=expected,
            actual=actual,
            supporting_claim_ids=ids or [],
            code=None if passed else code,
        )

    if {
        ObservationPattern.reported_radio_degradation,
        ObservationPattern.stable_reported_radio,
    } <= set(findings.observation_patterns):
        return result(
            False,
            "stable and degraded reported radio labels conflict",
            code=VerificationCode.observation_pattern_conflict,
        )
    if (
        not context.allowed_window.contains(reference)
        or not context.allowed_window.contains(event)
        or reference.end_ms > event.start_ms
        or any(
            t % 1000 for t in (reference.start_ms, reference.end_ms, event.start_ms, event.end_ms)
        )
    ):
        return result(False, "representative radio windows must be aligned, disjoint and in scope")
    supporting_claims = [
        claim
        for claim in findings.factual_claims
        if isinstance(claim, MetricChangeClaim)
        and claim.claim_id in passed_claim_ids
        and claim.metric == MetricName.reported_rsrp_dbm
        and claim.statistic == Statistic.median
        and claim.reference_window == reference
        and claim.event_window == event
    ]
    if not supporting_claims:
        return result(
            False,
            "radio label needs a verified median RSRP change on both representative windows",
        )
    supporting = [supporting_claims[0].claim_id]
    basis = supporting_claims[0].basis
    expected["basis"] = basis.value
    actual: dict[str, object] = {}
    medians = []
    try:
        for name, window in (("reference", reference), ("event", event)):
            on_query("reported RSRP " + name + " structural support")
            structure = repository.drop_window(
                context.session_id, context.segment_id, window, event.end_ms
            )
            on_query("reported RSRP " + name + " median and coverage")
            summary = repository.summarize(
                context.session_id, window, MetricName.reported_rsrp_dbm, basis
            )
            native = summary
            if basis != AggregationBasis.native:
                on_query("reported RSRP " + name + " native range flags")
                native = repository.summarize(
                    context.session_id,
                    window,
                    MetricName.reported_rsrp_dbm,
                    AggregationBasis.native,
                )
            actual[name] = {"structure": asdict(structure), "radio": summary.as_dict()}
            if summary.applicability is None or not summary.applicability.supported:
                return result(
                    False,
                    "reported RSRP label requires a supported measurement source in both windows",
                    actual,
                    supporting,
                    VerificationCode.metric_not_applicable,
                )
            if (
                not complete_support(structure)
                or summary.coverage is None
                or summary.coverage < configuration.minimum_radio_coverage
                or summary.median is None
                or native.outside_documented_range_count
            ):
                return result(
                    False,
                    "radio windows need available observations in this segment, "
                    "sufficient RSRP coverage and no out-of-range values",
                    actual,
                    supporting,
                )
            medians.append(summary.median)
    except Exception as failure_reason:
        return result(
            False,
            f"radio recomputation failed: {type(failure_reason).__name__}",
            actual,
            supporting,
            VerificationCode.recomputation_failed,
        )
    delta = medians[1] - medians[0]
    actual["rsrp_delta_db"] = delta
    passed = (
        abs(delta) <= configuration.rsrp_stability_tolerance_db
        if pattern == ObservationPattern.stable_reported_radio
        else delta < -configuration.rsrp_stability_tolerance_db
    )
    return result(
        passed,
        "reported median RSRP meets the observation rule; physical cause "
        "and unmeasured radio health remain unresolved"
        if passed
        else "reported median RSRP does not meet this observation rule",
        actual,
        supporting,
    )
