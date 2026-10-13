"""Executable verification of typed factual claims.

For each claim the verifier resolves every cited evidence id, checks run,
snapshot, dictionary and content hash, confirms that at least one cited
record actually covers the claim, and recomputes the value from the snapshot.
A model cannot pass a check by agreement or confidence, and a reviewer
decision cannot override a failure here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from celltrace.config import DetectorSection, ObservationSection
from celltrace.contracts.common import (
    RELATIVE_CHANGE_METRICS,
    AggregationBasis,
    MetricName,
    TimeWindow,
)
from celltrace.contracts.evidence import RunContext
from celltrace.contracts.findings import (
    ActivityClaim,
    AvailabilityClaim,
    FactualClaim,
    HypothesisName,
    HypothesisStatus,
    InvestigationFindings,
    MetricChangeClaim,
    MetricStatisticClaim,
    ObservationPattern,
    TransitionClaim,
)
from celltrace.contracts.report import ClaimCheck, VerificationCode, VerificationResult
from celltrace.storage.run_store import RunStore
from celltrace.tools.dispatch import (
    compare_summaries,
    evidence_content_hash,
    transitions_in_window,
)
from celltrace.tools.repository import MeasurementRepository
from celltrace.verification.patterns import check_active_rate_drop, check_reported_radio

ABSOLUTE_TOLERANCE = 1e-6
# The archive has no block error, scheduling, interference, cell load, transport or
# server fields, so nothing in it can rule these out (stable RSRP does not rule
# out every radio issue). "Consistent" means not ruled out and stays
# allowed.
UNOBSERVABLE_EXCLUSIONS = frozenset(
    {
        HypothesisName.radio_degradation,
        HypothesisName.shared_resource_load,
        HypothesisName.transport_or_server,
    }
)
RELATIVE_TOLERANCE = 1e-4


def numbers_match(actual: float | None, expected: float | None) -> bool:
    """Declared engineering tolerance in canonical units. None only matches None."""
    if actual is None or expected is None:
        return actual is None and expected is None
    return abs(actual - expected) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(expected)


def _fail(claim_id: str, code: VerificationCode, message: str, **values: Any) -> ClaimCheck:
    return ClaimCheck(
        claim_id=claim_id,
        passed=False,
        code=code,
        message=message,
        expected=values.get("expected"),
        actual=values.get("actual"),
    )


class Verifier:
    def __init__(
        self,
        context: RunContext,
        store: RunStore,
        repository: MeasurementRepository,
        on_query: Callable[[str], None] | None = None,
        detector_configuration: DetectorSection | None = None,
        observation_configuration: ObservationSection | None = None,
    ):
        self.context = context
        self.store = store
        self.repository = repository
        self.on_query = on_query or (lambda description: None)
        self.query_count = 0
        self.detector_configuration = detector_configuration or DetectorSection()
        self.observation_configuration = observation_configuration or ObservationSection()

    def _query(self, description: str) -> None:
        self.query_count += 1
        self.on_query(description)

    # Entry point

    def verify(
        self, findings: InvestigationFindings, review: dict[str, Any] | None = None
    ) -> VerificationResult:
        checks = [self.check_claim(claim) for claim in findings.factual_claims]
        failed_ids = {check.claim_id for check in checks if not check.passed}
        policy = self._policy_checks(findings, failed_ids, review)
        pattern_checks = []
        if ObservationPattern.active_rate_drop in findings.observation_patterns:
            pattern_checks.append(
                check_active_rate_drop(
                    self.context,
                    findings,
                    {check.claim_id for check in checks if check.passed},
                    self.repository,
                    self.detector_configuration,
                    self._query,
                )
            )
        for pattern in (
            ObservationPattern.reported_radio_degradation,
            ObservationPattern.stable_reported_radio,
        ):
            if pattern in findings.observation_patterns:
                pattern_checks.append(
                    check_reported_radio(
                        self.context,
                        findings,
                        pattern,
                        {check.claim_id for check in checks if check.passed},
                        self.repository,
                        self.observation_configuration,
                        self._query,
                    )
                )
        passed = all(check.passed for check in checks + policy + pattern_checks)
        return VerificationResult(
            run_id=self.context.run_id,
            passed=passed,
            claim_checks=checks,
            policy_errors=[check for check in policy if not check.passed],
            pattern_checks=pattern_checks,
            verification_queries=self.query_count,
        )

    def _policy_checks(
        self,
        findings: InvestigationFindings,
        failed_ids: set[str],
        review: dict[str, Any] | None,
    ) -> list[ClaimCheck]:
        results = []
        if findings.assessment_scope == "whole_session":
            results.append(
                _fail(
                    "assessment_scope",
                    VerificationCode.out_of_scope,
                    "case-bound evidence cannot establish a whole-session interpretation",
                    expected="cited_windows",
                    actual=findings.assessment_scope,
                )
            )
        if findings.event_id != self.context.candidate.event_id:
            results.append(
                _fail(
                    "findings",
                    VerificationCode.out_of_scope,
                    "findings refer to a different event",
                    expected=self.context.candidate.event_id,
                    actual=findings.event_id,
                )
            )
        for hypothesis in findings.hypotheses:
            if (
                hypothesis.status == HypothesisStatus.contradicted
                and hypothesis.hypothesis_id in UNOBSERVABLE_EXCLUSIONS
            ):
                results.append(
                    _fail(
                        f"hypothesis:{hypothesis.hypothesis_id.value}",
                        VerificationCode.exclusion_not_observable,
                        "this archive has no measurement that can rule out this hypothesis",
                        actual=hypothesis.status.value,
                    )
                )
            cited_failures = sorted(
                set(hypothesis.supporting_claim_ids + hypothesis.contradicting_claim_ids)
                & failed_ids
            )
            if cited_failures:
                results.append(
                    _fail(
                        f"hypothesis:{hypothesis.hypothesis_id.value}",
                        VerificationCode.evidence_does_not_support,
                        "hypothesis cites claims that failed verification",
                        actual=cited_failures,
                    )
                )
        if review is not None:
            decision = review["review"]["decision"]
            if (
                decision == "accept"
                and review["independent_query_count"] == 0
                and not review["review"].get("review_limitation")
            ):
                results.append(
                    _fail(
                        "review",
                        VerificationCode.review_incomplete,
                        "acceptance without an independent query or a recorded limitation",
                    )
                )
        return results

    # Evidence provenance

    def _resolve(self, claim_id: str, evidence_ids: list[str]) -> list[dict[str, Any]] | ClaimCheck:
        records = []
        for evidence_id in evidence_ids:
            record = self.store.get_evidence(evidence_id)
            if record is None:
                return _fail(
                    claim_id, VerificationCode.unknown_evidence, f"unknown evidence {evidence_id}"
                )
            if record["run_id"] != self.context.run_id:
                return _fail(
                    claim_id,
                    VerificationCode.evidence_wrong_run,
                    f"{evidence_id} belongs to another run",
                )
            if record["snapshot_hash"] != self.context.snapshot_hash:
                return _fail(
                    claim_id,
                    VerificationCode.evidence_wrong_snapshot,
                    f"{evidence_id} comes from a different snapshot",
                    expected=self.context.snapshot_hash,
                    actual=record["snapshot_hash"],
                )
            if record["dictionary_hash"] != self.context.dictionary_hash:
                return _fail(
                    claim_id,
                    VerificationCode.evidence_wrong_dictionary,
                    f"{evidence_id} used a different metric dictionary",
                )
            if evidence_content_hash(record) != record["content_sha256"]:
                return _fail(
                    claim_id,
                    VerificationCode.evidence_content_altered,
                    f"{evidence_id} content does not match its stored hash",
                )
            records.append(record)
        return records

    def _in_scope(self, claim_id: str, *windows: TimeWindow) -> ClaimCheck | None:
        for window in windows:
            if not self.context.allowed_window.contains(window):
                return _fail(
                    claim_id,
                    VerificationCode.out_of_scope,
                    "claim window lies outside the run's allowed window",
                    expected=self.context.allowed_window.model_dump(),
                    actual=window.model_dump(),
                )
        return None

    # Claim dispatch

    def check_claim(self, claim: FactualClaim) -> ClaimCheck:
        resolved = self._resolve(claim.claim_id, claim.evidence_ids)
        if isinstance(resolved, ClaimCheck):
            return resolved
        try:
            if isinstance(claim, MetricStatisticClaim):
                return self._metric_statistic(claim, resolved)
            if isinstance(claim, MetricChangeClaim):
                return self._metric_change(claim, resolved)
            if isinstance(claim, ActivityClaim):
                return self._activity(claim, resolved)
            if isinstance(claim, TransitionClaim):
                return self._transition(claim, resolved)
            return self._availability(claim, resolved)
        except Exception as failure:  # recomputation must never pass silently
            return _fail(
                claim.claim_id,
                VerificationCode.recomputation_failed,
                f"recomputation failed: {type(failure).__name__}",
            )

    # Support predicates: a cited record covers the claim if its arguments include it.

    @staticmethod
    def _covers_summary(
        record: dict[str, Any], metric: str, window: TimeWindow, basis: str
    ) -> bool:
        arguments = record["validated_arguments"]
        if arguments.get("basis") != basis or metric not in arguments.get("metrics", []):
            return False
        window_dump = window.model_dump()
        if record["tool_name"] == "summarize_window":
            return bool(
                {"start_ms": arguments["start_ms"], "end_ms": arguments["end_ms"]} == window_dump
            )
        if record["tool_name"] == "compare_windows":
            reference = {
                "start_ms": arguments["reference_start_ms"],
                "end_ms": arguments["reference_end_ms"],
            }
            event = {"start_ms": arguments["event_start_ms"], "end_ms": arguments["event_end_ms"]}
            return window_dump in (reference, event)
        return False

    @staticmethod
    def _covers_availability(
        record: dict[str, Any], metric: str, window: TimeWindow, basis: str
    ) -> bool:
        # Session context reports native sample availability for the allowed window.
        if record["tool_name"] == "get_session_context":
            result = record["result"]
            return (
                basis == "native"
                and result.get("allowed_window") == window.model_dump()
                and metric in result.get("metric_availability_in_allowed_window", {})
            )
        return Verifier._covers_summary(record, metric, window, basis)

    def _metric_statistic(
        self, claim: MetricStatisticClaim, records: list[dict[str, Any]]
    ) -> ClaimCheck:
        scope = self._in_scope(claim.claim_id, claim.window)
        if scope:
            return scope
        if not any(
            self._covers_summary(record, claim.metric.value, claim.window, claim.basis.value)
            for record in records
        ):
            return _fail(
                claim.claim_id,
                VerificationCode.evidence_does_not_support,
                "no cited evidence summarizes this metric, window and basis",
            )
        self._query(f"summary {claim.metric.value}")
        summary = self.repository.summarize(
            self.context.session_id, claim.window, claim.metric, claim.basis
        )
        if summary.applicability is not None and not summary.applicability.supported:
            return _fail(
                claim.claim_id,
                VerificationCode.metric_not_applicable,
                "populated logger fields do not establish this radio measurement on these rows",
                actual=summary.applicability.as_dict(),
            )
        if claim.unit.value != summary.unit:
            return _fail(
                claim.claim_id,
                VerificationCode.unit_mismatch,
                "unit differs from the metric unit",
                expected=summary.unit,
                actual=claim.unit.value,
            )
        expected_value = summary.statistic(claim.statistic.value)
        if not numbers_match(claim.value, expected_value):
            return _fail(
                claim.claim_id,
                VerificationCode.value_mismatch,
                f"{claim.statistic.value} differs from the recomputed value",
                expected=expected_value,
                actual=claim.value,
            )
        if (claim.valid_count, claim.missing_count) != (summary.valid_count, summary.missing_count):
            return _fail(
                claim.claim_id,
                VerificationCode.count_mismatch,
                "valid or missing count differs",
                expected=[summary.valid_count, summary.missing_count],
                actual=[claim.valid_count, claim.missing_count],
            )
        return ClaimCheck(claim_id=claim.claim_id, passed=True)

    def _metric_change(self, claim: MetricChangeClaim, records: list[dict[str, Any]]) -> ClaimCheck:
        scope = self._in_scope(claim.claim_id, claim.reference_window, claim.event_window)
        if scope:
            return scope
        metric, basis = claim.metric.value, claim.basis.value
        covered_reference = any(
            self._covers_summary(record, metric, claim.reference_window, basis)
            for record in records
        )
        covered_event = any(
            self._covers_summary(record, metric, claim.event_window, basis) for record in records
        )
        if not (covered_reference and covered_event):
            return _fail(
                claim.claim_id,
                VerificationCode.evidence_does_not_support,
                "cited evidence does not cover both windows for this metric and basis",
            )
        if claim.event_window.start_ms < claim.reference_window.end_ms:
            return _fail(
                claim.claim_id,
                VerificationCode.chronology_invalid,
                "the event window must start after the reference window ends",
            )
        self._query(f"change {metric}")
        reference = self.repository.summarize(
            self.context.session_id, claim.reference_window, claim.metric, claim.basis
        )
        event = self.repository.summarize(
            self.context.session_id, claim.event_window, claim.metric, claim.basis
        )
        for name, summary in (("reference", reference), ("event", event)):
            if summary.applicability is not None and not summary.applicability.supported:
                return _fail(
                    claim.claim_id,
                    VerificationCode.metric_not_applicable,
                    f"{name} window has an unsupported or unknown radio measurement source",
                    actual=summary.applicability.as_dict(),
                )
        comparison = compare_summaries(
            claim.metric, reference.as_dict(), event.as_dict(), claim.statistic.value
        )
        if claim.unit.value != reference.unit:
            return _fail(
                claim.claim_id,
                VerificationCode.unit_mismatch,
                "unit differs from the metric unit",
                expected=reference.unit,
                actual=claim.unit.value,
            )
        for name, actual, expected in (
            ("reference_value", claim.reference_value, comparison["reference_value"]),
            ("event_value", claim.event_value, comparison["event_value"]),
            ("delta", claim.delta, comparison["delta_event_minus_reference"]),
        ):
            if not numbers_match(actual, expected):
                return _fail(
                    claim.claim_id,
                    VerificationCode.value_mismatch,
                    f"{name} differs from the recomputed value",
                    expected=expected,
                    actual=actual,
                )
        if claim.relative_decrease is not None:
            if claim.metric not in RELATIVE_CHANGE_METRICS:
                return _fail(
                    claim.claim_id,
                    VerificationCode.relative_change_not_allowed,
                    "relative change is not defined for dB or dBm metrics, use the delta",
                )
            if not numbers_match(claim.relative_decrease, comparison["relative_decrease"]):
                return _fail(
                    claim.claim_id,
                    VerificationCode.value_mismatch,
                    "relative_decrease differs or is not defined for this reference "
                    f"({comparison['relative_decrease_status']})",
                    expected=comparison["relative_decrease"],
                    actual=claim.relative_decrease,
                )
        return ClaimCheck(claim_id=claim.claim_id, passed=True)

    def _activity(self, claim: ActivityClaim, records: list[dict[str, Any]]) -> ClaimCheck:
        scope = self._in_scope(claim.claim_id, claim.window)
        if scope:
            return scope
        window_dump = claim.window.model_dump()
        if not any(
            record["tool_name"] == "get_activity_state"
            and record["validated_arguments"].get("basis") == claim.basis.value
            and {
                "start_ms": record["validated_arguments"]["start_ms"],
                "end_ms": record["validated_arguments"]["end_ms"],
            }
            == window_dump
            for record in records
        ):
            return _fail(
                claim.claim_id,
                VerificationCode.evidence_does_not_support,
                "no cited activity evidence for this window and basis",
            )
        self._query("activity")
        activity = self.repository.activity(self.context.session_id, claim.window, claim.basis)
        expected_fraction = activity["fractions"][claim.state.value]
        if activity["denominator"] != claim.denominator:
            return _fail(
                claim.claim_id,
                VerificationCode.count_mismatch,
                "activity denominator differs",
                expected=activity["denominator"],
                actual=claim.denominator,
            )
        if not numbers_match(claim.fraction, expected_fraction):
            return _fail(
                claim.claim_id,
                VerificationCode.value_mismatch,
                f"{claim.state.value} fraction differs",
                expected=expected_fraction,
                actual=claim.fraction,
            )
        return ClaimCheck(claim_id=claim.claim_id, passed=True)

    def _transition(self, claim: TransitionClaim, records: list[dict[str, Any]]) -> ClaimCheck:
        supporting = [
            record
            for record in records
            if record["tool_name"] == "get_cell_transitions"
            and record["validated_arguments"]["start_ms"] <= claim.later_time_ms
            and claim.later_time_ms < record["validated_arguments"]["end_ms"]
        ]
        if not supporting:
            return _fail(
                claim.claim_id,
                VerificationCode.evidence_does_not_support,
                "no cited transition evidence covers the later observation time",
            )
        arguments = supporting[0]["validated_arguments"]
        window = TimeWindow(start_ms=arguments["start_ms"], end_ms=arguments["end_ms"])
        scope = self._in_scope(claim.claim_id, window)
        if scope:
            return scope
        self._query("transitions")
        recomputed = transitions_in_window(
            self.repository, self.context.session_id, window, self.context.allowed_window
        )
        claimed = {
            "transition_type": claim.transition_type.value,
            "earlier_cell_id": claim.earlier_cell_id,
            "later_cell_id": claim.later_cell_id,
            "earlier_time_ms": claim.earlier_time_ms,
            "later_time_ms": claim.later_time_ms,
            "gap_ms": claim.gap_ms,
            "identity_basis": claim.identity_basis.value,
        }
        for transition in recomputed["transitions"]:
            if all(transition[key] == value for key, value in claimed.items()):
                return ClaimCheck(claim_id=claim.claim_id, passed=True)
        return _fail(
            claim.claim_id,
            VerificationCode.transition_not_found,
            "no recomputed qualified transition matches every claimed field",
            expected=recomputed["transitions"][:5],
            actual=claimed,
        )

    def _availability(self, claim: AvailabilityClaim, records: list[dict[str, Any]]) -> ClaimCheck:
        scope = self._in_scope(claim.claim_id, claim.window)
        if scope:
            return scope
        if not any(
            self._covers_availability(record, claim.metric.value, claim.window, claim.basis.value)
            for record in records
        ):
            return _fail(
                claim.claim_id,
                VerificationCode.evidence_does_not_support,
                "no cited evidence summarizes this metric, window and basis",
            )
        self._query(f"availability {claim.metric.value}")
        summary = self.repository.summarize(
            self.context.session_id,
            claim.window,
            MetricName(claim.metric),
            AggregationBasis(claim.basis),
        )
        if (claim.valid_count, claim.missing_count) != (summary.valid_count, summary.missing_count):
            return _fail(
                claim.claim_id,
                VerificationCode.count_mismatch,
                "valid or missing count differs",
                expected=[summary.valid_count, summary.missing_count],
                actual=[claim.valid_count, claim.missing_count],
            )
        return ClaimCheck(claim_id=claim.claim_id, passed=True)
