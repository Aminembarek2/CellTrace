"""Blind paired review of agent and template reports on the same real incidents.

prepare: renders both reports of each incident in one neutral format without arm, run, model,
cost or review identity, assigns A and B by a private random key, interleaves calibration pairs
whose one side carries a planted overclaim, and writes a rubric and an empty rating sheet.
rate: an automated model rater fills the rating sheet from the packet alone, each pair in both
orders, every provider call charged to a durable ledger.
score: unblinds rating sheets and runs the preregistered analysis described in
docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from celltrace.canonical import content_hash
from celltrace.clock import SystemClock
from celltrace.providers.base import (
    FailureKind,
    ModelAdapter,
    ProviderFailure,
    ProviderRequest,
    ProviderResponse,
)
from celltrace.providers.rate_card import RateCard
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunLimits,
    RunStore,
)

CRITERIA = ("grounded", "cause_open", "discriminating_next", "no_overclaim", "clarity")
CALIBRATION_PAIRS = 6
MINIMUM_CALIBRATION_DETECTED = 5
PLANTED_SUMMARIES = (
    "Congestion in the serving cell is what reduced the download rate in this window.",
    "Given the rate delivered here, the cell clearly had no spare capacity left for this user.",
)
RUBRIC = """# Blind review rubric

You will read pairs of reports, A and B, about the same measured incident. Do not try to guess
how a report was produced. Rate each report on its own, then say which one you would act on.

Scores from 1 (poor) to 5 (excellent):

- grounded: every statement rests on the measurements listed in the report. 5 means nothing is
  asserted beyond them, 1 means important statements have no support.
- cause_open: the report keeps the physical cause open when the measurements cannot decide it.
  5 means alternatives are kept and named, 1 means a cause is asserted.
- discriminating_next: the next measurements would actually separate the open explanations.
  5 means each named measurement tells two explanations apart, 1 means none does.
- no_overclaim: no causal, capacity or whole session claim beyond the evidence. 5 means none,
  1 means a clear overclaim.
- clarity: an engineer could act on it in two minutes. 5 means yes, 1 means no.

preference: A, B or tie, for the report you would act on.

Write a short comment when a score is 1 or 2. Rate pairs in the order given. Take a break after
twelve pairs. About four minutes per pair.
"""


def _number(value: Any) -> str:
    """Same rounding for both sources, so float noise cannot reveal where a report came from."""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _claim_text(claim: dict[str, Any], rendering: int = 1) -> str:
    window = claim.get("window") or claim.get("event_window") or {}
    span = f"{window.get('start_ms', '?')} to {window.get('end_ms', '?')} ms" if window else ""
    kind = claim.get("claim_type")
    if kind == "metric_change":
        before, after = _number(claim.get("reference_value")), _number(claim.get("event_value"))
        return (
            f"{claim.get('metric')} {claim.get('statistic')} changed from {before} to {after} "
            f"{claim.get('unit', '')} ({span})"
        )
    if kind == "metric_statistic":
        value = f"{_number(claim.get('value'))} {claim.get('unit', '')}"
        return f"{claim.get('metric')} {claim.get('statistic')} = {value} ({span})"
    if kind == "activity":
        if rendering >= 2:
            percent = f"{100 * float(claim.get('fraction') or 0):.0f}%"
            return (
                f"{claim.get('state')} in {percent} of {claim.get('denominator')} samples ({span})"
            )
        share, count = _number(claim.get("fraction")), claim.get("denominator")
        return f"{claim.get('state')} share {share} of {count} samples ({span})"
    details = {
        key: value
        for key, value in claim.items()
        if key not in ("claim_id", "evidence_ids", "basis", "window")
        and value not in (None, [], {})
    }
    described = ", ".join(f"{key} {_number(value)}" for key, value in sorted(details.items()))
    return f"{described} ({span})" if span else described


def _hypothesis_line(
    hypothesis: dict[str, Any], claims: dict[str, dict[str, Any]], rendering: int
) -> str:
    line = f"- {hypothesis['hypothesis_id']}: {hypothesis['status']}."
    if rendering >= 2:
        for field, heading in (
            ("supporting_claim_ids", "Supported by"),
            ("contradicting_claim_ids", "Contradicted by"),
        ):
            cited = [claims[key] for key in hypothesis.get(field) or [] if key in claims]
            if cited:
                described = " | ".join(_claim_text(claim, rendering) for claim in cited)
                line += f" {heading}: {described}."
        if hypothesis.get("assumptions"):
            line += " Assumes: " + " ".join(hypothesis["assumptions"])
    discriminating = hypothesis.get("discriminating_measurement", "not stated")
    return f"{line} Would be decided by: {discriminating}"


def render(report: dict[str, Any], summary_override: str | None = None, rendering: int = 1) -> str:
    """One neutral format for both sources, without identity, cost or review fields.
    Rendering 2 prints activity as a percentage and shows the measurements and assumptions
    each explanation rests on. Rendering 1 reproduces the first packet."""
    findings = report.get("final_findings") or {}
    summary = summary_override or findings.get("decision_summary") or "No summary delivered."
    lines = ["### Summary", "", summary, "", "### Observation labels checked in code", ""]
    patterns = report.get("verified_observation_patterns") or []
    lines += [f"- {pattern}" for pattern in patterns] or ["- none"]
    lines += ["", "### Verified measurements", ""]
    claims = report.get("verified_claims", [])
    lines += [f"- {_claim_text(claim, rendering)}" for claim in claims[:20]]
    if len(report.get("verified_claims", [])) > 20:
        lines.append(f"- and {len(report['verified_claims']) - 20} more")
    lines += ["", "### Explanations considered", ""]
    by_identifier = {claim.get("claim_id"): claim for claim in claims}
    for hypothesis in report.get("qualified_hypotheses") or []:
        lines.append(_hypothesis_line(hypothesis, by_identifier, rendering))
    if not report.get("qualified_hypotheses"):
        lines.append("- none")
    lines += ["", "### Missing evidence", ""]
    lines += [f"- {item}" for item in report.get("missing_evidence") or []] or ["- none"]
    lines += ["", "### Next measurements", ""]
    lines += [f"- {item}" for item in report.get("next_measurements") or []] or ["- none"]
    return "\n".join(lines) + "\n"


def _side(key: str, pair_id: str) -> str:
    """Which side the agent report takes, from the private key."""
    return "A" if int(hashlib.sha256(f"{key}:{pair_id}".encode()).hexdigest(), 16) % 2 else "B"


def prepare(agent_dir: Path, template_dir: Path, out: Path, seed: int, rendering: int = 1) -> None:
    out.mkdir(parents=True, exist_ok=False)
    agent = {
        json.loads(p.read_text())["event_id"]: json.loads(p.read_text())
        for p in sorted(agent_dir.glob("*.json"))
    }
    template = {
        json.loads(p.read_text())["event_id"]: json.loads(p.read_text())
        for p in sorted(template_dir.glob("*.json"))
    }
    if set(agent) != set(template):
        raise ValueError("agent and template reports cover different incidents")
    generator = random.Random(seed)
    key = secrets.token_hex(16)
    events = sorted(agent)
    calibration_events = generator.sample(
        [event for event in events if agent[event].get("final_findings")], CALIBRATION_PAIRS
    )
    pairs: list[dict[str, Any]] = []
    for event in events:
        pairs.append(
            {
                "kind": "real",
                "event_id": event,
                "first": ("agent", agent[event], None),
                "second": ("template", template[event], None),
            }
        )
    for index, event in enumerate(calibration_events):
        planted = PLANTED_SUMMARIES[index % len(PLANTED_SUMMARIES)]
        pairs.append(
            {
                "kind": "calibration",
                "event_id": event,
                "first": ("planted", copy.deepcopy(agent[event]), planted),
                "second": ("original", agent[event], None),
            }
        )
    generator.shuffle(pairs)
    mapping = []
    sheet_rows = []
    packet = [
        "# Blind review packet",
        "",
        "Rate every pair with `rubric.md`. Record scores in `ratings_template.csv`, "
        "one copy per rater.",
        "",
    ]
    for number, pair in enumerate(pairs, start=1):
        pair_id = f"P{number:02d}"
        first_side = _side(key, pair_id)
        sides = {first_side: pair["first"], ("B" if first_side == "A" else "A"): pair["second"]}
        packet += [f"## Pair {pair_id}", ""]
        for label in ("A", "B"):
            _, report, override = sides[label]
            packet += [f"## Report {label}", "", render(report, override, rendering)]
        mapping.append(
            {
                "pair_id": pair_id,
                "kind": pair["kind"],
                "event_id": pair["event_id"],
                "A": sides["A"][0],
                "B": sides["B"][0],
            }
        )
        row: dict[str, str] = {"pair_id": pair_id}
        for criterion in CRITERIA:
            row[f"A_{criterion}"] = ""
            row[f"B_{criterion}"] = ""
        row.update(preference="", comment="")
        sheet_rows.append(row)
    (out / "packet.md").write_text("\n".join(packet))
    (out / "rubric.md").write_text(RUBRIC)
    with (out / "ratings_template.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet_rows[0]))
        writer.writeheader()
        writer.writerows(sheet_rows)
    private = out / "private_unblinding"
    private.mkdir()
    (private / "mapping.json").write_text(json.dumps({"key": key, "pairs": mapping}, indent=1))
    digest = hashlib.sha256((private / "mapping.json").read_bytes()).hexdigest()
    (out / "mapping_sha256.txt").write_text(digest + "\n")
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "real": len(events),
                "calibration": CALIBRATION_PAIRS,
                "mapping_sha256": digest,
            },
            indent=1,
        )
    )


def sign_test(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    tail = (
        sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
        / 2**trials
    )
    return min(1.0, 2 * tail)


def clopper_pearson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None

    def tail_at_least(count: int, share: float) -> float:
        return sum(
            math.comb(trials, k) * share**k * (1 - share) ** (trials - k)
            for k in range(count, trials + 1)
        )

    def solve(function: Any, target: float) -> float:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2
            low, high = (middle, high) if function(middle) < target else (low, middle)
        return (low + high) / 2

    lower = 0.0 if successes == 0 else solve(lambda share: tail_at_least(successes, share), 0.025)
    upper = (
        1.0
        if successes == trials
        else solve(lambda share: tail_at_least(successes + 1, share), 0.975)
    )
    return [lower, upper]


def krippendorff_alpha(units: list[list[float | str]], metric: str) -> float | None:
    """Alpha for two or more raters per unit, nominal or interval distance."""

    def distance(first: float | str, second: float | str) -> float:
        if metric == "nominal":
            return 0.0 if first == second else 1.0
        return (float(first) - float(second)) ** 2

    pairable = [values for values in units if len(values) >= 2]
    total = sum(len(values) for values in pairable)
    if total < 2:
        return None
    observed = (
        sum(
            sum(
                distance(a, b) for i, a in enumerate(values) for j, b in enumerate(values) if i != j
            )
            / (len(values) - 1)
            for values in pairable
        )
        / total
    )
    pooled = [value for values in pairable for value in values]
    expected = sum(
        distance(a, b) for i, a in enumerate(pooled) for j, b in enumerate(pooled) if i != j
    ) / (total * (total - 1))
    return None if expected == 0 else 1 - observed / expected


def score(packet: Path, ratings: list[Path], out: Path, labels: list[str] | None = None) -> None:
    mapping = json.loads((packet / "private_unblinding" / "mapping.json").read_text())
    expected = (packet / "mapping_sha256.txt").read_text().strip()
    if (
        hashlib.sha256((packet / "private_unblinding" / "mapping.json").read_bytes()).hexdigest()
        != expected
    ):
        raise ValueError("unblinding key changed after the packet was written")
    pairs = {entry["pair_id"]: entry for entry in mapping["pairs"]}
    raters = [list(csv.DictReader(path.open())) for path in ratings]
    result: dict[str, Any] = {"raters": len(raters), "per_rater": []}
    preference_units: dict[str, list[str]] = {}
    for rater_index, rows in enumerate(raters):
        agent_wins = template_wins = ties = 0
        criterion_differences: dict[str, list[float]] = {name: [] for name in CRITERIA}
        calibration_hits = calibration_total = 0
        for row in rows:
            entry = pairs[row["pair_id"]]
            preferred = row["preference"].strip().upper()
            if entry["kind"] == "real":
                agent_label = "A" if entry["A"] == "agent" else "B"
                other = "B" if agent_label == "A" else "A"
                if preferred == "TIE":
                    ties += 1
                elif preferred == agent_label:
                    agent_wins += 1
                elif preferred == other:
                    template_wins += 1
                preference_units.setdefault(row["pair_id"], []).append(
                    "agent"
                    if preferred == agent_label
                    else ("template" if preferred == other else "tie")
                )
                for name in CRITERIA:
                    a, b = row[f"{agent_label}_{name}"], row[f"{other}_{name}"]
                    if a and b:
                        criterion_differences[name].append(float(a) - float(b))
            else:
                planted = "A" if entry["A"] == "planted" else "B"
                original = "B" if planted == "A" else "A"
                calibration_total += 1
                lower = (
                    row[f"{planted}_no_overclaim"]
                    and row[f"{original}_no_overclaim"]
                    and float(row[f"{planted}_no_overclaim"])
                    < float(row[f"{original}_no_overclaim"])
                )
                calibration_hits += bool(lower) and preferred != planted
        decided = agent_wins + template_wins
        result["per_rater"].append(
            {
                "rater": labels[rater_index] if labels else rater_index + 1,
                "real_pairs": agent_wins + template_wins + ties,
                "agent_preferred": agent_wins,
                "template_preferred": template_wins,
                "ties": ties,
                "agent_share_of_decided": agent_wins / decided if decided else None,
                "agent_share_interval_95": clopper_pearson(agent_wins, decided),
                "sign_test_p_two_sided": sign_test(agent_wins, decided),
                "criterion_mean_difference_agent_minus_template": {
                    name: (sum(values) / len(values) if values else None)
                    for name, values in criterion_differences.items()
                },
                "criterion_sign_test_p": {
                    name: sign_test(sum(v > 0 for v in values), sum(v != 0 for v in values))
                    for name, values in criterion_differences.items()
                },
                "calibration_detected": f"{calibration_hits} of {calibration_total}",
                "qualified": calibration_hits >= MINIMUM_CALIBRATION_DETECTED,
            }
        )
    if len(raters) >= 2:
        units = list(preference_units.values())
        agreement = sum(len(set(values)) == 1 for values in units if len(values) >= 2) / max(
            1, sum(len(values) >= 2 for values in units)
        )
        result["agreement"] = {
            "preference_percent": agreement,
            "preference_alpha_nominal": krippendorff_alpha(units, "nominal"),
        }
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


RATER_ROLE = "blind_rater"
RATER_TOOL = "submit_rating"
ORDERS = ("presented", "swapped")
PREFERENCES = ("A", "B", "tie")
RATING_FIELDS = ("report_A", "report_B", "preference", "comment")
ASSESSMENT_METHOD = "automated model rater, not a human rating"
STOPPING_STATUSES = frozenset({"budget_prevented", "provider_uncertain", "provider_rejected"})
MAXIMUM_CONSECUTIVE_UNRATED = 3
RATER_INSTRUCTIONS = (
    RUBRIC
    + """
You are one of two independent raters and see one pair at a time. Read both reports, then call
the submit_rating tool exactly once with integer scores from 1 to 5 for each report on every
criterion, a preference of A, B or tie, and a short comment. Rate only what the reports say.
"""
)


@dataclass(frozen=True)
class RaterSettings:
    model_id: str
    maximum_usd: float
    maximum_output_tokens: int = 16384
    maximum_attempts: int = 2
    timeout_seconds: int = 120
    maximum_usd_per_presentation: float = 0.30
    strict_schema: bool = True

    def run_limits(self) -> RunLimits:
        return RunLimits(
            0,
            self.maximum_attempts,
            100000,
            self.maximum_attempts * self.maximum_output_tokens,
            self.maximum_usd_per_presentation,
            self.maximum_attempts * self.timeout_seconds * 1000,
            0,
            1,
        )


def parse_packet(packet: Path) -> dict[str, dict[str, str]]:
    """Both report texts of every pair, exactly as a human rater reads them."""
    pairs: dict[str, dict[str, str]] = {}
    for section in (packet / "packet.md").read_text().split("\n## Pair ")[1:]:
        pair_id, _, body = section.partition("\n")
        first, separator, second = body.partition("\n## Report B\n")
        if not separator or "## Report A\n" not in first:
            raise ValueError(f"pair {pair_id} is malformed")
        pairs[pair_id.strip()] = {
            "A": first.split("## Report A\n", 1)[1].strip(),
            "B": second.strip(),
        }
    return pairs


def pair_message(pair_id: str, reports: dict[str, str], order: str) -> str:
    first, second = (reports["A"], reports["B"])
    if order == "swapped":
        first, second = second, first
    return f"## Pair {pair_id}\n\n## Report A\n\n{first}\n\n## Report B\n\n{second}\n"


def rating_schema() -> dict[str, Any]:
    scores = {
        "type": "object",
        "additionalProperties": False,
        "required": list(CRITERIA),
        "properties": {name: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for name in CRITERIA},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(RATING_FIELDS),
        "properties": {
            "report_A": scores,
            "report_B": copy.deepcopy(scores),
            "preference": {"type": "string", "enum": list(PREFERENCES)},
            "comment": {"type": "string"},
        },
    }


def validate_rating(arguments: dict[str, Any]) -> dict[str, Any]:
    if set(arguments) != set(RATING_FIELDS):
        raise ValueError(f"rating fields must be exactly {list(RATING_FIELDS)}")
    for side in ("report_A", "report_B"):
        scores = arguments[side]
        if not isinstance(scores, dict) or set(scores) != set(CRITERIA):
            raise ValueError(f"{side} must score exactly {list(CRITERIA)}")
        for name, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{side}.{name} must be an integer from 1 to 5")
    if arguments["preference"] not in PREFERENCES:
        raise ValueError("preference must be A, B or tie")
    if not isinstance(arguments["comment"], str):
        raise ValueError("comment must be a string")
    return arguments


def in_packet_frame(rating: dict[str, Any], order: str) -> dict[str, Any]:
    """Map a rating back to the packet's A and B labels."""
    if order == "presented":
        return {
            "A": rating["report_A"],
            "B": rating["report_B"],
            "preference": rating["preference"],
        }
    flipped = {"A": "B", "B": "A", "tie": "tie"}
    return {
        "A": rating["report_B"],
        "B": rating["report_A"],
        "preference": flipped[rating["preference"]],
    }


def sheet_row(pair_id: str, framed: list[dict[str, Any]], comments: list[str]) -> dict[str, str]:
    """Average scores over both orders. A preference counts only when both orders agree."""
    row = {"pair_id": pair_id}
    for name in CRITERIA:
        for label in ("A", "B"):
            values = [rating[label][name] for rating in framed]
            row[f"{label}_{name}"] = _number(sum(values) / len(values)) if values else ""
    preferences = {rating["preference"] for rating in framed}
    if len(framed) < len(ORDERS):
        row["preference"] = ""
    else:
        row["preference"] = preferences.pop() if len(preferences) == 1 else "tie"
    row["comment"] = " | ".join(comment.replace("\n", " ") for comment in comments)[:600]
    return row


def _rating_request(message: str, settings: RaterSettings) -> ProviderRequest:
    return ProviderRequest(
        role=RATER_ROLE,
        system_prompt=RATER_INSTRUCTIONS,
        messages=[{"role": "user", "content": message}],
        tools=[
            {
                "name": RATER_TOOL,
                "description": "Submit the scores of both reports and the preference.",
                "input_schema": rating_schema(),
                "strict": settings.strict_schema,
            }
        ],
        maximum_output_tokens=settings.maximum_output_tokens,
        timeout_seconds=settings.timeout_seconds,
        effort="high",
    )


def _validated(response: ProviderResponse) -> dict[str, Any]:
    if response.stop_reason in ("length", "refusal") or len(response.tool_calls) != 1:
        raise ValueError(
            f"expected one complete rating call, stop reason {response.stop_reason}, "
            f"{len(response.tool_calls)} tool calls"
        )
    call = response.tool_calls[0]
    if call.name != RATER_TOOL:
        raise ValueError("unexpected tool")
    return validate_rating(call.arguments)


def _ask_for_correction(
    request: ProviderRequest, response: ProviderResponse, error: Exception
) -> ProviderRequest:
    """Answer a malformed tool call with its error. Without a tool call, repeat the request."""
    if not response.tool_calls:
        return request
    feedback = [
        {
            "type": "tool_result",
            "tool_use_id": call.tool_use_id,
            "content": json.dumps({"error": "INVALID_RATING", "message": str(error)[:300]}),
            "is_error": True,
        }
        for call in response.tool_calls
    ]
    messages = [
        *request.messages,
        {"role": "assistant", "content": response.assistant_content},
        {"role": "user", "content": feedback},
    ]
    return ProviderRequest(
        role=request.role,
        system_prompt=request.system_prompt,
        messages=messages,
        tools=request.tools,
        maximum_output_tokens=request.maximum_output_tokens,
        timeout_seconds=request.timeout_seconds,
        effort=request.effort,
    )


def rate_presentation(
    store: RunStore,
    adapter: ModelAdapter,
    rates: RateCard,
    settings: RaterSettings,
    experiment_id: str,
    presentation_id: str,
    message: str,
) -> dict[str, Any]:
    """Rate one presentation. An attempt with unknown outcome is never dispatched again."""
    run_id = f"{experiment_id}:{presentation_id}"
    result: dict[str, Any] = {
        "presentation_id": presentation_id,
        "run_id": run_id,
        "model_id": settings.model_id,
        "assessment_method": ASSESSMENT_METHOD,
        "input_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "rating": None,
        "status": "unrated",
    }
    if store.connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone():
        result["status"] = "unresolved_prior_attempt"
        return result
    store.create_run(
        run_id,
        presentation_id,
        presentation_id,
        "blind_rating",
        adapter.execution_mode.value,
        settings.model_id,
        result["input_sha256"],
        {"input_sha256": result["input_sha256"], "instructions": content_hash(RATER_INSTRUCTIONS)},
        {"assessment_method": ASSESSMENT_METHOD},
        settings.run_limits(),
        experiment_id=experiment_id,
    )
    request = _rating_request(message, settings)
    for attempt_number in range(1, settings.maximum_attempts + 1):
        estimate = adapter.estimate_input_tokens(request)
        reservation = Reservation(
            input_tokens=estimate.input_tokens,
            output_tokens=request.maximum_output_tokens,
            usd=rates.reservation_cost(estimate.input_tokens, request.maximum_output_tokens),
            estimation_method=estimate.method,
        )
        try:
            action_id = store.reserve_action(
                run_id,
                f"{run_id}:{attempt_number}",
                f"{run_id}:{RATER_ROLE}",
                RATER_ROLE,
                ActionKind.provider_attempt,
                {"input_sha256": result["input_sha256"], "attempt": attempt_number},
                reservation,
                attempt_number=attempt_number,
            )
        except BudgetExhausted as exhausted:
            result.update(status="budget_prevented", reason=exhausted.limit_name)
            break
        store.mark_started(action_id)
        started = store.clock.monotonic_ms()
        try:
            response = adapter.complete(request)
        except ProviderFailure as failure:
            duration = store.clock.monotonic_ms() - started
            store.add_active_ms(run_id, duration)
            if failure.kind == FailureKind.uncertain:
                store.mark_uncertain(action_id, str(failure), duration)
                result.update(status="provider_uncertain", reason=str(failure))
                break
            store.complete_action(
                action_id,
                ActionState.failed,
                duration,
                error_code="PROVIDER_FAILURE",
                error=failure.to_record(),
                usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            )
            permanent = failure.kind == FailureKind.permanent
            result.update(
                status="provider_rejected" if permanent else "provider_failed", reason=str(failure)
            )
            if permanent:
                break
            continue
        duration = store.clock.monotonic_ms() - started
        store.complete_action(
            action_id,
            ActionState.completed,
            duration,
            response=response.to_record(),
            usage={**response.usage.__dict__, "usd": rates.cost(response.usage)},
            provider_request_id=response.request_id,
        )
        store.add_active_ms(run_id, duration)
        if not response.usage.usage_known or response.effective_settings.get("settings_errors"):
            result.update(status="provider_uncertain", reason="usage or routing not established")
            break
        try:
            rating = _validated(response)
        except (ValueError, KeyError, TypeError) as error:
            result.update(status="invalid_rating", reason=f"{type(error).__name__}: {error}"[:300])
            request = _ask_for_correction(request, response, error)
            continue
        result.update(status="rated", rating=rating, attempts=attempt_number)
        result.pop("reason", None)
        break
    result["usage"] = store.usage(run_id).__dict__
    store.mark_terminal(run_id, "completed" if result["rating"] else "failed", result["status"])
    return result


def rate(
    packet: Path,
    out: Path,
    adapter: ModelAdapter,
    rates: RateCard,
    settings: RaterSettings,
    freeze_sha256: str,
) -> dict[str, Any]:
    """Fill one rating sheet from the packet alone. Running again resumes without new charges
    for presentations already recorded."""
    pairs = parse_packet(packet)
    presentations = out / "presentations"
    presentations.mkdir(parents=True, exist_ok=True)
    experiment_id = f"blind_rating:{settings.model_id}:{freeze_sha256[:16]}"
    store = RunStore(out / "ledger.sqlite3", SystemClock())
    unrated_in_a_row = 0
    stopped_by = None
    try:
        store.create_experiment(
            experiment_id,
            settings.maximum_usd,
            {"freeze_sha256": freeze_sha256, "model_id": settings.model_id},
        )
        for pair_id, reports in pairs.items():
            for order in ORDERS:
                presentation_id = f"{pair_id}_{order}"
                destination = presentations / f"{presentation_id}.json"
                if destination.exists():
                    continue
                result = rate_presentation(
                    store,
                    adapter,
                    rates,
                    settings,
                    experiment_id,
                    presentation_id,
                    pair_message(pair_id, reports, order),
                )
                destination.write_text(json.dumps(result, indent=1) + "\n")
                print(json.dumps({"presentation": presentation_id, "status": result["status"]}))
                unrated_in_a_row = 0 if result["rating"] else unrated_in_a_row + 1
                if result["status"] in STOPPING_STATUSES:
                    stopped_by = result["status"]
                elif unrated_in_a_row >= MAXIMUM_CONSECUTIVE_UNRATED:
                    stopped_by = "consecutive_unrated"
                if stopped_by:
                    break
            if stopped_by:
                break
        used_usd, ceiling_usd = store.experiment_usd(experiment_id)
    finally:
        store.close()
    rows, consistent, complete = [], 0, 0
    for pair_id in pairs:
        framed, comments = [], []
        for order in ORDERS:
            path = presentations / f"{pair_id}_{order}.json"
            saved = json.loads(path.read_text()) if path.exists() else {"rating": None}
            if saved["rating"]:
                framed.append(in_packet_frame(saved["rating"], order))
                comments.append(f"{order}: {saved['rating']['comment']}")
        if len(framed) == len(ORDERS):
            complete += 1
            consistent += framed[0]["preference"] == framed[1]["preference"]
        rows.append(sheet_row(pair_id, framed, comments))
    with (packet / "ratings_template.csv").open() as handle:
        fieldnames = next(csv.reader(handle))
    with (out / "ratings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "assessment_method": ASSESSMENT_METHOD,
        "model_id": settings.model_id,
        "execution_mode": adapter.execution_mode.value,
        "freeze_sha256": freeze_sha256,
        "pairs": len(pairs),
        "pairs_rated_in_both_orders": complete,
        "order_consistent_preferences": consistent,
        "stopped_by": stopped_by,
        "known_or_reserved_usd": used_usd,
        "ceiling_usd": ceiling_usd,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _live_rater(config_path: Path) -> tuple[ModelAdapter, RateCard, RaterSettings]:
    """A rater from a fixed profile whose endpoint and prices were verified within a day."""
    from celltrace.config import load_config
    from celltrace.providers.openrouter_adapter import OpenRouterAdapter
    from celltrace.providers.readiness import verified_rate_card
    from celltrace.secrets import load_openrouter_key

    runtime = load_config(config_path).runtime
    rates = verified_rate_card(runtime)
    load_openrouter_key(Path(".env"), os.environ)
    settings = RaterSettings(
        model_id=runtime.model_id,
        maximum_usd=runtime.maximum_estimated_usd_per_experiment,
        maximum_output_tokens=runtime.maximum_output_tokens_per_attempt,
        timeout_seconds=int(runtime.provider_timeout_seconds),
        strict_schema=runtime.strict_tool_schemas,
    )
    return OpenRouterAdapter(runtime.model_id), rates, settings


class TieEverythingPolicy:
    """Offline negative control: rates every report 3 and prefers neither."""

    def next_turn(self, request: ProviderRequest) -> Any:
        from celltrace.providers.scripted import ScriptedTurn

        scores = {name: 3 for name in CRITERIA}
        arguments = {
            "report_A": scores,
            "report_B": dict(scores),
            "preference": "tie",
            "comment": "scripted fixture, no model inference",
        }
        return ScriptedTurn("", RATER_TOOL, arguments)


def source_means(packet: Path, sheet: Path) -> dict[str, dict[str, float]]:
    """Mean score per source and criterion over the real pairs, after unblinding."""
    mapping = json.loads((packet / "private_unblinding" / "mapping.json").read_text())
    pairs = {entry["pair_id"]: entry for entry in mapping["pairs"]}
    values: dict[str, dict[str, list[float]]] = {
        source: {name: [] for name in CRITERIA} for source in ("agent", "template")
    }
    for row in csv.DictReader(sheet.open()):
        entry = pairs[row["pair_id"]]
        if entry["kind"] != "real":
            continue
        for label in ("A", "B"):
            for name in CRITERIA:
                if row[f"{label}_{name}"]:
                    values[entry[label]][name].append(float(row[f"{label}_{name}"]))
    return {
        source: {name: sum(scores) / len(scores) for name, scores in criteria.items() if scores}
        for source, criteria in values.items()
    }


def figure(packet: Path, sheets: list[Path], labels: list[str], title: str, out: Path) -> None:
    """Dumbbell plot of mean agent and template scores per criterion, one panel per rater."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    colors = {"agent": "#2a78d6", "template": "#eb6834"}
    names = {
        "grounded": "Grounded in measurements",
        "cause_open": "Keeps cause open",
        "discriminating_next": "Next steps discriminate",
        "no_overclaim": "No overclaim",
        "clarity": "Clarity",
    }
    figure_handle, axes = plt.subplots(
        1, len(sheets), figsize=(12, 4.6), sharex=True, sharey=True, facecolor=surface
    )
    for axis, sheet, label in zip(axes, sheets, labels, strict=True):
        means = source_means(packet, sheet)
        axis.set_facecolor(surface)
        for row, name in enumerate(reversed(CRITERIA)):
            agent, template = means["agent"][name], means["template"][name]
            axis.plot([template, agent], [row, row], color=grid, linewidth=3, zorder=1)
            for source, value in (("template", template), ("agent", agent)):
                axis.scatter(
                    value, row, s=90, color=colors[source], edgecolor=surface, linewidth=2, zorder=2
                )
        axis.set_yticks(range(len(CRITERIA)), [names[name] for name in reversed(CRITERIA)])
        axis.set_xlim(1, 5.2)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(axis="x", color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=muted, length=0, labelsize=10.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(label, loc="left", fontsize=11.5, color=ink, pad=26)
        axis.set_xlabel("Mean score over 18 real incidents, 1 poor to 5 excellent", color=muted)
    first = source_means(packet, sheets[0])
    widest = max(CRITERIA, key=lambda name: abs(first["agent"][name] - first["template"][name]))
    row = len(CRITERIA) - 1 - CRITERIA.index(widest)
    for source, text in (("agent", "Agent report"), ("template", "Scripted template")):
        axes[0].annotate(
            text,
            (first[source][widest], row),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=muted,
        )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=colors[source])
        for source in ("agent", "template")
    ]
    figure_handle.legend(
        handles,
        ["Agent report", "Scripted template"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=muted,
    )
    figure_handle.suptitle(title, x=0.012, ha="left", fontsize=15, color=ink, y=0.99)
    figure_handle.text(
        0.012,
        0.915,
        "Blind pairs, each rated in both A and B orders by a model rater. Not a human rating.",
        fontsize=10.5,
        color=muted,
    )
    figure_handle.tight_layout(rect=(0, 0.07, 1, 0.92))
    figure_handle.savefig(out, dpi=120, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--agent-reports", type=Path, required=True)
    prepare_parser.add_argument("--template-reports", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20261012)
    prepare_parser.add_argument("--rendering", type=int, choices=(1, 2), default=1)
    rate_parser = actions.add_parser("rate")
    rate_parser.add_argument("--packet", type=Path, required=True)
    rate_parser.add_argument("--out", type=Path, required=True)
    rate_parser.add_argument("--freeze", type=Path, required=True)
    rate_parser.add_argument("--config", type=Path, help="verified rater configuration")
    rate_parser.add_argument("--live", action="store_true", help="paid provider calls")
    score_parser = actions.add_parser("score")
    score_parser.add_argument("--packet", type=Path, required=True)
    score_parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    score_parser.add_argument("--labels", nargs="+")
    score_parser.add_argument("--out", type=Path, required=True)
    figure_parser = actions.add_parser("figure")
    figure_parser.add_argument("--packet", type=Path, required=True)
    figure_parser.add_argument("--ratings", type=Path, nargs="+", required=True)
    figure_parser.add_argument("--labels", nargs="+", required=True)
    figure_parser.add_argument("--title", required=True)
    figure_parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare(
            arguments.agent_reports,
            arguments.template_reports,
            arguments.out,
            arguments.seed,
            arguments.rendering,
        )
    elif arguments.action == "rate":
        freeze_sha256 = json.loads(arguments.freeze.read_text())["sha256"]
        if arguments.live:
            if arguments.config is None:
                parser.error("--live needs --config")
            adapter, rates, settings = _live_rater(arguments.config)
        else:
            from celltrace.providers.scripted import ScriptedModelAdapter

            adapter = ScriptedModelAdapter({RATER_ROLE: TieEverythingPolicy()})
            rates = RateCard(
                model_id="scripted-fixture",
                currency="USD",
                unit="per_million_tokens",
                input=0,
                output=0,
                cache_write_5m=0,
                cache_write_1h=0,
                cache_read=0,
                tool_use_system_prompt_tokens_auto=0,
                source_url="none",
                checked_date="none",
                cache_treatment="no provider",
                extra_charges="none",
            )
            settings = RaterSettings(model_id="scripted-fixture", maximum_usd=0.01)
        print(
            json.dumps(
                rate(arguments.packet, arguments.out, adapter, rates, settings, freeze_sha256),
                indent=1,
            )
        )
    elif arguments.action == "figure":
        figure(
            arguments.packet, arguments.ratings, arguments.labels, arguments.title, arguments.out
        )
    else:
        score(arguments.packet, arguments.ratings, arguments.out, arguments.labels)


if __name__ == "__main__":
    main()
