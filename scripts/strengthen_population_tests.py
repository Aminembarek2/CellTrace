"""Low rate persistence against IAAFT surrogates and the order of drops and reported cell
changes on one snapshot. Writes JSON, Markdown and one figure.

Method: docs/TECHNICAL_REPORT.md. Refuses to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from celltrace.analysis import persistence, population
from celltrace.analysis import transition_timing as timing
from celltrace.config import DetectorSection
from celltrace.data.snapshot import open_snapshot
from celltrace.release import source_tree_hash

CONTROL_ARCHIVES = 20
NEGATIVE_LIMIT = 3
POSITIVE_MINIMUM = 16
PERSISTENT = (
    "low rate stretches are longer than the rate's own autocorrelation and distribution predict"
)


CONTROL_FAMILIES = (
    ("negative_coefficient_0_88", 0.88, 0),
    ("negative_coefficient_0_95", 0.95, 0),
    ("positive", 0.88, 2),
)


METHOD_LABELS = {"gaussianized_iaaft": "rank Gaussianized IAAFT", "iaaft": "IAAFT"}
CONTROL_LABELS = {
    "negative_coefficient_0_88": "negative controls with lag one coefficient 0.88:",
    "negative_coefficient_0_95": "negative controls with coefficient 0.95:",
    "positive": "positive controls:",
}


def run_controls(draws: int, seed: int, gaussianize: bool) -> dict[str, Any]:
    """Synthetic archives with the archive's gap rate: rejections of the long share at 0.05."""
    generator = np.random.default_rng([seed, 1])
    rejections = {name: 0 for name, _, _ in CONTROL_FAMILIES}
    p_values: dict[str, list[float | None]] = {name: [] for name, _, _ in CONTROL_FAMILIES}
    for index in range(CONTROL_ARCHIVES):
        for name, coefficient, outages in CONTROL_FAMILIES:
            archive = persistence.synthetic_archive(
                generator, coefficient=coefficient, planted_outages=outages
            )
            result = persistence.surrogate_test(archive, draws, seed + index, gaussianize)
            p_value = result["p_value"]["long_share"]
            p_values[name].append(p_value)
            rejections[name] += p_value is not None and p_value <= 0.05
    passed = all(
        count <= NEGATIVE_LIMIT if name.startswith("negative") else count >= POSITIVE_MINIMUM
        for name, count in rejections.items()
    )
    return {
        "archives_per_family": CONTROL_ARCHIVES,
        "draws": draws,
        "seed": seed,
        "method": "gaussianized_iaaft" if gaussianize else "iaaft",
        "rejections_at_0_05": rejections,
        "required": {"negative_at_most": NEGATIVE_LIMIT, "positive_at_least": POSITIVE_MINIMUM},
        "passed": passed,
        "p_values": p_values,
    }


def persistence_decision(result: dict[str, Any]) -> str:
    adjusted = result["holm_p_value"]
    rejected = [adjusted.get(name, 1.0) < 0.05 for name in ("episodes", "long_share")]
    if all(rejected):
        return PERSISTENT
    if not any(rejected):
        return "the stretches are what ordinary autocorrelation produces"
    return "mixed"


def order_decision(result: dict[str, Any]) -> str:
    share, p_value = result["share_change_after_onset"], result["p_value_two_sided"]
    if share is None or p_value is None or p_value >= 0.05:
        return "order is not resolved at one second resolution"
    if share > 0.5:
        return "drops tend to begin before the reported change"
    return "the reported change tends to come first"


def figure(profile: dict[str, Any], path: Path) -> None:
    offsets = list(range(-timing.PROFILE_SECONDS, timing.PROFILE_SECONDS + 1))
    values = [profile["median_ratio_by_offset"][str(offset)] for offset in offsets]
    drawing, axis = plt.subplots(figsize=(9.6, 4.8))
    axis.axhline(1.0, color=population.INK["muted"], linewidth=0.8, linestyle=(0, (3, 3)))
    axis.axvline(0, color=population.INK["secondary"], linewidth=1)
    axis.plot(
        offsets, values, color=population.SERIES_PRIMARY, linewidth=2, marker="o", markersize=4
    )
    finite = [(offset, value) for offset, value in zip(offsets, values, strict=True) if value]
    if not finite:
        title = "Rate around a reported cell change, relative to 20 to 10 s before"
    else:
        trough_offset, trough = min(finite, key=lambda pair: pair[1])
        recovered = next(
            (offset for offset, value in finite if offset > trough_offset and value >= 0.9), None
        )
        recovery = "" if recovered is None else f", above 90% {recovered - trough_offset} s later"
        title = (
            f"Median rate falls to {100 * trough:.0f}% of baseline at {trough_offset:+d} s "
            f"from a logged cell change{recovery}"
        )
    subtitle = (
        f"{profile['changes']} isolated changes in {profile['sessions']} driving sessions. "
        "Dashed: baseline. Vertical line: the second the new identity is first logged"
    )
    population._style(axis, title, subtitle)
    ink = population.INK["secondary"]
    axis.set_xlabel("Seconds from the reported change", fontsize=9, color=ink)
    axis.set_ylabel("Median rate relative to baseline", fontsize=9, color=ink)
    axis.grid(axis="y", color=population.GRID, linewidth=0.8)
    population._save(drawing, path)
    plt.close(drawing)


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _s1_lines(
    controls: dict[str, Any], s1: dict[str, Any] | None, decision: str | None
) -> list[str]:
    counts, required = controls["rejections_at_0_05"], controls["required"]
    verdict = "passed" if controls["passed"] else "failed, so no archive statistic was computed"
    families = ", ".join(
        f"{CONTROL_LABELS.get(name, name)} {count}" for name, count in counts.items()
    )
    lines = [
        f"## Low rate persistence against {METHOD_LABELS[controls['method']]} surrogates",
        "",
        f"Controls first, seed {controls['seed']}, {controls['archives_per_family']} synthetic "
        f"archives per family with one empty bin in seven and {controls['draws']} surrogates "
        f"each. Rejections of the long share at 0.05: {families}. Required: at most "
        f"{required['negative_at_most']} in each negative family and at least "
        f"{required['positive_at_least']} in the positive family. Controls {verdict}.",
        "",
    ]
    if s1 is None:
        return lines
    lines += [
        "| Statistic | Observed | IAAFT null mean | 95% null range | p | Holm p |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, label in (
        ("episodes", "Episodes of 10 s or more under 1.5 Mbps"),
        ("long_share", "Share of low seconds inside such episodes"),
    ):
        digits = ".0f" if name == "episodes" else ".3f"
        low, high = s1["null_interval_95"][name]
        observed = _fmt(s1["observed"][name], digits)
        mean = _fmt(s1["null_mean"][name], digits)
        null_range = f"{_fmt(low, digits)} to {_fmt(high, digits)}"
        raw = _fmt(s1["p_value"][name], ".3f")
        adjusted = _fmt(s1["holm_p_value"].get(name), ".3f")
        lines.append(f"| {label} | {observed} | {mean} | {null_range} | {raw} | {adjusted} |")
    return [*lines, "", f"Decision by the frozen rule: {decision}.", ""]


def _s2_lines(s2: dict[str, Any], profile: dict[str, Any], decision: str) -> list[str]:
    low, high = s2["null_interval_95"] or [None, None]
    lag = None if s2["median_lag_ms"] is None else s2["median_lag_ms"] / 1000
    lines = [
        "## Order of drop onset and reported cell change, driving sessions",
        "",
        f"{s2['candidates_with_nearby_change']} candidates have a reported change within 10 s "
        f"of onset. The change comes after onset in {_percent(s2['share_change_after_onset'])} "
        f"of them (ties count half), median lag {_fmt(lag, '.1f')} s. With change times shifted "
        f"within eligible runs ({s2['draws']} shifts) the share averages "
        f"{_percent(s2['null_mean'])} (95% of shifts {_percent(low)} to {_percent(high)}), two "
        f"sided p = {_fmt(s2['p_value_two_sided'], '.3f')}.",
        "",
        f"Decision by the frozen rule: {decision}. A logged identity can trail the radio "
        "handover, so this is the order of logged events at one second resolution.",
        "",
        "![Rate around a reported cell change](cell_change_profile.png)",
        "",
        "| Contrast, after over before | Mean log10 ratio | Factor | 95% interval | Sessions |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for name, label in (
        ("one_second", "1 s after against 1 s before"),
        ("five_seconds", "5 s after against 5 s before"),
    ):
        entry = profile["log10_post_over_pre"][name]
        mean, interval = entry["mean"], entry["interval_95"]
        factor = _fmt(None if mean is None else 10**mean, ".2f")
        span = "n/a" if interval is None else f"{10 ** interval[0]:.2f} to {10 ** interval[1]:.2f}"
        lines.append(f"| {label} | {_fmt(mean, '.3f')} | {factor} | {span} | {entry['sessions']} |")
    return [
        *lines,
        "",
        f"{profile['changes']} changes with no other change within 20 s, inside one segment, "
        "with at least 80% active samples and a baseline of at least 0.5 Mbps from 20 to 10 s "
        "before. Intervals delete one session at a time (jackknife, Student t).",
        "",
    ]


def report(summary: dict[str, Any]) -> str:
    lines = [
        "# Persistence and order checks",
        "",
        f"Snapshot `{summary['snapshot_id']}`, source `{summary['source_sha256'][:16]}…`. Rules "
        "fixed in an internal plan before these results, summarized in "
        "`docs/TECHNICAL_REPORT.md`, appendix. "
        "Exploratory: this archive was analysed before.",
        "",
        *_s1_lines(summary["s1_controls"], summary["s1_persistence"], summary["s1_decision"]),
        *_s2_lines(summary["s2_order"], summary["s2_profile"], summary["s2_decision"]),
        "Limits: one operator in one city, a transition proxy rather than a handover record, "
        "and route and technology confounding unadjusted.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=199)
    parser.add_argument("--control-draws", type=int, default=99)
    parser.add_argument("--shifts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--control-seed", type=int, default=20261013)
    parser.add_argument(
        "--raw-iaaft", action="store_true", help="plain IAAFT, without rank Gaussian scores"
    )
    arguments = parser.parse_args()
    gaussianize = not arguments.raw_iaaft
    arguments.out.mkdir(parents=True, exist_ok=False)
    detector = DetectorSection()
    connection, manifest = open_snapshot(arguments.snapshot)
    try:
        controls = run_controls(arguments.control_draws, arguments.control_seed, gaussianize)
        s1 = None
        if controls["passed"]:
            sessions = persistence.load_archive(connection, detector)
            s1 = persistence.surrogate_test(sessions, arguments.draws, arguments.seed, gaussianize)
        timing_sessions = timing.load_sessions(connection, detector)
        s2 = timing.order_test(timing_sessions, arguments.shifts, arguments.seed)
        profiles = [item for session in timing_sessions for item in timing.change_profiles(session)]
        profile = timing.profile_summary(profiles)
    finally:
        connection.close()
    summary = {
        "snapshot_id": manifest["snapshot_id"],
        "source_sha256": source_tree_hash(),
        "plan": "internal plan, summarized in docs/TECHNICAL_REPORT.md, appendix",
        "s1_controls": controls,
        "s1_persistence": s1,
        "s1_decision": None if s1 is None else persistence_decision(s1),
        "s2_order": s2,
        "s2_decision": order_decision(s2),
        "s2_profile": profile,
    }
    (arguments.out / "persistence_and_order.json").write_text(json.dumps(summary, indent=1))
    figure(profile, arguments.out / "cell_change_profile.png")
    (arguments.out / "persistence_and_order.md").write_text(report(summary))
    print(json.dumps({key: summary[key] for key in ("s1_decision", "s2_decision")}, indent=1))


if __name__ == "__main__":
    main()
