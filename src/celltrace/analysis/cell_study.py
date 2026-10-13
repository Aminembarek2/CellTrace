"""Cell study: do cells keep their rate across days, and which are reliably slow?

Runs the Q7 leave one day out test and the empirical Bayes step on one snapshot and one
set of mobility cohorts, then writes a summary, a report and two figures. The same code
serves the Raca archive and each operator of the 2018 LTE dataset. Method:
docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from celltrace.analysis import cells, shrinkage
from celltrace.analysis.population import (
    AXIS,
    GRID,
    INK,
    SERIES_PRIMARY,
    SURFACE,
    _fmt,
    _percent,
    _save,
    _style,
    day_pattern,
)
from celltrace.data.snapshot import open_snapshot
from celltrace.release import source_tree_hash

METHOD_VERSION = "cell_study_v1"
SUMMARY_NAME = "cell_study_summary.json"
REPORT_NAME = "cell_study_report.md"
FIGURES = {"funnel": "cell_funnel.png", "forecast": "cell_held_out_forecast.png"}
TABLE_ROWS = 10


def run_cell_study(
    snapshot_dir: Path,
    out_dir: Path,
    label: str,
    cohort: str,
    mobilities: Sequence[str],
    seed: int,
    permutations: int = cells.PERMUTATIONS,
    accept_proxy_identity: bool = False,
) -> dict[str, Any]:
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        replication = cells.cell_replication(
            connection, cohort, seed, permutations, mobilities, accept_proxy_identity
        )
        visits = cells.load_visits(
            connection, cohort, cells.MINIMUM_VISIT_SAMPLES, mobilities, accept_proxy_identity
        )
    finally:
        connection.close()
    summary = {
        "label": label,
        "method_version": METHOD_VERSION,
        "source_sha256": source_tree_hash(),
        "snapshot_id": manifest["snapshot_id"],
        "cohort": cohort,
        "mobilities": list(mobilities),
        "cell_identity": "proxy: technology family, site coordinates and local number"
        if accept_proxy_identity
        else "reported serving cell with a known identity basis",
        "seed": seed,
        "replication": replication,
        "shrinkage": shrinkage.analyse(visits, seed) if _has_repeat_cells(visits) else None,
        "figures": FIGURES,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary["shrinkage"] is not None:
        figure_funnel(summary["shrinkage"], out_dir / FIGURES["funnel"])
        figure_forecast(visits, summary["shrinkage"], out_dir / FIGURES["forecast"])
    (out_dir / REPORT_NAME).write_text(render(summary), encoding="utf-8")
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def _has_repeat_cells(visits: Sequence[cells.Visit]) -> bool:
    cell_set = shrinkage.cell_data(visits, shrinkage.technology_means(visits))
    return len(cell_set) >= 3 and any(item.visits >= 2 for item in cell_set)


def _factor(log_effect: float) -> str:
    return f"{10**log_effect:.2f}"


def figure_funnel(result: dict[str, Any], path: Path) -> None:
    """Raw cell effect against visits, with limits for noise alone and for real spread."""
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    rows = result["table"]
    visits = np.array([row["visits"] for row in rows])
    effects = np.array([row["raw_effect"] for row in rows])
    span = np.linspace(1, max(int(visits.max()), 2) + 1, 200)
    noise = np.sqrt(result["within_variance"] / span)
    spread = np.sqrt(result["within_variance"] / span + result["between_variance"])
    for multiplier, style in ((1.96, "-"), (3.09, ":")):
        axis.plot(span, -multiplier * noise, color=INK["muted"], linestyle=style, linewidth=1)
        axis.plot(span, multiplier * noise, color=INK["muted"], linestyle=style, linewidth=1)
    axis.plot(span, -1.96 * spread, color=INK["secondary"], linestyle="--", linewidth=1)
    axis.plot(span, 1.96 * spread, color=INK["secondary"], linestyle="--", linewidth=1)
    axis.axhline(0, color=AXIS, linewidth=1)
    axis.axhline(shrinkage.SLOW_THRESHOLD, color=INK["muted"], linewidth=0.8, linestyle="-.")
    selected = np.array([row["selected"] for row in rows])
    axis.scatter(
        visits[~selected],
        effects[~selected],
        s=28,
        color=INK["muted"],
        edgecolors=SURFACE,
        zorder=3,
    )
    axis.scatter(
        visits[selected],
        effects[selected],
        s=44,
        color=SERIES_PRIMARY,
        edgecolors=SURFACE,
        zorder=4,
    )
    for position, row in enumerate(rows, start=1):
        if row["selected"]:
            axis.annotate(
                str(position),
                (row["visits"], row["raw_effect"]),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=8,
                color=INK["primary"],
            )
    ticks = [-1.0, -0.5, -0.301, 0.0, 0.301, 0.5]
    axis.set_yticks(ticks, [f"×{_factor(tick)}" for tick in ticks])
    axis.set_xlabel("Visits to the cell", fontsize=9, color=INK["secondary"])
    axis.set_ylabel("Rate relative to the cell's technology", fontsize=9, color=INK["secondary"])
    count = int(selected.sum())
    title = (
        f"{count} of {len(rows)} cells are reliably at least 2× slower than their technology"
        if count != 1
        else f"1 of {len(rows)} cells is reliably at least 2× slower than its technology"
    )
    subtitle = (
        "Noise limits: 95% solid, 99.8% dotted. With real cell spread: 95% dashed. "
        "Blue: selected, numbered as in the table"
    )
    _style(axis, title, subtitle)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    _save(figure, path)


def figure_forecast(visits: Sequence[cells.Visit], result: dict[str, Any], path: Path) -> None:
    """Held out deviations against raw and shrunken predictions, with fitted slopes."""
    forecast = result["held_out_forecast"]
    pairs = [item[1:] for item in shrinkage.held_out_pairs(visits)]
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    if pairs:
        actual = np.array([item[0] for item in pairs])
        raw = np.array([item[1] for item in pairs])
        shrunk = np.array([item[2] for item in pairs])
        axis.scatter(raw, actual, s=18, color=INK["muted"], label="Raw cell mean", zorder=3)
        axis.scatter(
            shrunk, actual, s=18, color=SERIES_PRIMARY, label="Shrunken estimate", zorder=4
        )
        span = np.linspace(min(raw.min(), shrunk.min()), max(raw.max(), shrunk.max()), 50)
        axis.plot(span, span, color=INK["secondary"], linewidth=1, label="Slope one")
        for values, color, name in (
            (raw, INK["muted"], "slope_raw"),
            (shrunk, SERIES_PRIMARY, "slope_shrunken"),
        ):
            slope = forecast[name]
            if slope is not None:
                intercept = float(actual.mean() - slope * values.mean())
                axis.plot(span, intercept + slope * span, color=color, linewidth=1.5)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.set_xlabel(
        "Predicted from other days, log10 rate relative to technology",
        fontsize=9,
        color=INK["secondary"],
    )
    axis.set_ylabel("Held out day, same scale", fontsize=9, color=INK["secondary"])
    shrunken_slope = _fmt(forecast.get("slope_shrunken"), ".2f")
    raw_slope = _fmt(forecast.get("slope_raw"), ".2f")
    title = (
        f"Shrunken effects predict held out days with slope {shrunken_slope}, raw means {raw_slope}"
    )
    subtitle = "A slope below one means the raw ranking exaggerates. Leave one day out"
    _style(axis, title, subtitle)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    _save(figure, path)


def _interval(values: list[float] | None, digits: str = ".2f") -> str:
    return "n/a" if not values else f"{values[0]:{digits}} to {values[1]:{digits}}"


def render(summary: dict[str, Any]) -> str:
    replication = summary["replication"]
    primary = replication["primary"]
    result = summary["shrinkage"]
    by_session = replication["sensitivity"]["leave_one_session_out"]
    larger = replication["sensitivity"]["minimum_visit_samples_20"]
    lines = [
        f"# Cell study: {summary['label']}",
        "",
        f"Snapshot `{summary['snapshot_id']}`, {summary['cohort']} sessions, mobility "
        f"{', '.join(summary['mobilities'])}. Cell identity: {summary['cell_identity']}. "
        f"Method `{summary['method_version']}`, method "
        "in `docs/TECHNICAL_REPORT.md`. Rates are delivered application rates, not "
        "capacity, and no cause is assessed.",
        "",
        "## Does a cell keep its rate across days?",
        "",
        f"{replication['visits']} visits on {replication['cells']} cells from "
        f"{replication['sessions']} sessions over {replication['days']} days. "
        f"{primary['scored_visits']} visits on {primary['scored_cells']} cells meet the same "
        "cell on another day. Leave one day out errors on log10 rate: same cell "
        f"{_fmt(primary['mae_cell'], '.3f')}, same technology "
        f"{_fmt(primary['mae_technology'], '.3f')}. Skill against technology "
        f"{_percent(primary['skill_against_technology'])}, permutation null mean "
        f"{_percent(primary['null_mean'])}, p = {_fmt(primary['p_value'], '.3f')}. Decision by "
        f"the frozen rule: {replication['decision']}.",
        "",
        "Skill by held out day: "
        + ", ".join(
            f"{item['fold']} {_percent(item['skill'])} ({item['visits']} visits)"
            for item in replication["per_day"]
        )
        + f". {day_pattern(replication, explain=False)}",
        "",
        "Sensitivity: leaving one session out instead of one day gives "
        f"{_percent(by_session['skill_against_technology'])} "
        f"(p = {_fmt(by_session['p_value'], '.3f')}), and requiring 20 samples per visit gives "
        f"{_percent(larger['skill_against_technology'])} (p = {_fmt(larger['p_value'], '.3f')}).",
        "",
    ]
    if result is None:
        lines += ["Too few cells were seen on two or more days for the shrinkage step.", ""]
        return "\n".join(lines)
    forecast = result["held_out_forecast"]
    split = result["split_sample"]
    synthetic = result["synthetic_recovery"]
    selected = [row for row in result["table"] if row["selected"]]
    lines += [
        "## Which cells are reliably slow for their technology?",
        "",
        f"![Funnel plot]({FIGURES['funnel']})",
        "",
        f"{result['cells']} cells seen on two or more days. A cell effect is the mean of its "
        "visits' log10 rate minus their technology mean. Within cell variance "
        f"{result['within_variance']:.4f}, between cell variance {result['between_variance']:.4f}. "
        f"Mean reliability {result['mean_reliability']:.2f}, and "
        f"{_percent(result['share_reliability_below_0_7'])} of cells fall below 0.7. "
        f"{len(selected)} cells are selected as at least 2× slower than their technology at a 10% "
        f"false discovery limit (expected false selections "
        f"{result['expected_false_among_selected']:.2f}). The nonparametric prior selects "
        f"{result['npmle_selected']}.",
        "",
        "| Cell | Technology | Days | Visits | Raw rate factor | Reliability | Shrunken factor "
        "| Probability at least 2× slower | Rank, 90% interval | Selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *(
            f"| {row['cell']} | {row['technology']} | {row['days']} | {row['visits']} | "
            f"×{_factor(row['raw_effect'])} | {row['reliability']:.2f} | "
            f"×{_factor(row['shrunken_effect'])} | {row['tail_probability']:.2f} | "
            f"{row['rank_interval_90'][0]:.0f} to {row['rank_interval_90'][1]:.0f} | "
            f"{'yes' if row['selected'] else 'no'} |"
            for row in result["table"][:TABLE_ROWS]
        ),
        "",
        "Factors are relative to the technology mean, so ×0.50 is half its typical rate. Ranks "
        "count from the slowest. A wide rank interval means the order is not known.",
        "",
        "## Verification",
        "",
        f"![Held out forecast]({FIGURES['forecast']})",
        "",
        "| Check | Result | What it shows |",
        "| --- | --- | --- |",
        f"| Held out slope, raw cell means | {_fmt(forecast.get('slope_raw'), '.2f')} "
        f"({_interval(forecast.get('slope_raw_interval_95'))}) | Below one means a raw ranking "
        "exaggerates differences |",
        f"| Held out slope, shrunken estimates | {_fmt(forecast.get('slope_shrunken'), '.2f')} "
        f"({_interval(forecast.get('slope_shrunken_interval_95'))}) | Near one means calibrated |",
        f"| Held out error, raw, shrunken, technology only | "
        f"{_fmt(forecast.get('mae_raw'), '.3f')}, {_fmt(forecast.get('mae_shrunken'), '.3f')}, "
        f"{_fmt(forecast.get('mae_technology_only'), '.3f')} | Lower is better |",
        f"| Split sample reliability | {_fmt(split.get('reliability'), '.2f')} "
        f"({_interval(split.get('interval_90'))}) against model mean "
        f"{result['mean_reliability']:.2f} | Two independent estimates agree when close |",
        f"| Synthetic recovery, {synthetic['replications']} archives | false discovery "
        f"{_percent(synthetic['mean_false_discovery_proportion'])}, power "
        f"{_percent(synthetic['mean_power'])}, slope "
        f"{_fmt(synthetic['mean_held_out_slope'], '.2f')} | The rule keeps its 10% limit when "
        "the model is right |",
        "",
        "Intervals: 95% cluster bootstrap over cells for slopes, 90% of random splits for "
        "reliability. The held out slopes use deviations from technology means estimated on the "
        "other days only.",
        "",
        "## Limits",
        "",
        "What is measured is a cell's area as driven on these routes. Coverage, load, backhaul "
        "and configuration are not separated. Time of day is not controlled. Selection controls "
        "the expected share of false selections, so a small selected set is a strength, not a "
        "gap.",
        "",
    ]
    return "\n".join(lines)
