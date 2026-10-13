"""Automated semantic review, the evaluation stage after code verification.

Code already recomputes every numerical claim. This stage asks fixed model judges whether a
report's wording overstates cause, misses a supported alternative, or proposes a measurement
that could separate its explanations. Every verdict must quote the report exactly, and code
rejects quotes, pointers and evidence that do not match. Judges first answer synthetic
controls with known intended verdicts, and a judge that fails them reviews no real report.

The stage is labeled automated in every record. Its verdicts are model opinions under a frozen
rubric, not human judgments and not physical causes.

Layers, each importable on its own:
    rubric    frozen rubric, typed verdict schema and exact quote validation (pure)
    controls  synthetic control reports with construction expectations (pure)
    judge     one paid judge call with durable budget and one correction attempt
    plan      frozen plan, schedule and tamper checks
    runner    batch execution with quality gates, stop file and ledger replay
    summary   qualification, panel verdicts, agreement and spend
    commands  the ``celltrace review`` command group
"""
