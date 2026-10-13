"""Interpretation rules over preserved logger fields, separate from normalization.

The Irish logger does not identify an LTE/NR measurement source on non-LTE/5G
rows. A populated column is not sufficient serving-radio evidence there. The
values remain intact; this rule neither imputes them nor relabels them as RSCP.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from celltrace.contracts.common import MetricName
from celltrace.data.raca import technology_family

APPLICABILITY_VERSION = "raca_reference_signal_v1"
REFERENCE_SIGNAL_METRICS = frozenset({MetricName.reported_rsrp_dbm, MetricName.reported_rsrq_db})


def reference_signal_supported(modes: Iterable[str | None]) -> bool:
    observed = set(modes)
    return bool(observed) and all(technology_family(mode) == "lte_nr" for mode in observed)


@dataclass(frozen=True)
class MetricApplicability:
    sample_count: int
    unsupported_sample_count: int
    observed_modes: tuple[str | None, ...]

    @property
    def supported(self) -> bool:
        return self.sample_count > 0 and self.unsupported_sample_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "supported": self.supported,
        }
