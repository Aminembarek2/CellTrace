"""Versioned metric dictionary for the Raca 5G production archive.

Every entry states what is documented and what is only inferred. The audit
on 2026-10-08 could not retrieve the 2020 dataset paper, so unit and state
semantics carry ``inferred_unverified`` status until a primary source is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from celltrace.canonical import content_hash
from celltrace.contracts.common import METRIC_UNITS, MetricName

DICTIONARY_VERSION = "raca-5g-dictionary-1"
SOURCE_ID = "raca_irish_5g"

ANDROID_CELLINFO_SOURCE = (
    "AOSP frameworks/base telephony CellInfo.java (UNAVAILABLE = Integer.MAX_VALUE), "
    "https://github.com/aosp-mirror/platform_frameworks_base, read 2026-10-08"
)
ANDROID_LTE_RANGE_SOURCE = (
    "AOSP CellSignalStrengthLte.java javadoc ranges (rsrp [-140,-43] dBm, rsrq [-34,3] dB, "
    "rssnr [-20,30] dB, rssi [-113,-51] dBm, cqi [0,15]), current main branch read 2026-10-08. "
    "The 2019 to 2020 handset release may differ."
)
DASH_SENTINEL_SOURCE = (
    "Observed in every archive file. The authors' 4G dataset page "
    "(https://www.ucc.ie/en/misl/research/datasets/ivid_4g_lte_dataset/) documents '-' for "
    "missing values. Not confirmed for the 5G archive by a primary source."
)

ANDROID_UNAVAILABLE = 2147483647


@dataclass(frozen=True)
class MetricDefinition:
    metric: str
    raw_field: str
    raw_unit: str
    canonical_unit: str
    conversion: str
    divisor: float
    technology_applicability: str
    missing_sentinels: tuple[str, ...]
    suspected_fill_values: tuple[float, ...]
    documented_range: tuple[float, float] | None
    definition_source: str
    audit_status: str
    limitations: str


def _definition(
    metric: MetricName,
    raw_field: str,
    raw_unit: str,
    conversion: str,
    audit_status: str,
    limitations: str,
    definition_source: str,
    divisor: float = 1.0,
    documented_range: tuple[float, float] | None = None,
    suspected_fill_values: tuple[float, ...] = (),
    technology_applicability: str = "all reported network modes",
) -> MetricDefinition:
    return MetricDefinition(
        metric=metric.value,
        raw_field=raw_field,
        raw_unit=raw_unit,
        canonical_unit=METRIC_UNITS[metric].value,
        conversion=conversion,
        divisor=divisor,
        technology_applicability=technology_applicability,
        missing_sentinels=("-", str(ANDROID_UNAVAILABLE)),
        suspected_fill_values=suspected_fill_values,
        documented_range=documented_range,
        definition_source=definition_source,
        audit_status=audit_status,
        limitations=limitations,
    )


_RATE_LIMITATION = (
    "Application level delivered traffic logged by G-NetTrack Pro, not available capacity. "
    "Unit kbps is inferred from magnitude (maximum 532905 would be 533 Mbps as kbps and "
    "0.53 Mbps as bit/s) and was not confirmed by the dataset paper. Relative drops do not "
    "depend on this unit, absolute thresholds do."
)

METRIC_DICTIONARY: dict[MetricName, MetricDefinition] = {
    MetricName.downlink_mbps: _definition(
        MetricName.downlink_mbps,
        "DL_bitrate",
        "kbps (inferred)",
        "Mbps = kbps / 1000 (decimal)",
        "inferred_unverified",
        _RATE_LIMITATION,
        "Raca et al. MMSys 2020 dataset, field name only. Paper text not retrieved.",
        divisor=1000.0,
    ),
    MetricName.uplink_mbps: _definition(
        MetricName.uplink_mbps,
        "UL_bitrate",
        "kbps (inferred)",
        "Mbps = kbps / 1000 (decimal)",
        "inferred_unverified",
        _RATE_LIMITATION,
        "Raca et al. MMSys 2020 dataset, field name only. Paper text not retrieved.",
        divisor=1000.0,
    ),
    MetricName.reported_rsrp_dbm: _definition(
        MetricName.reported_rsrp_dbm,
        "RSRP",
        "dBm",
        "identity",
        "unit_conventional_semantics_unverified",
        "Reported reference signal received power of the logged serving cell. For 5G mode rows "
        "the measuring technology (LTE anchor or NR) is not documented. -200 is treated as a "
        "suspected fill value (73 rows, 60 dB below the documented floor).",
        ANDROID_LTE_RANGE_SOURCE,
        documented_range=(-140.0, -43.0),
        suspected_fill_values=(-200.0,),
    ),
    MetricName.reported_rsrq_db: _definition(
        MetricName.reported_rsrq_db,
        "RSRQ",
        "dB",
        "identity",
        "unit_conventional_semantics_unverified",
        "Reported reference signal received quality. Never derived from RSRP.",
        ANDROID_LTE_RANGE_SOURCE,
        documented_range=(-34.0, 3.0),
    ),
    MetricName.reported_snr_db: _definition(
        MetricName.reported_snr_db,
        "SNR",
        "dB",
        "identity",
        "unit_conventional_semantics_unverified",
        "Reported SNR field. Not renamed to SINR or SS-SINR because no source defines it.",
        ANDROID_LTE_RANGE_SOURCE,
        documented_range=(-20.0, 30.0),
    ),
    MetricName.reported_cqi: _definition(
        MetricName.reported_cqi,
        "CQI",
        "index",
        "identity",
        "unit_conventional_semantics_unverified",
        "Channel quality indicator index 0 to 15. Not a rate in Mbps. Table index unknown.",
        ANDROID_LTE_RANGE_SOURCE,
        documented_range=(0.0, 15.0),
    ),
    MetricName.reported_rssi_dbm: _definition(
        MetricName.reported_rssi_dbm,
        "RSSI",
        "dBm",
        "identity",
        "unit_conventional_semantics_unverified",
        "Reported received signal strength indicator. 2101 values above -51 dBm are kept and "
        "flagged because the logger may not apply the current Android clamp.",
        ANDROID_LTE_RANGE_SOURCE,
        documented_range=(-113.0, -51.0),
    ),
    MetricName.reported_nrx_rsrp_dbm: _definition(
        MetricName.reported_nrx_rsrp_dbm,
        "NRxRSRP",
        "dBm (inferred)",
        "identity",
        "semantics_unverified",
        "Meaning not documented. Present on GPRS, UMTS and LTE rows, so it is not exclusively a "
        "5G NR measurement. It may describe a neighbouring cell. -205 is a suspected fill value.",
        "Field name only. No primary definition found.",
        documented_range=(-140.0, -43.0),
        suspected_fill_values=(-205.0,),
    ),
    MetricName.reported_nrx_rsrq_db: _definition(
        MetricName.reported_nrx_rsrq_db,
        "NRxRSRQ",
        "dB (inferred)",
        "identity",
        "semantics_unverified",
        "Meaning not documented, see reported_nrx_rsrp_dbm. -204 and -225 are suspected fill "
        "values.",
        "Field name only. No primary definition found.",
        documented_range=(-34.0, 3.0),
        suspected_fill_values=(-204.0, -225.0),
    ),
    MetricName.speed_kph: _definition(
        MetricName.speed_kph,
        "Speed",
        "km/h (inferred)",
        "identity",
        "inferred_unverified",
        "Device speed, unit not documented for this archive. Static folder sessions report "
        "nonzero speed in about 21 percent of rows (maximum 91), so mobility labels and speed "
        "can disagree.",
        "Field name only. No primary definition found.",
    ),
    MetricName.ping_average_ms: _definition(
        MetricName.ping_average_ms,
        "PINGAVG",
        "ms (inferred)",
        "identity",
        "inferred_unverified",
        "Average ping latency. Present in under 2 percent of rows, so absence is the norm.",
        "Field name only. No primary definition found.",
    ),
}

STATE_MAPPING: dict[str, str] = {"D": "active", "I": "idle"}
STATE_MAPPING_NOTE = (
    "D maps to active and I to idle. In file download sessions I rows always have zero "
    "downlink (268 of 268). V and VD are undocumented and map to unknown with the raw code "
    "preserved. VD rows carry nonzero traffic, V rows carry none."
)

APPLICATION_FOLDERS: dict[str, str] = {
    "Download": "file_download",
    "Netflix": "video_netflix",
    "Amazon_Prime": "video_amazon_prime",
}
MOBILITY_FOLDERS: dict[str, str] = {"Static": "static", "Driving": "driving"}
COHORT_MAPPING_NOTE = (
    "Cohorts come from repository folder names. The repository README describes two "
    "mobility patterns (static and car) and two application patterns (video streaming and "
    "file download), which matches the folders."
)

TIMESTAMP_FORMAT = "%Y.%m.%d_%H.%M.%S"
TIMESTAMP_BASIS = "local_wall_clock_timezone_unknown_second_resolution"


def dictionary_payload() -> dict[str, object]:
    return {
        "dictionary_version": DICTIONARY_VERSION,
        "source_id": SOURCE_ID,
        "metrics": {metric.value: asdict(entry) for metric, entry in METRIC_DICTIONARY.items()},
        "state_mapping": STATE_MAPPING,
        "state_mapping_note": STATE_MAPPING_NOTE,
        "missing_sentinel_source": DASH_SENTINEL_SOURCE,
        "unavailable_sentinel_source": ANDROID_CELLINFO_SOURCE,
    }


def dictionary_hash() -> str:
    return content_hash(dictionary_payload())
