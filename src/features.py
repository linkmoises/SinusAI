"""Vector clínico cerrado desde `Signal12` (RF-03/RF-04, carril LC).

Convierte una `Signal12` (serie temporal 1D, nunca píxeles — RF-03) en el
`FeatureVector` de `docs/contracts.md` §2.4: la lista cerrada de 8 medidas clínicas
interpretables (§2.4.1) — duración QRS, QT/QTc, frecuencia cardiaca, eje QRS y HRV
en dominio de tiempo. El `status` es `completo` si salen las 8, `parcial` si sale al
menos una, e `imposible` si no sale ninguna (§2.4).

La extracción recorre la derivación de ritmo (lead II nominal) con `neurokit2`
(filtrado, detección de picos R y delineación del complejo QRS y de la onda T) y
calcula el eje eléctrico desde las derivaciones I y III del triángulo de Einthoven.
Todas las medidas provienen de `Signal12.data` (mV, 1D) — ninguna rama toca una
imagen.
"""

from __future__ import annotations

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np

from src.ingest_signal import Signal12

# contracts §2.4.1 — lista cerrada de features (orden canónico)
FEATURE_KEYS = [
    "qrs_duration_ms",
    "qt_ms",
    "qtc_ms",
    "hr_bpm",
    "qrs_axis_deg",
    "hrv_sdnn_ms",
    "hrv_rmssd_ms",
    "hrv_pnn50_pct",
]

LEAD_RHYTHM = "II"  # derivación de referencia para ritmo/QRS/QT (nominal)
LEADS_AXIS = ("I", "III")  # triángulo de Einthoven para el eje QRS
QTC_METHOD = "bazett"  # QTc = QT / sqrt(RR), con RR medio


@dataclass
class FeatureVector:
    """contracts §2.4 — vector de medidas clínicas cerradas de una `Signal12`."""

    record_id: str
    values: dict[str, float]
    missing_features: list[str]
    status: str


def extract_features(signal12: Signal12) -> FeatureVector:
    """Extrae el vector cerrado de 8 features (`§2.4.1`) desde `signal12`.

    `status == completo` si las 8 feature keys están en `values`; `parcial` si al
    menos una está (pero no todas); `imposible` si ninguna pudo extraerse. La
    extracción opera solo sobre `Signal12.data` (1D), nunca sobre píxeles (RF-03).
    """
    try:
        measures = _measure(signal12)
    except Exception:
        return FeatureVector(
            record_id=signal12.record_id,
            values={},
            missing_features=list(FEATURE_KEYS),
            status="imposible",
        )

    values: dict[str, float] = {}
    missing: list[str] = []
    for key in FEATURE_KEYS:
        try:
            value = _EXTRACTORS[key](measures)
            if value is None or not np.isfinite(value):
                raise ValueError(f"valor no finito para {key!r}")
            values[key] = float(value)
        except Exception:
            missing.append(key)

    status = "completo" if not missing else ("parcial" if values else "imposible")
    return FeatureVector(
        record_id=signal12.record_id,
        values=values,
        missing_features=sorted(missing),
        status=status,
    )


def _measure(signal12: Signal12) -> dict:
    """Procesa la derivación de ritmo y devuelve las primitivas por beat.

    Lanza excepción si la derivación de ritmo no es utilizable o la delineación no
    produce ventanas QRS/QT válidas (p. ej. señal plana -> `imposible` aguas abajo).
    """
    fs = float(signal12.fs)
    col = _lead_column(signal12, LEAD_RHYTHM)
    if col is None:
        raise ValueError(f"derivación de ritmo {LEAD_RHYTHM} no utilizable")

    _, info = nk.ecg_process(signal12.data[:, col], sampling_rate=fs)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if peaks.size < 2:
        raise ValueError("menos de 2 picos R para estimar ritmo")

    rr_ms = np.diff(peaks) / fs * 1000.0

    r_on = np.asarray(info.get("ECG_R_Onsets", []), dtype=float)
    r_off = np.asarray(info.get("ECG_R_Offsets", []), dtype=float)
    t_off = np.asarray(info.get("ECG_T_Offsets", []), dtype=float)
    delin = (r_on > 0) & (r_off > 0) & (t_off > 0)
    if not delin.any():
        raise ValueError("delineación QRS/QT sin ventanas válidas")

    qrs_ms = float(np.median((r_off[delin] - r_on[delin]) / fs * 1000.0))
    qt_ms = float(np.median((t_off[delin] - r_on[delin]) / fs * 1000.0))

    windows = list(
        zip(r_on[delin].astype(int), r_off[delin].astype(int), strict=False)
    )
    net = {}
    for lead in LEADS_AXIS:
        lead_col = _lead_column(signal12, lead)
        if lead_col is None:
            net[lead] = np.nan
        else:
            per_beat = [
                np.nanmean(signal12.data[a:b, lead_col]) for a, b in windows
            ]
            net[lead] = float(np.nanmean(per_beat))

    return {
        "rr_ms": rr_ms,
        "qrs_duration_ms": qrs_ms,
        "qt_ms": qt_ms,
        "net_I": net["I"],
        "net_III": net["III"],
    }


def _lead_column(signal12: Signal12, lead: str) -> int | None:
    """Columna canónica de `lead` si está usable, `None` en otro caso."""
    if lead not in signal12.lead_names:
        return None
    col = signal12.lead_names.index(lead)
    if signal12.lead_flags.get(lead, "ok") == "missing":
        return None
    if not np.isfinite(signal12.data[:, col]).any():
        return None
    return col


def _qtc_ms(m: dict) -> float:
    rr_s = np.mean(m["rr_ms"]) / 1000.0
    return m["qt_ms"] / np.sqrt(rr_s)  # Bazett


def _hr_bpm(m: dict) -> float:
    return 60000.0 / np.mean(m["rr_ms"])


def _qrs_axis_deg(m: dict) -> float:
    net_i, net_iii = m["net_I"], m["net_III"]
    if not np.isfinite(net_i) or not np.isfinite(net_iii):
        raise ValueError("eje QRS requiere las derivaciones I y III")
    # Einthoven: ángulo desde las proyecciones del vector cardíaco en I (0°) y III (120°)
    return float(np.degrees(np.arctan2((2.0 * net_iii + net_i) / np.sqrt(3.0), net_i)))


def _hrv_measures(m: dict) -> dict[str, float]:
    diffs = np.diff(m["rr_ms"])
    return {
        "hrv_sdnn_ms": float(np.std(m["rr_ms"], ddof=1)),
        "hrv_rmssd_ms": float(np.sqrt(np.mean(diffs**2.0))),
        "hrv_pnn50_pct": float(100.0 * np.mean(np.abs(diffs) > 50.0)),
    }


_EXTRACTORS = {
    "qrs_duration_ms": lambda m: m["qrs_duration_ms"],
    "qt_ms": lambda m: m["qt_ms"],
    "qtc_ms": _qtc_ms,
    "hr_bpm": _hr_bpm,
    "qrs_axis_deg": _qrs_axis_deg,
    "hrv_sdnn_ms": lambda m: _hrv_measures(m)["hrv_sdnn_ms"],
    "hrv_rmssd_ms": lambda m: _hrv_measures(m)["hrv_rmssd_ms"],
    "hrv_pnn50_pct": lambda m: _hrv_measures(m)["hrv_pnn50_pct"],
}