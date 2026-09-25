"""Ingesta de señal nativa WFDB/PTB-XL a 100 Hz (RF-01, carril LA).

Convierte un par WFDB (`.hea`/`.dat`) en `ECGRecord` + `Signal12` según
`docs/contracts.md` §2.2–§2.3: normaliza las columnas al orden canónico de 12
derivaciones (§1.2) y expone los metadatos del encabezado. La señal se trata
como serie temporal 1D (nunca píxeles, RF-03).

Estado nominal de la señal (todo derivación presente, n_samples >= 1000@100 Hz):
`Signal12.status == "completo"`; cualquier degradación (derivación faltante o menos
muestras de las nominales) produce `Signal12.status == "parcial"` sobre un registro
`reconocido` (RF-08). Si `wfdb` no puede leer la entrada (cabecera ilegible, archivo
corrupto o inexistente), no es reconocible como EKG y se devuelve un `ECGRecord` con
`status == "rechazo"`, `signal is None` y el motivo en `metadata.reason` (RF-08b).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

CANONICAL_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
]
NOMINAL_FS = 100
NOMINAL_N_SAMPLES = 1000
NOMINAL_DURATION_S = NOMINAL_N_SAMPLES / NOMINAL_FS  # 10.0 s
REJECTION_REASON = "entrada no reconocible como EKG: cabecera WFDB ilegible o corrupta"


@dataclass
class Signal12:
    """contracts §2.3 — señal 1D normalizada a 12 derivaciones canónicas."""

    record_id: str
    source: str
    fs: float
    duration_s: float
    n_samples: int
    data: np.ndarray
    lead_names: list[str]
    lead_flags: dict[str, str]
    status: str


@dataclass
class ECGRecord:
    """contracts §2.2 — registro EKG reconocido (o rechazado) con su señal."""

    record_id: str
    source: str
    status: str
    fs: float
    duration_s: float
    n_leads_valid: int
    leads_present: list[str]
    metadata: dict
    signal: Signal12 | None = None


def ingest_signal(record_path: str | Path) -> ECGRecord:
    """Lee un par WFDB nativo (`.hea`/`.dat`) y devuelve su `ECGRecord`.

    `record_path` es la ruta del registro sin extensión (p. ej.
    `records100/00000/00001_lr` o `tests/fixtures/nominal/nominal`).

    Una entrada que `wfdb` no puede leer como EKG (cabecera ilegible, `.hea`/`.dat`
    corruptos o inexistentes) produce `status == "rechazo"`, `signal is None` y el
    motivo en `metadata.reason` (RF-08b).
    """
    record_path = Path(record_path)
    record_id = str(record_path.with_suffix(""))

    try:
        sig, meta = wfdb.rdsamp(record_path)
    except Exception as exc:
        return ECGRecord(
            record_id=record_id,
            source="signal",
            status="rechazo",
            fs=0.0,
            duration_s=0.0,
            n_leads_valid=0,
            leads_present=[],
            metadata={
                "record_path": str(record_path.with_suffix("")),
                "reason": REJECTION_REASON,
                "error": f"{type(exc).__name__}: {exc}",
            },
            signal=None,
        )

    fs = float(meta["fs"])
    n_samples = int(sig.shape[0])
    duration_s = n_samples / fs
    header_leads = list(meta["sig_name"])

    data, lead_flags = _normalize_to_canonical(sig, header_leads)
    leads_present = [lead for lead, flag in lead_flags.items() if flag != "missing"]
    status = (
        "completo"
        if len(leads_present) == 12 and n_samples >= NOMINAL_N_SAMPLES
        else "parcial"
    )

    signal12 = Signal12(
        record_id=record_id,
        source="signal",
        fs=fs,
        duration_s=duration_s,
        n_samples=n_samples,
        data=data,
        lead_names=list(CANONICAL_LEADS),
        lead_flags=lead_flags,
        status=status,
    )

    return ECGRecord(
        record_id=record_id,
        source="signal",
        status="reconocido",
        fs=fs,
        duration_s=duration_s,
        n_leads_valid=len(leads_present),
        leads_present=leads_present,
        metadata={
            "record_path": str(record_path.with_suffix("")),
            "name": record_path.name,
            "n_leads": int(meta["n_sig"]),
            "n_samples": n_samples,
            "fs": fs,
            "units": list(meta["units"]),
            "sig_name": header_leads,
        },
        signal=signal12,
    )


def _normalize_to_canonical(
    sig: np.ndarray, header_leads: list[str]
) -> tuple[np.ndarray, dict[str, str]]:
    """Reordena `sig` (n, n_leads) a las 12 columnas canónicas (contracts §1.2).

    La derivación ausente se marca `missing` y su columna queda NaN; el resto
    `ok`. Las columnas de `Signal12.data[:, i]` corresponden a `CANONICAL_LEADS[i]`.
    """
    header_col = {lead: i for i, lead in enumerate(header_leads)}
    data = np.full((sig.shape[0], len(CANONICAL_LEADS)), np.nan, dtype=float)
    lead_flags: dict[str, str] = {}
    for i, lead in enumerate(CANONICAL_LEADS):
        col = header_col.get(lead)
        if col is not None:
            data[:, i] = sig[:, col]
            lead_flags[lead] = "ok"
        else:
            lead_flags[lead] = "missing"
    return data, lead_flags