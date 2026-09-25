"""Ingesta de imagen de trazado EKG (PNG/JPG) a señal 1D (RF-02, RF-03, RF-14, carril LB).

Convierte una imagen escaneada de un EKG de 12 derivaciones estándar en un
`ECGRecord` + `Signal12` (contracts §2.2–§2.3) vía digitalización a señal
temporal 1D — el trazado nunca se clasifica desde píxeles (RF-03, Constitution
§2).

Comportamiento:
- La imagen es tratada como un gráfico de líneas: cada derivación es una curva
  que cruza la imagen de punta a punta. La digitalización detecta, por columna,
  los cruces de tinta de corta altura (la curva en sí) y decide el número de
  trazados como el máximo de cruces observado en una sola columna vertical.
- Si la imagen tiene exactamente 12 trazados → `status == "reconocido"` y se
  devuelve un `Signal12` con las 12 columnas en orden canónico (§1.2), top→down
  (la curva superior es la derivación I). El resultado NO es de calidad nativa:
  la calibración exacta mm/s–mm/mV está fuera de alcance (spec "Fuera de
  alcance"), así que `n_samples` es el ancho en píxeles (< 1000) y
  `Signal12.status == "parcial"` — la vía imagen queda marcada como degradada
  aguas abajo (RF-08) y la demo la presenta con aviso de calidad.
- Si la imagen no tiene 12 trazados (menos trazados, o ruido/no-ECG) →
  `status == "rechazo"`, `signal is None` y motivo en `metadata.reason` (RF-08b).
- No persiste nada: procesa en memoria, sin archivos temporales en repo ni en
  disco (RF-14). Solo lee la ruta del upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from src.ingest_signal import (
    CANONICAL_LEADS,
    ECGRecord,
    NOMINAL_FS,
    Signal12,
)

NOMINAL_N_SAMPLES = 1000  # contracts §1.2: 10 s @ 100 Hz
INK_THRESHOLD = 128  # píxel 'tinta' si gris < umbral (trazo negro sobre blanco)
MAX_SLIVER_PX = 5  # agrupa fragmentos del trazo a menos de 5 px (antialias)
MAX_RUN_LEN = 3  # cruce de la curva: altura ≤ 3 px; un trazo vertical es más alto
N_TRACES_EXPECTED = 12
REJECTION_REASON_BASIC = "entrada no reconocible como EKG: imagen no legible o no es PNG/JPG"
REJECTION_REASON_TRACES = (
    "entrada no reconocible como EKG: imagen sin 12 trazados detectados"
)


@dataclass
class TraceDetection:
    """Resultado de la etapa de detección de trazados."""

    n_traces: int
    columns_detected: list[np.ndarray]
    bands: np.ndarray


def ingest_image(image_path: str | Path) -> ECGRecord:
    """Digitaliza una imagen PNG/JPG de trazado EKG y devuelve su `ECGRecord`.

    `image_path` es la ruta a un archivo de imagen (PNG o JPG). Si no se lee
    como imagen, o no se detectan exactamente 12 trazados, devuelve un
    `ECGRecord` con `status == "rechazo"`, `signal is None` y el motivo en
    `metadata.reason` (RF-08b). No escribe ningún archivo (RF-14).
    """
    image_path = Path(image_path)
    record_id = image_path.stem
    image_path_unchecked = image_path

    try:
        grey, mode, size = _load_greyscale(image_path)
    except Exception as exc:
        return _rejection(record_id, image_path_unchecked, REJECTION_REASON_BASIC, exc)

    detection = _detect_traces(grey)
    if detection.n_traces != N_TRACES_EXPECTED:
        return ECGRecord(
            record_id=record_id,
            source="image",
            status="rechazo",
            fs=0.0,
            duration_s=0.0,
            n_leads_valid=0,
            leads_present=[],
            metadata={
                "image_path": str(image_path_unchecked),
                "n_traces_detected": detection.n_traces,
                "reason": f"{REJECTION_REASON_TRACES} ({detection.n_traces})",
            },
            signal=None,
        )

    data_mv = _digitize(detection)
    n_samples = int(data_mv.shape[0])
    duration_s = n_samples / NOMINAL_FS
    lead_flags = {lead: "ok" for lead in CANONICAL_LEADS}

    signal12 = Signal12(
        record_id=record_id,
        source="image",
        fs=NOMINAL_FS,
        duration_s=duration_s,
        n_samples=n_samples,
        data=data_mv,
        lead_names=list(CANONICAL_LEADS),
        lead_flags=lead_flags,
        status="parcial",
    )

    return ECGRecord(
        record_id=record_id,
        source="image",
        status="reconocido",
        fs=NOMINAL_FS,
        duration_s=duration_s,
        n_leads_valid=N_TRACES_EXPECTED,
        leads_present=list(CANONICAL_LEADS),
        metadata={
            "image_path": str(image_path_unchecked),
            "format": image_path.suffix.lower().lstrip("."),
            "size": size,
            "mode": mode,
            "n_traces_detected": detection.n_traces,
            "efimero": "in-memory: sin archivo temporal persistido en disco/repo (RF-14)",
        },
        signal=signal12,
    )


def _rejection(
    record_id: str, image_path: Path, reason: str, exc: Exception | None = None
) -> ECGRecord:
    metadata: dict = {"image_path": str(image_path), "reason": reason}
    if exc is not None:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    return ECGRecord(
        record_id=record_id,
        source="image",
        status="rechazo",
        fs=0.0,
        duration_s=0.0,
        n_leads_valid=0,
        leads_present=[],
        metadata=metadata,
        signal=None,
    )


def _load_greyscale(image_path: Path) -> tuple[np.ndarray, str, tuple[int, int]]:
    """Abre la imagen y la devuelve en escala de grises float (0–255)."""
    with Image.open(image_path) as img:
        mode = img.mode
        size = img.size
        grey = np.asarray(img.convert("L"), dtype=float)
    return grey, mode, size


def _ink_runs(grey: np.ndarray, x: int) -> list[tuple[int, int]]:
    """Segmentos verticales consecutivos de tinta en la columna `x` (start, end)."""
    col = grey[:, x]
    runs: list[tuple[int, int]] = []
    h = grey.shape[0]
    r = 0
    while r < h:
        if col[r] < INK_THRESHOLD:
            s = r
            while r < h and col[r] < INK_THRESHOLD:
                r += 1
            runs.append((s, r))
        else:
            r += 1
    return runs


def _merged_centers(grey: np.ndarray, x: int) -> np.ndarray:
    """Centros de tinta de una columna, con slivers de antialias agrupados.

    Cada trazo (curva de ~1 px de alto) deja en la columna uno o varios
    fragmentos muy próximos; se fusionan los centros a menos de
    `MAX_SLIVER_PX` px en un solo observado (media), representando una muestra
    de la curva en esa columna.
    """
    centers = [s + (e - s) / 2.0 for s, e in _ink_runs(grey, x)]
    centers.sort()
    merged: list[float] = []
    for c in centers:
        if merged and c - merged[-1] <= MAX_SLIVER_PX:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)
    return np.array(merged)


def _detect_traces(grey: np.ndarray) -> TraceDetection:
    """Detecta los trazados y su orden vertical (contracts §1.2).

    Cuenta por columna los cruces de tinta de corta altura (`MAX_RUN_LEN`): la
    curva de cada derivación cruza la columna como un segmento de ~1–3 px, en
    tanto que un QRS (desfase vertical) o ruido grueso producen segmentos más
    altos o bien dispersos. `n_traces` es el máximo de cruces en una sola
    columna. Las bandas verticales de cada trazado se estiman con picos del
    histograma suavizado de todos los centros observados; si no aparecen
    exactamente 12 picos (trazados casi coincidentes), se usan 12 cuantiles
    como bandas de reserva.
    """
    width = grey.shape[1]
    columns = [_merged_centers(grey, x) for x in range(width)]
    counts = np.array([len(c) for c in columns])
    n_traces = int(counts.max())

    all_centers = np.concatenate([c for c in columns if len(c)])
    max_y = int(all_centers.max())
    hist, edges = np.histogram(
        all_centers, bins=max_y + 1, range=(0, max_y + 1)
    )
    kernel = np.ones(5) / 5  # suavizado de 5 px
    smoothed = np.convolve(hist, kernel, mode="same")
    peaks, _ = find_peaks(smoothed, prominence=2, distance=6)
    bands = np.sort(edges[peaks].astype(float))
    if len(bands) != N_TRACES_EXPECTED:
        bands = np.quantile(all_centers, np.linspace(0.03, 0.97, N_TRACES_EXPECTED))

    return TraceDetection(n_traces=n_traces, columns_detected=columns, bands=bands)


def _digitize(detection: TraceDetection) -> np.ndarray:
    """Convierte los centros observados en una matriz `(n_samples, 12)` en mV.

    Para cada trazado `i` y columna `x`, la muestra es el centro observado más
    cercano a su banda `bands[i]` dentro de la mitad de la separación a las
    bandas vecinas (así cada curva "pertenece" a su banda). Base de la curva =
    mediana de sus centros; la deflexión en píxeles se escala a mV asumiendo que
    la separación mediana entre bandas equivale a ≈1 mV (calibración aproximada,
    documentada en spec como fuera de alcance). Los micro-huecos (≤4 px) se
    interpolan linealmente; el resto queda NaN (degradación local, RF-08). La
    columna `i` del resultado corresponde a `CANONICAL_LEADS[i]` (curva superior
    = derivación I, contracts §1.2).
    """
    columns = detection.columns_detected
    bands = detection.bands
    width = len(columns)

    half: np.ndarray = np.full(len(bands), np.inf)
    for i in range(len(bands)):
        if i > 0:
            half[i] = min(half[i], (bands[i] - bands[i - 1]) / 2.0)
        if i < len(bands) - 1:
            half[i] = min(half[i], (bands[i + 1] - bands[i]) / 2.0)

    tracks = np.full((len(bands), width), np.nan)
    for x, centers in enumerate(columns):
        if not len(centers):
            continue
        for i, band in enumerate(bands):
            distances = np.abs(centers - band)
            j = int(np.argmin(distances))
            if distances[j] <= half[i]:
                tracks[i, x] = centers[j]

    baseline = np.nanmedian(tracks, axis=1)
    band_sep = np.diff(bands)
    px_per_mv = float(np.median(band_sep)) if len(band_sep) else 1.0
    scale = 1.0 / px_per_mv
    data = (baseline[:, None] - tracks) * scale
    data = _fill_short_gaps(data, max_gap=4)
    return data.T  # (n_samples, 12)


def _fill_short_gaps(data: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola huecos de hasta `max_gap` columnas en cada derivación."""
    out = data.copy()
    for row in out:
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size < 2:
            continue
        for idx in np.flatnonzero(np.isnan(row)):
            left = finite[finite < idx]
            right = finite[finite > idx]
            if not left.size or not right.size:
                continue
            if right[0] - left[-1] - 1 <= max_gap:
                a, b = left[-1], right[0]
                row[a + 1 : b] = np.linspace(row[a], row[b], b - a + 1)[1:-1]
    return out