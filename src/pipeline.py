"""Pegamento de integración señal→demo Streamlit (T16, carril LH).

Cierra la vía nominal en vivo del taller: una señal WFDB nativa (T04/T05,
`src.ingest_signal`) se convierte en `Signal12`; `src.features` (T07/T08)
extrae el vector clínico cerrado de 8 medidas; el modelo entrenado (T12,
`train.predict_label_proba`) infiere las probabilidades de las 12 salidas
evaluadas de los tres ejes de `docs/contracts.md` §1.4 (5 superclase + 5 ritmo +
2 específico — la noción v1.0 de 8 salidas quedó supersedida por el contrato
v1.2, igual que en T14); y `src.policy` (T10/T11) las convierte en `Prediction`
con los avisos de §2.6.1, que la demo (T14/T15, `app.py`) renderiza.

La entrada es siempre `Signal12` (serie 1D); nunca píxeles (RF-03). El modelo se
carga una vez por proceso (`load_model`, lru_cache) desde `model.pkl` (contracts
§2.7/§3). El pegamento no pide ni persiste datos de pacientes (RF-14).

Mapeo de calidad (contracts §2.1, RF-08): `FeatureVector.status == "imposible"`
→ `Prediction` de incertidumbre sin etiquetas; vector `parcial` o `Signal12`
`parcial` (degradación reconocible) → predicción con `quality == "degraded"`
(`quality08` prevalece sobre el umbral) usando `FEATURE_FALLBACKS` para las
pocas features ausentes — esos valores nunca se muestran en la app y la salida
siempre se presenta como no confiable.
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path

import numpy as np

from src.features import FEATURE_KEYS, FeatureVector, extract_features
from src.ingest_signal import Signal12
from src.labels import (
    EVALUATED_TARGETS,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
)
from src.policy import Prediction, build_prediction
from train import predict_label_proba

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.pkl"

MODEL_MISSING_MSG = (
    "no se encontró el modelo entrenado (`model.pkl`): ejecutar `python train.py` "
    "para generar la predicción en vivo de la demo"
)

# Respaldos fisiológicos típicos para las features ausentes en la vía degradada.
# Solo se aplican cuando `FeatureVector.status == "parcial"` (contracts §2.1,
# RF-08): la salida se marca siempre `quality08` (no confiable), nunca se muestran
# en la app (RF-09) y jamás se usan con vector completo.
FEATURE_FALLBACKS: dict[str, float] = {
    "qrs_duration_ms": 100.0,
    "qt_ms": 400.0,
    "qtc_ms": 420.0,
    "hr_bpm": 75.0,
    "qrs_axis_deg": 60.0,
    "hrv_sdnn_ms": 50.0,
    "hrv_rmssd_ms": 40.0,
    "hrv_pnn50_pct": 20.0,
}

# Indices de cada eje dentro del vector de `EVALUATED_TARGETS` (§1.4, §2.7).
_SUPERCLASS_IDX = [EVALUATED_TARGETS.index(key) for key in SUPERCLASS_ORDER]
_RHYTHM_IDX = [EVALUATED_TARGETS.index(key) for key in RHYTHM_EVALUATED]
_SPECIFIC_IDX = [EVALUATED_TARGETS.index(key) for key in SPECIFIC_EVALUATED]


@functools.lru_cache(maxsize=2)
def load_model(model_path: str | Path = MODEL_PATH) -> dict:
    """Carga una vez por proceso el artefacto `model.pkl` (contracts §3).

    El artefacto (`train.save_model`) contiene `model`, `target_labels`,
    `features_used` y `metadata`; no contiene datos de pacientes (RF-14). Si la
    ruta no existe lanza `FileNotFoundError` — el demo lo muestra como aviso.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"modelo entrenado no encontrado: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def feature_vector_to_X(fv: FeatureVector) -> np.ndarray:
    """`FeatureVector` → matriz `(1, 8)` de features en el orden de §2.4.1.

    Con vector `completo` se usan los valores reales; con `parcial` (vía
    degradada, RF-08) las features ausentes se completan con `FEATURE_FALLBACKS`
    para que el modelo pueda emitir una probabilidad — la salida se marca
    siempre `quality08`, nunca se presenta como confiable.
    """
    values: dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = fv.values.get(key)
        if value is None or not np.isfinite(value):
            if fv.status == "completo":
                raise ValueError(f"feature {key!r} no finita con status completo")
            values[key] = FEATURE_FALLBACKS[key]
        else:
            values[key] = float(value)
    return np.asarray([[values[key] for key in FEATURE_KEYS]], dtype=float)


def probs_vector_to_axes(
    proba: np.ndarray | list,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Vector `(12,)` del modelo → dicts de `probs` por eje (contracts §2.6.0).

    Alineado con `EVALUATED_TARGETS` (orden canónico §1.4): 5 superclase, 5
    ritmo, 2 específico. Cada dict usa las claves canónicas de su eje en orden.
    """
    proba = np.asarray(proba, dtype=float).reshape(-1)
    if proba.shape[0] != len(EVALUATED_TARGETS):
        raise ValueError(
            f"se esperaban {len(EVALUATED_TARGETS)} probabilidades por "
            f"EVALUATED_TARGETS, se recibió {proba.shape[0]}"
        )
    superclass_probs = {key: float(proba[idx]) for key, idx in zip(SUPERCLASS_ORDER, _SUPERCLASS_IDX)}
    rhythm_probs = {key: float(proba[idx]) for key, idx in zip(RHYTHM_EVALUATED, _RHYTHM_IDX)}
    specific_probs = {key: float(proba[idx]) for key, idx in zip(SPECIFIC_EVALUATED, _SPECIFIC_IDX)}
    return superclass_probs, rhythm_probs, specific_probs


def _quality(signal12: Signal12, fv: FeatureVector) -> str:
    """§2.1: degradada ⇔ señal parcial o vector de features parcial."""
    if signal12.status == "parcial" or fv.status == "parcial":
        return "degraded"
    return "nominal"


def predict_signal(signal12: Signal12, model_path: str | Path = MODEL_PATH) -> Prediction:
    """Vía nominal completa de una `Signal12`: ingesta→features→modelo→política.

    `fv.status == "imposible"` → `Prediction` de incertidumbre, sin etiquetas
    (RF-08, RF-07c no aplica). En otro caso infiere las 12 probabilidades con el
    modelo y construye el `Prediction` con la política de avisos: `nominal` si la
    señal y el vector están completos, `degraded` (quality08) en caso contrario.
    """
    if signal12 is None:
        raise ValueError("predict_signal requiere una Signal12 (record.signal)")

    fv = extract_features(signal12)
    if fv.status == "imposible":
        # `quality == "impossible"` ignora las probs (short-circuit de la política,
        # §2.6.1/C9): se pasa un dict vacío-eléctrico solo por la firma obligatoria
        # de `superclass_probs`.
        return build_prediction(
            signal12.record_id,
            {key: 0.0 for key in SUPERCLASS_ORDER},
            quality="impossible",
        )

    artifact = load_model(model_path)
    proba = predict_label_proba(artifact["model"], feature_vector_to_X(fv))[0]
    superclass_probs, rhythm_probs, specific_probs = probs_vector_to_axes(proba)

    return build_prediction(
        signal12.record_id,
        superclass_probs,
        rhythm_probs,
        specific_probs,
        quality=_quality(signal12, fv),
    )