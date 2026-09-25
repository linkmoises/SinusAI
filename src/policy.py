"""Política de avisos de la predicción (RF-06, RF-07a–d, carril LE).

Convierte las probabilidades por eje (contracts §2.6.0) y la calidad de la entrada
en el `Prediction` de `docs/contracts.md` §2.6, aplicando caso por caso la tabla de
avisos del Apéndice A (T03) con las constantes canónicas de §2.6.1:

- Umbral único `THRESHOLD = 0.60`.
- `lowconf07b` sobre **cada clave** con p < 0,60, evaluado por eje por separado
  (RF-07b; caso C2).
- Eje 1 nunca vacío: la superclase siempre expone su `top_label` (caso C5).
- Eje 2/eje 3 vacíos (todas sus p < 0,60): se expone el top-1 del eje como
  `top_label` + `lowconf07b`, conservando las probabilidades (RF-07c; casos C3–C4).
- Catálogo declarado-no-evaluado §1.4.4 (incl. `VT`/`VF`/NODAL, sin statement) se
  lista **siempre** en `declared_not_evaluated` con el aviso fijo `insuf14` y
  **nunca** una probabilidad — RF-07a y AGENTS regla 3 (casos D1–D3). El caso v1.0
  "VT con p=0,30 → avisos 07a+07b apilados" NO existe en v1.2: VT es declarada y su
  único aviso es `insuf14` (contracts §2.6.1 «Cuándo NO aplica»).
- Entrada degradada: `quality08` a nivel de `Prediction` en `prediction_alerts`; los
  avisos concurrentes se apilan, nunca se suprimen (RF-07d; caso C8).

En un `Prediction` con `outcome == "prediction"`, `alerts` contiene una entrada por
cada clave presente de los ejes (las 5 de superclase +, si vienen, las 5 de ritmo y
las 2 de específico), con sus códigos §2.6.1. Los valores son **códigos**
(`insuf14`/`lowconf07b`/`quality08`); su **texto literal canónico** (obligatorio,
AGENTS regla 3 / §2.6.1) vive en `ALERT_TEXTS` y en las constantes `*_TEXT`.

RF-06 (exponer las probabilidades de los tres ejes) se cumple en el formato: las
`probs` de cada eje van completas en `SuperclassPrediction5`/`RhythmPrediction`/
`SpecificDiagnosticPrediction`, y las claves declaradas no ocupan posición de `probs`
(§2.6, invariante 3).

El fallo total de extracción (`quality == "impossible"`) produce `outcome ==
"incertidumbre"` sin ejes, sin `alerts` ni `declared_not_evaluated` (RF-08, caso C9;
RF-07c no aplica). La entrada degradada con una o más claves p ≥ 0,60 sale como no
confiable con `quality08` a nivel de `Prediction` sin agregar `lowconf07b` a esas
claves (C6: RF-08 prevalece sobre RF-07b). Ambos casos tienen cobertura de tests en
`tests/test_policy.py`. El rechazo de entrada no-EKG (RF-08b) se produce en la
ingesta y no pasa por esta función.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.labels import (
    DECLARED_CATALOG,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
)

# constants §2.6.1 — umbral y textos literales canónicos (AGENTS regla 3)
THRESHOLD = 0.60

CODE_INSUF14 = "insuf14"
CODE_LOWCONF = "lowconf07b"
CODE_QUALITY08 = "quality08"

INSUF14_TEXT = "dato insuficiente en el set de entrenamiento"
LOWCONF_TEXT = "confianza baja"
QUALITY08_TEXT = "entrada degradada: predicción no confiable"

ALERT_TEXTS = {
    CODE_INSUF14: INSUF14_TEXT,
    CODE_LOWCONF: LOWCONF_TEXT,
    CODE_QUALITY08: QUALITY08_TEXT,
}


@dataclass
class SuperclassPrediction5:
    """contracts §2.6.0 — eje 1 (superclase diagnóstica), siempre presente."""

    probs: dict[str, float]  # exactamente las 5 claves del eje 1 (§1.4.1), [0, 1]
    top_label: str  # mayor `probs` (desempate: primer orden canónico §1.4.1)


@dataclass
class RhythmPrediction:
    """contracts §2.6.0 — eje 2 (ritmo evaluado); solo junto al eje 1."""

    probs: dict[str, float]  # exactamente las 5 claves evaluables del eje 2 (§1.4.2), [0, 1]
    top_label: str  # mayor `probs`; si el eje queda vacío, se expone el top-1 (C3)


@dataclass
class SpecificDiagnosticPrediction:
    """contracts §2.6.0 — eje 3 (diagnóstico específico evaluado); solo junto al eje 1."""

    probs: dict[str, float]  # exactamente las 2 claves evaluables del eje 3 (§1.4.3), [0, 1]
    top_label: str  # mayor `probs`; si el eje queda vacío, se expone el top-1 (C4)


@dataclass
class Prediction:
    """contracts §2.6 — salida de la política para una entrada reconocida como EKG."""

    record_id: str
    outcome: str  # `prediction` | `incertidumbre` (el `rechazo` se produce en ingesta)
    quality: str  # `nominal` | `degraded` | `impossible`
    superclass: SuperclassPrediction5 | None
    rhythm: RhythmPrediction | None
    specific_diagnostic: SpecificDiagnosticPrediction | None
    declared_not_evaluated: list[str]
    alerts: dict[str, list[str]]
    prediction_alerts: list[str]


def build_prediction(
    record_id: str,
    superclass_probs: dict[str, float],
    rhythm_probs: dict[str, float] | None = None,
    specific_probs: dict[str, float] | None = None,
    quality: str = "nominal",
) -> Prediction:
    """Construye el `Prediction` a partir de probabilidades por eje (§2.6.0).

    `superclass_probs` es obligatorio: el eje 1 se muestra siempre (§2.6, invariante
    1). `rhythm_probs`/`specific_probs` opcionales → eje 2/3 ausente (`None`); los
    ejes presentes nunca se muestran solos (invariante 2). Cada dict debe contener
    exactamente las claves del eje en su orden canónico, con valores en [0, 1].

    `quality`: `nominal` → sin aviso de calidad; `degraded` → `quality08` en
    `prediction_alerts` (prevalece sobre el umbral, RF-08); `impossible` →
    `outcome == "incertidumbre"` sin etiquetas (RF-07c no aplica, C9).
    """
    if quality not in ("nominal", "degraded", "impossible"):
        raise ValueError(
            f"quality inválida: {quality!r} (esperada: nominal, degraded o impossible)"
        )
    if quality == "impossible":
        return Prediction(
            record_id=record_id,
            outcome="incertidumbre",
            quality=quality,
            superclass=None,
            rhythm=None,
            specific_diagnostic=None,
            declared_not_evaluated=[],
            alerts={},
            prediction_alerts=[CODE_QUALITY08],
        )

    superclass_probs = _validated(superclass_probs, SUPERCLASS_ORDER, "superclass")
    superclass = SuperclassPrediction5(
        probs=superclass_probs,
        top_label=_top_label(superclass_probs, SUPERCLASS_ORDER),
    )

    rhythm = None
    if rhythm_probs is not None:
        rhythm_probs = _validated(rhythm_probs, RHYTHM_EVALUATED, "rhythm")
        rhythm = RhythmPrediction(
            probs=rhythm_probs,
            top_label=_top_label(rhythm_probs, RHYTHM_EVALUATED),
        )

    specific_diagnostic = None
    if specific_probs is not None:
        specific_probs = _validated(specific_probs, SPECIFIC_EVALUATED, "specific_diagnostic")
        specific_diagnostic = SpecificDiagnosticPrediction(
            probs=specific_probs,
            top_label=_top_label(specific_probs, SPECIFIC_EVALUATED),
        )

    alerts: dict[str, list[str]] = {}
    for axis_probs in (superclass_probs, rhythm_probs, specific_probs):
        if axis_probs is None:
            continue
        for key in axis_probs:
            alerts[key] = [CODE_LOWCONF] if axis_probs[key] < THRESHOLD else []

    prediction_alerts = [CODE_QUALITY08] if quality == "degraded" else []

    return Prediction(
        record_id=record_id,
        outcome="prediction",
        quality=quality,
        superclass=superclass,
        rhythm=rhythm,
        specific_diagnostic=specific_diagnostic,
        declared_not_evaluated=list(DECLARED_CATALOG),
        alerts=alerts,
        prediction_alerts=prediction_alerts,
    )


def declared_entries(catalog: list[str] | None = None) -> list[tuple[str, str]]:
    """Pares (clave declarada, código `insuf14`) del catálogo §1.4.4.

    Todas las entradas declaradas del Apéndice A.2 llevan el aviso fijo `insuf14` y
    ninguna lleva probabilidad calculada (RF-07a). El texto literal de cada código
    está en `ALERT_TEXTS` (§2.6.1).
    """
    keys = DECLARED_CATALOG if catalog is None else catalog
    return [(key, CODE_INSUF14) for key in keys]


def _validated(probs: dict[str, float], order: list[str], axis: str) -> dict[str, float]:
    """Valida claves y valores de un eje; devuelve los valores como float en [0, 1]."""
    if set(probs) != set(order):
        raise ValueError(
            f"probs de {axis}: se esperaban exactamente {order}, "
            f"se recibió {sorted(probs)}"
        )
    values: dict[str, float] = {}
    for key in order:
        try:
            value = float(probs[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probs de {axis}[{key}] no numérico: {probs[key]!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"probs de {axis}[{key}] fuera de [0, 1]: {value}")
        values[key] = value
    return values


def _top_label(probs: dict[str, float], order: list[str]) -> str:
    """Etiqueta con mayor `probs`; desempate por primer orden canónico (§2.6.0)."""
    best_key: str | None = None
    best_value = -math.inf
    for key in order:
        if probs[key] > best_value:
            best_value = probs[key]
            best_key = key
    assert best_key is not None  # `order` nunca es vacío para los ejes del contrato
    return best_key