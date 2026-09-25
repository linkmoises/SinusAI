"""Integración señal→demo Streamlit (T16, carril LH).

Verifica el «Hecho cuando» de T16: una señal nominal recorre
ingesta→features→modelo→política→demo Streamlit con las probabilidades en vivo
de las 12 salidas evaluadas de los tres ejes (contracts v1.2 §1.4/§2.6; la
noción v1.0 de 8 probabilidades quedó supersedida, igual que en T14).

El «pegamento» vive en `src.pipeline.predict_signal`; la demo la cablea en
`_render_upload` al cargar una señal WFDB nativa. Los tests de AppTest cargan la
fixture nominal (`tests/fixtures/nominal/nominal`) y comprueban que lo renderizado
es la predicción real del modelo `model.pkl` (no la sintética de T14).

`model.pkl` no se versiona (`.gitignore`; contracts §3 lo lista como artefacto
local del entrenamiento T12): los tests que lo requieren se saltan si hace falta
en el entorno (en el laptop del taller existe y corren).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

import app as app_module  # noqa: E402
from src.features import FEATURE_KEYS, extract_features  # noqa: E402
from src.ingest_signal import ingest_signal  # noqa: E402
from src.labels import (  # noqa: E402
    EVALUATED_TARGETS,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
)
from src.pipeline import (  # noqa: E402
    MODEL_PATH,
    FEATURE_FALLBACKS,
    feature_vector_to_X,
    predict_signal,
    probs_vector_to_axes,
)
from src.policy import CODE_QUALITY08  # noqa: E402

APP = ROOT / "app.py"
NOMINAL = ROOT / "tests" / "fixtures" / "nominal" / "nominal"
NOMINAL_HEA = NOMINAL.with_suffix(".hea")
NOMINAL_DAT = NOMINAL.with_suffix(".dat")
DEGRADADA = ROOT / "tests" / "fixtures" / "degradada" / "degradada"
FALLO_TOTAL = ROOT / "tests" / "fixtures" / "fallo_total" / "fallo_total"

_HAS_MODEL = MODEL_PATH.exists()
_NEEDS_MODEL = pytest.mark.skipif(not _HAS_MODEL, reason="modelo entrenado (model.pkl) ausente")

# Prohibidos por RF-09 (contracts §2.6 / spec RF-09): ni features ni gráficas.
BANNED = FEATURE_KEYS + ["ROC", "AUC", "matriz de confusión", "importancia"]


def test_probs_vector_to_axes_respeta_el_orden_canonico_del_contrato():
    proba = np.arange(len(EVALUATED_TARGETS), dtype=float) + 0.5
    superclass_probs, rhythm_probs, specific_probs = probs_vector_to_axes(proba)
    assert list(superclass_probs) == SUPERCLASS_ORDER
    assert list(rhythm_probs) == RHYTHM_EVALUATED
    assert list(specific_probs) == SPECIFIC_EVALUATED
    for key in SUPERCLASS_ORDER:
        assert superclass_probs[key] == proba[EVALUATED_TARGETS.index(key)]
    for key in RHYTHM_EVALUATED:
        assert rhythm_probs[key] == proba[EVALUATED_TARGETS.index(key)]
    for key in SPECIFIC_EVALUATED:
        assert specific_probs[key] == proba[EVALUATED_TARGETS.index(key)]


def test_probs_vector_de_longitud_incorrecta_lanza_error():
    with pytest.raises(ValueError):
        probs_vector_to_axes(np.zeros(8))


def test_predict_signal_rechaza_signal_none():
    with pytest.raises(ValueError):
        predict_signal(None, model_path=NOMINAL)  # nunca llega al modelo


def test_fallo_total_produce_incertidumbre_sin_etiquetas():
    rec = ingest_signal(FALLO_TOTAL)
    assert rec.status == "reconocido" and rec.signal is not None
    prediction = predict_signal(rec.signal)
    assert prediction.outcome == "incertidumbre"
    assert prediction.quality == "impossible"
    assert prediction.superclass is None
    assert prediction.declared_not_evaluated == []
    assert prediction.prediction_alerts == [CODE_QUALITY08]


def test_modelo_ausente_lanza_filenotfound(tmp_path):
    rec = ingest_signal(NOMINAL)
    with pytest.raises(FileNotFoundError):
        predict_signal(rec.signal, model_path=tmp_path / "no_hay.pkl")


def test_feature_vector_parcial_usa_fallback_solo_de_la_via_degradada():
    fv = extract_features(ingest_signal(NOMINAL).signal)
    assert fv.status == "completo"
    x = feature_vector_to_X(fv)
    assert x.shape == (1, 8)
    assert np.isfinite(x).all()

    fv_degradada = extract_features(ingest_signal(DEGRADADA).signal)
    assert fv_degradada.status == "parcial"
    x_degradada = feature_vector_to_X(fv_degradada)
    for key in fv_degradada.missing_features:
        assert x_degradada[0, FEATURE_KEYS.index(key)] == FEATURE_FALLBACKS[key]
    for key in set(FEATURE_KEYS) - set(fv_degradada.missing_features):
        assert x_degradada[0, FEATURE_KEYS.index(key)] == fv_degradada.values[key]


@_NEEDS_MODEL
def test_senal_nominal_recorre_ingesta_features_politica_con_probabilidad_en_vivo():
    rec = ingest_signal(NOMINAL)
    assert rec.status == "reconocido" and rec.signal is not None
    assert extract_features(rec.signal).status == "completo"

    prediction = predict_signal(rec.signal)
    assert prediction.outcome == "prediction"
    assert prediction.quality == "nominal"
    assert prediction.record_id == str(NOMINAL.with_suffix(""))

    # RF-06 / contracts v1.2: las 12 probabilidades de los tres ejes presentes.
    axes = (prediction.superclass, prediction.rhythm, prediction.specific_diagnostic)
    assert all(axis is not None for axis in axes)
    assert set(prediction.superclass.probs) == set(SUPERCLASS_ORDER)
    assert set(prediction.rhythm.probs) == set(RHYTHM_EVALUATED)
    assert set(prediction.specific_diagnostic.probs) == set(SPECIFIC_EVALUATED)
    for axis in axes:
        for prob in axis.probs.values():
            assert np.isfinite(prob) and 0.0 <= prob <= 1.0
        assert axis.top_label == max(axis.probs, key=lambda k: axis.probs[k])

    # RF-07a: catálogo declarado presente con aviso fijo en política.
    assert prediction.declared_not_evaluated
    assert prediction.superclass.top_label == "NORM"
    assert prediction.rhythm.top_label == "SR"


@_NEEDS_MODEL
def test_probabilidad_en_vivo_no_es_la_sintetica_de_la_demo_t14():
    live = predict_signal(ingest_signal(NOMINAL).signal)
    sintetica = app_module._demo_prediction()
    assert live.superclass.probs != sintetica.superclass.probs


@_NEEDS_MODEL
def test_degradada_reconocible_sale_no_confiable_con_quality08():
    rec = ingest_signal(DEGRADADA)
    assert rec.status == "reconocido"
    prediction = predict_signal(rec.signal)
    assert prediction.outcome == "prediction"
    assert prediction.quality == "degraded"  # RF-08: prevalece sobre el umbral
    assert prediction.prediction_alerts == [CODE_QUALITY08]
    assert prediction.superclass is not None


# ---------------------------------------------------------------------------
# AppTest — la demo Streamlit muestra la predicción en vivo de una señal nominal
# ---------------------------------------------------------------------------


def _app_upload_signal(files):
    """AppTest con una carga de señal WFDB seteada y su temporal efímero."""
    at = AppTest.from_file(APP).run()
    at.file_uploader[0].set_value(files).run()
    assert not at.exception
    return at


def _text(at) -> str:
    return "\n".join(e.value for e in at.markdown)


@_NEEDS_MODEL
def test_app_muestra_prediccion_en_vivo_al_cargar_senal_nominal():
    at = _app_upload_signal(
        [
            ("nominal.hea", NOMINAL_HEA.read_bytes(), "text/plain"),
            ("nominal.dat", NOMINAL_DAT.read_bytes(), "application/octet-stream"),
        ]
    )
    try:
        assert not at.exception
        assert [e.value for e in at.warning] == []
        assert [e.value for e in at.error] == []

        expected = predict_signal(ingest_signal(NOMINAL).signal)
        text = _text(at)

        # las 12 etiquetas evaluadas con su probabilidad del modelo (RF-06)
        for key in SUPERCLASS_ORDER + RHYTHM_EVALUATED + SPECIFIC_EVALUATED:
            assert f"**{key}**" in text, key
        assert f"**NORM** — {expected.superclass.probs['NORM']:.2f}" in text
        assert f"**SR** — {expected.rhythm.probs['SR']:.2f}" in text
        # top por eje (el de la predicción real, no la sintética de T14)
        assert f"**Diagnóstico principal: {expected.superclass.top_label}**" in text
        assert f"**Ritmo principal: {expected.rhythm.top_label}**" in text
        assert f"**Hallazgo principal: {expected.specific_diagnostic.top_label}**" in text

        # aviso de RF-07a y contenido exacto RF-09 (solo texto ni features/gráficas)
        assert "**VT**" in text
        for banned in BANNED:
            assert banned not in text, banned
        assert len(at.image) == 1  # trazado del EKG (RF-09)
    finally:
        app_module.cleanup_uploads(at)