"""Integración imagen→demo Streamlit (T17)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

import app as app_module
from src.ingest_image import ingest_image
from src.labels import RHYTHM_EVALUATED, SPECIFIC_EVALUATED, SUPERCLASS_ORDER
from src.pipeline import MODEL_PATH, predict_signal
from src.policy import CODE_QUALITY08

APP = ROOT / "app.py"
IMAGE = ROOT / "tests" / "fixtures" / "imagen" / "toy_ecg_12_lead.png"
_NEEDS_MODEL = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="modelo entrenado (model.pkl) ausente"
)


def _app_upload_image(filename: str, content_type: str):
    at = AppTest.from_file(APP).run()
    at.file_uploader[0].set_value([(filename, IMAGE.read_bytes(), content_type)]).run()
    return at


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("scan.png", "image/png"), ("scan.jpg", "image/jpeg")],
)
@_NEEDS_MODEL
def test_imagen_recorre_signal1d_y_muestra_calidad_degradada(filename, content_type):
    record = ingest_image(IMAGE)
    prediction = predict_signal(record.signal)

    assert record.source == "image"
    assert record.signal is not None
    assert record.signal.data.ndim == 2
    assert record.signal.data.shape[1] == 12
    assert prediction.outcome == "prediction"
    assert prediction.quality == "degraded"
    assert prediction.prediction_alerts == [CODE_QUALITY08]
    assert prediction.record_id == record.record_id
    assert set(prediction.superclass.probs) == set(SUPERCLASS_ORDER)
    assert set(prediction.rhythm.probs) == set(RHYTHM_EVALUATED)
    assert set(prediction.specific_diagnostic.probs) == set(SPECIFIC_EVALUATED)

    at = _app_upload_image(filename, content_type)
    try:
        assert not at.exception
        text = "\n".join(element.value for element in at.markdown)
        for key in SUPERCLASS_ORDER + RHYTHM_EVALUATED + SPECIFIC_EVALUATED:
            assert f"**{key}**" in text
        assert "entrada degradada: predicción no confiable" in [
            element.value for element in at.warning
        ]
        assert len(at.image) == 1
    finally:
        app_module.cleanup_uploads(at)
