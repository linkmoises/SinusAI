"""Tests de la ingesta de imagen PNG/JPG → señal 1D (RF-02/RF-03/RF-08b/RF-14, tarea T06).

Cubre: el PNG/JPG de juguete con 12 trazados
(`tests/fixtures/imagen/toy_ecg_12_lead.{png,jpg}`) produce un `ECGRecord`
reconocido con `Signal12` (contracts §2.2–§2.3) en orden canónico; las imágenes
sin 12 trazados (`toy_tres_trazos.png`) y no-ECG (`toy_no_ecg.png`) aplican
rechazo con mensaje explicativo (RF-08b); y la vía imagen no persiste ningún
archivo, procesando en memoria con temporales efímeros (RF-14).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest_image import (  # noqa: E402
    CANONICAL_LEADS,
    N_TRACES_EXPECTED,
    REJECTION_REASON_BASIC,
    REJECTION_REASON_TRACES,
    ingest_image,
)
from src.ingest_signal import NOMINAL_FS, ECGRecord, Signal12  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
IMAGEN = FIXTURES / "imagen"
PNG_12 = IMAGEN / "toy_ecg_12_lead.png"
JPG_12 = IMAGEN / "toy_ecg_12_lead.jpg"
PNG_3 = IMAGEN / "toy_tres_trazos.png"
PNG_NO_ECG = IMAGEN / "toy_no_ecg.png"


def test_ingest_png_12_trazos_reconocido():
    rec = ingest_image(PNG_12)
    assert isinstance(rec, ECGRecord)
    assert rec.status == "reconocido"
    assert isinstance(rec.signal, Signal12)
    assert rec.source == "image"


def test_signal12_from_png_12_trazos():
    sig12 = ingest_image(PNG_12).signal
    assert sig12.data.ndim == 2
    assert sig12.data.shape[1] == 12
    assert sig12.data.shape[0] == sig12.n_samples
    assert sig12.data.dtype.kind == "f"
    assert sig12.lead_names == CANONICAL_LEADS
    assert set(sig12.lead_flags.values()) == {"ok"}
    assert all(sig12.lead_flags[lead] == "ok" for lead in CANONICAL_LEADS)


def test_signal12_from_png_orden_canonico_top_abajo():
    sig12 = ingest_image(PNG_12).signal
    assert sig12.data[:, 0].shape == (sig12.n_samples,)
    assert not np.isnan(sig12.data[:, 0]).all()


def test_imagen_produce_senial_1d_no_pixeles():
    sig12 = ingest_image(PNG_12).signal
    assert sig12.data.shape[0] > 0
    assert sig12.data.shape[1] == N_TRACES_EXPECTED == 12
    finite_per_col = np.isfinite(sig12.data).sum(axis=0)
    assert (finite_per_col > 0).all()


def test_imagen_calidad_degradada_se_marca_parcial():
    sig12 = ingest_image(PNG_12).signal
    assert sig12.fs == NOMINAL_FS
    assert sig12.n_samples < 1000
    assert sig12.status == "parcial"
    assert sig12.duration_s == pytest.approx(sig12.n_samples / NOMINAL_FS)


def test_imagen_signal_con_amplitud_de_seno():
    sig12 = ingest_image(PNG_12).signal
    assert np.nanmax(sig12.data) > 0.2
    assert np.nanmin(sig12.data) < -0.2


def test_ingest_jpg_12_trazos_reconocido():
    rec = ingest_image(JPG_12)
    assert rec.status == "reconocido"
    assert rec.signal is not None
    assert rec.signal.status == "parcial"
    assert rec.n_leads_valid == 12


def test_metadatos_imagen_y_efimeridad():
    rec = ingest_image(PNG_12)
    md = rec.metadata
    assert "image_path" in md
    assert "efimero" in md
    assert "in-memory" in md["efimero"]
    assert md["n_traces_detected"] == 12


def test_record_id_de_imagen_es_un_id_corto():
    rec = ingest_image(PNG_12)
    assert rec.record_id == "toy_ecg_12_lead"
    assert rec.signal.record_id == rec.record_id


def test_imagen_determinista():
    a = ingest_image(PNG_12)
    b = ingest_image(PNG_12)
    assert np.allclose(a.signal.data, b.signal.data, equal_nan=True)


def test_imagen_sin_12_trazados_se_rechaza():
    rec = ingest_image(PNG_3)
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert REJECTION_REASON_TRACES in rec.metadata["reason"]
    assert 1 <= rec.metadata["n_traces_detected"] < 12


def test_imagen_no_ecg_se_rechaza():
    rec = ingest_image(PNG_NO_ECG)
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert rec.metadata["reason"]


def test_imagen_ilegible_o_corrupta_se_rechaza():
    rec = ingest_image(FIXTURES / "no_ekg" / "no_ekg.dat")
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert rec.metadata["reason"] == REJECTION_REASON_BASIC


def test_ruta_inexistente_se_rechaza():
    rec = ingest_image(IMAGEN / "no_existe.png")
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert rec.metadata["reason"] == REJECTION_REASON_BASIC
    assert "error" in rec.metadata


def test_rechazos_sin_senal():
    for img in (PNG_3, PNG_NO_ECG):
        rec = ingest_image(img)
        assert rec.fs == 0.0
        assert rec.duration_s == 0.0
        assert rec.n_leads_valid == 0
        assert rec.leads_present == []


def test_rf14_no_persiste_archivos():
    def tree(folder: Path) -> set[tuple[str, int]]:
        return {
            (str(p.relative_to(folder)), p.stat().st_size)
            for p in folder.rglob("*")
            if p.is_file()
        }

    before = tree(FIXTURES)
    ingest_image(PNG_12)
    ingest_image(JPG_12)
    ingest_image(PNG_3)
    ingest_image(PNG_NO_ECG)
    ingest_image(IMAGEN / "no_existe.png")
    assert tree(FIXTURES) == before