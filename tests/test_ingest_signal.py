"""Tests de la ingesta de señal nativa WFDB/PTB-XL (RF-01/RF-08/RF-08b, tareas T04–T05).

Cubre el caso nominal de `tests/fixtures/nominal/nominal` (12 derivaciones × 100 Hz
× 10 s, orden canónico, metadatos del header), el caso degradado de
`tests/fixtures/degradada/degradada` (derivación `I` faltante → reconocido pero
`parcial`, nunca rechazo) y el rechazo de entradas no legibles como EKG
(`tests/fixtures/no_ekg/no_ekg` y rutas inexistentes).
"""

import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest_signal import (  # noqa: E402
    CANONICAL_LEADS,
    ECGRecord,
    NOMINAL_DURATION_S,
    NOMINAL_FS,
    NOMINAL_N_SAMPLES,
    REJECTION_REASON,
    Signal12,
    ingest_signal,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOMINAL = FIXTURES / "nominal" / "nominal"
DEGRADADA = FIXTURES / "degradada" / "degradada"
NO_EKG = FIXTURES / "no_ekg" / "no_ekg"


def test_ingest_nominal_devuelve_ecgrecord_y_signal12():
    rec = ingest_signal(NOMINAL)
    assert isinstance(rec, ECGRecord)
    assert isinstance(rec.signal, Signal12)


def test_signal12_nominal_shape_y_orden_canonico():
    sig12 = ingest_signal(NOMINAL).signal
    assert sig12.data.shape == (NOMINAL_N_SAMPLES, 12)
    assert sig12.data.dtype.kind == "f"
    assert np.isfinite(sig12.data).all()
    assert sig12.lead_names == CANONICAL_LEADS
    assert sig12.n_samples == NOMINAL_N_SAMPLES


def test_columnas_signal12_corresponden_al_header():
    raw, meta = wfdb.rdsamp(NOMINAL)
    sig12 = ingest_signal(NOMINAL).signal
    header_col = {name: i for i, name in enumerate(meta["sig_name"])}
    assert len(meta["sig_name"]) == 12
    for i, lead in enumerate(CANONICAL_LEADS):
        assert np.allclose(sig12.data[:, i], raw[:, header_col[lead]])


def test_estados_nominal():
    rec = ingest_signal(NOMINAL)
    assert rec.status == "reconocido"
    assert rec.source == "signal"
    assert rec.signal.status == "completo"


def test_metadatos_temporales_y_fisicos():
    rec = ingest_signal(NOMINAL)
    assert rec.fs == NOMINAL_FS
    assert rec.duration_s == NOMINAL_DURATION_S
    assert rec.signal.fs == NOMINAL_FS
    assert rec.signal.duration_s == NOMINAL_DURATION_S


def test_derivaciones_y_flags():
    rec = ingest_signal(NOMINAL)
    assert rec.n_leads_valid == 12
    assert rec.leads_present == CANONICAL_LEADS
    assert set(rec.signal.lead_flags.values()) == {"ok"}
    assert all(rec.signal.lead_flags[k] == "ok" for k in CANONICAL_LEADS)


def test_metadatos_del_header():
    rec = ingest_signal(NOMINAL)
    md = rec.metadata
    assert md["name"] == "nominal"
    assert md["n_leads"] == 12
    assert md["fs"] == NOMINAL_FS
    assert md["n_samples"] == NOMINAL_N_SAMPLES
    assert md["units"] == ["mV"] * 12
    assert md["sig_name"] == CANONICAL_LEADS


def test_record_id_ruta_sin_extension():
    rec = ingest_signal(NOMINAL)
    expected = str(NOMINAL.with_suffix(""))
    assert rec.record_id == expected
    assert rec.signal.record_id == expected


def test_corte_real_ptbxl_nominal_si_presente():
    cut = FIXTURES / "corte_real" / "ptbxl_cut"
    headers = sorted(cut.rglob("*.hea"))
    if not headers:
        import pytest

        pytest.skip("corte real ausente: ejecutar tests/fixtures/corte_real/make_cut.py")
    sample = str(headers[0]).removesuffix(".hea")
    rec = ingest_signal(sample)
    assert rec.status == "reconocido"
    assert rec.signal.status == "completo"
    assert rec.n_leads_valid == 12
    assert rec.signal.data.shape == (NOMINAL_N_SAMPLES, 12)
    assert rec.metadata["units"] == ["mV"] * 12


def test_ingest_degradada_se_marca_degradada_no_rechaza():
    rec = ingest_signal(DEGRADADA)
    assert rec.status == "reconocido"
    assert isinstance(rec.signal, Signal12)
    assert rec.signal.status == "parcial"


def test_ingest_degradada_11_derivaciones_falta_I():
    rec = ingest_signal(DEGRADADA)
    assert rec.n_leads_valid == 11
    assert rec.leads_present == [lead for lead in CANONICAL_LEADS if lead != "I"]
    assert rec.signal.lead_flags["I"] == "missing"
    assert all(
        rec.signal.lead_flags[lead] == "ok"
        for lead in CANONICAL_LEADS
        if lead != "I"
    )
    i_col = CANONICAL_LEADS.index("I")
    assert np.isnan(rec.signal.data[:, i_col]).all()
    assert rec.signal.data.shape == (NOMINAL_N_SAMPLES, 12)


def test_ingest_degradada_resto_nominal():
    rec = ingest_signal(DEGRADADA)
    assert rec.fs == NOMINAL_FS
    assert rec.duration_s == NOMINAL_DURATION_S
    raw_deg, _ = wfdb.rdsamp(DEGRADADA)
    deg_leads = [lead for lead in CANONICAL_LEADS if lead != "I"]
    header_col = {name: i for i, name in enumerate(deg_leads)}
    for lead in deg_leads:
        assert np.allclose(rec.signal.data[:, CANONICAL_LEADS.index(lead)], raw_deg[:, header_col[lead]])


def test_ingest_header_ilegible_se_rechaza_con_mensaje():
    rec = ingest_signal(NO_EKG)
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert rec.metadata["reason"] == REJECTION_REASON


def test_ingest_rechazo_sin_senal_ni_prediccion():
    rec = ingest_signal(NO_EKG)
    assert rec.fs == 0.0
    assert rec.duration_s == 0.0
    assert rec.n_leads_valid == 0
    assert rec.leads_present == []


def test_ingest_ruta_inexistente_se_rechaza_con_mensaje():
    rec = ingest_signal(FIXTURES / "no_existe" / "no_existe")
    assert rec.status == "rechazo"
    assert rec.signal is None
    assert rec.metadata["reason"] == REJECTION_REASON
    assert rec.metadata["error"]


def test_ingest_rechazo_error_con_detalle():
    rec = ingest_signal(NO_EKG)
    assert "error" in rec.metadata
    assert "HeaderSyntaxError" in rec.metadata["error"]