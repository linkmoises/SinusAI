"""Tests del vector clínico cerrado (RF-03/RF-04/RF-08, tareas T07–T08, carril LC).

- T07 (nominal): de la `Signal12` de `tests/fixtures/nominal/nominal` (serie temporal
  1D, nunca píxeles — RF-03) se extrae el vector cerrado de 8 features de
  `docs/contracts.md` §2.4.1 (QRS, QT/QTc, FC, eje QRS y HRV) con
  `FeatureVector.status == "completo"`.
- T08 (degradación): la fixture degradada (`tests/fixtures/degradada/degradada`,
  derivación I faltante) da `status == "parcial"` sin ocultar el resto de features, y
  la de fallo total (`tests/fixtures/fallo_total/fallo_total`, trazados planos) da
  `status == "imposible"` con `values` vacíos — sin probabilidades que rankear, por
  lo que RF-07c (top-1 por umbral) no aplica (§2.1, §2.6.1, RF-08).
"""

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FEATURE_KEYS,
    FeatureVector,
    extract_features,
)
from src.ingest_signal import CANONICAL_LEADS, ingest_signal  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOMINAL = FIXTURES / "nominal" / "nominal"
DEGRADADA = FIXTURES / "degradada" / "degradada"
FALLO_TOTAL = FIXTURES / "fallo_total" / "fallo_total"


def _sig12():
    return ingest_signal(NOMINAL).signal


def test_extract_features_devuelve_featurevector():
    fv = extract_features(_sig12())
    assert isinstance(fv, FeatureVector)


def test_vector_cerrado_completo_con_las_8_features():
    fv = extract_features(_sig12())
    assert set(fv.values) == set(FEATURE_KEYS)
    assert len(fv.values) == 8
    assert fv.missing_features == []
    assert fv.status == "completo"


def test_record_id_se_propaga():
    fv = extract_features(_sig12())
    assert fv.record_id == str(NOMINAL.with_suffix(""))


def test_solo_depende_de_la_senal_no_del_record_id():
    fv = extract_features(_sig12())
    renamed = replace(_sig12(), record_id="renombrado")
    assert extract_features(renamed).values == fv.values


def test_determinista():
    a = extract_features(_sig12()).values
    b = extract_features(_sig12()).values
    assert a == b


def test_valores_finitos_y_orden_canonico():
    fv = extract_features(_sig12())
    assert list(fv.values) == FEATURE_KEYS
    assert all(np.isfinite(x) for x in fv.values.values())


def test_rangos_fisiologicos_plausibles():
    v = extract_features(_sig12()).values
    assert 60 <= v["hr_bpm"] <= 85  # fixture simulada a 72 bpm
    assert 60 <= v["qrs_duration_ms"] <= 220
    assert 250 <= v["qt_ms"] <= 600
    assert 300 <= v["qtc_ms"] <= 650
    assert 0 <= v["qrs_axis_deg"] <= 90
    assert 0 <= v["hrv_sdnn_ms"] <= 300
    assert 0 <= v["hrv_rmssd_ms"] <= 300
    assert 0 <= v["hrv_pnn50_pct"] <= 100


def test_fc_coherente_con_los_picos_r():
    sig = _sig12()
    _, info = nk.ecg_process(sig.data[:, CANONICAL_LEADS.index("II")], sampling_rate=sig.fs)
    rr_ms = np.diff(np.asarray(info["ECG_R_Peaks"])) / sig.fs * 1000.0
    v = extract_features(sig).values
    assert v["hr_bpm"] == pytest.approx(60000.0 / np.mean(rr_ms), rel=1e-6)


def test_qtc_bazett_con_rr_medio():
    sig = _sig12()
    _, info = nk.ecg_process(sig.data[:, CANONICAL_LEADS.index("II")], sampling_rate=sig.fs)
    rr_s = np.mean(np.diff(np.asarray(info["ECG_R_Peaks"])) / sig.fs)
    v = extract_features(sig).values
    assert v["qtc_ms"] == pytest.approx(v["qt_ms"] / np.sqrt(rr_s), rel=1e-6)


def test_qrs_y_qt_coherentes_con_la_delineacion():
    sig = _sig12()
    _, info = nk.ecg_process(sig.data[:, CANONICAL_LEADS.index("II")], sampling_rate=sig.fs)
    r_on = np.asarray(info["ECG_R_Onsets"], dtype=float)
    r_off = np.asarray(info["ECG_R_Offsets"], dtype=float)
    t_off = np.asarray(info["ECG_T_Offsets"], dtype=float)
    delin = (r_on > 0) & (r_off > 0) & (t_off > 0)
    v = extract_features(sig).values
    assert v["qrs_duration_ms"] == pytest.approx(
        np.median((r_off[delin] - r_on[delin]) / sig.fs * 1000.0), rel=1e-6
    )
    assert v["qt_ms"] == pytest.approx(
        np.median((t_off[delin] - r_on[delin]) / sig.fs * 1000.0), rel=1e-6
    )


def test_entrada_es_signal12_1d_no_imagen():
    sig = inspect.signature(extract_features)
    assert list(sig.parameters) == ["signal12"]


# ---------------------------------------------------------------------------
# T08 — estados parcial/imposible (RF-08)
# ---------------------------------------------------------------------------


def test_degradada_da_estado_parcial_con_eje_faltante():
    rec = ingest_signal(DEGRADADA)
    assert rec.signal.status == "parcial"  # ingesta ya la degrada (falta derivación I)
    fv = extract_features(rec.signal)
    assert fv.status == "parcial"
    assert len(fv.values) >= 1  # parcial: al menos una feature presente
    assert fv.missing_features == ["qrs_axis_deg"]
    assert set(fv.missing_features) <= set(FEATURE_KEYS)


def test_degradada_no_oculta_el_resto_de_features():
    sig = ingest_signal(DEGRADADA).signal
    fv = extract_features(sig)
    assert set(fv.values) == set(FEATURE_KEYS) - {"qrs_axis_deg"}
    assert all(np.isfinite(x) for x in fv.values.values())
    assert fv.values["hr_bpm"] >= 1.0


def test_degradada_record_id_se_propaga():
    fv = extract_features(ingest_signal(DEGRADADA).signal)
    assert fv.record_id == str(DEGRADADA.with_suffix(""))


def test_fallo_total_da_imposible_sin_valores():
    fv = extract_features(ingest_signal(FALLO_TOTAL).signal)
    assert fv.status == "imposible"
    assert fv.values == {}
    assert fv.missing_features == FEATURE_KEYS


def test_imposible_bloquea_rf_07c_porque_no_hay_probabilidades():
    fv = extract_features(ingest_signal(FALLO_TOTAL).signal)
    # RF-07c (top-1 si ninguna p supera el umbral) requiere probabilidades; el
    # vector imposible no expone ninguna (contracts §2.1, §2.6.1), así la política
    # aguas abajo no puede rankear ni mostrar top-1.
    assert fv.status == "imposible"
    assert fv.values == {}

    parcial = extract_features(ingest_signal(DEGRADADA).signal)
    assert parcial.status == "parcial"
    assert parcial.values  # sí hay medidas, luego la política puede predecir con aviso