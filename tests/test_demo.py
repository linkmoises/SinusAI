"""Tests de la demo Streamlit — contenido exacto de RF-09 (T14, carril LG) + aviso
pedagógico permanente y uploads efímeros (RF-10, RF-14 — T15, carril LG).

T14 verifica el «Hecho cuando»: `streamlit run app.py` muestra trazado +
probabilidades de las etiquetas de los tres ejes + avisos aplicables y nada de
features ni gráficas de evaluación, validado con `Prediction` sintética.

T15 verifica el «Hecho cuando» propio: el aviso pedagógico no-clínico (RF-10) es
visible siempre (predicción y rechazo incluidos), y los uploads de la demo se
guardan en temporales efímeros fuera del repo que se eliminan al cerrar la sesión
sin persistir en disco ni en el repo (RF-14).

La salida se expone según contracts v1.2 (§1.4, §2.6): tres ejes independientes
(5 superclase + 5 ritmo + 2 específico = 12 probabilidades evaluadas). La
`Prediction8` de v1.0 quedó supersedida (contracts §2.6 y la nota de consumo de
T14); la app muestra siempre todas las probabilidades de los ejes presentes
(RF-06) y el catálogo declarado-no-evaluado con `insuf14` (§1.4.4, RF-07a):

- C1 nominal: sin avisos salvo el catálogo declarado.
- C2 p < 0,60 → `lowconf07b` sobre cada clave afectada (RF-07b), texto literal
  `confianza baja` (§2.6.1).
- C6/C8 entrada degradada → `quality08` a nivel de `Prediction` (RF-08 prevalece
  sobre el umbral) y, si hay claves con p < 0,60, ambos avisos se apilan (RF-07d).
- D1–D3 declaradas: siempre visibles con `insuf14` y sin probabilidad calculada.
- C9 fallo total → `outcome == incertidumbre`, sin etiquetas, con aviso de calidad.
- C10 rechazo → mensaje de rechazo sin predicción.

Los tests de render usan un grabador que sustituye la API de Streamlit
(`st.markdown`/`st.warning`/`st.error`/`st.pyplot`) para inspeccionar el texto
emitido; el de integración lanza `app.py` de verdad con `AppTest`.
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from app import (  # noqa: E402
    INCERTIDUMBRE_TEXT,
    PEDAGOGICAL_NOTICE,
    RECHAZO_TEXT,
    UPLOAD_PREFIX,
    _UPLOAD_DIR_KEY,
    cleanup_uploads,
    render_notice,
    render_prediction,
    sweep_stale_uploads,
    upload_dir,
)
from src.features import FEATURE_KEYS  # noqa: E402
from src.ingest_signal import ingest_signal  # noqa: E402
from src.labels import (  # noqa: E402
    DECLARED_CATALOG,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
)
from src.policy import (  # noqa: E402
    INSUF14_TEXT,
    LOWCONF_TEXT,
    QUALITY08_TEXT,
    THRESHOLD,
    Prediction,
    build_prediction,
)

APP = ROOT / "app.py"
NOMINAL = ROOT / "tests" / "fixtures" / "nominal" / "nominal"
NOMINAL_HEA = NOMINAL.with_suffix(".hea")
NOMINAL_DAT = NOMINAL.with_suffix(".dat")
NO_EKG_HEA = ROOT / "tests" / "fixtures" / "no_ekg" / "no_ekg.hea"
NO_EKG_DAT = ROOT / "tests" / "fixtures" / "no_ekg" / "no_ekg.dat"
TOY_12_LEAD = ROOT / "tests" / "fixtures" / "imagen" / "toy_ecg_12_lead.png"

# Prohibidos por RF-09: ni features (contracts §2.4.1) ni gráficas de evaluación.
BANNED = FEATURE_KEYS + ["ROC", "AUC", "matriz de confusión", "importancia"]


class _Recorder:
    """Sustituto de la API de Streamlit que captura el contenido renderizado."""

    def __init__(self):
        self.markdowns: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.figs: list = []

    def markdown(self, value, **kwargs):
        self.markdowns.append(str(value))

    def warning(self, value, **kwargs):
        self.warnings.append(str(value))

    def error(self, value, **kwargs):
        self.errors.append(str(value))

    def pyplot(self, fig, **kwargs):
        self.figs.append(fig)


def _probs(order, **overrides):
    values = {key: 0.90 for key in order}
    values.update(overrides)
    return values


def _super(**overrides):
    return _probs(SUPERCLASS_ORDER, **overrides)


def _rhyth(**overrides):
    return _probs(RHYTHM_EVALUATED, **overrides)


def _spec(**overrides):
    return _probs(SPECIFIC_EVALUATED, **overrides)


def _render(prediction, signal=None):
    recorder = _Recorder()
    render_prediction(prediction, signal, recorder)
    return recorder


def _text(recorder):
    return "\n".join(recorder.markdowns)


# ---------------------------------------------------------------------------
# T14 «Hecho cuando» — la app corre y muestra el contenido exacto de RF-09
# ---------------------------------------------------------------------------


def test_app_muestra_trazado_probabilidades_avisos_y_nada_mas():
    at = AppTest.from_file(APP).run()
    assert not at.exception

    # trazado: exactamente una figura de 12 derivaciones (RF-09); en AppTest el
    # `st.pyplot` se expone como elemento de tipo `image` (elemento `imgs`).
    assert len(at.image) == 1
    assert "image" in [e.type for e in at.image]

    text = "\n".join(e.value for e in at.markdown)
    # probabilidad de las 12 etiquetas evaluadas de los tres ejes (RF-06)
    for key in SUPERCLASS_ORDER + RHYTHM_EVALUATED + SPECIFIC_EVALUATED:
        assert f"**{key}**" in text, key
    # top por eje
    assert "**Diagnóstico principal: NORM**" in text
    assert "**Ritmo principal: SR**" in text
    assert "**Hallazgo principal: ASMI**" in text

    # avisos aplicables: insuf14 del catálogo declarado (RF-07a, AGENTS regla 3)
    assert INSUF14_TEXT in text
    assert "**VT**" in text and "**3AVB**" in text

    # contenido exacto: nada de features ni gráficas de evaluación
    for banned in BANNED:
        assert banned not in text, banned


def test_app_nominal_sin_errores_ni_rechazo():
    at = AppTest.from_file(APP).run()
    assert [e.value for e in at.warning] == []
    assert [e.value for e in at.error] == []


# ---------------------------------------------------------------------------
# Render con Prediction sintética — avisos por clave (RF-07b, §2.6.1)
# ---------------------------------------------------------------------------


def test_render_lowconf07b_por_clave_con_prediction_sintetica():
    pred = build_prediction(
        "rec_01",
        _super(MI=0.30),
        _rhyth(AFIB=0.45),
        _spec(**{"1AVB": 0.40}),
        quality="nominal",
    )
    rec = _render(pred)
    lines = _text(rec)
    assert "**MI** — 0.30 — confianza baja" in lines
    assert "**AFIB** — 0.45 — confianza baja" in lines
    assert "**1AVB** — 0.40 — confianza baja" in lines
    # solo las claves con p < 0,60 llevan el aviso (RF-07b evaluado por clave)
    assert "**NORM** — 0.90 — confianza baja" not in lines
    assert "**SR** — 0.90 — confianza baja" not in lines
    assert LOWCONF_TEXT == "confianza baja"  # texto literal (AGENTS regla 3)


def test_render_umbral_060_no_avisa_y_059_avisa():
    rec_ok = _render(build_prediction("rec_01", _super(NORM=0.60)))
    assert "confianza baja" not in _text(rec_ok)
    rec_baja = _render(build_prediction("rec_01", _super(NORM=0.59)))
    assert "**NORM** — 0.59 — confianza baja" in _text(rec_baja)
    assert THRESHOLD == 0.60


# ---------------------------------------------------------------------------
# RF-08 prevalece sobre el umbral — aviso de calidad (T11, casos C6/C8)
# ---------------------------------------------------------------------------


def test_render_degradada_quality08_sin_fingir_confianza():
    pred = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="degraded")
    rec = _render(pred)
    # p alta en todas las claves → sin lowconf, pero presentada como no confiable
    assert "confianza baja" not in _text(rec)
    assert QUALITY08_TEXT == "entrada degradada: predicción no confiable"
    assert QUALITY08_TEXT in rec.warnings


def test_render_degradada_mas_lowconf_apilados_rf07d():
    pred = build_prediction("rec_01", _super(MI=0.30), quality="degraded")
    rec = _render(pred)
    assert QUALITY08_TEXT in rec.warnings  # RF-08 a nivel de Prediction
    # y el lowconf no se suprime (RF-07d): ambos avisos presentes
    assert "**MI** — 0.30 — confianza baja" in _text(rec)


# ---------------------------------------------------------------------------
# Catálogo declarado-no-evaluado (RF-07a, casos D1–D3)
# ---------------------------------------------------------------------------


def test_render_declaradas_insuf14_siempre_y_nunca_con_probabilidad():
    pred = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="nominal")
    rec = _render(pred)
    lines = _text(rec)
    for key in DECLARED_CATALOG:
        assert f"- **{key}** — {INSUF14_TEXT}" in lines, key
        assert f"**{key}** — {0.0}" not in lines and f"**{key}** — 0." not in lines
        assert INSUF14_TEXT == "dato insuficiente en el set de entrenamiento"


# ---------------------------------------------------------------------------
# Fallo total (C9) y rechazo (C10)
# ---------------------------------------------------------------------------


def test_render_incertidumbre_sin_etiquetas():
    pred = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="impossible")
    assert pred.outcome == "incertidumbre"
    rec = _render(pred)
    assert QUALITY08_TEXT in rec.warnings  # aviso de calidad/incertidumbre
    assert _text(rec) == INCERTIDUMBRE_TEXT  # sin etiquetas ni ejes
    for banned in BANNED + SUPERCLASS_ORDER + RHYTHM_EVALUATED + SPECIFIC_EVALUATED:
        assert banned not in _text(rec), banned


def test_render_rechazo_sin_prediccion():
    pred = Prediction(
        record_id="no_ekg",
        outcome="rechazo",
        quality="",
        superclass=None,
        rhythm=None,
        specific_diagnostic=None,
        declared_not_evaluated=[],
        alerts={},
        prediction_alerts=[],
    )
    rec = _render(pred)
    assert rec.markdowns == []
    assert rec.errors and RECHAZO_TEXT in rec.errors[0]


# ---------------------------------------------------------------------------
# Trazado
# ---------------------------------------------------------------------------


def test_render_trazado_con_signal12_12_derivaciones():
    signal = ingest_signal(NOMINAL).signal
    assert signal is not None and signal.data.shape[1] == 12
    rec = _render(build_prediction("rec_01", _super()), signal)
    assert len(rec.figs) == 1
    assert len(rec.figs[0].axes) == 12


def test_render_sin_signal12_no_dibuja_trazado():
    rec = _render(build_prediction("rec_01", _super()))
    assert rec.figs == []


# ---------------------------------------------------------------------------
# T15 — RF-10: aviso pedagógico permanente y visible
# ---------------------------------------------------------------------------


def test_render_notice_muestra_el_texto_pedagogico():
    rec = _Recorder()
    render_notice(rec)
    assert PEDAGOGICAL_NOTICE in rec.markdowns
    # RF-10: la demo se declara apoyo pedagógico y NO diagnóstico clínico
    assert "pedagógico" in PEDAGOGICAL_NOTICE.lower()
    assert "no es un diagnóstico clínico" in PEDAGOGICAL_NOTICE.lower()


def test_app_aviso_pedagogico_visible_siempre():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    text = "\n".join(e.value for e in at.markdown)
    assert PEDAGOGICAL_NOTICE in text
    # no es un aviso de alerta de calidad: la corrida nominal sigue sin warnings
    assert [e.value for e in at.warning] == []
    assert [e.value for e in at.error] == []


def test_app_aviso_pedagogico_visible_tambien_en_rechazo():
    """El aviso precede a cualquier resultado: presente aun en rechazo (C10)."""
    at = AppTest.from_file(APP).run()
    at.file_uploader[0].set_value(
        [
            ("no_ekg.hea", NO_EKG_HEA.read_bytes(), "text/plain"),
            ("no_ekg.dat", NO_EKG_DAT.read_bytes(), "application/octet-stream"),
        ]
    ).run()
    assert not at.exception
    text = "\n".join(e.value for e in at.markdown)
    assert PEDAGOGICAL_NOTICE in text
    assert [e.value for e in at.error] and RECHAZO_TEXT in at.error[0].value


# ---------------------------------------------------------------------------
# T15 — RF-14: uploads efímeros, fuera del repo, eliminados al cerrar sesión
# ---------------------------------------------------------------------------


def _session_upload(files):
    """AppTest con una carga de archivos seteada y sus archivos en el temporal."""
    at = AppTest.from_file(APP).run()
    at.file_uploader[0].set_value(files).run()
    assert not at.exception
    tmp = Path(at.session_state[_UPLOAD_DIR_KEY])
    assert tmp.is_dir() and ROOT not in tmp.parents  # RF-14: nunca dentro del repo
    assert tmp.name.startswith(UPLOAD_PREFIX)
    return at, tmp


def test_upload_signal_va_a_temporal_efimero_fuera_del_repo():
    at, tmp = _session_upload(
        [
            ("nominal.hea", NOMINAL_HEA.read_bytes(), "text/plain"),
            ("nominal.dat", NOMINAL_DAT.read_bytes(), "application/octet-stream"),
        ]
    )
    try:
        assert (tmp / "nominal.hea").exists() and (tmp / "nominal.dat").exists()
    finally:
        cleanup_uploads(at)


def test_upload_imagen_va_a_temporal_efimero_fuera_del_repo():
    at, tmp = _session_upload(("toy_12.png", TOY_12_LEAD.read_bytes(), "image/png"))
    try:
        assert (tmp / "toy_12.png").exists()
        # la vía imagen ingesta a Signal12 y la demo dibuja su trazado (T15)
        assert any("Carga recibida" in m.value for m in at.markdown)
    finally:
        cleanup_uploads(at)


def test_uploads_se_eliminan_al_cerrar_sesion():
    at, tmp = _session_upload(
        [
            ("nominal.hea", NOMINAL_HEA.read_bytes(), "text/plain"),
            ("nominal.dat", NOMINAL_DAT.read_bytes(), "application/octet-stream"),
        ]
    )
    assert (tmp / "nominal.hea").exists()
    cleanup_uploads(at)  # teardown de cierre de sesión
    assert not tmp.exists()
    assert _UPLOAD_DIR_KEY not in at.session_state


def test_upload_dir_reusado_dentro_de_la_misma_sesion():
    at = AppTest.from_file(APP).run()
    first = upload_dir(at)
    second = upload_dir(at)
    assert first == second  # RF-14: un solo temporal por sesión
    cleanup_uploads(at)
    assert not first.exists()


def test_sweep_elimina_huerfanos_viejos_pero_no_recientes():
    tdir = Path(tempfile.gettempdir())
    stale = tdir / f"{UPLOAD_PREFIX}huerfano_{int(time.time() * 1000)}"
    stale.mkdir()
    (stale / "leftover.bin").write_bytes(b"leftover")
    older = time.time() - 7200
    os.utime(stale, (older, older))
    fresh = Path(tempfile.mkdtemp(prefix=UPLOAD_PREFIX))
    try:
        removed = sweep_stale_uploads(min_age_s=3600)
        assert stale in removed and not stale.exists()  # huérfano viejo: eliminado
        assert fresh.is_dir()  # temporal reciente: no se toca
    finally:
        shutil.rmtree(stale, ignore_errors=True)
        shutil.rmtree(fresh, ignore_errors=True)