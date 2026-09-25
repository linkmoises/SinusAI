"""Demo Streamlit de taller — contenido exacto (RF-09, T14, carril LG) + aviso
pedagógico permanente y uploads efímeros (RF-10, RF-14, T15).

Muestra únicamente lo que exige RF-09 (contrato de presentación de `docs/
contracts.md` §2.6): el trazado del EKG, los diagnósticos sugeridos (top por eje),
la probabilidad de las etiquetas de los tres ejes (§1.4: 5 superclase + 5 ritmo +
2 específico — las 12 evaluadas; en v1.2 ya no existe la `Prediction8` de v1.0) y
los avisos aplicables cuando apliquen (§2.6.1): `lowconf07b` por clave con p <
0,60, `insuf14` para las claves declaradas-no-evaluadas (§1.4.4) y `quality08`
para entradas degradadas. Nada de features ni gráficas de evaluación (RF-09,
D-08): esa evidencia vive en `reports/`, no en la app.

T15 añade dos piezas:
- RF-10: un aviso pedagógico no-clínico (`render_notice`) se exhibe en TODA
  ejecución, antes de cualquier resultado (predicción, incertidumbre o rechazo).
- RF-14: un cargador de archivos (señal WFDB `.hea`/`.dat` o imagen PNG/JPG). Los
  uploads se escriben a un directorio temporal efímero por sesión, fuera del repo
  (`tempfile`), se eliminan al cerrar la sesión/proceso (`cleanup_uploads` +
  purge en `atexit`) y nunca persisten en disco/repo. El barrido de huérfanos de
  sesiones previas se hace al arrancar (`sweep_stale_uploads`).

La vía completa ingesta→features→política→predicción se integra en T16/T17: en
T15 la carga se procesa hasta la ingesta (M1/M2), se reconoce y se desplega su
trazado. Sin carga, la demo sigue mostrando el caso sintético autónomo de T14
(fixture nominal de `tests/fixtures/` + `Prediction` sintética de `src.policy`).
"""

import atexit
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest_image import ingest_image  # noqa: E402
from src.ingest_signal import ECGRecord, Signal12, ingest_signal  # noqa: E402
from src.labels import RHYTHM_EVALUATED, SPECIFIC_EVALUATED, SUPERCLASS_ORDER  # noqa: E402
from src.policy import (  # noqa: E402
    ALERT_TEXTS,
    INSUF14_TEXT,
    QUALITY08_TEXT,
    Prediction,
    build_prediction,
)

# Ruta de la señal sintética nominal que la demo usa como trazado en T14 (§1.2).
DEMO_RECORD = ROOT / "tests" / "fixtures" / "nominal" / "nominal"

RECHAZO_TEXT = "entrada no reconocible como EKG: no se emite predicción"
INCERTIDUMBRE_TEXT = "extracción de features imposible: no se emiten etiquetas para esta entrada"

# RF-10 (spec §RF-10): carácter pedagógico no-clínico, permanente y visible.
PEDAGOGICAL_NOTICE = (
    "**Herramienta de apoyo pedagógico para el taller de interpretación de EKG.** "
    "No es un diagnóstico clínico ni sustituye la interpretación médica."
)

# RF-14 (contracts §3): uploads efímeros de procesamiento, fuera del repo, nunca
# versionados. Prefijo de directorio, clave de sesión y antigüedad del barrido.
UPLOAD_PREFIX = "sinusai_upload_"
_UPLOAD_DIR_KEY = "_sinusai_upload_dir"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_STALE_AGE_S = 3600.0  # temporales de sesiones previas (huérfanos) al arrancar

# Títulos de sección y caption del top de cada eje en el orden de presentación
# (§2.6): (título, top_caption). El eje 1 (superclase) se muestra siempre; ritmo y
# específico solo junto a él (invariante de presentación).
_AXES = (
    ("Superclase diagnóstica", "Diagnóstico principal"),
    ("Ritmo evaluado", "Ritmo principal"),
    ("Diagnóstico específico evaluado", "Hallazgo principal"),
)


def render_trazado(st, signal12: Signal12) -> None:
    """Dibuja las 12 derivaciones del trazado de `signal12` (RF-09, nunca píxeles)."""
    fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(signal12.data[:, i], lw=0.6, color="black")
        ax.set_ylabel(signal12.lead_names[i], fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(0, signal12.n_samples)
        if i < 11:
            ax.set_xticks([])
    st.pyplot(fig)
    plt.close(fig)


def _prob_line(key: str, prob: float, texts: list[str]) -> str:
    """Línea de una etiqueta con su probabilidad y sus avisos literales (§2.6.1)."""
    line = f"- **{key}** — {prob:.2f}"
    for text in texts:
        line += f" — {text}"
    return line


def render_axis(st, *, title: str, order: list[str], probs: dict, top: str, alerts: dict, top_caption: str) -> None:
    """Eje de predicción: una línea por etiqueta con probabilidad + avisos + top."""
    st.markdown(f"### {title}")
    for key in order:
        texts = [ALERT_TEXTS[code] for code in alerts.get(key, [])]
        st.markdown(_prob_line(key, probs[key], texts))
    st.markdown(f"**{top_caption}: {top}**")


def render_declared(st, declared: list[str]) -> None:
    """Catálogo declarado-no-evaluado (§1.4.4): siempre visible, aviso fijo, sin p."""
    st.markdown("### Patrones declarados — sin soporte en los datos de entrenamiento")
    for key in declared:
        st.markdown(f"- **{key}** — {INSUF14_TEXT}")


def render_prediction(prediction: Prediction, signal12: Signal12 | None = None, st=st) -> None:
    """Renderiza el contenido exacto de RF-09 para un `Prediction` (§2.6).

    `outcome == prediction`: trazado (si hay `Signal12`), los tres ejes que estén
    presentes (el eje 1 siempre — invariante 1; ejes 2/3 solo junto a él —
    invariante 2), el catálogo declarado y el aviso de calidad `quality08` cuando
    aplique. `outcome == incertidumbre`: solo advertencia, sin etiquetas (C9).
    `outcome == rechazo`: mensaje de rechazo, sin predicción (C10). No se muestran
    features ni gráficas de evaluación.
    """
    if prediction.outcome == "rechazo":
        st.error(RECHAZO_TEXT)
        return

    if prediction.outcome == "incertidumbre":
        st.warning(QUALITY08_TEXT)
        st.markdown(INCERTIDUMBRE_TEXT)
        return

    if signal12 is not None:
        render_trazado(st, signal12)

    for code in prediction.prediction_alerts:
        st.warning(ALERT_TEXTS[code])

    # Eje 1 se muestra siempre (invariante 1); ejes 2/3 solo junto a él (invariante 2).
    if prediction.superclass is not None:
        render_axis(
            st,
            title=_AXES[0][0],
            order=SUPERCLASS_ORDER,
            probs=prediction.superclass.probs,
            top=prediction.superclass.top_label,
            alerts=prediction.alerts,
            top_caption=_AXES[0][1],
        )
    if prediction.rhythm is not None:
        render_axis(
            st,
            title=_AXES[1][0],
            order=RHYTHM_EVALUATED,
            probs=prediction.rhythm.probs,
            top=prediction.rhythm.top_label,
            alerts=prediction.alerts,
            top_caption=_AXES[1][1],
        )
    if prediction.specific_diagnostic is not None:
        render_axis(
            st,
            title=_AXES[2][0],
            order=SPECIFIC_EVALUATED,
            probs=prediction.specific_diagnostic.probs,
            top=prediction.specific_diagnostic.top_label,
            alerts=prediction.alerts,
            top_caption=_AXES[2][1],
        )

    render_declared(st, prediction.declared_not_evaluated)


def render_notice(st=st) -> None:
    """RF-10: aviso pedagógico no-clínico, visible en toda ejecución y resultado."""
    st.markdown(PEDAGOGICAL_NOTICE)


def upload_dir(st) -> Path:
    """Directorio temporal efímero por sesión — en temp del sistema, nunca en el repo (RF-14)."""
    key = _UPLOAD_DIR_KEY
    existing = st.session_state.get(key)
    if existing and Path(existing).is_dir():
        return Path(existing)
    tmp = Path(tempfile.mkdtemp(prefix=UPLOAD_PREFIX))
    st.session_state[key] = str(tmp)
    return tmp


def save_upload(st, uploaded, dest_dir: Path | None = None) -> Path:
    """Escribe los bytes del upload a un temporal efímero y devuelve su ruta."""
    tmp = dest_dir or upload_dir(st)
    name = Path(uploaded.name).name  # sanitiza el nombre: descarta cualquier ruta
    path = tmp / name
    path.write_bytes(uploaded.getvalue())
    return path


def cleanup_uploads(st) -> None:
    """Elimina el directorio temporal efímero de la sesión (RF-14).

    Es el teardown del cierre de sesión local: el carril LG lo usa al final de la
    sesión y `atexit` lo hace al cerrar el proceso de la app.
    """
    key = _UPLOAD_DIR_KEY
    tmp = st.session_state.get(key)
    if tmp:
        shutil.rmtree(str(tmp), ignore_errors=True)
        del st.session_state[key]


def sweep_stale_uploads(min_age_s: float = _STALE_AGE_S) -> list[Path]:
    """Borra temporales `sinusai_upload_*` huérfanos de sesiones previas (> `min_age_s`)."""
    removed: list[Path] = []
    now = time.time()
    for tmp in Path(tempfile.gettempdir()).glob(f"{UPLOAD_PREFIX}*"):
        if not tmp.is_dir():
            continue
        try:
            if now - tmp.stat().st_mtime <= min_age_s:
                continue
        except OSError:
            continue
        shutil.rmtree(tmp, ignore_errors=True)
        removed.append(tmp)
    return removed


def _atexit_purge_uploads() -> None:
    """Al terminar el proceso (cerrar la app) purga todos los temporales de la demo."""
    for tmp in Path(tempfile.gettempdir()).glob(f"{UPLOAD_PREFIX}*"):
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)


atexit.register(_atexit_purge_uploads)


def _ingest_upload(paths: list[Path]) -> ECGRecord:
    """Dirige la ingesta por tipo de archivo: imagen PNG/JPG o señal WFDB.

    Una carga mixta con imagen tiene prioridad de imagen; sin `.hea`/`.dat` (ni
    imagen) la carga no es reconocible y se rechaza (RF-08b).
    """
    images = [p for p in paths if p.suffix.lower() in _IMAGE_SUFFIXES]
    if images:
        return ingest_image(images[0])
    hea = [p for p in paths if p.suffix.lower() == ".hea"]
    if not hea:
        return ECGRecord(
            record_id="upload",
            source="signal",
            status="rechazo",
            fs=0.0,
            duration_s=0.0,
            n_leads_valid=0,
            leads_present=[],
            metadata={"reason": RECHAZO_TEXT},
            signal=None,
        )
    return ingest_signal(hea[0].with_suffix(""))


def _render_upload(uploads, st) -> None:
    """Guarda la carga en un temporal efímero e ingesta para reconocerla (T15, RF-14).

    En T15 la carga (señal o imagen) se procesa hasta la ingesta (M1/M2): si es
    reconocible se muestra su trazado; si no, se rechaza. La vía completa
    ingesta→features→política→predicción se integra en T16/T17. Los archivos se
    eliminan al cerrar la sesión y no persisten en disco ni en el repo.
    """
    tmp = upload_dir(st)
    paths = [save_upload(st, uploaded, tmp) for uploaded in uploads]
    record = _ingest_upload(paths)

    if record.status == "rechazo":
        st.error(RECHAZO_TEXT)
        return

    st.markdown(
        "**Carga recibida** — procesada solo en un directorio temporal efímero: "
        "los archivos se eliminan al cerrar la sesión y no persisten en disco ni "
        "en el repositorio (RF-14)."
    )
    if record.signal is not None:
        render_trazado(st, record.signal)
    st.markdown(
        f"Registro: `{record.record_id}` · derivaciones válidas: "
        f"{record.n_leads_valid}/12 · duración: {record.duration_s:.1f} s · "
        f"origen: {record.source}"
    )


def _demo_prediction() -> Prediction:
    """Prediction sintética del caso nominal de la demo (T14, sin pipeline aún)."""
    superclass_probs = {"NORM": 0.92, "MI": 0.35, "STTC": 0.50, "CD": 0.20, "HYP": 0.10}
    rhythm_probs = {"SR": 0.85, "AFIB": 0.40, "STACH": 0.25, "SARRH": 0.15, "SBRAD": 0.60}
    specific_probs = {"ASMI": 0.55, "1AVB": 0.30}
    return build_prediction(
        "demo_nominal",
        superclass_probs,
        rhythm_probs,
        specific_probs,
        quality="nominal",
    )


def _demo_signal() -> Signal12 | None:
    """Señal del fixture nominal para el trazado; `None` si el fixture falta."""
    rec = ingest_signal(DEMO_RECORD)
    return rec.signal if rec.status == "reconocido" else None


def _main() -> None:
    render_notice(st)  # RF-10: antes de cualquier resultado, siempre visible
    sweep_stale_uploads()  # RF-14: huérfanos de sesiones previas
    uploads = st.file_uploader(
        "Carga un EKG para la demo (señal WFDB .hea/.dat o imagen PNG/JPG)",
        accept_multiple_files=True,
    )
    if uploads:
        _render_upload(uploads, st)
    else:
        render_prediction(_demo_prediction(), _demo_signal(), st)


if __name__ == "__main__":
    _main()