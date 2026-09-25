"""Evaluación uno-vs-resto por etiqueta (RF-11, RF-12, T13, carril LF).

A partir de las predicciones de una corrida — `y_true` (binario, una columna por
target evaluado de contracts §1.4) y `y_score` (probabilidad en [0,1] por target)
— genera en `reports/` (contracts §3) los artefactos de evaluación que exige
RF-11:

  - Matriz de confusión por etiqueta (umbral de decisión 0,50).
  - Curva ROC/AUC por cada una de las 12 salidas evaluadas (uno-vs-resto).
  - Importancia de variables por etiqueta: el entrenamiento usa un clasificador
    binario por salida (`MultiOutputClassifier`, contracts §2.7
    `architecture`), así que la importancia se expone una por etiqueta,
    derivada de los `feature_importances_` de cada estimator.

Además escribe `evaluation_metadata.json` con la versión de PTB-XL, el split y
la seed de la corrida junto a las métricas por etiqueta (RF-11), y registra las
declaradas-no-evaluadas de §1.4.4 como «no evaluadas por diseño» para no ocultar
el punto ciego (RF-12). La versión/split/seed también se incrusta en el
encabezado de cada figura.

Sin suavizar (RF-12): si una etiqueta carece de muestras suficientes en test
para una curva ROC fiable (`MIN_TEST_POSITIVES`/`MIN_TEST_NEGATIVES`), la
gráfica lo indica explícitamente en lugar de omitirla y la etiqueta se registra
en `TrainingRun.rare_insufficient_test` (contracts §2.7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: compatibilidad con CI/laptop sin display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from sklearn.metrics import auc as _sk_auc
from sklearn.metrics import confusion_matrix, roc_curve

from src.labels import DECLARED_CATALOG

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"  # contracts §3

# Umbral de decisión para la matriz de confusión (separación binaria de p).
DECISION_THRESHOLD = 0.50
# Muestras mínimas por clase en test para considerar fiable una curva ROC/AUC.
MIN_TEST_POSITIVES = 5
MIN_TEST_NEGATIVES = 5
# Claves de metadatos de corrida que RF-11 exige junto a las gráficas (§2.7).
_METADATA_KEYS = ("ptbxl_version", "split", "seed")


def _to_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} debe ser (n_registros, n_targets), se obtuvo forma {arr.shape}"
        )
    return arr


def per_label_metrics(
    y_true, y_score, threshold: float = DECISION_THRESHOLD
) -> dict[int, dict]:
    """Métricas uno-vs-resto por etiqueta (colección ordenada por columna).

    Por cada target devuelve `{n_pos, n_neg, tn, fp, fn, tp, auc,
    insufficient}`. La matriz se calcula con `y_score >= threshold` y el AUC
    desde las probabilidades. `auc` es `None` e `insufficient` es `True` cuando
    el target no tiene ambas clases en test (no existe curva fiable) o cuando
    los positivos/negativos de test no alcanzan el mínimo de fiabilidad (RF-11):
    la gráfica lo nota explícitamente en lugar de omitir la etiqueta.
    """
    y_true = _to_array(y_true, "y_true")
    y_score = _to_array(y_score, "y_score")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true y y_score deben tener la misma forma; se obtuvo "
            f"{y_true.shape} y {y_score.shape}"
        )
    y_pred = (y_score >= float(threshold)).astype(int)
    metrics: dict[int, dict] = {}
    for j in range(y_true.shape[1]):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j]
        ys = y_score[:, j]
        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        auc = None
        if n_pos > 0 and n_neg > 0:
            fpr, tpr, _ = roc_curve(yt, ys)
            auc = float(_sk_auc(fpr, tpr))
        insufficient = (
            n_pos < MIN_TEST_POSITIVES or n_neg < MIN_TEST_NEGATIVES or auc is None
        )
        metrics[j] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "auc": auc,
            "insufficient": insufficient,
        }
    return metrics


def _roc_data(y_true_col: np.ndarray, y_score_col: np.ndarray) -> tuple:
    fpr, tpr, _ = roc_curve(y_true_col, y_score_col)
    return fpr, tpr


def _set_metadata(fig, metadata: dict | None) -> None:
    """Incrusta versión/split/seed de la corrida en el encabezado de la figura."""
    if not metadata:
        return
    bits = [f"{key}: {metadata[key]}" for key in _METADATA_KEYS if metadata.get(key)]
    if bits:
        fig.suptitle("  |  ".join(bits), fontsize=8, wrap=True)


def _save_figure(fig, out_path: Path) -> Path:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid_shape(n: int) -> tuple[int, int]:
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    return rows, cols


def plot_confusion_matrices(
    metrics: dict,
    target_labels: list[str],
    out_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Matrices de confusión por etiqueta en una sola figura (RF-11, RF-12)."""
    target_labels = list(target_labels)
    n = len(target_labels)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3.2, nrows * 3.0))
    for j, (label, ax) in enumerate(zip(target_labels, axes.ravel())):
        m = metrics[j]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        vmax = max(1, int(cm.max()))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["real 0", "real 1"], fontsize=7)
        ax.set_xlabel(
            f"n_pos={m['n_pos']}, n_neg={m['n_neg']}", fontsize=6
        )
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > vmax / 2 else "black"
                ax.text(c, r, f"{cm[r, c]}", ha="center", va="center", color=color)
    _set_metadata(fig, metadata)
    return _save_figure(fig, Path(out_path))


def plot_roc_curves(
    y_true,
    y_score,
    metrics: dict,
    target_labels: list[str],
    out_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Curvas ROC/AUC por etiqueta (uno-vs-resto); nota explícita de insuficiencia.

    Si una etiqueta no tiene muestras suficientes en test para una curva fiable,
    su subplot indica el motivo explícitamente en lugar de omitirse (RF-11).
    """
    y_true = _to_array(y_true, "y_true")
    y_score = _to_array(y_score, "y_score")
    target_labels = list(target_labels)
    n = len(target_labels)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3.2, nrows * 3.0))
    for j, (label, ax) in enumerate(zip(target_labels, axes.ravel())):
        m = metrics[j]
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        if m["auc"] is not None:
            fpr, tpr = _roc_data(y_true[:, j], y_score[:, j])
            ax.plot(fpr, tpr, lw=1.4, label=f"AUC = {m['auc']:.3f}")
            ax.legend(loc="lower right", fontsize=7)
        if m["insufficient"]:
            ax.text(
                0.5,
                0.9,
                "muestras insuficientes en test para curva ROC fiable\n"
                f"(n_pos={m['n_pos']}, n_neg={m['n_neg']})",
                ha="center",
                va="top",
                fontsize=7,
                color="crimson",
                transform=ax.transAxes,
            )
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    _set_metadata(fig, metadata)
    return _save_figure(fig, Path(out_path))


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    target_labels: list[str],
    out_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Importancia de variables por etiqueta (una por clasificador por salida).

    `importance` es (n_targets, n_features) alineado con `target_labels` y
    `feature_names`; cada fila suma ~1 (importancias de un Random Forest).
    """
    importance = np.asarray(importance, dtype=float)
    target_labels = list(target_labels)
    feature_names = list(feature_names)
    if importance.shape != (len(target_labels), len(feature_names)):
        raise ValueError(
            f"importance debe ser ({len(target_labels)}, {len(feature_names)}), "
            f"se obtuvo {importance.shape}"
        )
    n = len(target_labels)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3.4, nrows * 2.6))
    for j, (label, ax) in enumerate(zip(target_labels, axes.ravel())):
        row = importance[j]
        order = np.argsort(row)
        ax.barh(range(len(feature_names)), row[order], color="steelblue")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=6)
        ax.set_title(label, fontsize=10)
    _set_metadata(fig, metadata)
    return _save_figure(fig, Path(out_path))


def importance_from_model(model, feature_names: list[str]) -> np.ndarray:
    """Importancia por etiqueta desde un `MultiOutputClassifier` (per-classifier).

    `model.estimators_` contiene un estimador binario por salida; cada uno expone
    `feature_importances_` (contracts §2.7 `architecture`). Devuelve
    (n_targets, n_features) en el orden de `model` (el de `EVALUATED_TARGETS`).
    """
    try:
        estimators = model.estimators_
    except AttributeError as exc:
        raise ValueError(
            "el modelo no expone estimators_; se espera MultiOutputClassifier "
            "con un clasificador binario por salida"
        ) from exc
    rows: list[np.ndarray] = []
    for estimator in estimators:
        try:
            fi = np.asarray(estimator.feature_importances_, dtype=float)
        except AttributeError as exc:
            raise ValueError(
                "el estimator no expone feature_importances_; se espera un "
                "estimador de árbol (Random Forest)"
            ) from exc
        if fi.shape != (len(feature_names),):
            raise ValueError(
                f"feature_importances_ {fi.shape} no coincide con "
                f"feature_names ({len(feature_names)} features)"
            )
        rows.append(fi)
    return np.asarray(rows)


def evaluate_run(
    y_true,
    y_score,
    target_labels: list[str],
    feature_importance: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    metadata: dict | None = None,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict:
    """Evalúa uno-vs-resto y escribe los artefactos de RF-11 en `reports_dir`.

    Parámetros:
      y_true: (n_registros, n_targets) binario; y_score: mismas dimensiones, en
        [0,1]; ambas alineadas a `target_labels` (orden de `EVALUATED_TARGETS`,
        contracts §1.4).
      feature_importance: (n_targets, n_features) opcional; si es `None` no se
        genera la gráfica de importancia (el resto sí).
      feature_names: nombres de las features (§2.4.1); requerido junto con
        `feature_importance`.
      metadata: dict con `ptbxl_version`, `split`, `seed` (contracts §2.7).
      reports_dir: directorio de salida (default `reports/`, contracts §3).

    Devuelve `{metrics, rare_insufficient_test, report_artifacts, figures,
    metadata_path}` listo para documentarse en el `TrainingRun` (§2.7).
    """
    target_labels = list(target_labels)
    y_true = _to_array(y_true, "y_true")
    y_score = _to_array(y_score, "y_score")
    if y_true.shape[1] != len(target_labels):
        raise ValueError(
            f"y_true tiene {y_true.shape[1]} columnas pero target_labels "
            f"define {len(target_labels)} salidas"
        )
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true y y_score deben tener la misma forma; se obtuvo "
            f"{y_true.shape} y {y_score.shape}"
        )
    if feature_importance is not None:
        if feature_names is None:
            raise ValueError("feature_names es requerido junto con feature_importance")
        importance = np.asarray(feature_importance, dtype=float)
        if importance.shape != (len(target_labels), len(feature_names)):
            raise ValueError(
                f"feature_importance debe ser ({len(target_labels)}, "
                f"{len(feature_names)}), se obtuvo {importance.shape}"
            )
    else:
        importance = None

    metrics = per_label_metrics(y_true, y_score)
    rare_insufficient_test = [
        label for j, label in enumerate(target_labels) if metrics[j]["insufficient"]
    ]

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})

    figures = {
        "confusion_matrix": plot_confusion_matrices(
            metrics,
            target_labels,
            out_dir / "confusion_matrix.png",
            metadata=meta,
        ),
        "roc_auc": plot_roc_curves(
            y_true,
            y_score,
            metrics,
            target_labels,
            out_dir / "roc_auc.png",
            metadata=meta,
        ),
    }
    if importance is not None:
        figures["feature_importance"] = plot_feature_importance(
            importance,
            feature_names,
            target_labels,
            out_dir / "feature_importance.png",
            metadata=meta,
        )

    evaluation = {
        "ptbxl_version": meta.get("ptbxl_version"),
        "split": meta.get("split"),
        "seed": meta.get("seed"),
        "n_test_registros": int(y_true.shape[0]),
        "target_labels": target_labels,
        "decision_threshold": DECISION_THRESHOLD,
        "min_test_positives": MIN_TEST_POSITIVES,
        "min_test_negatives": MIN_TEST_NEGATIVES,
        "rare_insufficient_test": rare_insufficient_test,
        "declared_not_evaluated": list(DECLARED_CATALOG),
        "metrics_by_label": {
            label: dict(metrics[j]) for j, label in enumerate(target_labels)
        },
    }
    meta_path = out_dir / "evaluation_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, ensure_ascii=False, indent=2)

    report_artifacts = [str(meta_path), *[str(p) for p in figures.values()]]
    return {
        "metrics": {label: dict(metrics[j]) for j, label in enumerate(target_labels)},
        "rare_insufficient_test": rare_insufficient_test,
        "report_artifacts": report_artifacts,
        "figures": {key: str(path) for key, path in figures.items()},
        "metadata_path": str(meta_path),
    }