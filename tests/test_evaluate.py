"""Tests de la evaluación uno-vs-resto (RF-11/RF-12, T13, carril LF).

Verifica el «Hecho cuando» de T13 con predicciones **sintéticas** (sin depender
del dataset PTB-XL): `src/evaluate.evaluate_run` genera los artefactos de RF-11
en `reports/` — matriz de confusión por etiqueta, curva ROC/AUC por cada una de
las 12 salidas evaluadas (contracts §1.4) e importancia de variables por
etiqueta — y `docs/contracts.md` §2.7 `TrainingRun.rare_insufficient_test`.

Cubre, sin ocultar clases (RF-12):

- Matriz de confusión correcta en el umbral 0,50 y AUC por etiqueta.
- Nota de insuficiencia explícita cuando una etiqueta no tiene muestras
  suficientes en test para una curva fiable (RF-11): la gráfica se genera con
  la nota y la etiqueta queda en `rare_insufficient_test` / el metadata JSON.
- Importancia por etiqueta desde un `MultiOutputClassifier` real pequeño y
  persistencia de los tres artefactos + `evaluation_metadata.json` con
  versión/split/seed (RF-11) y `declared_not_evaluated` (RF-12).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    DECISION_THRESHOLD,
    MIN_TEST_NEGATIVES,
    MIN_TEST_POSITIVES,
    importance_from_model,
    per_label_metrics,
    evaluate_run,
)
from src.features import FEATURE_KEYS  # noqa: E402
from src.labels import DECLARED_CATALOG, EVALUATED_TARGETS  # noqa: E402

N_TARGETS = len(EVALUATED_TARGETS)
N_FEATURES = len(FEATURE_KEYS)
N = 200  # registros sintéticos de test

# Predicciones sintéticas: 12 columnas alineadas a EVALUATED_TARGETS.
# - Índices 0..9: ambas clases en test (curva fiable, scores perfectos).
# - Índice 10 (ASMI): 2 positivos → curva computable pero insuficiente (< 5).
# - Índice 11 (1AVB): sin positivos → sin curva (auc None), insuficiente.
# Esto ejercita la «nota de insuficiencia» de RF-11 sobre especies raras.
_ASMI_IDX = EVALUATED_TARGETS.index("ASMI")
_1AVB_IDX = EVALUATED_TARGETS.index("1AVB")


def _synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.zeros((N, N_TARGETS), dtype=int)
    for j in range(10):
        y_true[: 100 - j * 4, j] = 1  # mezcla de clases en 0..9, sin vacíos
    y_true[:2, _ASMI_IDX] = 1  # exactamente 2 positivos (n_pos=2 < 5)
    return y_true, y_true.astype(float)  # scores perfectos para 0..9 y ASMI


def _metadata() -> dict:
    return {
        "ptbxl_version": "1.0.3",
        "split": "train = strat_fold != 10, test = strat_fold == 10 (canónico PTB-XL, §1.5)",
        "seed": 42,
    }


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Métricas por etiqueta (matriz + AUC)
# ---------------------------------------------------------------------------


def test_per_label_metrics_matriz_y_auc_perfectos_en_el_umbral():
    y_true, y_score = _synthetic_data()
    metrics = per_label_metrics(y_true, y_score)

    for j in range(10):
        m = metrics[j]
        assert m["tp"] == m["n_pos"]
        assert m["tn"] == m["n_neg"]
        assert m["fp"] == 0 and m["fn"] == 0
        assert m["auc"] == pytest.approx(1.0)
        assert m["insufficient"] is False

    m_asmi = metrics[_ASMI_IDX]
    assert m_asmi["n_pos"] == 2 < MIN_TEST_POSITIVES
    assert m_asmi["auc"] == pytest.approx(1.0)  # curvable pero no fiable
    assert m_asmi["insufficient"] is True

    m_1avb = metrics[_1AVB_IDX]
    assert m_1avb["n_pos"] == 0
    assert m_1avb["auc"] is None
    assert m_1avb["insufficient"] is True


def test_matriz_refleja_el_umbral_de_decision_en_predicciones_imperfectas():
    y_true = np.zeros((12, N_TARGETS), dtype=int)
    y_true[:3, 0] = 1  # NORM: 3 positivos
    y_score = np.full((12, N_TARGETS), 0.1)  # default negativo
    y_score[0, 0] = 0.9   # TP
    y_score[2, 0] = 0.9   # TP
    y_score[1, 0] = 0.1   # FN (below threshold)
    y_score[3, 0] = 0.8   # FP (negative con p alta)

    metrics = per_label_metrics(y_true, y_score)
    m = metrics[0]
    assert m["n_pos"] == 3 and m["n_neg"] == 9
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (8, 1, 1, 2)
    assert m["auc"] is not None  # ambas clases presentes → curvable
    assert m["insufficient"] is True  # n_pos=3 < 5 → no fiable (RF-11)
    assert DECISION_THRESHOLD == pytest.approx(0.50)


def test_formas_incoherentes_lanzan_error():
    y_true, y_score = _synthetic_data()
    with pytest.raises(ValueError):
        per_label_metrics(y_true, y_score[:, : N_TARGETS - 1])
    with pytest.raises(ValueError):
        per_label_metrics(y_true, y_score[:, 0])  # 1D


# ---------------------------------------------------------------------------
# evaluate_run — artefactos de RF-11 en reports/
# ---------------------------------------------------------------------------


def test_eval_sintetica_genera_los_tres_artefactos_y_metadatos(tmp_path):
    y_true, y_score = _synthetic_data()
    importance = np.full((N_TARGETS, N_FEATURES), 0.125)  # uniforme por etiqueta
    importance[0, 0] = 0.6  # NORM depende sobre todo de qrs_duration_ms

    result = evaluate_run(
        y_true,
        y_score,
        EVALUATED_TARGETS,
        feature_importance=importance,
        feature_names=FEATURE_KEYS,
        metadata=_metadata(),
        reports_dir=tmp_path / "reports",
    )

    out = tmp_path / "reports"
    for name in ("confusion_matrix.png", "roc_auc.png", "feature_importance.png"):
        assert (out / name).exists(), name

    assert result["rare_insufficient_test"] == ["ASMI", "1AVB"]
    assert "ASMI" in result["metrics"] and result["metrics"]["ASMI"]["insufficient"]
    assert result["metrics"]["1AVB"]["auc"] is None
    assert result["metrics"]["NORM"]["auc"] == pytest.approx(1.0)

    assert set(result["figures"]) == {"confusion_matrix", "roc_auc", "feature_importance"}
    expected_artifacts = [str(out / "evaluation_metadata.json"), *result["figures"].values()]
    assert result["report_artifacts"] == expected_artifacts
    assert all(Path(artifact).exists() for artifact in result["report_artifacts"])

    meta = _read_json(out / "evaluation_metadata.json")
    assert meta["ptbxl_version"] == "1.0.3"  # RF-11: versión junto a las gráficas
    assert "strat_fold != 10" in meta["split"]
    assert meta["seed"] == 42
    assert meta["n_test_registros"] == N
    assert meta["target_labels"] == EVALUATED_TARGETS
    assert meta["decision_threshold"] == pytest.approx(0.50)
    assert meta["rare_insufficient_test"] == ["ASMI", "1AVB"]
    assert meta["declared_not_evaluated"] == list(DECLARED_CATALOG)  # RF-12
    assert meta["metrics_by_label"]["NORM"]["auc"] == pytest.approx(1.0)
    assert meta["metrics_by_label"]["1AVB"]["auc"] is None


def test_sin_importancia_no_obliga_y_solo_genera_matriz_y_roc(tmp_path):
    y_true, y_score = _synthetic_data()
    result = evaluate_run(
        y_true, y_score, EVALUATED_TARGETS, metadata=_metadata(),
        reports_dir=tmp_path / "reports",
    )
    assert "feature_importance" not in result["figures"]
    assert (tmp_path / "reports" / "feature_importance.png").exists() is False
    assert (tmp_path / "reports" / "confusion_matrix.png").exists()
    assert (tmp_path / "reports" / "roc_auc.png").exists()


def test_importancia_con_forma_incoherente_lanza_error(tmp_path):
    y_true, y_score = _synthetic_data()
    with pytest.raises(ValueError):
        evaluate_run(
            y_true, y_score, EVALUATED_TARGETS,
            feature_importance=np.ones((N_TARGETS - 1, N_FEATURES)),
            feature_names=FEATURE_KEYS,
            reports_dir=tmp_path / "reports",
        )
    with pytest.raises(ValueError):
        evaluate_run(
            y_true, y_score, EVALUATED_TARGETS,
            feature_importance=np.ones((N_TARGETS, N_FEATURES)),
            feature_names=None,
            reports_dir=tmp_path / "reports",
        )


def test_target_labels_incoherente_con_y_true_lanza_error(tmp_path):
    y_true, y_score = _synthetic_data()
    with pytest.raises(ValueError):
        evaluate_run(
            y_true, y_score, EVALUATED_TARGETS[:-1], reports_dir=tmp_path / "reports"
        )


# ---------------------------------------------------------------------------
# Importancia por etiqueta desde un clasificador por salida (RF-11)
# ---------------------------------------------------------------------------


def test_importance_from_model_extrae_una_fila_por_etiqueta():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, N_FEATURES))
    pre = rng.normal(size=(60, N_TARGETS))
    Y = (pre + X[:, 0:1] * 2.0 > 0.0).astype(int)  # etiquetas con señal

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42),
        n_jobs=1,
    )
    model.fit(X, Y)
    importance = importance_from_model(model, FEATURE_KEYS)

    assert importance.shape == (N_TARGETS, N_FEATURES)
    assert np.isfinite(importance).all()
    assert (importance >= 0).all()
    row_sums = importance.sum(axis=1)
    assert np.allclose(row_sums, 1.0), row_sums

    # Cada etiqueta extrae su fila al orden de EVALUATED_TARGETS (model.estimators_).
    fi_first = model.estimators_[0].feature_importances_
    assert np.allclose(importance[0], fi_first)


def test_importancia_rechaza_modelo_sin_estimators():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=2, random_state=42)
    with pytest.raises(ValueError, match="estimators_"):
        importance_from_model(model, FEATURE_KEYS)