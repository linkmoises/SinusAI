"""Tests de entrenamiento reproducible (RF-05/RF-13/RF-14, T12, carril LF).

Verifica `train.py` contra un dataset PTB-XL-like sintético pequeño (generado en
`tmp_path`, sin depender del dataset completo ni del corte real gitignored):

- T12 «Hecho cuando»: `train.py` lee el dataset desde la ruta del contrato (§1.1),
  entrena solo con features de `src.features` (caja negra LC vía contratos) y
  produce `model.pkl` + metadatos (versión/split/seed/laptop/duración), sin GPU ni
  datos de pacientes.
- Los 12 targets de los tres ejes (contracts §1.4) son salidas binarias
  independientes (RF-05); se verifica que cada target tiene ambas clases en el
  corte de entrenamiento y que `predict_label_proba` devuelve probabilidades
  individuales en [0,1] para los 12 (RF-06).
- Reproducibilidad con la misma seed (contracts §1.5) y contabilización de
  registros incompletos (degradados/ilegibles, RF-08/08b) como `skipped`.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import neurokit2 as nk
import pandas as pd
import pytest
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from src.features import FEATURE_KEYS  # noqa: E402
from src.ingest_signal import CANONICAL_LEADS  # noqa: E402
from src.labels import EVALUATED_TARGETS  # noqa: E402

# Registros sintéticos: (ecg_id, strat_fold, heart_rate, seed_rng, scp_codes, existe)
# El corte de entrenamiento (fold != 10) cubre ambas clases de los 12 targets.
# HR 60 evita `ecg_simulate` (cuelga en neurokit2 0.2.13 con 100 Hz y esa frecuencia).
_RECORDS = [
    (1, 3, 70, 0, "{'NORM': 100.0, 'SR': 0.0}", True),
    (2, 3, 75, 1, "{'IMI': 100.0, 'SR': 0.0}", True),
    (3, 3, 80, 2, "{'NDT': 100.0, 'STD_': 0.0, 'SR': 0.0}", True),
    (4, 3, 85, 3, "{'LAFB': 100.0, 'SR': 0.0}", True),
    (5, 3, 90, 4, "{'LVH': 100.0, 'SR': 0.0}", True),
    (6, 3, 65, 5, "{'NORM': 80.0, 'AFIB': 0.0}", True),
    (7, 3, 95, 6, "{'NST_': 100.0, 'STACH': 0.0}", True),
    (8, 3, 100, 7, "{'ASMI': 50.0, '1AVB': 100.0, 'SARRH': 0.0, 'SBRAD': 0.0}", True),
    (9, 10, 66, 8, "{'NORM': 100.0, 'SR': 0.0}", True),  # fold 10 → test, excluido
    (10, 3, 67, 9, "{'NORM': 100.0, 'SR': 0.0}", False),  # sin archivo → skipped
]
_TRAIN_IDS_SKIPPED_BY_MISSING = 1
_TRAIN_ROWS = 8  # 9 en train menos el registro 10 sin archivo

N_COLS_VECTOR = 5 + 5 + 2  # eje 1 + eje 2 + eje 3 (contracts §1.4)


def _make_dataset(root: Path) -> Path:
    """Construye un dataset PTB-XL-like mínimo en `root` (§1.1/§1.3 layout)."""
    records_dir = root / "records100" / "00000"
    records_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ecg_id, fold, hr, seed_rng, scp_codes, exists in _RECORDS:
        filename_lr = f"records100/00000/{ecg_id:05d}_lr"
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": ecg_id + 10000,
                "age": 55,
                "sex": 1,
                "strat_fold": fold,
                "scp_codes": scp_codes,
                "filename_lr": filename_lr,
            }
        )
        if not exists:
            continue
        sig = nk.ecg_simulate(
            duration=10, sampling_rate=100, heart_rate=hr, random_state=seed_rng
        )
        sig12 = np.repeat(sig[:, None], 12, axis=1)
        wfdb.wrsamp(
            record_name=f"{ecg_id:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=CANONICAL_LEADS,
            p_signal=sig12,
            fmt=["16"] * 12,
            adc_gain=[1000] * 12,
            baseline=[0] * 12,
            write_dir=str(records_dir),
        )

    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    return root


def _artifact(model_path: Path) -> dict:
    with (model_path).open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# T12 — entrenamiento reproducible CPU-only
# ---------------------------------------------------------------------------


def test_run_produce_model_pkl_y_metadatos_completos(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    model_path = tmp_path / "model.pkl"

    metadata = train.run(dataset_root=dataset, model_path=model_path, jobs=1, seed=42)

    assert model_path.exists()
    artifact = _artifact(model_path)
    assert set(artifact) == {"model", "target_labels", "features_used", "metadata"}

    assert artifact["target_labels"] == EVALUATED_TARGETS
    assert len(artifact["target_labels"]) == N_COLS_VECTOR
    assert artifact["features_used"] == FEATURE_KEYS

    # contracts §2.7 — metadata de la corrida
    assert metadata["ptbxl_version"] == train.PTBXL_VERSION == "1.0.3"
    assert "strat_fold != 10" in metadata["split"]
    assert metadata["seed"] == 42
    assert metadata["features_used"] == FEATURE_KEYS
    assert metadata["label_mapping_path"] == train.LABEL_MAPPING_PATH
    assert metadata["target_labels"] == EVALUATED_TARGETS
    assert metadata["cpu_only"] is True  # RF-13
    assert metadata["duration_s"] > 0
    assert metadata["hardware"]["cores"] >= 1
    assert metadata["hardware"]["ram_gb"] is None or metadata["hardware"]["ram_gb"] > 0
    assert metadata["hardware"]["os"]
    assert metadata["n_features"] == 8
    assert metadata["n_targets"] == N_COLS_VECTOR
    assert metadata["n_rows_entrenadas"] == _TRAIN_ROWS


def test_solo_entrena_registros_con_vector_completo(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    metadata = train.run(dataset_root=dataset, model_path=tmp_path / "model.pkl", jobs=1)
    # registro 10 (sin archivo) → rechazo/ilegible (RF-08b): se contabiliza, no entra.
    assert metadata["records_incomplete_skipped"] == _TRAIN_IDS_SKIPPED_BY_MISSING
    assert metadata["n_rows_entrenadas"] == _TRAIN_ROWS


def test_matriz_de_entrenamiento_cubre_ambas_clases_de_los_12_targets(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    X, Y, record_ids, skipped = train.build_training_matrix(dataset, jobs=1)
    assert X.shape == (_TRAIN_ROWS, 8)
    assert Y.shape == (_TRAIN_ROWS, N_COLS_VECTOR)
    assert len(record_ids) == _TRAIN_ROWS
    assert skipped == _TRAIN_IDS_SKIPPED_BY_MISSING
    for j, target in enumerate(EVALUATED_TARGETS):
        assert set(Y[:, j]) <= {0, 1}, target
        assert 1 in Y[:, j], f"target {target} sin positivos en el corte de entrenamiento"
        assert 0 in Y[:, j], f"target {target} sin negativos en el corte de entrenamiento"


def test_predict_label_proba_da_probabilidad_por_cada_uno_de_los_12(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    train.run(dataset_root=dataset, model_path=tmp_path / "model.pkl", jobs=1)
    artifact = _artifact(tmp_path / "model.pkl")
    X, _, _, _ = train.build_training_matrix(dataset, jobs=1)

    proba = train.predict_label_proba(artifact["model"], X)
    assert proba.shape == (_TRAIN_ROWS, N_COLS_VECTOR)
    assert np.isfinite(proba).all()
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_reproducible_con_la_misma_seed_y_archivo_sin_datos_de_pacientes(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    model_a = tmp_path / "a.pkl"
    model_b = tmp_path / "b.pkl"
    train.run(dataset_root=dataset, model_path=model_a, jobs=1, seed=42)
    train.run(dataset_root=dataset, model_path=model_b, jobs=1, seed=42)
    a = _artifact(model_a)
    b = _artifact(model_b)
    assert a["metadata"]["seed"] == b["metadata"]["seed"] == 42
    X, _, _, _ = train.build_training_matrix(dataset, jobs=1)
    proba_a = train.predict_label_proba(a["model"], X)
    proba_b = train.predict_label_proba(b["model"], X)
    assert np.allclose(proba_a, proba_b)

    blob = model_a.read_bytes()
    for leaked in ("ptbxl_database", "scp_codes", "001_lr"):
        assert leaked.encode() not in blob  # RF-14: no señales ni scp_codes de pacientes


def test_dataset_fuera_de_la_ruta_del_contrato_lanza_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.read_database(tmp_path / "nada")
    with pytest.raises(FileNotFoundError):
        train.run(dataset_root=tmp_path / "nada", model_path=tmp_path / "m.pkl")
    assert train.main(["--dataset", str(tmp_path / "nada"), "--model", str(tmp_path / "m.pkl")]) == 1


def test_main_cli_entrena_desde_la_ruta_indicada(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    model_path = tmp_path / "model.pkl"
    assert train.main(["--dataset", str(dataset), "--model", str(model_path), "--jobs", "1"]) == 0
    assert model_path.exists()
    assert _artifact(model_path)["metadata"]["cpu_only"] is True


def test_lectura_del_database_con_respaldo_en_fixture_real(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    df = train.read_database(dataset)
    for column in ("ecg_id", "strat_fold", "scp_codes", "filename_lr"):
        assert column in df.columns
    assert list(df["strat_fold"]) == [row[1] for row in _RECORDS]