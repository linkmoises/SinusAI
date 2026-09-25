"""Entrenamiento CPU-only reproducible (RF-05, RF-13, RF-14, T12, carril LF).

Lee el dataset PTB-XL 100 Hz desde la ruta congelada en `docs/contracts.md` §1.1
(`data/ptb-xl/`), extrae el vector clínico cerrado de `src.features` por registro
(LC como caja negra vía contratos) y entrena un clasificador multietiqueta Random
Forest — un clasificador binario por salida, uno-vs-resto — sobre los 12 targets
evaluados de los tres ejes de `docs/contracts.md` §1.4 (`EVALUATED_TARGETS` de
`src.labels`, T09). Guarda `model.pkl` + metadatos de la corrida en la raíz del
repo, siguiendo el esquema `TrainingRun` de contracts §2.7 (versión/split/seed/
features/laptop/duración).

Decisiones de LF documentadas en el `TrainingRun` (contracts §2.5 «el manejo en
entrenamiento es decisión de LF»):

  - Cada una de las 12 salidas evaluadas se trata como etiqueta binaria
    independiente (multietiqueta real, RF-05): `MultiOutputClassifier` con base
    RandomForest y `class_weight="balanced"` por la asimetría de las clases raras
    evaluadas (ej. `1AVB`: 714 casos en folds 1–8). Así cada salida expone su
    probabilidad individual en [0,1] (RF-06) y se evalúa uno-vs-resto (RF-11).
  - El eje 1 (superclase) participa como 5 binarios independientes, sin lógica de
    exclusión de NORM frente a patológicas (RF-05) y tolerando co-ocurrencia.
  - Solo se entrenan registros con `FeatureVector.status == "completo"` (las 8
    features de §2.4.1). Los registros incompletos/degradados de PTB-XL no entran
    a la matriz de entrenamiento: se contabilizan como `records_incomplete_skipped`
    en los metadatos (RF-08 degradación, no hacer crecer la matriz con filas NaN).
  - Entrenamiento CPU-only (RF-13): ningún estimador requiere GPU; la extracción de
    features se paraleliza con `multiprocessing` (`--jobs`).
  - Solo datos públicos desidentificados PTB-XL (RF-14): no se incorporan señales ni
    `scp_codes` de pacientes al artefacto — el pickle contiene modelo + metadatos
    de corrida, no los datos crudos.

La evaluación uno-vs-resto (matriz, ROC/AUC, importancia) es T13 (`src/evaluate.*`);
este módulo solo entrena y persiste `model.pkl` + metadatos.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_KEYS
from src.labels import (
    EVALUATED_TARGETS,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
    build_label_vector,
)

REPO_ROOT = Path(__file__).resolve().parent

# contracts §1.1 — rutas congeladas del dataset PTB-XL (100 Hz)
DATASET_ROOT = REPO_ROOT / "data" / "ptb-xl"
DATABASE_CSV = DATASET_ROOT / "ptbxl_database.csv"

# contracts §2.7 — constantes de corrida
MODEL_PATH = REPO_ROOT / "model.pkl"
PTBXL_VERSION = "1.0.3"  # según `data/ptb-xl/ptbxl_v*_changelog.txt` (§1.1)
DEFAULT_SEED = 42
SPLIT_DESCRIPTION = (
    "train = strat_fold != 10, test = strat_fold == 10 (canónico PTB-XL, §1.5)"
)
LABEL_MAPPING_PATH = "docs/label_mapping.md"

RF_ESTIMATORS = 200  # budget CPU-only (RF-13): árboles pequeños sobre 8 features


def read_database(dataset_root: str | Path = DATASET_ROOT) -> pd.DataFrame:
    """Lee `ptbxl_database.csv` de `dataset_root` (contracts §1.1, §1.3).

    Devuelve el DataFrame original; las columnas requeridas por el entrenamiento
    son `ecg_id`, `strat_fold`, `scp_codes` y `filename_lr`.
    """
    database_csv = Path(dataset_root) / "ptbxl_database.csv"
    if not database_csv.exists():
        raise FileNotFoundError(
            f"no se encuentra {database_csv} — descargar PTB-XL 100 Hz y colocarlo "
            "en la ruta indicada en contracts §1.1 (data/ptb-xl/)"
        )
    return pd.read_csv(database_csv)


def train_rows(database: pd.DataFrame, max_records: int | None = None) -> pd.DataFrame:
    """Subconjunto de entrenamiento: `strat_fold != 10` (§1.5)."""
    rows = database[database["strat_fold"].astype(int) != 10]
    if max_records is not None:
        rows = rows.iloc[:max_records]
    return rows


def _extract_worker(args: tuple) -> tuple[int, list[float] | None]:
    """Carga un registro real y devuelve su vector de features (o `None`).

    Corre dentro de los procesos del pool; la tirada de registros puede ser
    degradada (RF-08) o de header ilegible (RF-08b) → `None` (se descarta y se
    contabiliza). Solo señales 1D (`src.ingest_signal` → `src.features`), nunca
    píxeles (RF-03).
    """
    dataset_root, row_index, filename_lr = args
    from src.features import extract_features
    from src.ingest_signal import ingest_signal

    rec = ingest_signal(Path(dataset_root) / filename_lr)
    if rec.status != "reconocido" or rec.signal is None:
        return row_index, None
    fv = extract_features(rec.signal)
    if fv.status != "completo":
        return row_index, None
    return row_index, [float(fv.values[key]) for key in FEATURE_KEYS]


def _labels_of(record_id: str, scp_codes: str | dict) -> list[int]:
    """Vector binario de los 12 targets evaluados en orden canónico (§1.4)."""
    lv = build_label_vector(record_id, scp_codes)
    values: list[int] = []
    for target in EVALUATED_TARGETS:
        if target in SUPERCLASS_ORDER:
            values.append(lv.superclass_labels[target])
        elif target in RHYTHM_EVALUATED:
            values.append(lv.rhythm_labels[target])
        elif target in SPECIFIC_EVALUATED:
            values.append(lv.specific_diagnostic_labels[target])
        else:  # pragma: no cover — los 12 targets siempre viven en un eje
            raise ValueError(f"target fuera de los tres ejes: {target!r}")
    return values


def build_training_matrix(
    dataset_root: str | Path = DATASET_ROOT,
    max_records: int | None = None,
    jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Construye X (n×8) e Y (n×12) para los registros de entrenamiento (§1.5).

    X proviene de `src.features` (caja negra LC vía contratos: 8 features de
    §2.4.1); Y de `src.labels` (T09) en el orden de `EVALUATED_TARGETS`. Devuelve
    `(X, Y, record_ids, skipped)` donde `skipped` es el número de registros de
    entrenamiento sin vector completo (degradados/ilegibles), contabilizado en los
    metadatos en lugar de rellenar la matriz con NaN.
    """
    database = read_database(dataset_root)
    rows = train_rows(database, max_records)

    labels: list[list[int]] = []
    filename_lr_list: list[str] = []
    for _, db_row in rows.iterrows():
        filename_lr = str(db_row["filename_lr"])
        filename_lr_list.append(filename_lr)
        labels.append(_labels_of(filename_lr, db_row["scp_codes"]))

    tasks = [
        (str(dataset_root), index, filename_lr)
        for index, filename_lr in enumerate(filename_lr_list)
    ]

    if jobs is None:
        jobs = os.cpu_count() or 1
    jobs = max(1, int(jobs))

    if jobs == 1:
        results = [_extract_worker(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_extract_worker, tasks))

    feature_rows: list[list[float]] = []
    kept_ids: list[str] = []
    skipped = 0
    for row_index, features in results:
        if features is None:
            skipped += 1
            continue
        feature_rows.append(features)
        kept_ids.append(filename_lr_list[row_index])

    if not feature_rows:
        raise ValueError(
            "ningún registro de entrenamiento produjo un vector de features completo; "
            "revisar la ruta del dataset o descartar el corte usado"
        )

    X = np.asarray(feature_rows, dtype=float)
    row_of = {filename_lr: label_row for filename_lr, label_row in zip(filename_lr_list, labels)}
    Y = np.asarray([row_of[record_id] for record_id in kept_ids], dtype=int)
    return X, Y, kept_ids, skipped


def train_model(
    X: np.ndarray, Y: np.ndarray, seed: int = DEFAULT_SEED
) -> object:
    """Entrena el clasificador multietiqueta CPU-only (RF-05, RF-13, D-02).

    `MultiOutputClassifier(RandomForestClassifier)` — un Random Forest binario por
    cada una de las 12 salidas, `class_weight="balanced"` por la asimetría de las
    clases evaluadas. `random_state=seed` hace la corrida reproducible (contracts
    §1.5, §2.7). Sin GPU.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    Y = np.asarray(Y)
    for j, target in enumerate(EVALUATED_TARGETS):
        classes = np.unique(Y[:, j])
        if len(classes) < 2:
            raise ValueError(
                f"target {target!r} con una sola clase en el corte de entrenamiento "
                f"({classes}); si se usó --limit, aumentar el tamaño del corte"
            )

    base = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model = MultiOutputClassifier(base, n_jobs=1)
    model.fit(X, Y)
    return model


def predict_label_proba(model: object, X: np.ndarray) -> np.ndarray:
    """Probabilidad individual en [0,1] por cada uno de los 12 targets.

    Devuelve `(n, 12)` alineado con `EVALUATED_TARGETS`: para cada salida binaria
    se toma la probabilidad de la clase positiva (contracts §2.6.0 — RF-06).
    """
    per_target = model.predict_proba(np.asarray(X))
    return np.column_stack([proba[:, 1] for proba in per_target])


def _hardware_info() -> dict:
    """Laptop de la corrida (contracts §2.7 `hardware`)."""
    cores = os.cpu_count() or 0
    ram_gb: float | None = None
    try:
        ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        ram_gb = round(ram_bytes / 1e9, 1)
    except (ValueError, AttributeError, OSError):
        pass
    cpu = platform.processor() or ""
    if not cpu:
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {"cpu": cpu or "unknown", "cores": cores, "ram_gb": ram_gb, "os": platform.platform()}


def make_metadata(
    dataset_root: str | Path,
    X: np.ndarray,
    skipped: int,
    seed: int,
    duration_s: float,
) -> dict:
    """Metadatos `TrainingRun`-compatibles (§2.7): versión/split/seed/laptop/duración."""
    return {
        "model_path": str(MODEL_PATH),
        "ptbxl_version": PTBXL_VERSION,
        "split": SPLIT_DESCRIPTION,
        "seed": seed,
        "features_used": list(FEATURE_KEYS),
        "label_mapping_path": LABEL_MAPPING_PATH,
        "target_labels": list(EVALUATED_TARGETS),
        "architecture": (
            "MultiOutputClassifier(RandomForestClassifier), un clasificador binario "
            "por salida (uno-vs-resto, 12 salidas), class_weight=balanced"
        ),
        "model_type": "random_forest",
        "hardware": _hardware_info(),
        "duration_s": float(duration_s),
        "cpu_only": True,
        "dataset_root": str(dataset_root),
        "n_registros_entrenamiento": int(X.shape[0]),
        "n_rows_entrenadas": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(len(EVALUATED_TARGETS)),
        "records_incomplete_skipped": int(skipped),
    }


def save_model(model: object, metadata: dict, path: str | Path = MODEL_PATH) -> Path:
    """Persiste `model.pkl` + metadatos de la corrida (contracts §3).

    El artefacto contiene `model`, `target_labels`, `features_used` y `metadata`;
    no contiene señales ni `scp_codes` de pacientes (RF-14).
    """
    path = Path(path)
    artifact = {
        "model": model,
        "target_labels": list(EVALUATED_TARGETS),
        "features_used": list(FEATURE_KEYS),
        "metadata": metadata,
    }
    with path.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def run(
    dataset_root: str | Path = DATASET_ROOT,
    model_path: str | Path = MODEL_PATH,
    seed: int = DEFAULT_SEED,
    jobs: int | None = None,
    max_records: int | None = None,
) -> dict:
    """Vía completa: dataset → features → modelo → `model.pkl` + metadatos (T12)."""
    start = time.time()
    X, Y, kept_ids, skipped = build_training_matrix(
        dataset_root, max_records=max_records, jobs=jobs
    )
    model = train_model(X, Y, seed=seed)
    duration_s = time.time() - start
    metadata = make_metadata(
        dataset_root, X, skipped, seed=seed, duration_s=duration_s
    )
    save_path = save_model(model, metadata, path=model_path)
    print(f"modelo guardado en {save_path}")
    print(f"filas entrenadas: {len(kept_ids)}, skipped: {skipped}, duración: {duration_s:.1f} s")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo SinusAI desde PTB-XL 100 Hz (RF-13, CPU-only)."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help=f"ruta de la raíz del dataset (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--model", default=str(MODEL_PATH), help="ruta de salida del pickle (default: model.pkl)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed (default: 42)")
    parser.add_argument(
        "--jobs", type=int, default=None, help="procesos de extracción de features (default: CPU cores)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limita el número de registros de entrenamiento (dev/tests)",
    )
    args = parser.parse_args(argv)

    try:
        run(
            dataset_root=args.dataset,
            model_path=args.model,
            seed=args.seed,
            jobs=args.jobs,
            max_records=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())