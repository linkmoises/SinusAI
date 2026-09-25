"""Mapeo de statements PTB-XL a `LabelVector` de tres ejes (RF-05, T09, carril LD).

Convierte los `scp_codes` de un registro (`ptbxl_database.csv`) en el `LabelVector`
de tres ejes independientes según `docs/contracts.md` §1.4 y §2.5:

  - Eje 1 — superclase diagnóstica (siempre evaluada): `NORM`, `MI`, `STTC`, `CD`,
    `HYP`. Escritura plana, sin lógica de exclusión frente a patológicas (RF-05):
    `NORM` no excluye y la co-ocurrencia deja activas todas las superclases presentes.
  - Eje 2 — ritmo evaluado (categoría `rhythm` con criterio de datos): `SR`, `AFIB`,
    `STACH`, `SARRH`, `SBRAD`.
  - Eje 3 — diagnóstico específico evaluado (subconjunto de categoría `diagnostic`):
    `ASMI`, `1AVB`.

Las claves declaradas-no-evaluadas (§1.4.4) NO ocupan claves en
`rhythm_labels`/`specific_diagnostic_labels` (contracts §2.5): no son targets de
entrenamiento ni de evaluación. Aquí se exponen como catálogo fijo (`DECLARED_*`)
para que LE/LG las listen con el aviso canónico `insuf14` sin probabilidad (§2.6.1).
El catálogo se define por el criterio de inclusión del contrato (>= `MIN_TOTAL_CASES`
totales y >= `MIN_TRAIN_CASES` en folds 1–8): solo lo genuinamente subrepresentado
se declara con `insuf14` — los statements con datos de sobra que no son targets se
listan aparte como `ABOVE_THRESHOLD_NOT_EVALUATED`, sin declarar insuficiencia
(Constitution §1).

Las categorías de anotación (`diagnostic`/`rhythm`/`form`) y el superclass de cada
statement provienen de `data/ptb-xl/scp_statements.csv` (contracts §1.4.5): la tabla
canónica se incrusta como `SUPERCLASS_OF` (generada del archivo real) y se puede
re-verificar contra el propio archivo con `load_statement_map()`. El detalle y los
conteos reales por `strat_fold` viven en `docs/label_mapping.md`.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCP_STATEMENTS_CSV = REPO_ROOT / "data" / "ptb-xl" / "scp_statements.csv"

# Órdenes canónicos de los tres ejes (contracts §1.4.1–§1.4.3).
SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
RHYTHM_EVALUATED = ["SR", "AFIB", "STACH", "SARRH", "SBRAD"]
SPECIFIC_EVALUATED = ["ASMI", "1AVB"]

# Total de salidas evaluadas (5 + 5 + 2): los targets de entrenamiento/evaluación.
EVALUATED_TARGETS = SUPERCLASS_ORDER + RHYTHM_EVALUATED + SPECIFIC_EVALUATED

# statement de categoría `diagnostic` -> superclase (diagnostic_class), generado de
# `scp_statements.csv` (44 entradas). Categorías verificadas en tests contra el archivo.
SUPERCLASS_OF = {
    "1AVB": "CD",
    "2AVB": "CD",
    "3AVB": "CD",
    "ALMI": "MI",
    "AMI": "MI",
    "ANEUR": "STTC",
    "ASMI": "MI",
    "CLBBB": "CD",
    "CRBBB": "CD",
    "DIG": "STTC",
    "EL": "STTC",
    "ILBBB": "CD",
    "ILMI": "MI",
    "IMI": "MI",
    "INJAL": "MI",
    "INJAS": "MI",
    "INJIL": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    "IPLMI": "MI",
    "IPMI": "MI",
    "IRBBB": "CD",
    "ISCAL": "STTC",
    "ISCAN": "STTC",
    "ISCAS": "STTC",
    "ISCIL": "STTC",
    "ISCIN": "STTC",
    "ISCLA": "STTC",
    "ISC_": "STTC",
    "IVCD": "CD",
    "LAFB": "CD",
    "LAO/LAE": "HYP",
    "LMI": "MI",
    "LNGQT": "STTC",
    "LPFB": "CD",
    "LVH": "HYP",
    "NDT": "STTC",
    "NORM": "NORM",
    "NST_": "STTC",
    "PMI": "MI",
    "RAO/RAE": "HYP",
    "RVH": "HYP",
    "SEHYP": "HYP",
    "WPW": "CD",
}

# Criterio de inclusión de los ejes evaluables (contracts §1.4.2–§1.4.3): un
# statement se evalúa solo si alcanza >= 500 casos totales Y >= 350 en los folds de
# entrenamiento (1–8 de `strat_fold`). Las claves que NO lo cumplen quedan en el
# catálogo declarado-no-evaluado (contracts §1.4.4) y NUNCA reciben probabilidad.
MIN_TOTAL_CASES = 500
MIN_TRAIN_CASES = 350

# Catálogo declarado-no-evaluado (contracts §1.4.4, claves exactas; Apéndice A.2).
# Eje 2 — declaradas de la categoría `rhythm` sin criterio (§1.4.2): el resto de la
# categoría son AFLT, SVARR, PACE, BIGU, SVTAC, PSVT, TRIGU (todas < 350; conteos en
# `docs/label_mapping.md`).
DECLARED_RHYTHM = ["AFLT", "BIGU", "PACE", "PSVT", "SVARR", "SVTAC", "TRIGU"]
# Eje 3 — declaradas de la categoría `diagnostic` sin criterio (incluye 2AVB/3AVB,
# §1.4.4). Derivado de la tabla real de conteos, no de una homonimia (también
# recomputable con `declared_catalog_from_counts()`).
DECLARED_SPECIFIC_DIAGNOSTIC = [
    "2AVB", "3AVB", "ALMI", "AMI", "ANEUR", "DIG", "EL", "ILBBB", "ILMI",
    "INJAL", "INJAS", "INJIL", "INJIN", "INJLA", "IPLMI", "IPMI", "ISCAN",
    "ISCAS", "ISCIL", "ISCIN", "ISCLA", "LAO/LAE", "LMI", "LNGQT", "LPFB",
    "PMI", "RAO/RAE", "RVH", "SEHYP", "WPW",
]
# Statements de la categoría `diagnostic` que SÍ cumplen el criterio numérico pero
# no integran el subconjunto evaluado fijado por el contrato (§1.4.3 = {ASMI, 1AVB}).
# Se documentan para no mezclarlas con el catálogo declarado: no son targets y no
# merecen `insuf14` (tienen datos de sobra; Constitution §1).
ABOVE_THRESHOLD_NOT_EVALUATED = [
    "CLBBB", "CRBBB", "IMI", "IRBBB", "ISCAL", "ISC_", "IVCD", "LAFB", "LVH",
    "NDT", "NORM", "NST_",
]
# Eje 2 — sin statement directo en PTB-XL (0 ocurrencias, definitivamente no evaluadas).
DECLARED_NO_STATEMENT = ["VT", "VF", "NODAL"]

DECLARED_CATALOG = (
    DECLARED_RHYTHM + DECLARED_SPECIFIC_DIAGNOSTIC + DECLARED_NO_STATEMENT
)


@dataclass
class LabelVector:
    """contracts §2.5 — etiquetas de un registro en los tres ejes independientes.

    Cada dict contiene exactamente las claves del eje en su orden canónico, con
    valores {0, 1}. Las claves declaradas-no-evaluadas no ocupan claves aquí.
    """

    record_id: str
    superclass_labels: dict[str, int]
    rhythm_labels: dict[str, int]
    specific_diagnostic_labels: dict[str, int]
    source_statements: dict[str, float]


def _parse_scp_codes(scp_codes: str | dict) -> dict[str, float]:
    """Acepta el string `"{'NORM': 100.0, ...}"` de `ptbxl_database.csv` o un dict."""
    if isinstance(scp_codes, dict):
        return {str(k): float(v) for k, v in scp_codes.items()}
    codes = ast.literal_eval(str(scp_codes))
    if not isinstance(codes, dict):
        raise TypeError(f"scp_codes inválido (se esperaba dict): {scp_codes!r}")
    return {str(k): float(v) for k, v in codes.items()}


def build_label_vector(record_id: str, scp_codes: str | dict) -> LabelVector:
    """Mapea `scp_codes` de un registro a su `LabelVector` de tres ejes (RF-05).

    Reglas (documentadas en `docs/label_mapping.md`):

      - Eje 1: se activa `SUPERCLASS_OF[statement]` por cada statement de categoría
        `diagnostic` presente. `NORM` es una etiqueta plana más: NO excluye
        patológicas. La co-ocurrencia deja activas todas las superclases presentes.
      - Eje 2: se activa cada statement de `RHYTHM_EVALUATED` presente. El resto de
        la categoría `rhythm` queda declarado, fuera del vector.
      - Eje 3: se activa cada statement de `SPECIFIC_EVALUATED` presente. El resto
        de la categoría `diagnostic` queda declarado, fuera del vector.
      - `source_statements` conserva los `scp_codes` tal cual vienen del dataset.
    """
    codes = _parse_scp_codes(scp_codes)

    superclass_labels = {key: 0 for key in SUPERCLASS_ORDER}
    for statement in codes:
        superclass = SUPERCLASS_OF.get(statement)
        if superclass in superclass_labels:
            superclass_labels[superclass] = 1

    rhythm_labels = {key: 0 for key in RHYTHM_EVALUATED}
    for statement in codes:
        if statement in rhythm_labels:
            rhythm_labels[statement] = 1

    specific_labels = {key: 0 for key in SPECIFIC_EVALUATED}
    for statement in codes:
        if statement in specific_labels:
            specific_labels[statement] = 1

    return LabelVector(
        record_id=record_id,
        superclass_labels=superclass_labels,
        rhythm_labels=rhythm_labels,
        specific_diagnostic_labels=specific_labels,
        source_statements=codes,
    )


def declared_codes_for(scp_codes: str | dict) -> list[str]:
    """Claves del catálogo declarado-no-evaluado presentes en un registro.

    Vacío si ninguna clave declarada aparece. No incluye `VT`/`VF`/`NODAL` (no
    existen como statement en PTB-XL, 0 ocurrencias). Sirve a LE/LG para listar el
    catálogo `declared_not_evaluated` con el aviso fijo `insuf14` (§2.6.1).
    """
    codes = _parse_scp_codes(scp_codes)
    return [code for code in DECLARED_RHYTHM + DECLARED_SPECIFIC_DIAGNOSTIC if code in codes]


def declared_catalog_from_counts(database_path: str | Path | None = None) -> dict:
    """Recomputa la partición evaluado/declarado desde los conteos reales (contracts §1.4.5).

    Lee `ptbxl_database.csv` y devuelve `{categoría: [codes]}` con:
      `declared_rhythm`, `declared_specific`, `above_threshold_not_evaluated`.
    Se usa para re-verificar (`tests/test_labels.py`) que la tabla incrustada
    coincide con los datos reales. Requiere la ruta del dataset (§1.1).
    """
    path = Path(database_path) if database_path else REPO_ROOT / "data" / "ptb-xl" / "ptbxl_database.csv"
    counts: dict[str, dict] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            for code in ast.literal_eval(row["scp_codes"]):
                entry = counts.setdefault(code, {"total": 0, "train18": 0})
                entry["total"] += 1
                if int(row["strat_fold"]) != 10:
                    entry["train18"] += 1

    def meets_criterion(code: str) -> bool:
        entry = counts.get(code)
        return bool(
            entry
            and entry["total"] >= MIN_TOTAL_CASES
            and entry["train18"] >= MIN_TRAIN_CASES
        )

    statement_map = load_statement_map()
    rhythm_categories = {
        code for code, info in statement_map.items() if "rhythm" in info["categories"]
    }
    diagnostic_categories = {
        code for code, info in statement_map.items() if "diagnostic" in info["categories"]
    }
    declared_rhythm = sorted(c for c in rhythm_categories - set(RHYTHM_EVALUATED) if not meets_criterion(c))
    declared_specific = sorted(
        c for c in diagnostic_categories - set(SPECIFIC_EVALUATED) if not meets_criterion(c)
    )
    above_threshold = sorted(
        c for c in diagnostic_categories - set(SPECIFIC_EVALUATED) if meets_criterion(c)
    )
    return {
        "declared_rhythm": declared_rhythm,
        "declared_specific": declared_specific,
        "above_threshold_not_evaluated": above_threshold,
    }


def load_statement_map(statements_path: str | Path | None = None) -> dict:
    """Lee `scp_statements.csv` y devuelve `{statement: {categories, diagnostic_class}}`.

    La categoría de anotación se toma de las columnas `diagnostic`/`rhythm`/`form`
    del archivo, no se asume por homonimia (contracts §1.4.5). Se usa para
    re-verificar las tablas incrustadas y en el pipeline de entrenamiento (T12).
    """
    path = Path(statements_path) if statements_path else SCP_STATEMENTS_CSV
    statement_map: dict[str, dict] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            code = (row.get("") or "").strip()
            if not code:
                continue
            categories = [
                cat
                for cat in ("diagnostic", "rhythm", "form")
                if (row.get(cat) or "").strip().startswith("1")
            ]
            statement_map[code] = {
                "categories": categories,
                "diagnostic_class": (row.get("diagnostic_class") or "").strip() or None,
            }
    return statement_map