"""Tests del mapeo de statements PTB-XL a `LabelVector` de tres ejes (RF-05, T09).

Cubre RF-05 según `docs/contracts.md` v1.2 (§1.4, §2.5): NORM como etiqueta plana
(sin exclusión frente a patológicas), co-ocurrencia de superclases, los tres ejes
independientes y la partición evaluado/declarado (contracts §1.4.4). Se verifica
con `scp_codes` sintéticos (cabeceras sintéticas) y, cuando el dataset está
presente, cruzando contra `tests/fixtures/corte_real/cut.csv` y el
`scp_statements.csv` real (contracts §1.4.5).
"""

import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import (  # noqa: E402
    ABOVE_THRESHOLD_NOT_EVALUATED,
    DECLARED_CATALOG,
    DECLARED_NO_STATEMENT,
    DECLARED_RHYTHM,
    DECLARED_SPECIFIC_DIAGNOSTIC,
    EVALUATED_TARGETS,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_OF,
    SUPERCLASS_ORDER,
    SCP_STATEMENTS_CSV,
    LabelVector,
    build_label_vector,
    declared_catalog_from_counts,
    declared_codes_for,
    load_statement_map,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CUT_CSV = FIXTURES / "corte_real" / "cut.csv"
DATABASE_CSV = ROOT / "data" / "ptb-xl" / "ptbxl_database.csv"


def _active_keys(label_dict):
    return {key for key, value in label_dict.items() if value}


def test_vector_tiene_las_12_salidas_evaluadas():
    lv = build_label_vector("synth_01", {"SR": 0.0})
    assert len(EVALUATED_TARGETS) == 12
    assert len(lv.superclass_labels) == 5
    assert len(lv.rhythm_labels) == 5
    assert len(lv.specific_diagnostic_labels) == 2


def test_ordenes_canonicos_por_eje():
    lv = build_label_vector("synth_01", {})
    assert list(lv.superclass_labels) == SUPERCLASS_ORDER
    assert list(lv.rhythm_labels) == RHYTHM_EVALUATED
    assert list(lv.specific_diagnostic_labels) == SPECIFIC_EVALUATED


def test_norm_es_etiqueta_plana_sin_exclusion():
    lv = build_label_vector("synth_01", {"NORM": 100.0, "IMI": 100.0, "SR": 0.0})
    assert lv.superclass_labels["NORM"] == 1
    assert lv.superclass_labels["MI"] == 1


def test_norm_aporta_superclass_norm():
    lv = build_label_vector("synth_01", {"NORM": 80.0})
    assert lv.superclass_labels["NORM"] == 1
    assert lv.superclass_labels["MI"] == 0


def test_superclass_unica_desde_diagnostic_class():
    lv = build_label_vector("synth_01", {"LVH": 100.0})
    assert _active_keys(lv.superclass_labels) == {"HYP"}


def test_coutrencia_de_superclasses_deja_todas_activas():
    lv = build_label_vector("synth_01", {"IMI": 15.0, "IVCD": 100.0, "SR": 0.0})
    assert _active_keys(lv.superclass_labels) == {"MI", "CD"}


def test_superclass_sin_mezclar_statements_form():
    lv = build_label_vector("synth_01", {"STD_": 0.0, "ABQRS": 0.0})
    assert set(lv.superclass_labels.values()) == {0}


def test_ritmo_evaluado_se_activa():
    lv = build_label_vector("synth_01", {"SR": 0.0, "AFIB": 100.0})
    assert lv.rhythm_labels["SR"] == 1
    assert lv.rhythm_labels["AFIB"] == 1
    assert lv.rhythm_labels["SBRAD"] == 0


def test_ritmo_declarado_no_ocupa_clave_evaluada():
    lv = build_label_vector("synth_01", {"AFLT": 100.0, "AFIB": 0.0})
    assert "AFLT" not in lv.rhythm_labels
    assert set(lv.rhythm_labels.values()) == {0, 1}
    assert lv.rhythm_labels["AFIB"] == 1


def test_specific_evaluado_se_activa():
    lv = build_label_vector("synth_01", {"ASMI": 50.0, "1AVB": 100.0})
    assert lv.specific_diagnostic_labels["ASMI"] == 1
    assert lv.specific_diagnostic_labels["1AVB"] == 1


def test_specific_declarado_no_ocupa_clave_evaluada():
    lv = build_label_vector("synth_01", {"2AVB": 100.0, "3AVB": 100.0})
    assert "2AVB" not in lv.specific_diagnostic_labels
    assert "3AVB" not in lv.specific_diagnostic_labels
    assert set(lv.specific_diagnostic_labels.values()) == {0}


def test_tres_ejes_independientes_a_la_vez():
    lv = build_label_vector(
        "synth_01", {"NST_": 100.0, "SBRAD": 0.0, "1AVB": 100.0}
    )
    # NST_ activa STTC y 1AVB activa CD en el eje 1; SBRAD y 1AVB son ejes aparte.
    assert _active_keys(lv.superclass_labels) == {"STTC", "CD"}
    assert _active_keys(lv.rhythm_labels) == {"SBRAD"}
    assert _active_keys(lv.specific_diagnostic_labels) == {"1AVB"}


def test_valores_binarios_en_todos_los_ejes():
    lv = build_label_vector("synth_01", {"NORM": 100.0, "SR": 0.0})
    for labels in (
        lv.superclass_labels,
        lv.rhythm_labels,
        lv.specific_diagnostic_labels,
    ):
        assert set(labels.values()) <= {0, 1}


def test_scp_codes_como_string_del_dataset():
    lv = build_label_vector("synth_01", "{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}")
    assert lv.superclass_labels["NORM"] == 1
    assert lv.rhythm_labels["SR"] == 1
    assert lv.source_statements["NORM"] == 100.0


def test_source_statements_conserva_scp_codes_original():
    lv = build_label_vector("synth_01", {"IMI": 35.0, "NST_": 0.0})
    assert lv.source_statements == {"IMI": 35.0, "NST_": 0.0}


def test_registro_vacio_da_todo_ceros():
    lv = build_label_vector("synth_01", {})
    assert set(lv.superclass_labels.values()) == {0}
    assert set(lv.rhythm_labels.values()) == {0}
    assert set(lv.specific_diagnostic_labels.values()) == {0}


def test_statement_desconocido_se_ignora_pero_no_se_pierde():
    lv = build_label_vector("synth_01", {"TAL_VEZ_NO_EXISTE": 100.0, "NORM": 100.0})
    assert _active_keys(lv.superclass_labels) == {"NORM"}
    assert lv.source_statements == {"TAL_VEZ_NO_EXISTE": 100.0, "NORM": 100.0}


def test_declared_catalog_contiene_las_claves_del_contrato():
    for code in ("AFLT", "SVTAC", "PSVT", "SVARR", "VT", "VF", "NODAL", "2AVB", "3AVB"):
        assert code in DECLARED_CATALOG


def test_declared_no_statements_no_tienen_statement():
    for code in DECLARED_NO_STATEMENT:
        assert code not in SUPERCLASS_OF
        assert code not in RHYTHM_EVALUATED
        assert code not in SPECIFIC_EVALUATED


def test_declared_specific_es_el_resto_del_eje_diagnostico():
    assert set(DECLARED_SPECIFIC_DIAGNOSTIC).issubset(
        set(SUPERCLASS_OF) - set(SPECIFIC_EVALUATED)
    )
    assert "2AVB" in DECLARED_SPECIFIC_DIAGNOSTIC
    assert "3AVB" in DECLARED_SPECIFIC_DIAGNOSTIC
    assert set(DECLARED_SPECIFIC_DIAGNOSTIC).isdisjoint(ABOVE_THRESHOLD_NOT_EVALUATED)


def test_declaradas_solo_las_subrepresentadas_no_norm():
    # NORM/IMI tienen datos de sobra: NUNCA declaradas como insuficientes (§1.4.4).
    assert "NORM" not in DECLARED_CATALOG
    assert "IMI" not in DECLARED_CATALOG
    assert "2AVB" in DECLARED_CATALOG
    assert "3AVB" in DECLARED_CATALOG


def test_above_threshold_not_evaluated_es_particion_limpia_de_diagnostic():
    complemento = (
        set(SUPERCLASS_OF)
        - set(SPECIFIC_EVALUATED)
        - set(DECLARED_SPECIFIC_DIAGNOSTIC)
    )
    assert complemento == set(ABOVE_THRESHOLD_NOT_EVALUATED)


def test_declared_rhythm_es_el_resto_de_la_categoria_rhythm():
    assert "AFLT" in DECLARED_RHYTHM
    assert "SVTAC" in DECLARED_RHYTHM
    assert set(RHYTHM_EVALUATED).isdisjoint(DECLARED_RHYTHM)


def test_declared_codes_para_un_registro():
    scp = {"AFLT": 0.0, "NORM": 100.0, "IMI": 35.0, "3AVB": 100.0}
    declared = declared_codes_for(scp)
    assert "AFLT" in declared
    assert "3AVB" in declared
    assert "NORM" not in declared


def test_consistencia_con_corte_real_si_presente():
    if not CUT_CSV.exists():
        import pytest

        pytest.skip("corte real ausente: ejecutar tests/fixtures/corte_real/make_cut.py")

    with CUT_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "corte vacío"
    for row in rows:
        rec_id = f"ecg_{row['ecg_id']}"
        lv = build_label_vector(rec_id, row["scp_codes"])
        expected_super = set(ast.literal_eval(row["diagnostic_superclass"]))
        expected_rhythm = set(ast.literal_eval(row["rhythm_eval"]))
        expected_spec = set(ast.literal_eval(row["specific_eval"]))
        assert _active_keys(lv.superclass_labels) == expected_super, row["ecg_id"]
        assert _active_keys(lv.rhythm_labels) == expected_rhythm, row["ecg_id"]
        assert _active_keys(lv.specific_diagnostic_labels) == expected_spec, row["ecg_id"]


def test_tablas_incrustadas_coinciden_con_scp_statements_si_presente():
    if not SCP_STATEMENTS_CSV.exists():
        import pytest

        pytest.skip("dataset ausente: no se puede re-verificar con scp_statements.csv")

    real = load_statement_map(SCP_STATEMENTS_CSV)

    for code, superclass in SUPERCLASS_OF.items():
        assert code in real, f"{code} debería estar en scp_statements.csv"
        assert "diagnostic" in real[code]["categories"], code
        assert real[code]["diagnostic_class"] == superclass, code

    real_diagnostic = {
        code
        for code, info in real.items()
        if "diagnostic" in info["categories"]
    }
    real_rhythm = {
        code
        for code, info in real.items()
        if "rhythm" in info["categories"]
    }
    assert set(SUPERCLASS_OF) == real_diagnostic
    assert set(DECLARED_SPECIFIC_DIAGNOSTIC).issubset(
        real_diagnostic - set(SPECIFIC_EVALUATED)
    )
    assert set(ABOVE_THRESHOLD_NOT_EVALUATED).issubset(
        real_diagnostic - set(SPECIFIC_EVALUATED)
    )
    assert (
        set(DECLARED_SPECIFIC_DIAGNOSTIC) | set(ABOVE_THRESHOLD_NOT_EVALUATED)
        == real_diagnostic - set(SPECIFIC_EVALUATED)
    )
    assert set(DECLARED_RHYTHM) == real_rhythm - set(RHYTHM_EVALUATED)
    assert set(RHYTHM_EVALUATED).issubset(real_rhythm)


def test_superclass_of_documenta_el_diagnostic_class_de_cada_statement():
    assert SUPERCLASS_OF["NORM"] == "NORM"
    assert SUPERCLASS_OF["1AVB"] == "CD"
    assert SUPERCLASS_OF["3AVB"] == "CD"
    assert SUPERCLASS_OF["ASMI"] == "MI"
    assert len(SUPERCLASS_OF) == 44


def test_catalogo_declarado_coincide_con_los_conteos_reales():
    if not DATABASE_CSV.exists():
        import pytest

        pytest.skip("dataset ausente: no se puede verificar con ptbxl_database.csv")

    derived = declared_catalog_from_counts(DATABASE_CSV)
    assert derived["declared_rhythm"] == DECLARED_RHYTHM
    assert derived["declared_specific"] == DECLARED_SPECIFIC_DIAGNOSTIC
    assert derived["above_threshold_not_evaluated"] == ABOVE_THRESHOLD_NOT_EVALUATED


def test_vector_es_instancia_de_labelvector():
    lv = build_label_vector("synth_01", {"SR": 0.0})
    assert isinstance(lv, LabelVector)
    assert lv.record_id == "synth_01"