"""Tests de la política de avisos de predicción (RF-06/RF-07a–d, T10, carril LE).

Verifica la tabla de casos del Apéndice A de `docs/contracts.md` (T03) en el alcance
de T10 (avisos 07a–07d; la prevalencia de RF-08 sobre `quality08` por entrada
degradada con p ≥ 0,60 y el fallo total tienen sus tests en T11):

- C1 nominal sin avisos (RF-06): todas las p ≥ 0,60 → `alerts` vacías.
- C2 p < 0,60 → `lowconf07b` sobre cada clave afectada, por eje (RF-07b), con el
  texto literal `confianza baja` (AGENTS regla 3 / contracts §2.6.1).
- C3/C4 eje vacío (todas sus p < 0,60) → top-1 del eje como `top_label` +
  `lowconf07b`, conservando las probabilidades (RF-07c).
- C5 superclase siempre con `top_label`, aun si todas sus p < 0,60 (RF-07c).
- D1–D3 catálogo declarado §1.4.4 (incl. `VT`/`VF`/NODAL, sin statement) siempre
  listado con el aviso fijo `insuf14` y sin probabilidad (RF-07a). El caso v1.0
  "VT con p=0,30 → avisos 07a+07b apilados" no existe en v1.2: VT es declarada y su
  único aviso es `insuf14` (contracts §2.6.1 «Cuándo NO aplica»).
- C8 apilado RF-07d: entrada degradada con clave p < 0,60 → `quality08` a nivel de
  `Prediction` + `lowconf07b` sobre la clave; los avisos se apilan, nunca se suprimen.
- C6 prevalece RF-08 sobre RF-07b (T11): entrada degradada con p ≥ 0,60 sale como
  NO confiable con aviso de calidad (`quality08` a nivel `Prediction`), aun sin
  ninguna clave bajo el umbral.
- C9 fallo total (T11): `quality == "impossible"` → `outcome == "incertidumbre"`
  sin ejes ni etiquetas ni catálogo declarado; RF-07c no aplica (no hay
  probabilidades que rankear).

Los textos literales obligatorios (§2.6.1) se verifican contra `ALERT_TEXTS`.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import (  # noqa: E402
    DECLARED_CATALOG,
    DECLARED_NO_STATEMENT,
    RHYTHM_EVALUATED,
    SPECIFIC_EVALUATED,
    SUPERCLASS_ORDER,
)
from src.policy import (  # noqa: E402
    ALERT_TEXTS,
    CODE_INSUF14,
    CODE_LOWCONF,
    CODE_QUALITY08,
    INSUF14_TEXT,
    LOWCONF_TEXT,
    QUALITY08_TEXT,
    THRESHOLD,
    Prediction,
    RhythmPrediction,
    SpecificDiagnosticPrediction,
    SuperclassPrediction5,
    build_prediction,
    declared_entries,
)


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


def test_c1_nominal_sin_avisos():
    p = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="nominal")
    assert isinstance(p, Prediction)
    assert p.outcome == "prediction"
    assert p.quality == "nominal"
    assert p.superclass is not None
    assert p.rhythm is not None
    assert p.specific_diagnostic is not None
    assert all(p.alerts[key] == [] for key in p.alerts)  # listas de alerts vacías
    assert p.prediction_alerts == []
    assert p.superclass.top_label == "NORM"


def test_c2_lowconf07b_por_clave_solo_donde_p_baja():
    p = build_prediction(
        "rec_01",
        _super(MI=0.30),
        _rhyth(AFIB=0.45),
        _spec(**{"1AVB": 0.10}),
        quality="nominal",
    )
    assert p.alerts["MI"] == [CODE_LOWCONF]
    assert p.alerts["AFIB"] == [CODE_LOWCONF]
    assert p.alerts["1AVB"] == [CODE_LOWCONF]
    assert p.alerts["NORM"] == []  # p=0.90 ≥ 0,60 → sin aviso
    assert p.alerts["SR"] == []
    assert p.alerts["ASMI"] == []


def test_lowconf07b_texto_literal_de_agentes_regla_3():
    assert LOWCONF_TEXT == "confianza baja"
    assert ALERT_TEXTS[CODE_LOWCONF] == LOWCONF_TEXT
    p = build_prediction("rec_01", _super(MI=0.30), quality="nominal")
    assert p.alerts["MI"] == [CODE_LOWCONF]
    assert ALERT_TEXTS[p.alerts["MI"][0]] == "confianza baja"


def test_umbral_exacto_060_no_genera_lowconf07b():
    assert THRESHOLD == 0.60
    assert build_prediction("rec_01", _super(NORM=0.60)).alerts["NORM"] == []
    assert build_prediction("rec_01", _super(NORM=0.59)).alerts["NORM"] == [CODE_LOWCONF]


def test_c3_ritmo_vacio_muestra_top1_con_lowconf():
    probs = _rhyth(SR=0.20, AFIB=0.40, STACH=0.10, SARRH=0.05, SBRAD=0.30)
    p = build_prediction("rec_01", _super(), probs, _spec(), quality="nominal")
    assert p.rhythm is not None
    assert p.rhythm.top_label == "AFIB"  # top-1 del eje expuesto (RF-07c)
    assert set(p.rhythm.probs) == set(RHYTHM_EVALUATED)  # se conservan las 5 probs
    assert p.alerts["AFIB"] == [CODE_LOWCONF]
    assert p.superclass.top_label == "NORM"  # eje 1 sigue presente


def test_c4_specific_vacio_muestra_top1_con_lowconf():
    probs = _spec(ASMI=0.10, **{"1AVB": 0.30})
    p = build_prediction("rec_01", _super(), _rhyth(), probs, quality="nominal")
    assert p.specific_diagnostic is not None
    assert p.specific_diagnostic.top_label == "1AVB"  # top-1 expuesto
    assert set(p.specific_diagnostic.probs) == set(SPECIFIC_EVALUATED)
    assert p.alerts["1AVB"] == [CODE_LOWCONF]


def test_c5_superclass_siempre_tiene_top_label():
    p = build_prediction(
        "rec_01",
        _super(NORM=0.20, MI=0.40, STTC=0.50, CD=0.30, HYP=0.10),
        quality="nominal",
    )
    assert p.superclass.top_label == "STTC"
    for key in SUPERCLASS_ORDER:
        assert p.alerts[key] == [CODE_LOWCONF]  # todas p < 0,60 → lowconf por clave


def test_desempate_top_label_por_primer_orden_canonico():
    p = build_prediction("rec_01", _super(NORM=0.90, MI=0.90, STTC=0.80))
    assert p.superclass.top_label == "NORM"  # NORM precede a MI en §1.4.1
    p2 = build_prediction("rec_01", _super(), _rhyth(AFIB=0.90, SR=0.90))
    assert p2.rhythm.top_label == "SR"  # SR precede a AFIB en §1.4.2


def test_declaradas_siempre_listadas_en_orden_del_catalogo():
    p = build_prediction("rec_01", _super())
    assert list(p.declared_not_evaluated) == DECLARED_CATALOG
    for key in ("AFLT", "SVTAC", "PSVT", "SVARR", "VT", "VF", "NODAL", "2AVB", "3AVB"):
        assert key in p.declared_not_evaluated


def test_insuf14_texto_literal_de_agentes_regla_3():
    assert INSUF14_TEXT == "dato insuficiente en el set de entrenamiento"
    assert ALERT_TEXTS[CODE_INSUF14] == INSUF14_TEXT
    for key, code in declared_entries():
        assert code == CODE_INSUF14
        assert ALERT_TEXTS[code] == INSUF14_TEXT


def test_declaradas_nunca_llevan_probabilidad_calculada():
    p = build_prediction("rec_01", _super(), _rhyth(), _spec())
    keys_con_p = set(p.superclass.probs)
    if p.rhythm is not None:
        keys_con_p |= set(p.rhythm.probs)
    if p.specific_diagnostic is not None:
        keys_con_p |= set(p.specific_diagnostic.probs)
    for key in p.declared_not_evaluated:
        assert key not in keys_con_p  # nunca ocupan clave de `probs` (§2.5)
        assert key not in p.alerts  # insuf14 es exclusivo de las declaradas


def test_vt_declarada_y_sin_07b_migracion_v12():
    # El caso v1.0 "VT con p=0,30 → avisos 07a+07b apilados" no existe en v1.2
    # (contracts §2.6.1 «Cuándo NO aplica»): VT se declara sin probabilidad; su
    # único aviso es `insuf14`, no se calcula ni apila `lowconf07b` sobre ella.
    for key in DECLARED_NO_STATEMENT:  # VT, VF, NODAL: sin statement directo
        assert key in DECLARED_CATALOG
        assert key not in RHYTHM_EVALUATED
        assert key not in SPECIFIC_EVALUATED
    p = build_prediction("rec_01", _super())
    assert "VT" in p.declared_not_evaluated
    assert "VT" not in p.alerts
    code_of_vt = dict(declared_entries())["VT"]
    assert code_of_vt == CODE_INSUF14
    assert ALERT_TEXTS[code_of_vt] == INSUF14_TEXT


def test_insuf14_no_se_apila_con_lowconf_sobre_la_misma_clave():
    p = build_prediction("rec_01", _super(), _rhyth(), _spec())
    for key in p.declared_not_evaluated:
        assert p.alerts.get(key) is None  # no hay p para calcular lowconf sobre ella
    assert "3AVB" not in p.superclass.probs
    assert "AFLT" not in p.rhythm.probs


def test_c8_apilado_rf07d_degradada_mas_lowconf_no_se_suprime():
    p = build_prediction(
        "rec_01",
        _super(MI=0.30),
        _rhyth(),
        _spec(),
        quality="degraded",
    )
    assert p.outcome == "prediction"
    assert p.quality == "degraded"
    assert p.prediction_alerts == [CODE_QUALITY08]  # aviso de calidad a nivel Prediction
    assert p.alerts["MI"] == [CODE_LOWCONF]  # no se suprime lowconf por estar degradada
    assert p.alerts["NORM"] == []  # degradada no inventa lowconf en claves con p alta


def test_quality08_texto_literal():
    assert QUALITY08_TEXT == "entrada degradada: predicción no confiable"
    assert ALERT_TEXTS[CODE_QUALITY08] == QUALITY08_TEXT


def test_ritmo_y_specific_ausentes_son_none_y_eje1_siempre():
    p = build_prediction("rec_01", _super())
    assert p.superclass is not None  # eje 1 se muestra siempre (invariante 1)
    assert set(p.superclass.probs) == set(SUPERCLASS_ORDER)
    assert p.rhythm is None
    assert p.specific_diagnostic is None


def test_tipos_por_eje():
    p = build_prediction("rec_01", _super(), _rhyth(), _spec())
    assert isinstance(p.superclass, SuperclassPrediction5)
    assert isinstance(p.rhythm, RhythmPrediction)
    assert isinstance(p.specific_diagnostic, SpecificDiagnosticPrediction)
    assert len(p.superclass.probs) == 5
    assert len(p.rhythm.probs) == 5
    assert len(p.specific_diagnostic.probs) == 2


def test_record_id_se_propaga():
    for p in (
        build_prediction("mi_registro", _super()),
        build_prediction("mi_registro", _super(), quality="degraded"),
    ):
        assert p.record_id == "mi_registro"


def test_claves_de_orden_canonico_obligatorias():
    with pytest.raises(ValueError):
        build_prediction("rec_01", {"NORM": 0.90})  # claves incompletas
    with pytest.raises(ValueError):
        build_prediction("rec_01", _super(), _rhyth(EXTRA=0.90))  # clave no canónica


def test_probabilidades_dentro_de_0_1():
    with pytest.raises(ValueError):
        build_prediction("rec_01", _super(NORM=1.50))
    with pytest.raises(ValueError):
        build_prediction("rec_01", _super(HYP=-0.10))


def test_quality_desconocida_raise():
    with pytest.raises(ValueError):
        build_prediction("rec_01", _super(), quality="rara")


def test_c6_degradada_p_alta_no_confiable_con_aviso_de_calidad():
    # RF-08 prevalece sobre RF-07b (T11): entrada degradada con todas sus p ≥ 0,60
    # NO se presenta como predcción confiada: lleva `quality08` (texto "entrada
    # degradada: predicción no confiable") sin que ninguna clave dispare lowconf07b.
    p = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="degraded")
    assert p.outcome == "prediction"
    assert p.quality == "degraded"
    assert p.superclass.top_label == "NORM"
    assert p.superclass.probs["NORM"] >= THRESHOLD  # p alta, umbral no dispara...
    assert all(p.alerts[key] == [] for key in p.alerts)  # ...ni lowconf sobre claves
    assert p.prediction_alerts == [CODE_QUALITY08]  # ...pero quality08 marca no confiable
    assert ALERT_TEXTS[p.prediction_alerts[0]] == QUALITY08_TEXT


def test_c6_degradada_con_p_justo_060_prevalece_quality08():
    # Borde del umbral: clave con p == 0,60 en entrada degradada no recibe
    # lowconf07b, pero el aviso de calidad sigue presente (prevalencia de RF-08).
    p = build_prediction("rec_01", _super(NORM=0.60), quality="degraded")
    assert p.alerts["NORM"] == []
    assert p.prediction_alerts == [CODE_QUALITY08]


def test_c6_degradada_con_top1_p_alta_no_finge_confianza():
    # Aun con todas las p ≥ 0,60 y top-1 claro, la entrada degradada nunca produce
    # una presentación confiada: `quality08` está presente (RF-08 > RF-06/RF-07b).
    probs = _super(MI=0.95, NORM=0.05)
    p = build_prediction("rec_01", probs, _rhyth(), _spec(), quality="degraded")
    assert p.superclass.top_label == "MI"
    assert p.superclass.probs["MI"] >= THRESHOLD
    assert p.alerts["MI"] == []
    assert p.prediction_alerts == [CODE_QUALITY08]


def test_c9_fallo_total_incertidumbre_sin_etiquetas():
    # RF-08 (T11): extracción imposible → `outcome == incertidumbre`, sin ejes,
    # sin etiquetas, sin catálogo declarado y sin alertas por clave; RF-07c no
    # aplica porque no existen probabilidades (C9).
    p = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="impossible")
    assert p.outcome == "incertidumbre"
    assert p.quality == "impossible"
    assert p.superclass is None  # ningún eje presente (invariante de presentación)
    assert p.rhythm is None
    assert p.specific_diagnostic is None
    assert p.declared_not_evaluated == []  # ni siquiera el catálogo declarado
    assert p.alerts == {}  # sin alertas por clave
    assert p.prediction_alerts == [CODE_QUALITY08]  # solo el aviso de calidad/incertidumbre
    assert ALERT_TEXTS[p.prediction_alerts[0]] == QUALITY08_TEXT


def test_c9_fallo_total_rf07c_no_aplica_aunque_hubiera_probs():
    # Aun si se pasaran probabilidades, con quality == "impossible" estas se
    # ignoran: no hay top-1, lowconf07b ni eje alguno (RF-07c no aplica, C9).
    p = build_prediction("rec_01", _super(), _rhyth(), _spec(), quality="impossible")
    for key in SUPERCLASS_ORDER:
        assert key not in p.alerts  # sin lowconf07b ni top-1
    assert p.prediction_alerts == [CODE_QUALITY08]


def test_c9_fallo_total_no_lista_declaradas():
    # El catálogo §1.4.4 solo aparece cuando hay predicción (outcome prediction);
    # en incertidumbre la salida no presenta ni siquiera las claves declaradas.
    for key in DECLARED_CATALOG:
        assert key not in build_prediction(
            "rec_01", _super(), quality="impossible"
        ).declared_not_evaluated