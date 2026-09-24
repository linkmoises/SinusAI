#!/usr/bin/env python3
"""Genera el corte reproducible del dataset PTB-XL para `tests/fixtures/corte_real/`.

Reproducible: la selección es determinista (primeros registros por `ecg_id` por rol),
sin aleatoriedad. Cubre los TRES EJES del contrato (contracts §1.4, v1.2):

  - Eje 1 — superclase diagnóstica: NORM, MI, STTC, CD, HYP (derivada de
    `diagnostic_class` de los statements con `diagnostic == 1`).
  - Eje 2 — ritmo evaluado: SR, AFIB, STACH, SARRH, SBRAD (categoría `rhythm` de
    `scp_statements.csv`, columna `rhythm == 1`).
  - Eje 3 — diagnóstico específico evaluado: ASMI (bajo MI), 1AVB (bajo CD).

Además incorpora muestras del catálogo declarado-no-evaluado (contracts §1.4.4) para
que LD pueda verificar conteos y `reports/` los liste: AFLT, SVTAC, PSVT, SVARR
(ritmo), 2AVB y 3AVB (diagnostic/CD).

Cobertura de VT/VF/ritmo nodal: no existen como statement en PTB-XL (0 ocurrencias,
contracts §1.4.4). Permanecen declaradas en el contrato, sin muestra en el corte ni en
el dataset.

La categoría de anotación (rhythm/diagnostic/form) se lee directamente de
`scp_statements.csv`, no se asume por homonimia (contracts §1.4.5).

Uso (requiere el dataset completo en `data/ptb-xl/`, ver contracts §1):
    python tests/fixtures/corte_real/make_cut.py

Los archivos de señal copiados en `ptbxl_cut/` están gitignored: no se versionan
(honra Done-8/RF-14). El manifest `cut.csv` sí se versiona.
"""

import ast
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = REPO / "data" / "ptb-xl"
OUT = HERE / "ptbxl_cut"
MANIFEST = HERE / "cut.csv"

S5 = ["NORM", "MI", "STTC", "CD", "HYP"]
RHYTHM_EVAL = ["SR", "AFIB", "STACH", "SARRH", "SBRAD"]
SPEC_EVAL = ["ASMI", "1AVB"]
DECLARED_RHYTHM = ["AFLT", "SVTAC", "PSVT", "SVARR"]
DECLARED_DIAG = ["2AVB", "3AVB"]
DECLARED = DECLARED_RHYTHM + DECLARED_DIAG
# VT/VF/ritmo nodal: sinon statement en PTB-XL (0 ocurrencias), catalogadas (v1.2).
VT_VF_NODAL = ["VT", "VF", "NODAL"]

PURE_EACH = 4           # registros puros por superclase (eje 1)
EVAL_EACH = 2           # registros por clave evaluada de ejes 2 y 3
FOLD10_EACH = 1         # garantiza >=1 de test (strat_fold == 10) por rol evaluado
MULTI_N = 3             # registros con superclases co-ocurrentes
DECLARED_EACH = 1       # registro por clave del catálogo declarado (si existe)


def build_statement_map():
    """scp_statements.csv -> {code: ("diagnostic"/"rhythm"/"form", diagnostic_class)}.

    La categoría se toma de las columnas `diagnostic`/`rhythm`/`form`, no se asume.
    Un statement puede tener varias a la vez (ej. NDT es diagnostic y form).
    """
    import csv

    mapping = {}
    with (DATA / "scp_statements.csv").open() as fh:
        for row in csv.DictReader(fh):
            code = row[""]
            if not code:
                continue
            cats = [c for c in ("diagnostic", "rhythm", "form")
                    if (row.get(c, "") or "").strip().startswith("1")]
            mapping[code] = {
                "categories": cats,
                "diagnostic_class": row.get("diagnostic_class", "").strip() or None,
            }
    return mapping


def iter_records():
    import csv as _csv
    with (DATA / "ptbxl_database.csv").open() as fh:
        yield from _csv.DictReader(fh)


def analyze():
    smap = build_statement_map()
    recs = {}
    for r in iter_records():
        codes = ast.literal_eval(r["scp_codes"])
        supers = sorted({smap[k]["diagnostic_class"] for k in codes
                         if k in smap and smap[k]["diagnostic_class"]
                         and "diagnostic" in smap[k]["categories"]})
        rhythm_hit = sorted(c for c in RHYTHM_EVAL if c in codes)
        spec_hit = sorted(c for c in SPEC_EVAL if c in codes)
        declared_hit = sorted(c for c in DECLARED if c in codes)
        recs[int(r["ecg_id"])] = {
            "ecg_id": int(r["ecg_id"]),
            "patient_id": r["patient_id"],
            "strat_fold": int(r["strat_fold"]),
            "filename_lr": r["filename_lr"].strip(),
            "scp_codes": dict(codes),
            "diagnostic_superclass": supers,
            "rhythm_eval": rhythm_hit,
            "specific_eval": spec_hit,
            "declared_codes": declared_hit,
        }
    return recs


def _take(recs, pred, n, fold10=True):
    chosen, f10 = [], []
    for ecg in sorted(recs):
        if pred(recs[ecg]):
            (f10 if recs[ecg]["strat_fold"] == 10 else chosen).append(ecg)
    picks = chosen[:n]
    if fold10 and not any(recs[e]["strat_fold"] == 10 for e in picks) and f10:
        picks.append(f10[0])
    return picks


def select_manifest(recs):
    picks = set()
    roles_of = {}

    def _add(ecgs, role):
        for e in ecgs:
            picks.add(e)
            roles_of.setdefault(e, set()).add(role)

    for cls in S5:
        _add(_take(recs, lambda r, c=cls: r["diagnostic_superclass"] == [c], PURE_EACH), cls)
    _add(_take(recs, lambda r: len(r["diagnostic_superclass"]) >= 2, MULTI_N), "MULTI")
    for code in RHYTHM_EVAL:
        _add(_take(recs, lambda r, c=code: c in r["rhythm_eval"], EVAL_EACH), f"RHYTHM:{code}")
    for code in SPEC_EVAL:
        _add(_take(recs, lambda r, c=code: c in r["specific_eval"], EVAL_EACH), f"SPEC:{code}")
    for code in DECLARED:
        _add(_take(recs, lambda r, c=code: c in r["declared_codes"], DECLARED_EACH), f"DECLARED:{code}")

    rows = []
    for e in sorted(picks):
        r = recs[e]
        roles = set(r["diagnostic_superclass"]) | roles_of[e]
        rows.append({
            "ecg_id": r["ecg_id"],
            "patient_id": r["patient_id"],
            "strat_fold": r["strat_fold"],
            "filename_lr": r["filename_lr"],
            "scp_codes": repr(r["scp_codes"]),
            "diagnostic_superclass": repr(r["diagnostic_superclass"]),
            "rhythm_eval": repr(r["rhythm_eval"]),
            "specific_eval": repr(r["specific_eval"]),
            "declared_codes": repr(r["declared_codes"]),
            "roles": repr(sorted(roles)),
        })
    return rows


def write_manifest(rows):
    import csv
    fields = ["ecg_id", "patient_id", "strat_fold", "filename_lr", "scp_codes",
              "diagnostic_superclass", "rhythm_eval", "specific_eval",
              "declared_codes", "roles"]
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def copy_records(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows:
        rel = Path(r["filename_lr"])
        for ext in ("hea", "dat"):
            src = DATA / f"{rel}.{ext}"
            dst = OUT / f"{rel}.{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main():
    if not (DATA / "ptbxl_database.csv").exists():
        sys.exit(f"ERROR: dataset completo no encontrado en {DATA} — descargar PTB-XL 100Hz (README).")
    recs = analyze()
    rows = select_manifest(recs)
    write_manifest(rows)
    copy_records(rows)

    coverage = {}
    for r in rows:
        for role in ast.literal_eval(r["roles"]):
            coverage[role] = coverage.get(role, 0) + 1
    print("corte generado:", OUT)
    print(f"  {len(rows)} registros en {MANIFEST.name}")
    print("  cobertura por rol:", ", ".join(f"{k}={v}" for k, v in sorted(coverage.items())))

    # Cobertura mínima: los 12 targets evaluados de los tres ejes.
    for cls in S5:
        assert coverage.get(cls, 0) >= 1, f"eje 1 debe incluir {cls}"
    for code in RHYTHM_EVAL:
        assert coverage.get(f"RHYTHM:{code}", 0) >= 1, f"eje 2 debe incluir {code}"
    for code in SPEC_EVAL:
        assert coverage.get(f"SPEC:{code}", 0) >= 1, f"eje 3 debe incluir {code}"
    # Catálogo declarado con statement presente en PTB-XL: al menos 1 muestra.
    for code in DECLARED:
        assert coverage.get(f"DECLARED:{code}", 0) >= 1, f"catálogo declarado debe incluir {code}"
    # Sin registros de VT/VF/ritmo nodal (no existen como statement).
    got_vt_vf = {code: next((r["ecg_id"] for r in rows if code in ast.literal_eval(r["scp_codes"])), None)
                 for code in VT_VF_NODAL}
    for code, ecg in got_vt_vf.items():
        assert ecg is None, f"{code} no debe aparecer en el corte (statement inexistente), se halló en {ecg}"
    print(f"  VT/VF/ritmo nodal: 0 ocurrencias confirmadas (declaradas por diseño).")


if __name__ == "__main__":
    main()