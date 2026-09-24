#!/usr/bin/env python3
"""Smoke-load: confirma que `wfdb` lee los registros reales del corte reproducible.

    python tests/fixtures/corte_real/smoke_load.py

Genera el corte previamente con `make_cut.py`. Sale con código != 0 si algún registro
no se lee o no cumple el nominal (12 derivaciones, 100 Hz, ~10 s, unidades mV).
"""

import sys
from pathlib import Path

import numpy as np
import wfdb

OUT = Path(__file__).resolve().parent / "ptbxl_cut"


def main():
    headers = sorted(OUT.rglob("*.hea"))
    if not headers:
        sys.exit(f"ERROR: sin registros en {OUT} — ejecutar make_cut.py primero.")

    ok = 0
    bad = 0
    for hea in headers:
        rec = str(hea).replace(".hea", "")
        sig, meta = wfdb.rdsamp(rec)
        problems = []
        if meta["fs"] != 100:
            problems.append(f"fs={meta['fs']}")
        if sig.shape[1] != 12:
            problems.append(f"n_leads={sig.shape[1]}")
        if sig.shape[0] < 800:
            problems.append(f"n_samples={sig.shape[0]}")
        if not set(meta["units"]) == {"mV"}:
            problems.append(f"units={set(meta['units'])}")
        if np.isnan(sig).any():
            problems.append("NaN en señal")
        status = "OK" if not problems else "FAIL: " + ", ".join(problems)
        if problems:
            bad += 1
        else:
            ok += 1
        print(f"{status}  {rec}")

    print(f"\n{len(headers)} registros revisados: {ok} OK, {bad} con problemas.")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()