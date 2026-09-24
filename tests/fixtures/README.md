# Fixtures (L0) — referenciadas por los contratos (`docs/contracts.md`)

Subcarpetas, una por caso. Solo el carril L0 edita esta carpeta.

| Caso | Subcarpeta | Contenido | Para qué carril | Contract |
|---|---|---|---|---|
| Señal nominal | `nominal/` | `nominal.hea/.dat` — 12 derivaciones × 100 Hz × 10 s (1000 muestras), formulato WFDB int16 gain 1000 → mV, idéntico a PTB-XL | LA (ingesta señal) | RF-01 / Done-1 |
| Señal degradada | `degradada/` | `degradada.hea/.dat` — 11 derivaciones (falta `I`), resto nominal | LA, LC | RF-08 / Done-3 |
| Fallo total de features | `fallo_total/` | `fallo_total.hea/.dat` — header válido + 12 trazados planos (0), extracción imposible | LC | RF-08 / Done-3 |
| No-EKG | `no_ekg/` | `no_ekg.hea/.dat` — header ilegible → rechazo sin predicción | LA | RF-08b / Done-4 |
| Imagen de juguete | `imagen/` | PNG/JPG con 12 trazados, con 3 trazados (rechazo) y no-ECG | LB | RF-02 / RF-08b |
| Corte real reproducible | `corte_real/` | `cut.csv` (manifest versionado) + `ptbxl_cut/` (gitignored) + `make_cut.py` + `smoke_load.py` | LD, LF, evaluaciones | RF-05 / Done-6 |

## Regenerar

```bash
python tests/fixtures/generate_all.py           # sintéticas (señal + imagen), determinista
python tests/fixtures/corte_real/make_cut.py    # corte real desde data/ptb-xl/ (requiere dataset completo)
python tests/fixtures/corte_real/smoke_load.py  # confirma wfdb lee el corte real
```

## Cómo consumir (sin tocar el dataset completo)

- Señal WFDB: `wfdb.rdsamp("tests/fixtures/<caso>/<caso>")` → `(p_signal, meta)` con
  `p_signal` en mV, commit de derivaciones en orden canónico (contracts §1.2).
  En `degradada/` la derivación `I` está ausente (11 canales, mismo orden canónico sin `I`).
- Corte real: leer `corte_real/cut.csv` para conocer `ecg_id`, `strat_fold`,
  `scp_codes`, `diagnostic_superclass`, `rhythm_eval`, `specific_eval`,
  `declared_codes` y `roles`; abrir cualquier registro con
  `wfdb.rdsamp("tests/fixtures/corte_real/ptbxl_cut/<filename_lr>")`. Los archivos de
  señal están gitignored (Done-8/RF-14): en una máquina nueva conviene correr
  `make_cut.py` primero (requiere `data/ptb-xl/`).
- Imágenes: abrir con PIL/`matplotlib.image`; el nombre indica el caso.

Cobertura del corte real (contracts v1.2, §1.4): el corte cubre los 12 targets
evaluados de los TRES ejes — eje 1 superclases (`NORM`, `MI`, `STTC`, `CD`, `HYP`),
eje 2 ritmos (`SR`, `AFIB`, `STACH`, `SARRH`, `SBRAD`) y eje 3 específicos (`ASMI`,
`1AVB`) — más un registro del catálogo declarado-no-evaluado (roles `DECLARED:*`:
`AFLT`, `SVTAC`, `PSVT`, `SVARR`, `2AVB`, `3AVB`). La categoría de anotación
(diagnostic/rhythm/form) se lee de `scp_statements.csv`, no se asume (§1.4.5).
`VT`, `VF` y ritmo nodal no existen como statement en PTB-XL (0 ocurrencias): quedan
declaradas por contrato, sin muestra en el corte ni en el dataset (RF-11, T09).
El cut se regenera con `python tests/fixtures/corte_real/make_cut.py`.