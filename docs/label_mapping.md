# Mapeo de etiquetas PTB-XL → salidas SinusAI

- **Versión:** vigente con `docs/contracts.md` v1.2 (T01 reabierto; TRES EJES independientes, §1.4).
- **Tarea:** T09 (carril LD). Referenciado por `TrainingRun.label_mapping_path` (contracts §2.7).
- **Código canónico:** `src/labels.py` (constantes `SUPERCLASS_OF`, `DECLARED_*`,
  `EVALUATED_TARGETS`). Las tablas de conteos de la sección §5 se pueden regenerar con
  `declared_catalog_from_counts()` y son re-verificadas por `tests/test_labels.py`.

Este documento cumple lo exigido por contracts §1.4.5: (a) la **categoría de anotación**
(`diagnostic`/`rhythm`/`form`) de cada statement usado, tomada directamente de
`scp_statements.csv` — no asumida por homonimia —, y (b) la **tabla completa de conteos
reales** (total y por folds de entrenamiento 1–8 de `strat_fold`) de las 12 claves
evaluadas y de cada entrada declarada de §1.4.4, de modo que la partición
evaluado/declarado quede trazable y verificable.

## 1. Resumen de los tres ejes

| Eje | Claves (orden canónico) | Categoría de anotación | Criterio de inclusión | Estado |
|---|---|---|---|---|
| Eje 1 — superclase diagnóstica | `NORM`, `MI`, `STTC`, `CD`, `HYP` | derivada de `diagnostic_class` de statements `diagnostic` | siempre evaluado | evaluado |
| Eje 2 — ritmo | `SR`, `AFIB`, `STACH`, `SARRH`, `SBRAD` | `rhythm` | ≥500 totales Y ≥350 en folds 1–8 | subconjunto evaluado |
| Eje 3 — diagnóstico específico | `ASMI`, `1AVB` | `diagnostic` | ≥500 totales Y ≥350 en folds 1–8 | subconjunto evaluado |

`VT`/`VF`/`NODAL` (bloqueo nodal) no existen como statement en PTB-XL (0 ocurrencias):
quedan en el catálogo declarado (§4). `NORM` participa como superclase **plana**, sin
lógica de exclusión frente a patológicas (RF-05, D-01).

## 2. Categorías de anotación — lectura de `scp_statements.csv`

La categoría se toma de las columnas `diagnostic`, `rhythm` y `form` del archivo
(1.0 significa pertenencia), no del nombre del statement. Un statement puede tener
varias categorías a la vez. Hallazgos verificados:

- 44 statements de categoría `diagnostic` (incluyen `NORM`).
- 12 statements de categoría `rhythm` (las 5 evaluadas + las 7 declaradas de §4).
- Statements con categoría doble `diagnostic`+`form`: `DIG`, `LNGQT`, `NDT`, `NST_`.
- `2AVB` y `3AVB` son categoría `diagnostic` con `diagnostic_class` `CD` y
  `diagnostic_subclass` `_AVB` — **no** son categoría `rhythm` (contracts §1.4.2 nota).
- Los statements de categoría `form` (ej. `STD_`, `ABQRS`, `PVC`, `QWAVE`, `LOWT`,
  `NT_`, `PAC`, `LPR`, `INVT`, `LVOLT`, `HVOLT`, `TAB_`, `STE_`, `PRC(S)`) **no**
  activan ningún eje: no son targets.

## 3. Reglas de mapeo (statement → etiqueta)

### 3.1 Eje 1 — superclase diagnóstica (5 salidas planas)

- Cada statement de categoría `diagnostic` presente en `scp_codes` activa su
  `diagnostic_class` (superclase). Tabla `SUPERCLASS_OF` completa en
  `src/labels.py` (44 statement → superclase, tomada de `scp_statements.csv`).
- `NORM` es etiqueta plana: un registro con `NORM` + patológica deja activas **ambas**
  (ej. `{'NORM': 80.0, '1AVB': 100.0}` → superclases `CD` y `NORM`).
- Ante co-ocurrencia de superclases, el mapeo activa **todas** las presentes
  (multietiqueta, RF-05). La resolución a `top_label` único para presentación es
  decisión de política de avisos (LE/T10), no del mapeo — el contrato (§1.4.1) llama
  a esto «superclase primaria»: consisten en que ninguna activa excluye a otra.

### 3.2 Eje 2 — ritmo evaluado (5 salidas)

- Solo los statements de `RHYTHM_EVALUATED` (`SR`, `AFIB`, `STACH`, `SARRH`, `SBRAD`)
  activan su clave binaria. El resto de la categoría `rhythm` queda en el catálogo
  declarado (§4) y **no** ocupa clave.

### 3.3 Eje 3 — diagnóstico específico evaluado (2 salidas)

- Solo los statements de `SPECIFIC_EVALUATED` (`ASMI`, `1AVB`) activan su clave.
  `ASMI` es el statement con `diagnostic_class` `MI` (subclass `AMI`); `1AVB` es el
  statement con `diagnostic_class` `CD` (subclass `_AVB`). El resto de la categoría
  `diagnostic` queda declarado (§4).

### 3.4 `source_statements`

`LabelVector.source_statements` conserva los `scp_codes` del registro tal cual
vienen de `ptbxl_database.csv` (statement → peso), para trazabilidad.

## 4. Catálogo declarado-no-evaluado (contracts §1.4.4)

**Definición:** un statement de la categoría `rhythm` o `diagnostic` es **declarado**
(y no target de entrenamiento/evaluación) cuando **no cumple el criterio**: totales
< 500 Ó folds 1–8 < 350 (`MIN_TOTAL_CASES = 500`, `MIN_TRAIN_CASES = 350`).
Las claves declaradas **nunca** llevan probabilidad calculada ni ocupan clave en los
vectores de ejes; se muestran en la presentación con el aviso fijo `insuf14`
(`dato insuficiente en el set de entrenamiento`, contracts §2.6.1).

| Eje | Códigos declarados | Conteo máximo | Aviso |
|---|---|---|---|
| 2 | `AFLT`, `SVTAC`, `PSVT`, `SVARR`, `PACE`, `BIGU`, `TRIGU` | 294 (`PACE`) | `insuf14` |
| 2 (sin statement) | `VT`, `VF`, `NODAL` | 0 | `insuf14` |
| 3 | `2AVB`, `3AVB`, `ALMI`, `AMI`, `ANEUR`, `DIG`, `EL`, `ILBBB`, `ILMI`, `INJAL`, `INJAS`, `INJIL`, `INJIN`, `INJLA`, `IPLMI`, `IPMI`, `ISCAN`, `ISCAS`, `ISCIL`, `ISCIN`, `ISCLA`, `LAO/LAE`, `LMI`, `LNGQT`, `LPFB`, `PMI`, `RAO/RAE`, `RVH`, `SEHYP`, `WPW` | 478 (`ILMI`) | `insuf14` |

**Nota de honestidad (Constitution §1):** existen 12 statements de categoría
`diagnostic` que **sí** superan el criterio numérico pero no integran el subconjunto
evaluado fijado por el contrato (§1.4.3 evalúa únicamente `ASMI` y `1AVB`). NO se
declaran con `insuf14` (mentiría: tienen datos de sobra); se listan aparte en la
columna *cumplen criterio, no evaluadas* de §5.6 (`ABOVE_THRESHOLD_NOT_EVALUATED`).
La partición evaluado/declarado queda así trazable sin mezclar «sin datos» con
«no es target».

Los point-blind de AGENTS regla 3 (taquicardia ventricular `VT`, fibrilación
ventricular `VF`, bloqueo AV completo `3AVB`) están absorbidos por el catálogo
declarado: **nunca** reciben probabilidad y **siempre** con `insuf14` (v1.2,
contracts §2.6.1 «Cuándo NO aplica»).

## 5. Conteos reales (total y folds de entrenamiento 1–8)

Dataset: PTB-XL v1.0.3, 100 Hz, `data/ptb-xl/ptbxl_database.csv` (21 799 registros).
Split canónico: train = `strat_fold != 10`, test = `strat_fold == 10` (contracts §1.5).

### 5.1 Eje 1 — superclases (registros cuya `diagnostic_class` incluye la clase)

| Superclase | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| NORM | (derivada) | — | 9514 | 8551 |
| MI | (derivada) | — | 5469 | 4919 |
| STTC | (derivada) | — | 5235 | 4714 |
| CD | (derivada) | — | 4898 | 4402 |
| HYP | (derivada) | — | 2649 | 2387 |

### 5.2 Eje 2 — ritmo evaluado

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| SR | rhythm | — | 16748 | 15074 |
| AFIB | rhythm | — | 1514 | 1362 |
| STACH | rhythm | — | 826 | 744 |
| SARRH | rhythm | — | 772 | 695 |
| SBRAD | rhythm | — | 637 | 573 |

### 5.3 Eje 3 — diagnóstico específico evaluado

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| ASMI | diagnostic | MI | 2357 | 2123 |
| 1AVB | diagnostic | CD | 793 | 714 |

### 5.4 Declaradas eje 2 (ritmo sin criterio)

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| AFLT | rhythm | — | 73 | 66 |
| BIGU | rhythm | — | 82 | 74 |
| PACE | rhythm | — | 294 | 266 |
| PSVT | rhythm | — | 24 | 22 |
| SVARR | rhythm | — | 157 | 143 |
| SVTAC | rhythm | — | 27 | 24 |
| TRIGU | rhythm | — | 20 | 18 |

### 5.5 Declaradas eje 3 (diagnostic sin criterio)

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| 2AVB | diagnostic | CD | 14 | 13 |
| 3AVB | diagnostic | CD | 16 | 14 |
| ALMI | diagnostic | MI | 288 | 261 |
| AMI | diagnostic | MI | 353 | 318 |
| ANEUR | diagnostic | STTC | 104 | 94 |
| DIG | diagnostic/form | STTC | 181 | 163 |
| EL | diagnostic | STTC | 96 | 87 |
| ILBBB | diagnostic | CD | 77 | 69 |
| ILMI | diagnostic | MI | 478 | 430 |
| INJAL | diagnostic | MI | 145 | 131 |
| INJAS | diagnostic | MI | 214 | 192 |
| INJIL | diagnostic | MI | 15 | 13 |
| INJIN | diagnostic | MI | 18 | 16 |
| INJLA | diagnostic | MI | 17 | 15 |
| IPLMI | diagnostic | MI | 51 | 46 |
| IPMI | diagnostic | MI | 33 | 30 |
| ISCAN | diagnostic | STTC | 44 | 40 |
| ISCAS | diagnostic | STTC | 169 | 152 |
| ISCIL | diagnostic | STTC | 179 | 161 |
| ISCIN | diagnostic | STTC | 218 | 196 |
| ISCLA | diagnostic | STTC | 140 | 127 |
| LAO/LAE | diagnostic | HYP | 426 | 384 |
| LMI | diagnostic | MI | 201 | 181 |
| LNGQT | diagnostic/form | STTC | 117 | 106 |
| LPFB | diagnostic | CD | 177 | 159 |
| PMI | diagnostic | MI | 17 | 15 |
| RAO/RAE | diagnostic | HYP | 99 | 89 |
| RVH | diagnostic | HYP | 126 | 114 |
| SEHYP | diagnostic | HYP | 29 | 27 |
| WPW | diagnostic | CD | 79 | 71 |

### 5.6 Cumplen criterio pero no evaluadas (eje 3, transparentes)

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| CLBBB | diagnostic | CD | 536 | 482 |
| CRBBB | diagnostic | CD | 541 | 487 |
| IMI | diagnostic | MI | 2676 | 2409 |
| IRBBB | diagnostic | CD | 1118 | 1006 |
| ISCAL | diagnostic | STTC | 659 | 593 |
| ISC_ | diagnostic | STTC | 1272 | 1144 |
| IVCD | diagnostic | CD | 787 | 708 |
| LAFB | diagnostic | CD | 1623 | 1461 |
| LVH | diagnostic | HYP | 2132 | 1918 |
| NDT | diagnostic/form | STTC | 1825 | 1643 |
| NORM | diagnostic | NORM | 9514 | 8551 |
| NST_ | diagnostic/form | STTC | 767 | 690 |

### 5.7 Sin statement directo (declaradas siempre)

| Clave | Categoría | `diagnostic_class` | Total | Folds 1–8 |
|---|---|---|---|---|
| VT | — | — | 0 | 0 |
| VF | — | — | 0 | 0 |
| NODAL | — | — | 0 | 0 |

## 6. Verificación y trazabilidad

- `tests/test_labels.py` verifica el mapeo con **`scp_codes` sintéticos** (RF-05,
  incluyendo NORM plana y co-ocurrencia) y, cuando el dataset está presente, cruza
  contra `tests/fixtures/corte_real/cut.csv`, re-verifica `SUPERCLASS_OF` contra
  `scp_statements.csv` y re-computa el catálogo declarado contra `ptbxl_database.csv`.
- Los conteos de §5 se regeneran con `declared_catalog_from_counts()`; cualquier
  cambio de partición evaluado/declarado es visible como diff del catálogo y debe
  pasar por PR que actualice esta documentación.