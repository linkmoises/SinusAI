# Contracts — SinusAI v1.2 (T01 reabierto; congelado)

Única vía de comunicación entre carriles (tasks.md). Los carriles solo editan sus
archivos; todo intercambio de datos se resuelve contra este documento, sin preguntar
formato ni ubicación. Este archivo es propiedad del carril L0.

## Estado y reglas de uso

- Versión: `1.2` — T01 reabierto y vuelto a congelar: el esquema de salidas pasa de
  DOS NIVELES jerárquicos (v1.1) a TRES EJES INDEPENDIENTES — superclase
  diagnóstica (eje 1, siempre evaluada), ritmo (eje 2, subconjunto evaluable de los
  12 rhythm statements) y diagnóstico específico dentro del eje diagnóstico (eje 3,
  subconjunto evaluable de los 44 diagnostic statements) — ver §1.4, §2.5, §2.6.
  `diagnostic` y `rhythm` son categorías de anotación independientes en
  `scp_statements.csv` (un registro puede tener un statement de cada una a la vez):
  NO es una relación padre-hijo. Consumidores que deben releer este contrato antes
  de continuar: T09, T10, T13, T14. Cambios posteriores se hacen por PR que
  actualice la versión y los consumidores.
- Los tipos se escriben como contratos nominales (nombres de campo, tipos, unidades,
  invariantes), no como clases Python. Cada carril los implementa libremente en su
  módulo respetando el contrato.
- Textos literales y constantes de política en §2.6 son referencia canónica de LE/LG;
  la tabla de casos de avisos (qué se emite y cómo se apila) vive en el apéndice, tarea
  T03, no en este cuerpo.

---

## 1. Layout del dataset PTB-XL (100 Hz)

### 1.1 Raíz y catálogo

| Artefacto | Ruta (relativa al repo) | Contenido |
|---|---|---|
| Raíz del dataset | `data/ptb-xl/` | nunca se versiona (`.gitignore`) |
| Metadatos por registro | `data/ptb-xl/ptbxl_database.csv` | 21 799 filas, una por `ecg_id`; columnas: `ecg_id`, `patient_id`, `age`, `sex`, `scp_codes`, `strat_fold`, `filename_lr`, `filename_hr`, `report`, entre otras |
| Tabla de statements SCP | `data/ptb-xl/scp_statements.csv` | 71 statements SCP-ECG con `diagnostic` ∈ {1.0, 0.0}, `diagnostic_class`, `diagnostic_subclass`, descripción |
| Listado de registros | `data/ptb-xl/RECORDS` | 43 597 rutas sin extensión: 21 799 de `records100/` + 21 799 de `records500/` (una línea cada una) |
| Veces de versión | `data/ptb-xl/ptbxl_v103_changelog.txt`, `ptbxl_v102_changelog.txt`, `SHA256SUMS.txt`, `LICENSE.txt` | versión descargada: 1.0.3 |

### 1.2 Registros de señal (`records100/*.hea/.dat`)

- Patrón de ruta: `records100/{carpeta}/{ecg_id:05d}_lr.{hea,dat}`
  - `carpeta` = `(ecg_id // 1000) * 1000` con `"{:05d}"` (ej.: ecg_id 1 → `00000/00001_lr`;
    ecg_id 15709 → `15000/15709_lr`).
- Cada registro es un par WFDB: cabecera `.hea` (texto) + señal `.dat` (binario int16).
- Nominal: `12` derivaciones, `fs=100` Hz, `n_samples=1000` (= 10 s), línea 1 del `.hea`
  en el formato `nombre n_leads fs n_samples`. Verificado en muestra: todos `100`×`1000`.
- La señal es de formato WFDB; `wfdb.rdsamp`/`wfdb.rdrecord` la devuelven en la unidad
  del `.hea` (mV): `p_signal` con shape `(1000, 12)`, dtype float. El `.dat` guarda
  int16 con ganancia 1000 (baseline=0) → mV ya escaladas.
- Orden canónico de derivaciones (el que trae WFDB en PTB-XL):
  `["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]`.
  `Signal12.data[:, i]` corresponde a `lead_names[i]`; la ingesta normaliza a este orden.
- Desviaciones del nominal (duración distinta a 10 s, fs ≠ 100, derivaciones
  faltantes/cortadas) son **degradación** (RF-08), no rechazo.

### 1.3 Metadatos por registro (`ptbxl_database.csv`)

- Índice: `ecg_id`. `scp_codes` es un string de dict `{statement: prob}` (ej.
  `"{'NORM': 100.0, 'SR': 0.0}"`); se parsea con `ast.literal_eval`.
- `strat_fold` ∈ {1..10}: convención PTB-XL de split — fold 10 = test, resto = train.
- `filename_lr` / `filename_hr`: relativa sin extensión a `data/ptb-xl/` (la `_hr` es la
  de 500 Hz, fuera de alcance v1).

### 1.4 Tabla de etiquetas PTB-XL — tres ejes independientes (v1.2)

- `scp_statements.csv` asigna a cada statement su `diagnostic`, `diagnostic_class`,
  `diagnostic_subclass` y su **categoría de anotación** (`diagnostic` / `rhythm` /
  `form`). Categorías son independientes entre sí según la tabla original: un mismo
  registro puede tener un statement de cada categoría simultáneamente — **no es una
  relación padre-hijo**.
- Hay 44 statements con `diagnostic == 1` (categoría diagnóstica), con jerarquía
  interna en 5 superclases: `NORM`, `MI`, `STTC`, `CD`, `HYP`. Hay 12 statements de
  categoría `rhythm` (SIN superclase, categoría aparte).
- La etiqueta objetivo del sistema son TRES EJES PARALELOS:
  - **Eje 1** — superclase diagnóstica (siempre evaluado).
  - **Eje 2** — ritmo (evaluado solo donde hay datos suficientes).
  - **Eje 3** — diagnóstico específico dentro del eje diagnóstico (evaluado solo
    donde hay datos suficientes).
- Los ejes 2 y 3 son subconjuntos evaluables de sus categorías de anotación, no
  niveles subordinados del eje 1. Toda salida válida incluye el eje 1; ningún eje
  específico se emite ni se muestra solo (invariante de presentación, §2.6).

#### 1.4.1 Eje 1 — Superclase diagnóstica (siempre evaluado)

- Orden canónico: `["NORM", "MI", "STTC", "CD", "HYP"]` (5 claves).
- Se entrena y evalúa como clasificador multiclase estándar: miles de casos por
  superclase en PTB-XL y AUC publicado ~0,92–0,93 en la literatura de referencia
  (Strodthoff et al. 2020, Tabla II). Alta confianza esperada; una superclase
  predicha por registro, con ROC/AUC uno-vs-resto sobre las 5 en `reports/`.
- Derivan de `diagnostic_class` de los statements presentes en `scp_codes`.
- `NORM` participa como cualquier otra superclase, sin lógica especial de exclusión
  frente a patológicas (RF-05). Ante co-ocurrencia de superclases en un registro, la
  superclase primaria la define el mapeo (T09) y queda documentada en
  `docs/label_mapping.md`.

#### 1.4.2 Eje 2 — Ritmo, evaluado por el modelo

- Fuente: statements de categoría `rhythm` de `scp_statements.csv` (12 en total).
- Orden canónico de las **evaluadas**: `["SR", "AFIB", "STACH", "SARRH", "SBRAD"]`
  (5 claves).
- Criterio de inclusión, verificado contra el conteo real del dataset: ≥500 casos
  totales Y ≥350 en los folds de entrenamiento 1–8 de `strat_fold`. El resto de la
  categoría rhythm (AFLT, SVTAC, PSVT, SVARR, ... ) **no alcanza el criterio** y
  queda declarado, no evaluado. Nota de clasificación: `2AVB` y `3AVB` caen en la
  categoría `diagnostic`/`CD` según `scp_statements.csv`, no en `rhythm` — verificar
  en `scp_statements.csv` antes de clasificar (T09, §1.4.4).
- Semántica: salidas binarias independientes por clave (multietiqueta), con
  probabilidad individual y umbral/avisos por etiqueta (§2.6.1).
- Se muestra SIEMPRE junto a su eje 1 (§2.6); nunca solo.

#### 1.4.3 Eje 3 — Diagnóstico específico dentro del eje diagnóstico, evaluado por el modelo

- Fuente: statements de categoría `diagnostic` de `scp_statements.csv` (44 en total).
- Orden canónico de las **evaluadas**: `["ASMI", "1AVB"]` (2 claves). `ASMI`
  corresponde a `diagnostic_subclass` de la superclase `MI`; `1AVB` corresponde a
  `diagnostic_subclass` de la superclase `CD`.
- Criterio de inclusión: ≥500 casos totales Y ≥350 en los folds de entrenamiento
  1–8 de `strat_fold`. `2AVB` y `3AVB` (mismo `diagnostic_subclass` `_AVB`, bajo
  `CD`) **no cumplen** el umbral y quedan declarados, no evaluados.
- Semántica: salidas binarias independientes por clave (multietiqueta), con
  probabilidad individual y umbral/avisos por etiqueta (§2.6.1).
- Se muestra SIEMPRE junto a su eje 1 (§2.6); nunca solo.

#### 1.4.4 Categorías y claves declaradas-no-evaluadas (punto ciego deliberado)

- Las claves declaradas de cada eje (ritmo y diagnóstico específico que no alcanzan
  el criterio) se catalogan fijo. Estas entradas SIEMPRE devuelven el aviso fijo
  `dato insuficiente en el set de entrenamiento` (§2.6.1, `insuf14`) y NUNCA una
  probabilidad calculada: no tienen campo `probs`.
- Eje 2 declaradas (resto de categoría rhythm): `AFLT`, `SVTAC`, `PSVT`, `SVARR`,
  y las sin statement directo en PTB-XL (`VT`/`VF`/ritmo nodal — 0 ocurrencias,
  no se derivan desde otros códigos para entrenar/evaluar).
- Eje 3 declaradas (resto de categoría diagnostic): `2AVB`, `3AVB` (subclass `_AVB`,
  bajo `CD`), y demás specific statements que no cumplen el umbral.
- Conteos y categoría de anotación de cada código — tomadas directamente de
  `scp_statements.csv`, no asumidas — se documentan en `docs/label_mapping.md`
  (§1.4.5, T09).
- Punto ciego deliberado del taller: se exhiben declaradas en la salida y en
  `reports/`, nunca se omiten ni se ocultan tras una confianza calculada (RF-11,
  RF-12, AGENTS regla 3).

#### 1.4.5 Mapeo y trazabilidad de conteos

- El mapeo detallado statements → tres ejes (con pesos, resolución de
  co-ocurrencias del eje 1, excepciones y versionado) vive en
  `docs/label_mapping.md`, referenciado por `TrainingRun.label_mapping_path`.
- T09 debe documentar allí: (a) la **categoría de anotación**
  (`diagnostic`/`rhythm`/`form`) de cada statement usado, tomada directamente de
  `scp_statements.csv` —no asumida por homonimia—, y (b) la tabla completa de
  conteos reales (total y por `strat_fold` 1–8) de las 12 claves evaluadas y de
  cada entrada declarada de §1.4.4, de modo que la partición evaluado/declarado
  quede trazable y verificable, no como un corte arbitrario.

### 1.5 Split y semilla

- Split canónico: train = `strat_fold != 10`, test = `strat_fold == 10` (2198 registros
  en test), según documentación oficial PTB-XL. Cualquier otro split se documenta en
  `TrainingRun.split`.
- `seed` por defecto: `42`. Se registra en `TrainingRun.seed`.

---

## 2. Modelo de datos

### 2.0 Convenciones generales

- `np.ndarray` para señales (float, unidad mV); dicts ordenados para vectores de
  etiquetas/probabilidad usando los órdenes canónicos de los tres ejes (§1.4).
- Enums por string: `completo`, `parcial`, `imposible`, `rechazo`, `incertidumbre` y los
  de cada tipo (§2.2–§2.7). Cualquier otro valor es error de contrato.
- `record_id`: para PTB-XL la ruta relativa sin extensión (ej.
  `records100/00000/00001_lr`); para fixtures sintéticas/imagen un identificador corto
  no ambiguo (ej. `synthetic_01`, `scan_abc`).

### 2.1 Estados del pipeline (completo/parcial/imposible, rechazo, incertidumbre)

| Estado | Dónde se produce | Significado | Efecto aguas abajo | RF |
|---|---|---|---|---|
| `rechazo` | Ingesta (M1/M2) | Entrada **no reconocible** como EKG: header ilegible, o imagen sin 12 trazados | Se aborta: no hay `ECGRecord`/`Signal12`/features ni predicción; mensaje explicativo; sin persistencia | RF-08b |
| `completo` | `Signal12.status`, `FeatureVector.status` | 12 derivaciones válidas + duración nominal + todas las features extraídas | Predicción normal; avisos solo por RF-07a/07b | RF-01, RF-08 |
| `parcial` | `Signal12.status`, `FeatureVector.status` | Reconocible pero degradado: ≥1 derivación faltante/ruidosa, o duración < nominal, o extracción parcial de features | Predicción **más aviso de calidad**: presentar como no confiable aun con p ≥ 0,60 (prevalece sobre RF-07b) | RF-08 |
| `imposible` | `FeatureVector.status` | Reconocible pero **extracción total de features imposible** | `Prediction` sin probabilidades en ningún eje: estado `incertidumbre`, sin etiquetas, con aviso de calidad; RF-07c no aplica (no hay probabilidades) | RF-08 |
| `incertidumbre` | `Prediction.outcome` | Salida del estado imposible | Sin top-1 ni avisos por etiqueta; solo aviso de calidad/incertidumbre | RF-08 |

Prohibido pasar de `parcial`/`imposible` a una predicción **confiada**: toda entrada
degradada se presenta siempre como no confiable (Constitution §1, RF-08).

### 2.2 `ECGRecord`

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `record_id` | str | §2.0 |
| `source` | enum | `signal` (señal wfdb nativa) \| `image` (digitalizada a 1D) |
| `status` | enum | `reconocido` \| `rechazo` |
| `fs` | float | nominal 100 Hz para señal nativa |
| `duration_s` | float | `n_samples / fs`; nominal 10,0 |
| `n_leads_valid` | int | 0..12 |
| `leads_present` | list[str] | derivaciones presentes (⊆ orden canónico) |
| `metadata` | dict | origen/procedencia: para imagen, ruta del temporal efímero; para señal, datos del `.hea` (nombre, n_leads, fs, n_samples, unidades) |
| `signal` | `Signal12` (nullable) | presente solo si `status == reconocido` |

### 2.3 `Signal12`

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `record_id` | str | §2.0 |
| `source` | enum | `signal` \| `image` |
| `fs` | float | nominal 100 |
| `duration_s` | float | nominal 10,0 |
| `n_samples` | int | nominal 1000 @100 Hz |
| `data` | np.ndarray | shape `(n_samples, 12)`, float, mV; columnas en orden canónico §1.2 |
| `lead_names` | list[str] | orden canónico §1.2 |
| `lead_flags` | dict[str→enum] | por derivación: `ok` \| `missing` \| `noisy` |
| `status` | enum | `completo` \| `parcial` |

Reglas: `completo` ⇔ las 12 derivaciones `ok` y `n_samples >= 1000`@100 Hz; en otro caso
`parcial`. Validado contra señal 1D nada más — nunca desde píxeles (RF-03).

### 2.4 `FeatureVector`

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `record_id` | str | §2.0 |
| `values` | dict[str→float] | lista cerrada §2.4.1 |
| `missing_features` | list[str] | nombres no extraídos (vacía si `completo`) |
| `status` | enum | `completo` \| `parcial` \| `imposible` |

- `completo`: las 8 features presentes. `parcial`: ≥1 presente y ≥1 ausente.
  `imposible`: 0 presentes.
- Se calcula exclusivamente sobre `Signal12` (o degradación de ella), nunca píxeles (RF-03).

#### 2.4.1 Lista cerrada de features (nombres canónicos)

| Feature | Clave | Unidad |
|---|---|---|
| Duración QRS | `qrs_duration_ms` | ms |
| Intervalo QT | `qt_ms` | ms |
| QT corregido | `qtc_ms` | ms |
| Frecuencia cardiaca | `hr_bpm` | bpm |
| Eje QRS | `qrs_axis_deg` | grados |
| HRV — SDNN | `hrv_sdnn_ms` | ms |
| HRV — RMSSD | `hrv_rmssd_ms` | ms |
| HRV — pNN50 | `hrv_pnn50_pct` | % |

La lista vigente usada en una corrida queda registrada en `TrainingRun.features_used`
(RF-04: "la lista vigente queda documentada con el entrenamiento"). Cualquier cambio a
la lista canónica actualiza esta sección por PR.

### 2.5 `LabelVector` (antes `LabelVector8`) — tres ejes

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `record_id` | str | §2.0 |
| `superclass_labels` | dict[str→int] | exactamente las 5 claves del eje 1 en orden canónico §1.4.1, valores {0, 1} |
| `rhythm_labels` | dict[str→int] | exactamente las 5 claves evaluables del eje 2 en orden canónico §1.4.2, valores {0, 1} |
| `specific_diagnostic_labels` | dict[str→int] | exactamente las 2 claves evaluables del eje 3 en orden canónico §1.4.3, valores {0, 1} |
| `source_statements` | dict[str→float] | `scp_codes` del registro (statement → peso) tal cual vienen de `ptbxl_database.csv` |

- Basado en `LabelVector` en lugar de píxeles: proviene del mapeo documentado
  `docs/label_mapping.md` (RF-05), que incluye la categoría de anotación
  (`diagnostic`/`rhythm`/`form`) de cada statement y la tabla de conteos reales por
  `strat_fold` que respalda la partición evaluado/declarado (§1.4.5).
- Las claves declaradas-no-evaluadas (§1.4.4) **no** ocupan claves en
  `rhythm_labels` ni en `specific_diagnostic_labels`: no son targets de
  entrenamiento ni de evaluación; su estado queda solo en el catálogo fijo del
  contrato.
- `rhythm` y `specific diagnostic` son ejes independientes entre sí y respecto de
  la superclase: un mismo registro puede tener claves activas en los tres a la vez
  (ej.: superclase `STTC` + ritmo `SBRAD` + `1AVB`).
- Todos-ceros permitido en cualquier eje (registro sin statements que mapeen); el
  manejo en entrenamiento es decisión de LF y se documenta en el `TrainingRun`.

### 2.6 `Prediction` (antes `Prediction8`) — tres ejes

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `record_id` | str | §2.0 |
| `outcome` | enum | `prediction` \| `incertidumbre` \| `rechazo` |
| `quality` | enum | `nominal` \| `degraded` \| `impossible` |
| `superclass` | `SuperclassPrediction5` | eje 1, presente solo si `outcome == prediction`; **siempre** se muestra |
| `rhythm` | `RhythmPrediction` (nullable) | eje 2, presente solo si `outcome == prediction`; solo el subconjunto evaluable §1.4.2; **nunca se muestra sin** `superclass` |
| `specific_diagnostic` | `SpecificDiagnosticPrediction` (nullable) | eje 3, presente solo si `outcome == prediction`; solo el subconjunto evaluable §1.4.3; **nunca se muestra sin** `superclass` |
| `declared_not_evaluated` | list[str] | catálogo fijo §1.4.4; presente solo si `outcome == prediction`; cada entrada lleva el aviso fijo `dato insuficiente en el set de entrenamiento`, sin probabilidad calculada |
| `alerts` | dict[str→list[str]] | por clave de `superclass`, `rhythm` y `specific_diagnostic` (los presentes), códigos §2.6.1; presente solo si `outcome == prediction`; listas vacías si nominal y sin avisos |

#### 2.6.0 Tipos por eje

| Tipo | Campo | Tipo | Valores / invariantes |
|---|---|---|---|
| `SuperclassPrediction5` | `probs` | dict[str→float] | exactamente las 5 claves eje 1 §1.4.1, valores [0, 1] |
| `SuperclassPrediction5` | `top_label` | str | etiqueta con mayor `probs` (desempate: primer orden canónico §1.4.1) |
| `RhythmPrediction` | `probs` | dict[str→float] | exactamente las 5 claves evaluables eje 2 §1.4.2, valores [0, 1] |
| `RhythmPrediction` | `top_label` | str (nullable) | mayor `probs` de eje 2 (desempate: primer orden canónico §1.4.2); `null` si ninguna supera el umbral y aplica RF-07c |
| `SpecificDiagnosticPrediction` | `probs` | dict[str→float] | exactamente las 2 claves evaluables eje 3 §1.4.3, valores [0, 1] |
| `SpecificDiagnosticPrediction` | `top_label` | str (nullable) | mayor `probs` de eje 3 (desempate: primer orden canónico §1.4.3); `null` si ninguna supera el umbral y aplica RF-07c |

Invariantes de presentación (obligatorios en `Prediction` y en cualquier salida):
1. Eje 1 (superclase) se muestra **siempre** — es la salida confiable y con soporte
   amplio de datos.
2. Eje 2 (ritmo) y eje 3 (diagnóstico específico) se muestran **solo junto a** el
   eje 1, nunca por su cuenta. Ritmo y hallazgo estructural/AV son preguntas
   **distintas e independientes**, no una jerarquía: la UI debe dejar visible que
   un registro puede tener superclase, ritmo y diagnosis específica en paralelo.
3. Las entradas de `declared_not_evaluated` se muestran declaradas, con su aviso
   fijo y sin probabilidad — jamás una probabilidad calculada para ellas.

#### 2.6.1 Códigos de aviso (textos literales canónicos)

| Código | Detonante | Texto literal (obligatorio) | RF |
|---|---|---|---|
| `insuf14` | clave ∈ `declared_not_evaluated` (§1.4.4) | `dato insuficiente en el set de entrenamiento` | §1.4.4, AGENTS regla 3 |
| `lowconf07b` | p de la etiqueta < 0,60 (por eje: 5 de eje 1, 5 de eje 2, 2 de eje 3) | `confianza baja` | RF-07b |
| `quality08` | entrada degradada (no confiable, aun con p ≥ 0,60) | `entrada degradada: predicción no confiable` | RF-08 |

Semántica por contrato (la tabla de casos completa es el apéndice de T03):
- RF-06: se exponen las 5 probabilidades de eje 1 y, cuando apliquen, las 5 de eje 2
  y las 2 de eje 3 (subconjuntos evaluables); las declaradas §1.4.4 no llevan
  probabilidad.
- RF-07a (etiquetas raras de v1.0 —`VT`, `VF`, `BAV`/`3AVB`—): quedan **absorbidas**
  por el catálogo declarado §1.4.4, donde nunca se calcula probabilidad y siempre se
  emite el aviso fijo `insuf14` (`dato insuficiente en el set de entrenamiento`). No
  hay ya claves raras **evaluadas** en los ejes 2/3: las evaluadas superan el criterio.
- `insuf14`: para las declaradas §1.4.4 el sistema **nunca** calcula ni muestra
  probabilidad; el único texto es el aviso fijo (AGENTS regla 3).
- RF-07b: `lowconf07b` se añade cuando p < 0,60, evaluado por eje por separado.
- RF-07c: si en un eje ninguna de sus probabilidades supera 0,60, el `top_label`
  de ese eje se expone + `lowconf07b` (no existe conjunto vacío silencioso). El
  eje 1 siempre presenta un `top_label` (5 superclases) — si `rhythm.top_label` o
  `specific_diagnostic.top_label` quedan sin definir, se muestra eje 1 con su aviso
  y el catálogo declarado, sin omitir los ejes presentes.
- RF-07d: los avisos se **apilan**, nunca se suprimen (ej.: ritmo con p=0,30 →
  `lowconf07b`; entrada degradada con p ≥ 0,60 → `quality08` + `lowconf07b` si
  aplica); `insuf14` es exclusivo de claves declaradas y no comparte etiqueta con p.
- RF-08: `quality08` presente si `quality == degraded`; prevalece sobre el umbral →
  la etiqueta se trata como no confiable aun si p ≥ 0,60. Si `quality == impossible`,
  `outcome == incertidumbre`, sin `superclass`/`rhythm`/`specific_diagnostic`/`alerts`
  y RF-07c no aplica.
- `rechazo`: entrada no-EKG → `outcome == rechazo`, sin predicción ni persistencia (RF-08b).

Constantes de política: umbral = `0.60`; catálogo declarado §1.4.4 (eje 2: `AFLT`,
`SVTAC`, `PSVT`, `SVARR`, `VT`, `VF`, ritmo nodal; eje 3: `2AVB`, `3AVB` y demás
diagnostic statements que no cumplen el criterio — claves exactas y categoría de cada
código fijadas en `docs/label_mapping.md`, T09).

### 2.7 `TrainingRun`

| Campo | Tipo | Valores / invariantes |
|---|---|---|
| `model_path` | str | `model.pkl` (raíz del repo) |
| `ptbxl_version` | str | `"1.0.3"` (según `data/ptbxl/`) |
| `split` | str | descripción, p. ej. `"strat_fold != 10 train, == 10 test (canónico PTB-XL)"` |
| `seed` | int | `42` por defecto |
| `features_used` | list[str] | subconjunto de la lista §2.4.1 efectivamente usado |
| `label_mapping_path` | str | `docs/label_mapping.md` (versión del mapeo) |
| `target_labels` | list[str] | las salidas de los tres ejes: 5 de eje 1 (§1.4.1) + 5 evaluables de eje 2 (§1.4.2) + 2 evaluables de eje 3 (§1.4.3); las declaradas §1.4.4 no son targets |
| `hardware` | dict | {`cpu`, `cores`, `ram_gb`, `os`} del laptop de la corrida |
| `duration_s` | float | duración de la corrida vía señal |
| `cpu_only` | bool | `true` siempre (RF-13) |
| `reports_dir` | str | `reports/` |
| `report_artifacts` | list[str] | matriz por etiqueta, ROC/AUC ×5 (eje 1) + ×5 (eje 2) + ×2 (eje 3), importancia; las declaradas §1.4.4 se listan como no evaluadas por diseño | 
| `rare_insufficient_test` | list[str] | raras sin suficientes muestras en test (→ nota explícita en la gráfica, RF-11) |

`model.pkl` y `reports/` se versionan; el dataset y los uploads nunca entran al repo
(RF-14, Done-8).

---

## 3. Rutas y artefactos congelados

| Artefacto | Ruta | Dueño |
|---|---|---|
| Dataset PTB-XL 100 Hz | `data/ptb-xl/` (§1) | todos leen; nadie edita |
| Mapa de etiquetas (detalle) | `docs/label_mapping.md` | LD (T09) |
| Modelo entrenado | `model.pkl` | LF (T12) |
| Evaluación de corrida | `reports/` (matriz, ROC/AUC ×5+×5+×2, importancia, metadatos) | LF/M6 |
| Uploads de la demo | temporales efímeros, descartados al cerrar sesión; nunca en disco/repo | LG/M8 |

---

## Apéndice A — Tabla de casos de avisos (T03)

Tabla de casos de avisos canónica: qué se emite y cómo se apila, caso por caso,
por eje. Nombres de código/estado/calidad según la versión canónica del texto en
§2.6.1 y §2.1. Los textos literales que LE y LG deben mostrar son los del cuerpo
(§2.6.1); este apéndice define **cuándo y sobre qué clave** se emite cada uno.
Cualquier caso no listado aquí se resuelve por los invariantes de presentación de
§2.6 y por prioridad de la RF indicada en la columna `RF`.

Definiciones de entrada usadas en la tabla:

- **Comparable p**: p del eje es la probabilidad **por clave** (§2.6.0). El umbral
  es `0.60` (§2.6.1). `p ≥ 0,60` → una clave OK del eje; un eje "vacío" ≠ tiene
  **todas** sus claves < 0,60.
- **Eje vacío**: eje 2 o eje 3 presente en la salida pero con todas sus
  probabilidades < 0,60. El eje 1 nunca está vacío: es multiclase con 5 claves y
  siempre arroja un `top_label` (§2.6.0, RF-07c).
- **Degradada**: `quality == degraded` (RF-08): reconocible, pero ≥1 derivación
  faltante/ruidosa, duración < nominal o features parciales (§2.2–§2.4).
- **Fallo total**: `quality == impossible` (RF-08): extracción de features
  imposible (0 features). Produce `outcome == incertidumbre`.
- **No-EKG**: entrada no reconocible (header ilegible, imagen sin 12 trazados) →
  `outcome == rechazo` (RF-08b).
- **Declarada**: clave del catálogo §1.4.4 (eje 2: `AFLT`, `SVTAC`, `PSVT`,
  `SVARR`, `VT`, `VF`, ritmo nodal; eje 3: `2AVB`, `3AVB` y demás specific sin
  criterio). Nunca lleva probabilidad calculada ni ocupa clave de `probs`.

Los avisos en la columna `Avisos emitidos` se generan **sobre la(s) clave(s)
indicada(s)**; si hay más de una clave afectada, el aviso se repite por clave.

### Tabla A.1 — Casos de avisos (por entrada)

| # | Caso (descripción) | quality | outcome | Salidas presentadas | Avisos emitidos (texto del código, §2.6.1) | RF |
|---|---|---|---|---|---|---|
| C1 | Predicción nominal, todas las claves del eje 1 con p ≥ 0,60 | `nominal` | `prediction` | superclass (5 probs + top_label); eje 2/3 según corresponda | ninguno (listas de `alerts` vacías) | RF-06 |
| C2 | Al menos una clave de un eje con p < 0,60 | `nominal` | `prediction` | probs completas del eje afectado (5 de eje 1, 5 de eje 2, 2 de eje 3) + top_label de cada eje | `lowconf07b` sobre **cada clave** con p < 0,60 | RF-07b |
| C3 | Eje 2 vacío (las 5 p de ritmo < 0,60) | `nominal` | `prediction` | superclass completa + top-1 del eje 2 (máximo de sus probs) como `rhythm.top_label` | `lowconf07b` sobre el `rhythm.top_label`; se conservan las 5 probs | RF-07c |
| C4 | Eje 3 vacío (las 2 p de specific < 0,60) | `nominal` | `prediction` | superclass completa + top-1 del eje 3 como `specific_diagnostic.top_label` | `lowconf07b` sobre el `specific_diagnostic.top_label`; se conservan las 2 probs | RF-07c |
| C5 | Eje 1 con `top_label` < 0,60 (todas las p de superclass < umbral, pero siempre hay top) | `nominal` | `prediction` | superclass (5 probs + top_label) | `lowconf07b` sobre el `superclass.top_label` | RF-07b/07c |
| C6 | Entrada degradada, key con p ≥ 0,60 | `degraded` | `prediction` | probs completas de los ejes + top_label por eje | `quality08` siempre y `lowconf07b` sobre las claves con p < 0,60 (ambos si aplican) | RF-08 |
| C7 | Entrada degradada, eje vacío (todas p < 0,60) | `degraded` | `prediction` | superclass + top-1 del eje vacío (eje 2/3) o top-1 del eje 1 | `quality08` + `lowconf07b` sobre el top-1 expuesto (apilados, RF-07d) | RF-08, RF-07d |
| C8 | Avisos concurrentes sobre la misma clave (p<0,60 Y degradada; o p<0,60 Y declarada) | `degraded` | `prediction` | probs completas + top_label | todos los avisos aplicables se apilan; **nunca se suprime uno** (RF-07d) | RF-07d, RF-11 |
| C9 | Fallo total de extracción (0 features) | `impossible` | `incertidumbre` | ninguna etiqueta; sin `superclass`/`rhythm`/`specific_diagnostic`/`declared_not_evaluated`/`alerts` | `quality08` como aviso de calidad/incertidumbre (nivel de `Prediction`); **RF-07c no aplica** (no hay probabilidades) | RF-08 |
| C10 | Entrada no-EKG (header ilegible o imagen sin 12 trazados) | — | `rechazo` | ninguna predicción; mensaje explicativo; sin persistencia | mensaje de rechazo (no es un aviso de alertas) | RF-08b |

Nota C1–C7: el eje 1 se muestra **siempre**; eje 2 y eje 3 **solo junto al eje 1**
(invariante de presentación, §2.6).

### Tabla A.2 — Claves declaradas-no-evaluadas

| # | Clave | Eje | Aviso fijo (§2.6.1) | ¿Lleva `probs`? |
|---|---|---|---|---|
| D1 | `AFLT`, `SVTAC`, `PSVT`, `SVARR` | 2 | `insuf14` | no |
| D2 | `VT`, `VF`, ritmo nodal | 2 (sin statement directo, §1.4.4) | `insuf14` | no |
| D3 | `2AVB`, `3AVB` y demás specific diagnostic sin criterio | 3 | `insuf14` | no |

Regla de la tabla A.2: las claves declaradas se listan en
`declared_not_evaluated` **siempre**, con su aviso fijo, **nunca** con probabilidad
y **nunca** omitidas de la presentación (punto ciego deliberado, RF-12). `insuf14`
no se combina con `lowconf07b` dentro de una misma clave: no existe p para calcularla.

### Cuándo NO aplica

- **"VT con p=0,30 → avisos 07a + 07b" (caso de v1.0): no existe en v1.2.** VT es
  declarada (§1.4.4, D2): no se calcula ni se muestra su probabilidad, solo
  `insuf14`. Toda clave rara de v1.0 quedó absorbida por el catálogo declarado; no
  hay clave evaluada con soporte insuficiente en los ejes 2/3.
- **`quality08` no se emite** cuando `quality == nominal` ni cuando `outcome == rechazo`
  (en rechazo no hay salida por predicción).
- **RF-07c no aplica**: cuando `outcome == incertidumbre` (C9) o `rechazo` (C10) — no
  existen probabilidades que rankear.