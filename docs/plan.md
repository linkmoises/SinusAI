# Plan — SinusAI v1 (diseño, sin código)

Derivado de `docs/spec.md` y `docs/constitution.md`. Describe el QUÉ 구조 — módulos, datos, decisiones y tests — con trazabilidad a RF. No prescribe implementación.

## 1. Módulos

| Módulo | Responsabilidad | RF que cubre |
|---|---|---|
| M1 Ingesta señal | Aceptar registro nominal PTB-XL/wfdb 12 derivaciones, 100 Hz, 10 s; validar encabezado y trazados | RF-01, RF-08b |
| M2 Ingesta imagen | Aceptar PNG/JPG de 12 derivaciones estándar; detectar 12 trazados y convertir a señal 1D; temporales efímeros | RF-02, RF-03, RF-08b, RF-14 |
| M3 Features clínicas | Producir vector cerrado (QRS, QT/QTc, FC, eje QRS, HRV temporal); señalar extracción parcial vs. total | RF-03, RF-04, RF-08 |
| M4 Etiquetado | Mapear statements PTB-XL a 8 salidas planas (5 superclases + VT/VF/BAV); NORM como etiqueta plana | RF-05 |
| M5 Entrenamiento | Entrenar clasificador multietiqueta CPU-only sobre features; registrar versión PTB-XL, split, seed | RF-05, RF-13, RF-14 |
| M6 Evaluación | Generar en `reports/`: matriz por etiqueta, ROC/AUC por salida, importancia (global o por etiqueta), metadatos de corrida; anotar raras sin muestras | RF-11, RF-12 |
| M7 Política de avisos | Aplicar RF-07a (clase rara), RF-07b (p<0,60), RF-07c (vacío→top-1), RF-07d (apilado), RF-08 (degradada→no confiable; fallo total→incertidumbre sin etiquetas), RF-08b (rechazo no-EKG) | RF-06 – RF-08b |
| M8 Demo taller | Mostrar trazado + 8 probabilidades + avisos + aviso pedagógico permanente; descartar uploads al cerrar sesión | RF-09, RF-10, RF-14 |

Flujo: M1/M2 → M3 → M5/M7 (inferencia) ; entrenamiento: M1+M4 → M5 → M6. La demo (M8) consume M1/M2+M3+M7, nunca píxeles directos (RF-03).

## 2. Modelo de datos

| Entidad / artefacto | Contenido | Origen → consumo | RF |
|---|---|---|---|
| `ECGRecord` | id, fuente (señal/imagen), metadatos (frecuencia, duración, nº derivaciones válidas) | M1/M2 → M3, M7 | RF-01, RF-02, RF-08 |
| `Signal12` | 12 series 1D a 100 Hz, duración nominal 10 s; marca de derivaciones faltantes/ruidosas | M1/M2 → M3 | RF-01, RF-02, RF-03 |
| `FeatureVector` | lista cerrada documentada (QRS, QT/QTc, FC, eje, HRV) + estado (completo/parcial/imposible) | M3 → M5/M7 | RF-04, RF-08 |
| `LabelVector8` | 8 binarios (NORM, MI, STTC, CD, HYP, VT, VF, BAV) según mapeo documentado | M4 → M5, M6 | RF-05 |
| `Prediction8` | 8 probabilidades [0,1] + avisos por etiqueta (07a/07b/calidad) o rechazo / incertidumbre sin etiquetas | M7 → M8 | RF-06 – RF-08b |
| `TrainingRun` | `model.pkl`, versión PTB-XL, split, seed, lista de features, mapeo de etiquetas, laptop y duración | M5 → M6, `reports/` | RF-11, RF-13 |
| `reports/` | matriz por etiqueta, ROC/AUC ×8 (o nota de muestras insuficientes), importancia, metadatos | M6 (insumo del taller) | RF-11, RF-12 |

Persistencia: solo `model.pkl` + `reports/` se versionan como artefactos; dataset PTB-XL y uploads nunca entran al repo (RF-14, Done-8).

## 3. Decisiones justificadas (con alternativa descartada)

- **D-01 Salidas planas 8 (M4).** Decisión: 8 binarios independientes, NORM sin exclusión. Por qué: co-ocurrencias reales + punto ciego visible (RF-05, §1). Descartada: taxonomía jerárquica con exclusión NORM — más fiel clínicamente pero añade lógica que v1 no necesita (§4).
- **D-02 Random Forest primero (M5).** Decisión: RF multietiqueta; XGBoost solo alternativa prevista. Por qué: auditable en una tarde, CPU-only <1 h (§3, §5, RF-13). Descartada: deep learning / CNN sobre trazado — más desempeño potencial pero caja negra sobre píxeles, ya fracasada y prohibida (RF-03, §2).
- **D-03 Features cerradas clínicas (M3).** Decisión: lista cerrada documentada por corrida. Por qué: explicabilidad al médico (§3, RF-04). Descartada: embeddings automáticos — mejor métrica posible pero inexplicables en taller.
- **D-04 Umbral fijo 0,60 + avisos separados (M7).** Decisión: 0,60 por etiqueta; textos 07a/07b separados y apilables. Por qué: distingue sesgo de dataset de incertidumbre del caso; contiene los términos de AGENTS regla 3 (RF-07a–d). Descartada: umbral calibrado por clase o mensaje único — más preciso o más simple, pero añade complejidad / mezcla causas (§4).
- **D-05 Degradada→aviso, fallo total→incertidumbre, no-EKG→rechazo (M7).** Decisión: triple vía con prevalencia explícita de RF-08 sobre RF-07b/07c. Por qué: discutir el fallo en vivo sin falsa seguridad; subordinación a §1. Descartada: siempre-predecir (deshonesto ante no-EKG) y siempre-rechazar (pierde el caso pedagógico).
- **D-06 Imagen acotada PNG/JPG 12 derivaciones (M2).** Decisión: solo ese mínimo; resto fuera de alcance. Por qué: cubre el aula con complejidad proporcional (§4); paridad de calidad no exigida. Descartada: PDF/multilayout + calibración exacta — cobertura mayor a costo desproporcionado para v1.
- **D-07 Evaluación uno-vs-resto con nota de insuficiencia (M6).** Decisión: matriz y ROC/AUC por salida; raras sin muestras se anotan, no se omiten. Por qué: evidencia completa incluyendo el punto ciego (RF-11, RF-12). Descartada: métrica global única — ocultaría las raras.
- **D-08 App exacta mínima, evidencia en `reports/` (M8/M6).** Decisión: app sin features ni gráficas; auditoría vía `reports/`. Por qué: sesión de una tarde, sin sobrecarga (§3, §4, RF-09). Descartada: SHAP/ROC embebidas — más transparencia en vivo a costo de alcance.
- **D-09 Privacidad por efimeralidad (M2/M8, RF-14).** Decisión: uploads solo temporales de procesamiento, eliminados al cerrar sesión. Por qué: §6 en inferencia además de entrenamiento. Descartada: persistir uploads para depuración — útil pero compromete datos reales.
- **D-10 Presupuesto <1 h solo vía señal (M5).** Decisión: digitalización excluida del presupuesto por ser interactiva. Por qué: hace verificable RF-13 en el hardware real (§5). Descartada: incluir digitalización masiva — impredecible y fuera del caso de entrenamiento.

## 4. Estrategia de tests (qué se verifica, sin cómo)

| Nivel | Qué cubre | RF / Done |
|---|---|---|
| Contratos de ingesta | Nominal 12×100 Hz×10 s acepta; header ilegible o imagen sin 12 trazados rechaza sin predecir ni persistir | RF-01, RF-02, RF-08b; Done-1, Done-4 |
| Features | Vector cerrado completo; parcial marcado; total imposible → estado imposible (bloquea RF-07c, habilita incertidumbre) | RF-04, RF-08 |
| Política de avisos (tabla de casos) | Rara con p alta → 07a; p<0,60 → 07b; VT p=0,30 → ambos (07d); vacío → top-1+aviso; degradada con p≥0,60 → calidad + no confiable; fallo total → sin etiquetas | RF-07a–d, RF-08; Done-2 – Done-4 |
| Casos demo en vivo | Señal nominal, imagen PNG/JPG, degradada reconocible, fallo total, no-EKG/corrupto; app muestra contenido exacto + aviso permanente y descarta uploads | RF-09, RF-10, RF-14; Done-3 – Done-5 |
| Entrenamiento reproducible | Una corrida genera matriz×etiqueta, ROC/AUC×8 (o nota de insuficiencia), importancia y metadatos (versión/split/seed/laptop); raras visibles sin suavizar | RF-11, RF-12, RF-13; Done-6, Done-7 |
| Presupuesto y privacidad | Vía señal <1 h CPU-only sin GPU; repo sin dataset, sin uploads, sin datos de pacientes; `pytest` en verde | RF-13, RF-14; Done-7, Done-8 |

## 5. Trazabilidad RF → plan

RF-01→M1+ingesta-tests; RF-02→M2+ingesta-tests; RF-03→M2/M3+D-02; RF-04→M3+D-03; RF-05→M4/M5+D-01; RF-06→M7/M8; RF-07a–d→M7+D-04+avisos-tests; RF-08/08b→M7+D-05+demo-tests; RF-09→M8+D-08; RF-10→M8; RF-11→M6+D-07; RF-12→M6; RF-13→M5+D-10; RF-14→M2/M5/M8+D-09. Fuera de alcance v1: §5 del spec (CNN, CSV/500 Hz/Holter, PACS, SHAP en app, exclusión NORM, MLOps, multisesión).
