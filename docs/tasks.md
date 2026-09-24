# Tasks — SinusAI v1 (tareas <30 min, paralelizables)

Fuente: `docs/plan.md` (M1–M8, D-01–D-10). Cada tarea indica RF, dependencia y línea `Hecho cuando:` verificable.

## Reglas de paralelismo (no pisarse)

| Carril | Dueño de archivos (solo ese carril los edita) |
|---|---|
| L0 contratos+base | `docs/contracts.md`, `tests/fixtures/` (solo añade subcarpeta propia), `requirements.txt` |
| LA señal | `src/ingest_signal.*`, `tests/test_ingest_signal.*` |
| LB imagen | `src/ingest_image.*`, `tests/test_ingest_image.*` |
| LC features | `src/features.*`, `tests/test_features.*` |
| LD etiquetas | `src/labels.*`, `tests/test_labels.*`, `docs/label_mapping.md` |
| LE política | `src/policy.*`, `tests/test_policy.*` |
| LF entrena+eval | `train.py`, `src/evaluate.*`, `tests/test_evaluate.*` (no toca `src/policy.*`) |
| LG demo Streamlit | `app.py` (Streamlit), `tests/test_demo.*` (lee contratos, no edita otros `src/`) |
| LH cierre | `reports/`, `requirements.txt`, `pytest.ini` / config (solo LH al final) |

Prohibido editar archivos de otro carril. La comunicación entre carriles es solo vía `docs/contracts.md` (congelado tras Fase 0).

## Fase 0 — Contratos (bloquea todo lo demás)

- [ ] **T01 L0 — Congelar contratos de datos.** RF: RF-01, RF-05, RF-06. Toca: `docs/contracts.md`. Depende de: nada.
  Hecho cuando: `contracts.md` define `ECGRecord/Signal12/FeatureVector/LabelVector8/Prediction8/TrainingRun` con campos y estados (completo/parcial/imposible, rechazo, incertidumbre), y también el layout del dataset (ruta `data/ptb-xl/`, archivos `.hea/.dat`, tabla de etiquetas PTB-XL) de modo que ningún carril necesite preguntar el formato ni la ubicación de los datos.
- [ ] **T02 L0 — Entorno base y fixtures mínimas.** RF: RF-01, RF-08, RF-08b. Toca: `tests/fixtures/` (una subcarpeta por caso), `requirements.txt`. Depende de: T01.
  Hecho cuando: `requirements.txt` fija las dependencias (neurokit2, wfdb, scikit-learn, streamlit y pytest) e instalables en el `.venv` con `pip install -r requirements.txt`; existen fixtures sintéticas nominal, degradada (derivación faltante), fallo total, no-EKG y PNG/JPG de juguete; además un subconjunto real cortado del dataset descargado (una ruta de corte reproducible que cubra las 8 clases incluyendo las raras) y un smoke-load que confirme que wfdb lee registros reales; todo usable por los carriles sin tocar el dataset completo.
- [ ] **T03 L0 — Tabla de casos de avisos.** RF: RF-07a–d, RF-08. Toca: `docs/contracts.md` (apéndice). Depende de: T01.
  Hecho cuando: la tabla rara-alta→07a, p<0,60→07b, VT 0,30→ambos, vacío→top-1, degradada p≥0,60→calidad+no-confiable, fallo total→sin etiquetas, no-EKG→rechazo queda escrita y LE/LG la citan sin ambigüedad.

## Fase 1 — Carriles paralelos (sin dependencias entre sí tras Fase 0)

- [ ] **T04 LA — Ingesta señal nominal.** RF: RF-01. Toca: `src/ingest_signal.*`, `tests/test_ingest_signal.*`. Depende de: T01–T02.
  Hecho cuando: `pytest tests/test_ingest_signal.*` acepta fixture nominal 12×100 Hz×10 s y expone metadatos.
- [ ] **T05 LA — Ingesta señal degradada vs. rechazo.** RF: RF-08, RF-08b. Toca: mismos que T04. Depende de: T04.
  Hecho cuando: fixture degradada se marca degradada (no se rechaza) y header ilegible se rechaza con mensaje, con tests en verde.
- [ ] **T06 LB — Ingesta imagen PNG/JPG → 1D.** RF: RF-02, RF-03, RF-14. Toca: `src/ingest_image.*`, `tests/test_ingest_image.*`. Depende de: T01–T02.
  Hecho cuando: PNG/JPG de juguete con 12 trazados produce `Signal12`; sin 12 trazados aplica rechazo; solo usa temporales efímeros.
- [ ] **T07 LC — Vector clínico cerrado.** RF: RF-03, RF-04. Toca: `src/features.*`, `tests/test_features.*`. Depende de: T01–T02.
  Hecho cuando: de una `Signal12` sintética sale el vector cerrado (QRS, QT/QTc, FC, eje, HRV) sin usar píxeles.
- [ ] **T08 LC — Estados parcial/imposible.** RF: RF-08. Toca: mismos que T07. Depende de: T07.
  Hecho cuando: fixture degradada da estado parcial y la de fallo total da imposible, bloqueando RF-07c según contratos.
- [ ] **T09 LD — Mapeo 8 salidas planas.** RF: RF-05. Toca: `src/labels.*`, `docs/label_mapping.md`, `tests/test_labels.*`. Depende de: T01.
  Hecho cuando: `label_mapping.md` documenta statements→8 salidas con NORM plana y los tests lo verifican con cabeceras sintéticas.
- [ ] **T10 LE — Avisos 07a–07d.** RF: RF-06, RF-07a–d. Toca: `src/policy.*`, `tests/test_policy.*`. Depende de: T01, T03.
  Hecho cuando: la tabla de contratos pasa caso por caso (incluido apilado VT 0,30 y vacío→top-1) con textos literales de AGENTS regla 3.
- [ ] **T11 LE — Prevalencias RF-08 sobre 07b/07c.** RF: RF-08, RF-08b. Toca: mismos que T10. Depende de: T10.
  Hecho cuando: degradada con p≥0,60 sale como no-confiable con aviso de calidad y el fallo total sale sin etiquetas (RF-07c no aplica).
- [ ] **T12 LF — Entrenamiento CPU-only reproducible.** RF: RF-05, RF-13, RF-14. Toca: `train.py`. Depende de: T01, T02, T09 (usa LC como caja negra vía contratos).
  Hecho cuando: `train.py` lee el dataset desde la ruta indicada en contratos (T01), entrena solo con PTB-XL features→`model.pkl` + metadatos (versión/split/seed/laptop/duración), sin GPU ni datos de pacientes.
- [ ] **T13 LF — Evaluación uno-vs-resto.** RF: RF-11, RF-12. Toca: `src/evaluate.*`, `tests/test_evaluate.*`. Depende de: T01.
  Hecho cuando: con predicciones sintéticas genera matriz×etiqueta, ROC/AUC×8 (o nota de insuficiencia) e importancia, sin ocultar raras.
- [ ] **T14 LG — Demo Streamlit contenido exacto.** RF: RF-09. Toca: `app.py` (Streamlit), `tests/test_demo.*`. Depende de: T01, T03.
  Hecho cuando: `streamlit run app.py` muestra trazado + 8 probabilidades + avisos aplicables y nada de features/gráficas, verificado por test con `Prediction8` sintética.
- [ ] **T15 LG — Aviso permanente y efimeralidad.** RF: RF-10, RF-14. Toca: mismos que T14. Depende de: T14.
  Hecho cuando: el aviso pedagógico es visible siempre y los uploads se eliminan al cerrar sesión sin persistir en disco/repo.

## Fase 2 — Integración y cierre (ordenada)

- [ ] **T16 LH — Integración señal→demo Streamlit.** RF: Done-1, Done-5. Toca: solo config/pegamento de LH. Depende de: T05, T08, T11, T14.
  Hecho cuando: una señal nominal recorre ingesta→features→policy→demo Streamlit con 8 probabilidades en vivo.
- [ ] **T17 LH — Integración imagen→demo.** RF: Done-1. Toca: solo LH. Depende de: T06, T16.
  Hecho cuando: un PNG/JPG recorre la misma vía por señal 1D (nunca píxeles), admitiendo aviso de calidad.
- [ ] **T18 LH — Corrida de evaluación real.** RF: Done-6. Toca: `reports/`. Depende de: T12, T13.
  Hecho cuando: `reports/` contiene matriz×etiqueta, ROC/AUC×8 (o nota), importancia y metadatos con raras visibles, sobre la corrida real del dataset (`data/ptb-xl`).
- [ ] **T19 LH — Presupuesto, privacidad y verde.** RF: Done-7, Done-8. Toca: `reports/` (tiempo de corrida), config pytest. Depende de: T18.
  Hecho cuando: vía señal <1 h CPU-only documentado, repo sin dataset/uploads/pacientes y `pytest` en verde.
- [ ] **T20 LH — Repaso de casos Done-2–Done-4.** RF: Done-2, Done-3, Done-4. Toca: solo LH. Depende de: T16–T17.
  Hecho cuando: quedan demostrados rara→07a, p<0,60→07b, degradada→calidad, fallo total→sin etiquetas, vacío→top-1 y no-EKG→rechazo.
