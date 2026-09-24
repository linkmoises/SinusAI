# Spec — SinusAI (demo de taller)

- **Estado:** v1 para taller de interpretación de EKG.
- **Propósito (POR QUÉ):** recurso pedagógico para mostrar en vivo dónde una IA acierta y dónde tiene puntos ciegos. No es producto clínico ni validación diagnóstica. La honestidad sobre los límites vale más que impresionar (Constitution §1, §4).
- **Flujo (QUÉ):** EKG (señal digital o imagen escaneada → señal 1D) → features clínicas → Random Forest multietiqueta (XGBoost solo como alternativa prevista en AGENTS.md, sin alterar salidas ni evaluación) → diagnóstico sugerido + confianza + gráficas de evaluación.

## Requisitos funcionales (EARS)

> Convención EARS: `Cuando <evento>`, `Si <condición no deseada>`, `Mientras <estado>`, `Donde <opcional>`, o ubicuo (`El sistema deberá…`). Solo QUÉ y POR QUÉ, sin CÓMO.

- **RF-01 (ubicuo):** El sistema deberá aceptar como entrada nominal señal digital de 12 derivaciones en formato nativo PTB-XL/wfdb a 100 Hz y duración nominal de 10 s.
  *Por qué: es el dato público, desidentificado y nativo disponible; fija el caso nominal contra el que se juzga la degradación (RF-08).*

- **RF-02 (ubicuo):** El sistema deberá aceptar como entrada imagen de trazado EKG limitada a foto o escaneo en PNG/JPG de un 12 derivaciones estándar, y transformarla a señal temporal 1D antes de cualquier análisis. Esta inclusión en v1 es excepción decidida al carácter "stretch" de AGENTS.md.
  *Por qué: el taller recibe EKG en papel; sin este puente la demo no cubre el caso de uso real del aula.*

- **RF-03 (ubicuo):** El sistema nunca deberá clasificar el EKG a partir de píxeles; toda predicción deberá derivar de la señal 1D o de features derivadas de ella.
  *Por qué: tratar el trazado como foto pierde la estructura fisiológica y ya fracasó en un proyecto previo (AGENTS.md regla 1, Constitution §2).*

- **RF-04 (ubicuo):** El sistema deberá caracterizar cada EKG exclusivamente con una lista cerrada de medidas clínicas interpretables (duración QRS, QT/QTc, frecuencia cardiaca, eje QRS, HRV en dominio tiempo). Toda medida de la lista deberá ser explicable a un médico del taller; la lista vigente queda documentada con el entrenamiento.
  *Por qué: el taller exige auditar el porqué de cada sugerencia; una representación no explicable rompe el propósito pedagógico (Constitution §3). La auditoría en vivo se hace vía `reports/` (RF-11), no vía la app (RF-09).*

- **RF-05 (ubicuo):** El sistema deberá sugerir diagnósticos en esquema multietiqueta real con 8 salidas planas e independientes: las 5 superclases PTB-XL (NORM, MI, STTC, CD, HYP) más las 3 clases raras demostrativas (taquicardia ventricular, fibrilación ventricular, bloqueo AV completo). El mapeo desde los statements PTB-XL a estas 8 salidas queda documentado con el entrenamiento. NORM se trata en v1 como una etiqueta plana más, sin lógica de exclusión frente a patológicas.
  *Por qué: un EKG real puede tener hallazgos co-ocurrentes, y las clases raras son el punto ciego deliberado que el taller debe exhibir, no ocultar.*

- **RF-06 (evento — Cuando):** Cuando el modelo emita una predicción, el sistema deberá mostrar las 8 etiquetas, cada una con su probabilidad individual en [0,1] como nivel de confianza.
  *Por qué: sin confianza explícita por etiqueta el usuario no puede calibrar cuándo confiar y cuándo no (Constitution §1). Mostrar las 8 sostiene la discusión del punto ciego.*

- **RF-07a (condición — Mientras):** Mientras la etiqueta predicha pertenezca a la lista fija poco representada (VT, VF, bloqueo AV completo), el sistema deberá mostrar explícitamente "patrón poco representado en los datos de entrenamiento", aun si su probabilidad es alta.
  *Por qué: una predicción confiada donde casi no hay ejemplos es peor que declarar incertidumbre. El texto contiene literalmente el término exigido en AGENTS.md regla 3.*

- **RF-07b (condición — Si):** Si la probabilidad individual de una etiqueta queda por debajo de 0,60, el sistema deberá mostrar explícitamente "confianza baja" para esa etiqueta, separada del aviso de clase rara.
  *Por qué: la incertidumbre caso a caso debe distinguirse del sesgo conocido del dataset; la regla combinada cubre ambas causas sin mezclarlas. El texto contiene literalmente el término exigido en AGENTS.md regla 3; ambos avisos satisfacen conjuntamente dicha regla.*

- **RF-07c (condición — Si):** Si la extracción produjo probabilidades pero ninguna de las 8 etiquetas supera el umbral de 0,60, el sistema deberá mostrar la etiqueta de mayor probabilidad acompañada del aviso de confianza baja (RF-07b).
  *Por qué: decisión de producto v1 — el conjunto vacío se resuelve con top-1 + aviso para poder discutirlo en vivo, sin presentar silencio como normalidad.*

- **RF-07d (ubicuo):** Si concurren varias causas de aviso sobre la misma etiqueta (p. ej. VT con p=0,30, o etiqueta degradada con p<0,60), los avisos aplicables se apilan, nunca se suprime uno.
  *Por qué: suprimir un aviso ocultaría una causa de incertidumbre y rompería la honestidad exigida (Constitution §1).*

- **RF-08 (no deseado — Si):** Si la entrada es reconocible como EKG pero está degradada (duración menor a la nominal, 1 o más derivaciones faltantes/ruidosas, o extracción parcial de features), el sistema deberá igualmente emitir una predicción acompañada de una advertencia explícita de calidad y presentarla como no confiable (aviso RF-07b aun si su valor numérico es ≥0,60), y nunca como predicción confiada. Esta regla prevalece explícitamente sobre el umbral de RF-07b. Si la extracción total de features es imposible, el sistema deberá mostrar incertidumbre sin etiquetas más la advertencia de calidad; en ese caso RF-07c no aplica por no existir probabilidades.
  *Por qué: lo decidido para v1 es "predecir con aviso" para poder discutir el fallo en vivo, sin inducir falsa seguridad. La subordinación a Constitution §1 es explícita: ante degradación total se declara incertidumbre sin predecir.*

- **RF-08b (no deseado — Si):** Si la entrada no es reconocible como EKG, el sistema deberá rechazarla sin emitir predicción y con mensaje explicativo. Criterio operativo v1: es reconocible si el encabezado wfdb es legible con trazados, o si la etapa de digitalización detecta 12 trazados en la imagen PNG/JPG; en otro caso aplica rechazo.
  *Por qué: predecir sobre no-EKG fabricaría un diagnóstico sin base fisiológica; el rechazo es el único comportamiento honesto.*

- **RF-09 (ubicuo):** La demo interactiva deberá mostrar exactamente el trazado del EKG, los diagnósticos sugeridos, la probabilidad de las 8 etiquetas y los avisos de RF-07a/RF-07b/RF-07d/RF-08 cuando apliquen. La app no deberá mostrar explicación por features ni gráficas de evaluación.
  *Por qué: es el contenido necesario para la discusión en vivo sin sobrecargar la sesión; la evidencia ampliada vive en `reports/`.*

- **RF-10 (ubicuo):** La demo deberá exhibir de forma permanente y visible que es una herramienta de apoyo pedagógico, no un diagnóstico clínico.
  *Por qué: delimita expectativas y riesgo de mal uso fuera del taller (Constitution §4).*

- **RF-11 (evento — Cuando):** Cuando se ejecute un entrenamiento, el sistema deberá generar en `reports/` la evaluación multietiqueta uno-vs-resto: matriz de confusión por etiqueta, curva ROC/AUC por cada una de las 8 salidas e importancia de variables global (una por etiqueta si el entrenamiento usa un clasificador por etiqueta), junto con la versión de PTB-XL, el split y la seed usados. Si una clase rara carece de muestras suficientes en test para una curva fiable, la gráfica deberá indicarlo explícitamente en lugar de omitirla.
  *Por qué: son insumo directo del taller y del manuscrito; sin ellas no hay evidencia que discutir, y omitir las raras equivaldría a ocultar el punto ciego.*

- **RF-12 (ubicuo):** El sistema deberá exponer sin suavizar las clases con mala performance en las gráficas y en la demo.
  *Por qué: el punto ciego es parte deliberada del mensaje pedagógico; maquillarlo destruye el valor del taller.*

- **RF-13 (ubicuo):** El pipeline de señal (features + entrenamiento sobre PTB-XL 100 Hz) deberá ser reejecutable en CPU-only, sin GPU, en menos de una hora en un laptop representativo documentado en `reports/` junto a las gráficas de la corrida. La digitalización de imágenes es interactiva por EKG y queda excluida de ese presupuesto de tiempo.
  *Por qué: es el hardware real disponible; lo que exija más infraestructura no es viable aquí (Constitution §5).*

- **RF-14 (ubicuo):** El sistema solo deberá entrenarse y evaluarse con datos públicos desidentificados (PTB-XL); ningún dato de paciente real del taller o práctica clínica deberá incorporarse ni versionarse. Los uploads de la demo en vivo no deberán persistir más allá de temporales efímeros de procesamiento, eliminados al cerrar la sesión, y nunca en el repo.
  *Por qué: compromiso innegociable de privacidad en entrenamiento e inferencia (Constitution §6).*

## Fuera de alcance (v1)

- Clasificadores sobre imagen (CNN 2D sobre el trazado renderizado) y cualquier framework de deep learning (PyTorch/TensorFlow).
- Entrada CSV genérica, formatos a 500 Hz, Holter prolongado o telemetría en tiempo real; en imagen, PDF multipágina y layouts distintos del 12 derivaciones estándar.
- Integración con historia clínica, PACS, dispositivos o uso clínico/validación diagnóstica.
- Digitalización perfecta de imágenes: se acepta degradación con aviso (RF-08); no se exige paridad de calidad señal-nativa ni calibración exacta mm/s–mm/mV.
- Explicabilidad avanzada en la app (SHAP, valores de features por predicción, curvas ROC embebidas): las gráficas viven en `reports/`, la app queda en el contenido exacto de RF-09.
- Lógica de exclusión NORM vs. patológicas: en v1 NORM es etiqueta plana más (RF-05).
- Infraestructura MLOps, GPUs, nube o reentrenamiento automático.
- Demo multiusuario concurrente, telemedicina u operación con SLO de latencia: en v1 la demo es local y monousuario en el laptop del taller.
- Versionar el dataset PTB-XL, persistir uploads de la demo o cualquier dato de paciente en el repo.

## Criterios de finalización (Done — verificables, sin CÓMO)

1. Se carga una señal PTB-XL 100 Hz nominal y se obtiene sugerencia multietiqueta de 8 salidas con probabilidad por etiqueta; se carga una imagen PNG/JPG y se obtiene el mismo tipo de salida vía señal 1D (nunca desde píxeles), admitiendo aviso de calidad en la vía imagen.
2. Cuando se sugiere una de las 3 clases raras, muestra siempre el aviso "patrón poco representado" (RF-07a), y cualquier etiqueta con probabilidad <0,60 muestra "confianza baja" (RF-07b) como aviso separado; si concurren, se apilan (RF-07d).
3. Existe un caso demostrable de EKG degradado pero reconocible que produce predicción + advertencia de calidad + confianza baja, nunca como predicción confiada; y un caso de fallo total de extracción que muestra incertidumbre sin etiquetas.
4. Existe un caso con ninguna etiqueta sobre 0,60 que muestra top-1 + aviso, y un archivo no-EKG/corrupto que se rechaza sin predicción y sin persistencia.
5. La demo muestra exactamente trazado + diagnóstico(s) + probabilidad de las 8 etiquetas + avisos cuando apliquen + aviso no-clínico permanente, sin explicaciones adicionales, y descarta uploads al cerrar sesión.
6. Una corrida de entrenamiento produce en `reports/` matriz por etiqueta, ROC/AUC por cada una de las 8 salidas (con nota explícita si una rara no tiene muestras suficientes) e importancia de variables, incluyendo clases con mala performance sin ocultarlas.
7. La corrida de la vía señal termina en <1 h en laptop CPU-only documentado, sin dependencias GPU; la digitalización queda fuera de ese presupuesto por ser interactiva.
8. El repositorio no contiene el dataset ni datos de pacientes ni uploads persistidos; `pytest` (si existen pruebas) pasa antes de dar por terminado un cambio.
