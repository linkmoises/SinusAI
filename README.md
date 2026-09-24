# SinusAI

Herramienta demostrativa de apoyo a la interpretación de electrocardiogramas (ECG/EKG),
pensada como recurso pedagógico para un taller. Muestra, en vivo, dónde una herramienta
de IA acierta y dónde tiene puntos ciegos.

> **Importante:** no es un producto clínico ni una validación diagnóstica. Su propósito
> es enseñar a cuándo confiar en una IA y cuándo no.

## Qué hace

El flujo general es:

1. **Carga del EKG**: se aceptan tanto señales digitales estándar como imágenes
   escaneadas o fotografiadas de un trazado en papel, que se convierten a señal.
2. **Medición de características clínicas**: del EKG se extraen medidas interpretables
   (frecuencia, duración del QRS, QT/QTc, eje, variabilidad, entre otras).
3. **Clasificación**: con esas medidas se obtiene una sugerencia diagnóstica sobre un
   conjunto de patrones (normal, infarto, alteraciones de ST-T, bloqueos, hipertrofia,
   y arritmias ventriculares), cada una con su nivel de confianza.
4. **Resultado en vivo**: se muestra el trazado, las probabilidades por diagnóstico y
   avisos cuando el modelo tiene confianza baja o el patrón está poco representado en
   los datos de entrenamiento.
5. **Evaluación**: cada entrenamiento genera sus gráficas de evaluación (matriz de
   confusión, curvas ROC/AUC e importancia de variables).

## Por qué es honesto por diseño

- El modelo trabaja sobre medidas clínicas, no sobre la imagen como "foto".
- Cuando un diagnóstico está poco representado en los datos (por ejemplo arritmias
  ventriculares), el sistema lo declara explícitamente en lugar de dar una respuesta
  confiada y engañosa.
- Las clases con mala performance no se ocultan: son parte deliberada del mensaje
  pedagógico del taller.

## Configuración

Clonar el repositorio y preparar el entorno:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Descargar el dataset **PTB-XL** (versión 100 Hz) dentro de la carpeta `data/`:

```bash
mkdir -p data
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./data/ptb-xl
```

La carpeta `data/` está ignorada por git: el dataset no se sube al repositorio.

## Uso

```bash
python train.py            # extrae features, entrena y genera las gráficas en reports/
streamlit run app.py       # levanta la demo interactiva del taller
```