# AGENTS.md

## Qué es este proyecto

Herramienta demostrativa de apoyo diagnóstico para un taller de interpretación de EKG.
No es un producto clínico ni de validación diagnóstica: es un recurso pedagógico para
mostrar, en vivo, dónde una herramienta de IA acierta y dónde tiene puntos ciegos.

Flujo: EKG (señal o imagen escaneada) → extracción de features clínicas → clasificación
con Random Forest → diagnóstico sugerido + nivel de confianza + gráficas de evaluación
(matriz de confusión, curva ROC/AUC, importancia de variables).

## Stack

- Python 3.11+
- `neurokit2` — extracción de features (QRS, QT, FC, eje, HRV) desde señal 1D
- `scikit-learn` — Random Forest / XGBoost sobre features tabulares
- `streamlit` — interfaz de carga y demo
- `wfdb` — lectura del dataset PTB-XL (formato nativo, NO imágenes)
- Opcional/stretch: `ecg-miner` o script propio de digitalización, solo para EKG escaneados

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dataset: descargar PTB-XL 100Hz (no la de 500Hz — no se necesita para esta demo) desde
PhysioNet y colocar en `data/ptbxl/`. No versionar el dataset en git.

## Comandos

- `python train.py` — extrae features, entrena el modelo, guarda `model.pkl` y las
  gráficas de evaluación en `reports/`
- `streamlit run app.py` — levanta la demo interactiva
- `pytest` — pruebas (si existen) antes de dar por terminado un cambio

## Reglas de diseño (no negociables para este proyecto)

1. **La señal es una serie temporal, no una imagen.** Nunca entrenar un modelo que trate
   el EKG como pixeles (CNN 2D sobre el trazado renderizado). Esto ya se intentó y falló
   rotundamente en un proyecto anterior — no se repite ese error aquí.
2. **Features clínicas, no caja negra.** El modelo debe operar sobre medidas
   interpretables (QRS, QT, FC, eje, HRV), nunca sobre representaciones que no se puedan
   explicar a un médico en el taller.
3. **Declarar confianza baja quando el diagnóstico está subrepresentado en PTB-XL**
   (taquicardia ventricular, fibrilación ventricular, bloqueo AV completo). No forzar una
   predicción confiada donde el dataset casi no tiene ejemplos — mostrar explícitamente
   "confianza baja / patrón poco representado en los datos de entrenamiento".
4. **Todo entrenamiento debe generar sus tres gráficas de evaluación** (matriz de
   confusión, ROC/AUC por clase, importancia de variables) en `reports/` — son insumo
   directo para el taller y para el manuscrito.
5. **CPU-only, sin GPU.** El entorno de desarrollo y demo es un laptop sin GPU dedicada.
   Nada de deep learning pesado; el pipeline completo (features + RF) debe poder
   reentrenarse en menos de una hora en ese hardware.
6. **Cambios pequeños y verificables.** Preferir ediciones puntuales sobre reescribir
   archivos completos. Evitar abstracciones o generalización que el alcance actual
   (una demo de taller) no necesita.

## Qué NO hacer

- No agregar frameworks de deep learning (PyTorch/TensorFlow) salvo que se pida
  explícitamente — no son necesarios para este alcance.
- No ocultar ni suavizar las clases con mala performance con tal de que la demo "se vea
  bien" — el punto ciego del modelo es parte deliberada del mensaje pedagógico.
- No subir el dataset PTB-XL ni datos de pacientes reales al repositorio.
