# Constitution

Principios que gobiernan las decisiones de diseño de esta herramienta. Cuando una
decisión técnica no esté cubierta en AGENTS.md, se resuelve contra estos principios,
en orden.

## 1. Honestidad sobre los límites, antes que impresionar

El objetivo de esta herramienta no es parecer inteligente — es enseñar, con evidencia,
cuándo confiar en una IA diagnóstica y cuándo no. Una predicción confiada en una clase
subrepresentada (ej. fibrilación ventricular) es peor que no dar predicción alguna.
Ante la duda entre "dar una respuesta" y "declarar incertidumbre", se declara
incertidumbre.

## 2. La representación de los datos es la decisión más importante

La señal de un EKG es una serie temporal fisiológica, no una fotografía. Cualquier
enfoque que trate el trazado como imagen pierde la estructura que le da sentido clínico
y ya demostró fallar en un proyecto previo. Toda arquitectura de este proyecto parte de
la señal 1D o de features derivadas de ella — nunca de píxeles.

## 3. Interpretabilidad por encima de desempeño marginal

Un punto porcentual más de accuracy no justifica perder la capacidad de explicar por
qué el modelo sugirió un diagnóstico. Se prefiere un modelo más simple que el médico
pueda auditar (features clínicas + Random Forest) sobre uno más complejo que no se
pueda explicar en un taller de una tarde.

## 4. Alcance proporcional al propósito

Esto es una demostración pedagógica para un taller, no un dispositivo diagnóstico ni
un producto en validación clínica. El alcance, el tiempo de desarrollo y la
complejidad del código deben ser proporcionales a ese propósito — no al máximo que
técnicamente sería posible construir.

## 5. Reproducibilidad simple sobre infraestructura sofisticada

El proyecto debe poder reentrenarse y correr en el hardware real disponible (laptop
sin GPU), sin dependencias de infraestructura que no existen para este contexto
(clústeres, GPUs en la nube, pipelines de MLOps). Si una solución requiere
infraestructura que no se tiene, no es la solución correcta para este proyecto.

## 6. Los datos de pacientes no se comprometen

Solo se usan datos públicos y desidentificados (PTB-XL). Ningún dato de paciente real
del taller o de la práctica clínica del autor se incorpora al entrenamiento o se sube
al repositorio, bajo ninguna circunstancia.

## 7. Cambios pequeños, verificables, reversibles

Se prefiere una serie de cambios pequeños y explicables sobre una reescritura grande
que sea difícil de revisar. Cada cambio debe poder explicarse en una frase y revertirse
sin afectar el resto del sistema.
