# Challenge Grupo 3

## Objetivo

Este proyecto busca clasificar imágenes de signos con un modelo de deep learning en PyTorch. El flujo es sencillo: preparar datos, entrenar, evaluar y guardar resultados.

## Formalización de la tarea

En inferencia, el modelo recibe una imagen RGB y predice la clase más probable. En entrenamiento, aprende a partir de ejemplos etiquetados de `train`, se ajusta con `val` y se comprueba con `test`.

## Estructura del proyecto

Los scripts principales están en `src/Challenge`: `dataset.py`, `model.py`, `train.py`, `evaluate.py` y `pipeline.py`. Los datos están en `dataset/` y los artefactos se guardan en `outs/`.

## Métricas de evaluación

Se usan métricas estándar de clasificación multiclase: accuracy global, precision/recall/F1 por clase y matriz de confusión. Con esto se ve tanto el rendimiento general como los errores por clase.

## Arquitectura y entrenamiento

La solución usa una arquitectura tipo VGG definida en `model.py`, con `CrossEntropyLoss` y `AdamW`. Los parámetros importantes de ejecución, como `BATCH_SIZE` e `IMAGE_SIZE`, están en `train.py`.

## Ejecución

Para ejecutar todo en orden:

```bash
python src/Challenge/pipeline.py
```

Si quieres hacerlo por pasos:

```bash
python src/Challenge/dataset.py
python src/Challenge/model.py
python src/Challenge/train.py
python src/Challenge/evaluate.py
```

## Resultados

El entrenamiento guarda `best_model.pth` y la curva de pérdida. La evaluación genera métricas y matrices de confusión para train, validation y test. Todo queda en `outs/`.

## Análisis de resultados

La curva de pérdida permite ver cómo evoluciona el aprendizaje por épocas. Si la pérdida de entrenamiento baja y la de validación se mantiene razonable, el modelo está aprendiendo de forma estable. Si se separan mucho, puede haber sobreajuste.

### Curva de pérdida

![Loss plot](./outs/Challenge/loss_plot.png)

Las matrices de confusión ayudan a entender en qué clases acierta más el modelo y en cuáles se confunde. La diagonal principal representa los aciertos; cuanto más marcada esté, mejor.

### Matriz de confusión (train)

![Confusion matrix train](./outs/Challenge/confusion_matrix_train.png)

### Matriz de confusión (validation)

![Confusion matrix validation](./outs/Challenge/confusion_matrix_validation.png)

### Matriz de confusión (test)

![Confusion matrix test](./outs/Challenge/confusion_matrix_test.png)

### Conclusiones

De forma general, el modelo aprende bien el patrón del problema. En entrenamiento alcanza una accuracy alta (0.9439), y en validación y test mantiene valores razonables (0.8342 y 0.8756). Esto sugiere una generalización aceptable, aunque con cierta caída respecto a train.

Las matrices de confusión muestran una diagonal bastante marcada, sobre todo en train y test. Aun así, hay clases que siguen siendo más difíciles. En test, por ejemplo, las clases **E**, **C** y **K** tienen los F1 más bajos, mientras que **B** y **P** se comportan mejor.

En resumen: el modelo funciona bien para una primera versión, pero todavía tiene margen de mejora en clases concretas. Un siguiente paso lógico sería reforzar esas clases más débiles con más datos o ajustes de entrenamiento.
