# Historias de usuario

Cargar cada una como issue en GitHub/GitLab con la plantilla `historia`. Los commits referencian el número del issue.

| ID | Historia | Criterios de aceptación | Semana |
| --- | --- | --- | --- |
| HU-01 | Como agricultor, quiero sacar una foto de una hoja con el celular para saber si está enferma. | La UI acepta foto desde cámara o galería en el navegador del celular. | 3 |
| HU-02 | Como agricultor, quiero ver la enfermedad y su probabilidad para saber cuánto confiar. | Se muestran nombre, cultivo y probabilidad en %. | 3 |
| HU-03 | Como agricultor, quiero una sugerencia de manejo para saber qué hacer. | Las 15 clases tienen sugerencia en `treatments.json`. | 4 |
| HU-04 | Como agricultor, quiero que el sistema me avise cuando no está seguro. | Si p < 0,5 se muestra "No estoy seguro" y se pide otra foto. | 4 |
| HU-05 | Como desarrollador, quiero un endpoint `/predict` documentado. | Swagger en `/docs`; tests de `/predict` pasan. | 2 |
| HU-06 | Como investigador, quiero un dataset unificado y documentado. | `prepare_data.py` reproduce los splits; `data/README.md` completo. | 1 |
| HU-07 | Como investigador, quiero comparar EfficientNet-B0 y ResNet50. | Tabla con accuracy, F1 macro, top-3 y latencia de ambos. | 5-6 |
| HU-08 | Como operador, quiero levantar todo con un comando. | `docker compose up --build` deja UI y API funcionando. | 5 |
| HU-09 | Como evaluador, quiero ver el tiempo de inferencia. | La respuesta incluye `tiempo_inferencia_ms` y la UI lo muestra. | 4 |
