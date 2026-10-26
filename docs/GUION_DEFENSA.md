# Guion de la defensa (10 min + 2 min de preguntas)

La demo en vivo vale más que cualquier diapositiva: dura 3 minutos y va en el medio, cuando el tribunal ya entendió el problema.

## Reparto

| Min | Bloque | Quién | Qué mostrar |
| --- | --- | --- | --- |
| 0:00–1:00 | Problema | Palmira | Soja récord 2025/26 (12,3 Mt); la roya puede quitar hasta 80% del rendimiento; alerta del SENAVE en enero 2026. Mandioca y sésamo: agricultura familiar. |
| 1:00–2:30 | Datos | Palmira | Tres datasets públicos → 15 clases. Limpieza: duplicados, **solo originales del sésamo (evita fuga)**, tope por clase. Mostrar el desbalance. |
| 2:30–4:00 | Modelo | Palmira | EfficientNet-B0 vs ResNet50, misma receta. Aumento de datos "de campo" y sampler ponderado. Ablación. |
| 4:00–5:30 | Resultados | Palmira | Tabla II del informe, matriz de confusión, latencia en CPU. Una frase: "elegimos EfficientNet-B0 porque ___". |
| 5:30–8:30 | **Demo en vivo** | Sebastián | Ver abajo. |
| 8:30–10:00 | Limitaciones y futuro | Sebastián | Sin fotos paraguayas; CMD/CBSD no están en América; filtro por cultivo; modelo offline; guaraní. |

## Demo en vivo (3 min)

Antes de entrar: `docker compose up` **ya corriendo**, pestañas abiertas, celular conectado al mismo wifi y con la URL abierta.

1. **Arquitectura en 15 s.** Terminal: `docker compose ps` (dos contenedores, *healthy*).
2. **Caso que acierta (celular).** Subir la foto de una hoja de soja con roya, de las imágenes de test. Leer en voz alta: enfermedad, probabilidad, sugerencia, tiempo en ms.
3. **Otro cultivo.** Hoja de mandioca o sésamo sana.
4. **Caso difícil, a propósito.** Foto de algo que no es una hoja (la mano, un cuaderno). Mostrar el aviso "No estoy seguro". Esto demuestra que pensaron en el uso real.
5. **API.** `http://localhost:8000/docs` → `POST /predict` → *Try it out*. Muestra que otra app (Flutter, PWA) puede usar el mismo backend.

### Plan B (si algo falla)

- Sin wifi: hacer la demo en la laptop (`localhost:8501`), sin celular.
- Docker no arranca: `make api` y `make ui` en dos terminales (sin Docker).
- Todo falla: video de 2 min grabado la semana 8 con la demo completa.
- Tener 5 fotos de prueba en el escritorio de la laptop, ya elegidas y probadas.

## Checklist del día

- [ ] `models/best_model.pt` es el modelo final (no el de prueba): `curl localhost:8000/health` muestra las métricas de validación.
- [ ] Imágenes de Docker ya construidas (el build tarda; no hacerlo en la defensa).
- [ ] Laptop enchufada, notificaciones apagadas, zoom del navegador al 125%.
- [ ] Celular con brillo alto y pantalla duplicada o cámara de documentos, si hay.
- [ ] Video del plan B en el escritorio.

## Preguntas probables del tribunal

| Pregunta | Respuesta corta |
| --- | --- |
| ¿Por qué F1 macro y no accuracy? | Las clases están desbalanceadas; la accuracy puede ser alta prediciendo bien solo las clases grandes. El F1 macro pesa igual a todas. |
| ¿Cómo evitaron la fuga de datos? | Duplicados exactos por MD5, solo imágenes originales del sésamo y división estratificada antes de aumentar datos (el aumento se aplica solo en train, en memoria). |
| ¿Funciona con fotos de Paraguay? | No lo sabemos con certeza: ninguna foto es local. Por eso hay umbral de confianza, y es el primer trabajo futuro. |
| ¿Por qué EfficientNet-B0 si ResNet50 tiene más parámetros? | ___ (completar con los resultados: casi siempre es "rinde igual o mejor con 6 veces menos parámetros y menos latencia"). |
| ¿Qué aporta el aumento de datos? | La ablación: F1 macro ___ sin las técnicas vs ___ con ellas. |
| ¿Por qué hay enfermedades que no existen en América? | Vienen del dataset de Uganda. Las dejamos para no sesgar la comparación con la literatura; la API advierte al usuario si las detecta. Se pueden quitar. |
| ¿Por qué Streamlit y no una app nativa? | Prioridad: un sistema completo en 8 semanas. La UI solo consume la API, así que se puede reemplazar por una PWA o Flutter sin tocar el backend. |
| ¿Cuánto tarda una predicción? | ___ ms en CPU, sin GPU: viable en una laptop o servidor barato. |
| ¿Qué pasa si el agricultor saca una foto borrosa? | El aumento de datos incluye desenfoque y cambios de luz; si igual no está seguro, el sistema pide otra foto. |
