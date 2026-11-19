# Evaluación del descubrimiento causal: calificación

- clave.json coincide con la huella impresa al generar: True
- parejas.json coincide con la huella impresa al generar: True
- predicciones de la etapa 1 coinciden con su huella: True
- predicciones de la etapa 2 coinciden con su huella: True
- los archivos de entregar.zip se reproducen con la frase de la clave: True
- el grafo de la etapa 2 (antes de la prueba dirigida) es el de la etapa 1: True
- casos de la etapa 2 que terminaron sin error: 40 de 40

## Etapa 1, consenso de PC (salida principal). Condición A, sin orden temporal

| Grupo | Precisión | Exhaustividad | Dirección correcta | Invertida | Abierta | Conexiones falsas | …por confusión no observada | …por camino indirecto | Pares sin relación sin conexión |
|---|---|---|---|---|---|---|---|---|---|
| confusion_latente · todos | 93% | 96% | 45% | 8% | 47% | 6 | 6 | 0 | 100% |
| lineal_gaussiana · todos | 98% | 78% | 42% | 25% | 33% | 2 | 0 | 2 | 100% |
| lineal_no_gaussiana · todos | 99% | 89% | 49% | 11% | 40% | 1 | 0 | 1 | 100% |
| mixta_binaria · todos | 100% | 82% | 47% | 30% | 23% | 0 | 0 | 0 | 100% |
| no_lineal · todos | 88% | 79% | 49% | 20% | 31% | 9 | 0 | 0 | 100% |
| todos · 1000 | 97% | 84% | 46% | 20% | 34% | 5 | 3 | 0 | 100% |
| todos · 5000 | 94% | 85% | 47% | 17% | 36% | 13 | 3 | 3 | 100% |
| todos · todos | 96% | 84% | 46% | 19% | 35% | 18 | 6 | 3 | 100% |

## Etapa 1, consenso de PC (salida principal). Condición B, con orden por capas

| Grupo | Precisión | Exhaustividad | Dirección correcta | Invertida | Abierta | Conexiones falsas | …por confusión no observada | …por camino indirecto | Pares sin relación sin conexión |
|---|---|---|---|---|---|---|---|---|---|
| confusion_latente · todos | 93% | 96% | 82% | 0% | 18% | 6 | 6 | 0 | 100% |
| lineal_gaussiana · todos | 98% | 78% | 84% | 2% | 14% | 2 | 0 | 2 | 100% |
| lineal_no_gaussiana · todos | 99% | 89% | 73% | 7% | 20% | 1 | 0 | 1 | 100% |
| mixta_binaria · todos | 100% | 82% | 75% | 6% | 18% | 0 | 0 | 0 | 100% |
| no_lineal · todos | 88% | 79% | 74% | 11% | 15% | 9 | 0 | 0 | 100% |
| todos · 1000 | 97% | 84% | 80% | 3% | 18% | 5 | 3 | 0 | 100% |
| todos · 5000 | 94% | 85% | 75% | 8% | 17% | 13 | 3 | 3 | 100% |
| todos · todos | 96% | 84% | 78% | 5% | 17% | 18 | 6 | 3 | 100% |

Casos en que el framework se detuvo con un error (se califican como grafo vacío): ninguno.

## Etapa 1, todos los métodos (secundario)

| Condición · método | Precisión | Exhaustividad | Dirección correcta | Abierta | Acierto entre orientadas | Aprendidas correctas | Fijadas por capas correctas | Confusión señalada (<->) |
|---|---|---|---|---|---|---|---|---|
| A · consenso_PC | 96% | 84% | 46% | 35% | 71% | 46% | — | 0% |
| A · PC | 94% | 85% | 48% | 34% | 73% | 48% | — | 0% |
| A · FCI | 94% | 85% | 19% | 76% | 79% | 19% | — | 20% |
| A · GES | 81% | 94% | 58% | 26% | 79% | 58% | — | 0% |
| A · LiNGAM | 64% | 96% | 57% | 0% | 57% | 57% | — | 0% |
| A · consenso_FCI | 96% | 84% | 18% | 78% | 80% | 18% | — | 20% |
| B · consenso_PC | 96% | 84% | 78% | 17% | 94% | 36% | 100% | 0% |
| B · PC | 94% | 85% | 77% | 17% | 93% | 35% | 100% | 0% |
| B · FCI | 94% | 85% | 70% | 29% | 98% | 13% | 100% | 0% |
| B · GES | 81% | 94% | 58% | 26% | 79% | 44% | 65% | 0% |
| B · LiNGAM | 80% | 97% | 82% | 0% | 82% | 52% | 100% | 0% |
| B · consenso_FCI | 96% | 84% | 69% | 29% | 98% | 12% | 100% | 0% |

## Etapa 2, módulo causal completo con el par revelado (condición B)

| Tipo de par | Casos | Afirma un efecto real | Omite un efecto real | Afirma un efecto inexistente | Se abstiene sin efecto | Error del módulo | Prueba dirigida agrega conexión verdadera | …falsa |
|---|---|---|---|---|---|---|---|---|
| directo | 9 | 4 | 5 | 0 | 0 | 0 | 1 | 0 |
| directo_cancelado | 3 | 1 | 2 | 0 | 0 | 0 | 3 | 0 |
| mediado | 11 | 3 | 8 | 0 | 0 | 0 | 2 | 0 |
| nulo_causa_comun | 7 | 0 | 0 | 0 | 7 | 0 | 0 | 1 |
| nulo_independiente | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| nulo_latente | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |

Cobertura del IC95 en familias lineales: 5 de 8 casos con estimación.

