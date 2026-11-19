# Evaluación interna del descubrimiento causal

Evalúa el módulo de descubrimiento de la versión 1.0.0 (etiqueta v1.0.0, commit 439bdf4), sin cambios, en 40 casos
sintéticos nuevos: cinco familias de mecanismos (lineal gaussiana, lineal no gaussiana, no lineal, mixta con variables
binarias y con confusión no observada), 1,000 y 5,000 observaciones y grafos aleatorios de 8 a 12 variables, con 448
conexiones verdaderas. Es una **evaluación interna**: el autor generó los casos, corrió el módulo y calificó. Sustituye a
una corrida anterior, descartada por un error en su registro de ejecución. Para que los casos no pudieran elegirse a
conveniencia:

1. El protocolo (`PROTOCOLO_INTERNO.md`, SHA-256 `6a309ec3216e06c09c86d8f3df84b3b2399425f13b6c0295d6f9ef4451e14432`) y
   los programas (`huellas_programas.sha256`) se fijaron antes de generar los datos. La huella del protocolo se registró
   en el [issue #1](https://github.com/0x141310/framework-causal-politicas-publicas/issues/1) el 19 de noviembre de 2026
   a las 19:53 UTC.
2. La frase que genera los casos es el campo `randomness` de la ronda 6656254 de drand (cadena `default` de la League of
   Entropy), emitida a las 20:04:00 UTC, 11 minutos después del registro: https://api.drand.sh/public/6656254.
   `verificar_drand.py` comprueba su firma BLS con la clave pública de la red (`pip install py_ecc`); el resultado está
   en `corrida_drand_6656254/verificacion_drand.txt`.
3. Una sola corrida automática (`evaluacion_interna.sh`) comprobó las huellas de los programas, generó los casos, apartó
   `parejas.json` y `clave.json`, corrió la etapa 1 (sólo los datos), la etapa 2 (con el par de cada caso) y calificó
   todos los casos; ninguno terminó con error. El registro está en `corrida_drand_6656254/registro.txt`.

## Resultados

Consenso de PC antes de la prueba dirigida, la salida principal:

| Medida | A: sólo los datos | B: con orden temporal |
| --- | --- | --- |
| Conexiones verdaderas recuperadas | 84% | 84% |
| Conexiones propuestas que son verdaderas | 95% | 95% |
| Dirección de las recuperadas: correcta, invertida, abierta | 46%, 19%, 35% | 78%, 5%, 17% |
| Conexiones falsas (con mecanismos no lineales o confusión no observada) | 18 (15) | 18 (15) |
| Pares sin relación sin conexión propuesta | 780 de 780 | 780 de 780 |

Con el orden temporal acertó la dirección de todas las conexiones que fijaban las capas y de 36% de las demás. FCI marcó
la confusión no observada en 2 de los 10 pares afectados sin orden temporal y en ninguno con él.

Con el par señalado y el orden temporal, el módulo completo afirmó 8 de 23 efectos reales (4 de 9 directos, 1 de 3 casi
cancelados y 3 de 11 mediados) y se abstuvo en los 17 pares sin efecto, incluidos los 2 con un confusor no observado
común: no afirmó ningún efecto inexistente. En 14 de sus 15 omisiones el dictamen fue «no identificable». En las
familias lineales, el IC95 cubrió el efecto verdadero en 5 de 8 casos con estimación.

Detalle por familia, tamaño y método: `corrida_drand_6656254/resultados/calificacion.md`. Ahí los porcentajes se
redondean dos veces (primero a tres decimales), por lo que la precisión aparece como 96%; el valor exacto es 378 de 396
conexiones propuestas, 95.45%.

## Reproducir

Desde la raíz del repositorio, con las dependencias de `requirements.txt`:

```
FRAMEWORK_DIR=$PWD bash evaluacion_interna/evaluacion_interna.sh 357af981de34fe517678050046310384d49a7615d57222f4bda3d0e268fc6cca otra_corrida
```

Una segunda corrida con la misma frase reprodujo los mismos casos (`entregar.zip`, SHA-256 `20bd4f79…`), las mismas
predicciones (salvo los tiempos) y la misma calificación.

`python3 -m pytest evaluacion_interna/test_evaluacion_interna.py` prueba el generador y la calificación.
