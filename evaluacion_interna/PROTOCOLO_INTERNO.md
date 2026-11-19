# Protocolo: evaluación interna del descubrimiento causal

**Pregunta.** Cuando nadie le indica qué relación buscar, ¿qué estructura recupera el módulo de descubrimiento del
framework, cuántas conexiones propone por error y cuándo deja una dirección abierta? ¿Cuánto ayuda conocer el orden
temporal? Y, una vez señalado un par de variables, ¿cuándo afirma un efecto y cuándo se abstiene?

**Carácter.** Evaluación interna: el autor genera los casos, corre el framework y califica; nadie externo custodia la
clave. Para que los casos no puedan elegirse a conveniencia, este protocolo y los programas se fijan antes de generar los
datos, y la frase que genera los casos es un valor público de drand que no existe al registrar el protocolo. Esta
evaluación sustituye a una corrida anterior, descartada por un error en su registro de ejecución; se hace con casos
nuevos, generados con una semilla nueva.

**Versión evaluada.** Framework 1.0.0 (etiqueta v1.0.0, commit 439bdf4510810755edd8c4737fefd583114f13a6), sin cambios:
una copia cuyos 129 archivos coinciden con MANIFEST.sha256. Antes de correr, `correr_descubrimiento.py` comprueba los
archivos de `modulo_causal/` contra ese manifiesto. Parámetros de la tesina: PC con prueba de Fisher z, α = 0.01,
partición aleatoria 50/50 con semilla 0, 50 remuestreos, umbral de consenso de 0.6 y de orientación de 0.5; FCI con 25
remuestreos; GES (BIC) y DirectLiNGAM (umbral de 0.05) en una corrida.

**Programas** (huellas SHA-256 al registrar este protocolo):
- `generar_casos.py` 0474ab597c163e9e7a959dc064680a35f4a01fa8801373372dc2c2acfe00828d
- `correr_descubrimiento.py` 251c542f06fff05be8cbf77899d576d81008b08a71cf07fad266b91691bc5526
- `calificar.py` fbed12f7eb0e2e1ac70acd70ae096c0eadf1f73e854999c171433140482dfc01
- `evaluacion_interna.sh` aca2fcefd7e7972758fbde2ea206600c8a400875b96d61714722c7218cd9d01e

Los programas se escribieron primero para una validación en la que una persona independiente custodiaría la clave. Para
esta evaluación se les añadió el manejo de fallos: si el framework se detiene en un caso, el error se registra, el caso
se califica como grafo vacío (etapa 1) o como «error del módulo» (etapa 2) y la corrida sigue con los demás. Además, sus
mensajes y textos se ajustaron a una corrida interna y la comprobación de `entregar.zip` compara el contenido de sus
archivos. La generación de los casos y las métricas no cambiaron: con la frase de control, el generador produce byte por
byte los mismos datos (los CSV, `capas.json`, `parejas.json` y `clave.json`; sólo cambia el texto de `LEEME.txt`) y, en
los 10 casos de control, los grafos, los dictámenes y la calificación son idénticos a los anteriores; sólo se agrega el
registro de errores, vacío.

**Semilla.** La huella de este protocolo se registra en un issue del repositorio público del framework. La frase del
generador es el campo `randomness` de la ronda de la cadena `default` de drand (League of Entropy; chain hash
8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce; esquema pedersen-bls-chained; génesis 1595431050;
periodo de 30 s) que se emite en el segundo 0 del minuto que resulta de sumar 11 minutos a la hora de creación del
issue, en UTC y truncada al minuto: si ese instante es t (segundos Unix), la ronda es (t − 1595431050)/30 + 1, y se
consulta en https://api.drand.sh/public/<ronda>. Así, el valor aleatorio se publica al menos 10 minutos después del
registro y nadie podía conocerlo al registrar el protocolo.

**Casos.** Diseño por omisión del generador: cinco familias de mecanismos (lineal gaussiana, lineal no gaussiana, no
lineal, mixta con variables binarias y con confusión no observada) × 1,000 y 5,000 observaciones × 4 réplicas = 40
casos de 8 a 12 variables anónimas, cada uno con un grafo aleatorio distinto. Incluyen efectos directos, mediación,
causas comunes y variables aisladas; en una cuarta parte de los casos lineales, el par de la etapa 2 tiene un efecto
casi cancelado por una causa común observada.

**Información previa.** Condición A: ninguna. Condición B: las capas temporales de `capas.json` (ninguna variable de una
capa posterior causa a una de una capa anterior). No se da ninguna otra información.

**Secuencia.** Una sola ejecución automática:
`FRAMEWORK_DIR=<copia de la versión 1.0.0> bash evaluacion_interna.sh <randomness> <salida>`. El script comprueba las
huellas de los programas; genera los casos y aparta `parejas.json` y `clave.json`; corre la etapa 1 sólo con los datos y
guarda la huella de sus predicciones; corre la etapa 2 con las parejas; y califica con la clave todos los casos. Se niega
a escribir sobre resultados existentes.

**Salida principal.** El grafo de consenso de PC antes de la prueba dirigida: el candidato que usa el módulo causal. La
prueba dirigida necesita un par, así que sólo se evalúa en la etapa 2.

**Métricas de la etapa 1** (por condición, familia y tamaño de muestra):
- precisión (conexiones propuestas que son verdaderas) y exhaustividad (conexiones verdaderas recuperadas);
- dirección de las conexiones verdaderas recuperadas: correcta, invertida o abierta (sin dirección, bidirigida o con
  círculos); en la condición B se separan las direcciones que ya fijaba el orden por capas;
- conexiones falsas según su origen: confusión no observada, camino indirecto, causa común observada u otro;
- pares sin relación (sin camino y sin causa común) en los que no se propuso conexión;
- casos en que el framework se detuvo con un error.

Secundarias: las mismas métricas para PC, FCI, GES y DirectLiNGAM en una corrida y para el consenso de FCI.

**Métricas de la etapa 2.** Con el par revelado se corre el módulo causal completo en la condición B. Se registra si el
dictamen afirma un efecto (identificado o identificado frágil), se abstiene o termina con error, frente a la verdad
(efecto directo, mediado o casi cancelado; o sin efecto, con causa común observada o no observada, o independiente).
También si la prueba dirigida agrega una conexión verdadera o falsa y, en las familias lineales, si el IC95 cubre el
efecto verdadero.

**Reglas.**
- No se modifica el framework ni sus parámetros, y no se repite ninguna etapa con otros ajustes ni con otra semilla.
- Si un programa de la evaluación (no el framework) tiene un error, se corrige, se documenta con su nueva huella y se
  reporta; los casos y la semilla no cambian.
- Se reporta todo, sea cual sea el resultado, incluidos los casos fallidos.
- Las corridas con la frase pública «control publico 2026» sólo sirven para comprobar los programas y no forman parte
  de la evidencia.

**Reporte.** En la tesina, una frase en la Sección VII-A y una tabla y un párrafo en el Anexo C; en la defensa, las
diapositivas 3 y 19. Los programas, los datos, las predicciones, la clave y el registro se publican en la versión 1.0.1
del repositorio.
