#!/bin/bash
# Evaluación interna del descubrimiento causal: una sola ejecución automática (véase PROTOCOLO_INTERNO.md).
#
#   FRAMEWORK_DIR=<carpeta de la versión 1.0.0> bash evaluacion_interna.sh <valor randomness de la ronda de drand> <salida>
#
# 1. Comprueba que los tres programas son los registrados (huellas_programas.sha256).
# 2. Genera los casos con el valor de drand como frase y aparta parejas.json y clave.json en <salida>/sobre.
# 3. Etapa 1: el módulo recibe sólo entregar.zip (CSV y capas); guarda sus grafos y la huella.
# 4. Etapa 2: con parejas.json corre el módulo causal completo para el par de cada caso.
# 5. Califica con clave.json todos los casos, incluidos los fallidos, y escribe calificacion.md y calificacion.json.
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
FRASE="${1:?Indique el valor randomness de la ronda de drand}"
SALIDA="${2:?Indique la carpeta de salida}"
: "${FRAMEWORK_DIR:?Indique FRAMEWORK_DIR, la carpeta de la versión 1.0.0 del framework}"
export FRAMEWORK_DIR PYTHONHASHSEED=0  # mismo orden de iteración que ejecutar_todo.sh de la versión 1.0.0
if [ -e "$SALIDA/resultados" ]; then echo "La carpeta $SALIDA/resultados ya existe: la evaluación se corre una sola vez." >&2; exit 1; fi
mkdir -p "$SALIDA/sobre"
LOG="$SALIDA/registro.txt"
{
  echo "inicio $(date -u +%FT%TZ)"
  echo "frase (randomness de drand) $FRASE"
  (cd "$AQUI" && sha256sum -c huellas_programas.sha256)
  python3 "$AQUI/generar_casos.py" --frase "$FRASE" --salida "$SALIDA/casos"
  mv "$SALIDA/casos/parejas.json" "$SALIDA/casos/clave.json" "$SALIDA/sobre/"
  echo "etapa 1 $(date -u +%FT%TZ)"
  python3 "$AQUI/correr_descubrimiento.py" etapa1 --datos "$SALIDA/casos/entregar.zip" --salida "$SALIDA/resultados"
  echo "etapa 2 $(date -u +%FT%TZ)"
  python3 "$AQUI/correr_descubrimiento.py" etapa2 --datos "$SALIDA/casos/entregar.zip" --parejas "$SALIDA/sobre/parejas.json" \
    --salida "$SALIDA/resultados"
  echo "calificación $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
HC=$(grep 'clave.json' "$LOG" | head -1 | awk '{print $2}')
python3 "$AQUI/calificar.py" --clave "$SALIDA/sobre/clave.json" --resultados "$SALIDA/resultados" \
  --parejas "$SALIDA/sobre/parejas.json" --datos "$SALIDA/casos/entregar.zip" \
  --huella-parejas "$(grep 'parejas.json' "$LOG" | head -1 | awk '{print $2}')" --huella-clave "$HC" > /dev/null
echo "fin $(date -u +%FT%TZ)" >> "$LOG"
echo "Listo: $SALIDA/resultados/calificacion.md"
