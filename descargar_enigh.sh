#!/usr/bin/env bash
# Descarga del INEGI los microdatos de la ENIGH, nueva serie (2016, 2018, 2020, 2022 y 2024),
# y los deja como los lee la ingesta: <carpeta>/<año>_<tabla>/<tabla>.csv
# Uso: bash descargar_enigh.sh microdatos      (unos 1.1 GB descomprimidos)
# Si una descarga falla, el archivo puede bajarse a mano desde
# https://www.inegi.org.mx/programas/enigh/nc/<año>/ (sección Microdatos, formato CSV).
set -euo pipefail
DEST="${1:-microdatos}"
mkdir -p "$DEST"
BASE="https://www.inegi.org.mx/contenidos/programas/enigh/nc"
for anio in 2016 2018 2020 2022 2024; do
  for tabla in concentradohogar hogares poblacion ingresos trabajos; do
    dir="$DEST/${anio}_${tabla}"
    if [ -f "$dir/$tabla.csv" ]; then
      echo "ya existe: $dir/$tabla.csv"
      continue
    fi
    mkdir -p "$dir"
    url="$BASE/$anio/microdatos/enigh${anio}_ns_${tabla}_csv.zip"
    echo "descargando $url"
    curl -fL --retry 3 -o "$dir/descarga.zip" "$url"
    rm -rf "$dir/descarga"
    unzip -o -q "$dir/descarga.zip" -d "$dir/descarga"
    csv="$(find "$dir/descarga" -type f -iname "${tabla}.csv" | head -1)"
    if [ -z "$csv" ]; then
      csv="$(find "$dir/descarga" -type f -ipath "*conjunto_de_datos/*${tabla}*.csv" | head -1)"
    fi
    if [ -z "$csv" ]; then
      echo "no encontré ${tabla}.csv dentro de $url" >&2
      exit 1
    fi
    mv "$csv" "$dir/$tabla.csv"
    rm -rf "$dir/descarga" "$dir/descarga.zip"
  done
done
echo "listo: $DEST"
