#!/usr/bin/env bash
# Ejecuta las pruebas y todos los análisis de la tesis, en orden, y al final compara lo producido
# con la versión publicada.
# Uso:
#   bash ejecutar_todo.sh                    # con los datos procesados incluidos en datos/
#   bash ejecutar_todo.sh <carpeta de CSV>   # rehace antes la ingesta desde los microdatos del INEGI
#                                            # (bash descargar_enigh.sh <carpeta de CSV> los descarga)
# Cada script escribe sus salidas en resultados/ y su registro en resultados/<nombre>.log. El paso más
# largo (descubrimiento_m3.py) tarda unos 40 min con 2 núcleos; PPO, unos 13; la predicción, unos 5.
# La primera ejecución guarda la versión publicada en publicado/. Cada ejecución empieza con resultados/
# vacía (la anterior pasa a resultados_anterior/) y, si se rehace la ingesta, también con datos/ vacía (la
# anterior pasa a datos_anterior/): así, un archivo que no se genere aparece como faltante. Termina con
# código distinto de cero si falla una prueba o un script, o si algún archivo difiere o falta.
set -euo pipefail
CSV=""
if [ $# -ge 1 ]; then
  CSV="$(cd "$1" && pwd)"
fi
cd "$(dirname "$0")"
export FRAMEWORK_DIR="$PWD"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
mkdir -p datos resultados

# copia de la versión publicada, para compararla al final (sólo en la primera ejecución)
if [ ! -d publicado ]; then
  mkdir publicado
  cp -r datos resultados publicado/
fi

# salidas vacías: resultados/ siempre; datos/ si se rehace la ingesta (si no, sólo se borran los archivos de
# datos/ que la ejecución vuelve a generar; los demás son sus entradas)
rm -rf resultados_anterior
mv resultados resultados_anterior
mkdir resultados
if [ -n "$CSV" ]; then
  rm -rf datos_anterior
  mv datos datos_anterior
  mkdir datos
else
  rm -f datos/categoria_indigena_2020.csv datos/inpi_municipios_2010.csv datos/diccionario_datos.csv datos/matriz_procedencia.csv
fi

# correr <registro> <comando...>: corre el comando y guarda lo que imprime en resultados/<registro>.log
correr() {
  local registro="$1"
  shift
  echo "== $registro"
  "$@" | tee "resultados/$registro.log"
}

# datos: la ingesta sólo si se pasa la carpeta de microdatos; si no, se usan los de datos/
if [ -n "$CSV" ]; then
  correr ingesta_2022_2024 python3 -I ingesta/enigh.py "$CSV" datos                            # análisis causal
  correr ingesta_2016_2024 python3 -I ingesta/enigh.py "$CSV" datos 2016,2018,2020,2022,2024   # 2016 a 2024
  correr ingesta_hogares python3 -I ingesta/hogares_m3.py "$CSV" datos/enigh_personas_2016_2024.parquet datos/enigh_hogares_2016_2024.parquet
fi
# municipios indígenas de la regla de elegibilidad de 2020: categoría A y B de 2020 (principal) y catálogo de 2010
correr ingesta_zap2020 python3 -I ingesta/zap2020.py fuentes/anexo_a_zap_2020.pdf datos/categoria_indigena_2020.csv
correr ingesta_inpi python3 -I ingesta/inpi.py fuentes/catalogo_de_localidades_indigenas_2010.csv datos/inpi_municipios_2010.csv

# pruebas automáticas (las de datos revisan los datos que se van a usar)
python -m pytest -q pruebas

# validación con verdad conocida (series y microdatos simulados, experimento NSW)
correr sim_temporal python experimentos/sim_temporal.py 100
correr sim_social python experimentos/sim_social.py 50
correr nsw_benchmark python experimentos/benchmark_nsw.py
correr nsw_pipeline python experimentos/nsw_pipeline.py
correr nsw_semillas python experimentos/nsw_semillas.py

# efectos causales con microdatos
correr piloto_b_pension python experimentos/piloto_b_pension.py
correr piloto_a_becas python experimentos/piloto_a_becas.py
correr escenario_pmb python experimentos/escenario_pmb.py
correr falsacion_regla python experimentos/falsacion_regla.py
correr falsacion_ventanas python experimentos/falsacion_ventanas.py
correr robustez_p104 python experimentos/robustez_p104.py
correr beca_universal_did python experimentos/beca_universal_did.py

# módulo predictivo
correr descubrimiento_m3 python experimentos/descubrimiento_m3.py
correr prediccion_m3 python experimentos/prediccion_m3.py
correr shap_m3 python experimentos/shap_m3.py
correr riesgo_pension_m3 python experimentos/riesgo_pension_m3.py
correr descomposicion_m3 python experimentos/descomposicion_m3.py

# escenarios
correr escenarios_m4 python experimentos/escenarios_m4.py
correr ppo_m4 python experimentos/ppo_m4.py

# OE1 y utilidad
correr diccionario python3 -I ingesta/diccionario.py
correr reforma_2021_exante python experimentos/reforma_2021_exante.py
correr utilidad_m4 python experimentos/utilidad_m4.py

# figuras
python figuras/figuras.py figuras datos/enigh_personas_2022_2024.parquet

# comparación con la versión publicada
estado=0
python comparar_resultados.py publicado/datos datos || estado=$?
python comparar_resultados.py publicado/resultados resultados || estado=$?
if [ "$estado" -eq 0 ]; then
  echo "listo: todos los archivos coinciden con la versión publicada."
else
  echo "atención: hay archivos distintos o faltantes (detalle arriba). Con otras versiones de las bibliotecas"
  echo "pueden variar decimales de PPO y de los modelos de boosting."
fi
exit "$estado"
