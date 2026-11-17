"""Módulo 1: detección de tipos, diccionario de datos, matriz de procedencia y conciliación (OE1).

Detecta el tipo de cada variable a partir de los datos (no de su nombre, salvo llaves y
variables de diseño muestral), lo compara con una clasificación de referencia escrita antes de
correr la detección, documenta de qué tabla y columnas sale cada variable en cada año y concilia
la prevalencia de inseguridad alimentaria con la cifra oficial publicada por CONEVAL.

Uso: python3 -I ingesta/diccionario.py
Salidas: datos/diccionario_datos.csv, datos/matriz_procedencia.csv, resultados/oe1_ingesta.json
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json

import numpy as np
import pandas as pd

BASE = ROOT
PERSONAS = f"{BASE}/datos/enigh_personas_2016_2024.parquet"
HOGARES = f"{BASE}/datos/enigh_hogares_2016_2024.parquet"
YEARS = [2016, 2018, 2020, 2022, 2024]

# --- variables cuyo papel no puede inferirse de los valores (se declaran por nombre)
LLAVES = {"folioviv", "foliohog", "numren"}
TERRITORIALES = {"ubica_geo", "entidad"}
TEMPORALES = {"anio"}
DISENO = {"factor", "upm", "est_dis"}


def inferir_tipo(nombre: str, s: pd.Series) -> tuple[str, str]:
    """Tipo inferido y regla aplicada."""
    if nombre in LLAVES:
        return "identificador", "llave declarada"
    if nombre in TERRITORIALES:
        return "territorial", "llave declarada"
    if nombre in TEMPORALES:
        return "temporal", "llave declarada"
    if nombre in DISENO:
        return "diseño muestral", "variable de diseño declarada"
    x = s.dropna()
    if x.empty:
        return "vacía", "sin valores"
    if not pd.api.types.is_numeric_dtype(x):
        frac = x.nunique() / len(x)
        return ("identificador", "texto casi único") if frac > 0.5 else ("nominal", "texto con pocas categorías")
    u = np.unique(x.to_numpy())
    entero = bool(np.all(np.equal(np.mod(u, 1), 0)))
    if len(u) <= 2:
        return "binaria", "dos valores distintos"
    if entero and len(u) <= 12:
        return "categórica", "enteros con 3 a 12 valores (códigos)"
    if entero and u.min() >= 0:
        return "conteo", "enteros no negativos con más de 12 valores"
    if entero:
        return "continua", "enteros con signo"
    if u.min() >= 0 and u.max() <= 1:
        return "continua", "proporción en [0, 1]"
    return "continua", "valores no enteros"


# --- clasificación de referencia (escrita antes de correr la detección), según el descriptor de INEGI
REF = {
    # personas
    "parentesco": "categórica", "sexo": "binaria", "edad": "conteo", "hablaind": "binaria", "asis_esc": "binaria",
    "nivel": "categórica", "grado": "categórica", "tipoesc": "categórica", "tiene_b": "binaria",
    "nivelaprob": "categórica", "gradoaprob": "categórica", "disc_ver": "categórica", "disc_camin": "categórica",
    "n_trabajos": "categórica", "trabaja": "binaria", "mujer": "binaria", "indigena": "binaria", "asiste": "binaria",
    "rural": "binaria", "educ_baja": "binaria", "ia_mod_sev": "binaria", "ia_severa": "binaria",
    "ia_adultos_alguna": "binaria", "ia_adultos": "categórica", "ia_menores": "categórica",
    "ing_cor": "continua", "ingtrab": "continua", "transfer": "continua", "jubilacion": "continua", "becas": "continua",
    "bene_gob": "continua", "ing_cor_pc": "continua", "ing_lab_pc": "continua", "log_ing_pc": "continua",
    "log_ing_lab_pc": "continua", "tam_loc": "categórica", "est_socio": "categórica", "sexo_jefe": "binaria",
    "edad_jefe": "conteo", "educa_jefe": "categórica", "tot_integ": "conteo", "menores": "categórica",
    "p65mas": "categórica",
    # hogares (módulo 3)
    "reg_centro_occ": "binaria", "reg_centro": "binaria", "reg_sur": "binaria", "jefa_mujer": "binaria",
    "habla_indigena_hog": "binaria", "ocupados_pc": "continua", "afiliacion_laboral": "binaria", "auto": "binaria",
    "refri": "binaria", "lavadora": "binaria", "computadora": "binaria", "internet": "binaria", "celular": "binaria",
    "recibe_remesas": "binaria", "recibe_programas": "binaria", "recibe_jubilacion": "binaria", "liconsa": "binaria",
    "ing_pc_real": "continua", "log_ing_pc_real": "continua", "prop_ing_trabajo": "continua",
    "prop_transferencias": "continua", "log_gasto_pc_real": "continua", "log_gasto_alim_pc_real": "continua",
    "prop_alimentos": "continua", "prop_alim_fuera": "continua", "prestamos_recibidos": "binaria",
}
for p in ("pam", "pam_p104", "beca_basica", "beca_ems", "beca_otra_gob", "pension_contrib", "prospera"):
    REF[p] = REF[f"hog_{p}"] = "binaria"
    REF[f"{p}_monto_tri"] = "continua"

# --- procedencia: tabla de INEGI, columnas de origen y transformación
CLAVES = {"pam": ("P044, P045", "P104, P045"), "pam_p104": ("P044", "P104"), "beca_basica": ("—", "P101"),
          "beca_ems": ("—", "P102"), "beca_otra_gob": ("P038", "P038"), "pension_contrib": ("P032, P033", "P032, P033"),
          "prospera": ("P042", "—")}
DIRECTAS_POB = ["parentesco", "sexo", "edad", "hablaind", "asis_esc", "nivel", "grado", "tipoesc", "tiene_b",
                "nivelaprob", "gradoaprob", "disc_ver", "disc_camin", "numren", "foliohog"]
DIRECTAS_CH = ["tam_loc", "est_socio", "est_dis", "upm", "factor", "sexo_jefe", "edad_jefe", "educa_jefe",
               "tot_integ", "menores", "p65mas", "ing_cor", "ingtrab", "transfer", "jubilacion", "becas", "bene_gob"]
DERIVADAS = {
    "folioviv": ("todas", "folioviv", "relleno a 10 dígitos; llave de vivienda"),
    "anio": ("nombre del archivo", "—", "año del levantamiento"),
    "entidad": ("todas", "folioviv", "dos primeros dígitos de folioviv"),
    "rural": ("todas", "folioviv", "tercer dígito de folioviv igual a 6: ámbito rural de la ENIGH (localidades de menos de 2,500 habitantes); difiere de tam_loc = 4 en 0.3% de los hogares"),
    "ubica_geo": ("concentradohogar", "ubica_geo", "clave de entidad y municipio a cinco dígitos; en 2016 se recortan los nueve dígitos publicados (los cuatro últimos son ceros)"),
    "mujer": ("poblacion", "sexo", "sexo = 2"),
    "indigena": ("poblacion", "hablaind", "hablaind = 1"),
    "asiste": ("poblacion", "asis_esc", "asis_esc = 1; faltante si no aplica"),
    "educ_baja": ("poblacion", "nivelaprob", "nivelaprob ≤ 2 (hasta primaria)"),
    "n_trabajos": ("trabajos", "id_trabajo", "número de trabajos de la persona"),
    "trabaja": ("trabajos", "id_trabajo", "1 si tiene al menos un trabajo"),
    "ia_adultos": ("hogares", "acc_alim2, 4, 5, 6, 7, 8", "reactivos de adultos de la EMSA respondidos con sí"),
    "ia_menores": ("hogares", "acc_alim11 a 16", "reactivos de menores de la EMSA respondidos con sí"),
    "ia_mod_sev": ("hogares", "acc_alim2 a 16", "con menores, al menos 4 de 12; sin menores, al menos 3 de 6"),
    "ia_severa": ("hogares", "acc_alim2 a 16", "con menores, al menos 8 de 12; sin menores, al menos 5 de 6"),
    "ia_adultos_alguna": ("hogares", "acc_alim2, 4, 5, 6, 7, 8", "al menos un reactivo de adultos con sí"),
    "ing_cor_pc": ("concentradohogar", "ing_cor, tot_integ", "ingreso corriente trimestral entre integrantes"),
    "ing_lab_pc": ("concentradohogar", "ingtrab, tot_integ", "ingreso del trabajo entre integrantes"),
    "log_ing_pc": ("concentradohogar", "ing_cor, tot_integ", "log(1 + ingreso per cápita nominal)"),
    "log_ing_lab_pc": ("concentradohogar", "ingtrab, tot_integ", "log(1 + ingreso laboral per cápita nominal)"),
    "fuente": ("—", "—", "etiqueta de procedencia por fila"),
    "reg_centro_occ": ("todas", "folioviv", "entidad en la región centro occidente"),
    "reg_centro": ("todas", "folioviv", "entidad en la región centro"),
    "reg_sur": ("todas", "folioviv", "entidad en la región sur"),
    "jefa_mujer": ("concentradohogar", "sexo_jefe", "sexo_jefe = 2"),
    "habla_indigena_hog": ("poblacion", "hablaind", "algún integrante habla lengua indígena"),
    "afiliacion_laboral": ("poblacion", "inscr_1", "algún integrante con inscripción por prestación laboral"),
    "ocupados_pc": ("concentradohogar", "ocupados, tot_integ", "ocupados entre integrantes"),
    "auto": ("hogares", "num_auto, num_van, num_pickup, num_pick", "algún vehículo"),
    "refri": ("hogares", "num_refri", "al menos un refrigerador"),
    "lavadora": ("hogares", "num_lavad", "al menos una lavadora"),
    "computadora": ("hogares", "num_compu, num_lap", "al menos una computadora o laptop"),
    "internet": ("hogares", "conex_inte", "conex_inte = 1"),
    "celular": ("hogares", "celular", "celular = 1"),
    "recibe_remesas": ("concentradohogar", "remesas", "remesas > 0"),
    "recibe_programas": ("concentradohogar", "bene_gob", "bene_gob > 0"),
    "recibe_jubilacion": ("concentradohogar", "jubilacion", "jubilacion > 0"),
    "liconsa": ("hogares", "bene_licon", "bene_licon = 1"),
    "ing_pc_real": ("concentradohogar + INPC", "ing_cor, tot_integ", "per cápita en pesos de agosto de 2024"),
    "log_ing_pc_real": ("concentradohogar + INPC", "ing_cor, tot_integ", "log(1 + ingreso per cápita real)"),
    "prop_ing_trabajo": ("concentradohogar", "ingtrab, ing_cor", "proporción del ingreso que viene del trabajo"),
    "prop_transferencias": ("concentradohogar", "transfer, ing_cor", "proporción del ingreso por transferencias"),
    "log_gasto_pc_real": ("concentradohogar + INPC", "gasto_mon, tot_integ", "log(1 + gasto monetario per cápita real)"),
    "log_gasto_alim_pc_real": ("concentradohogar + INPC", "alimentos, tot_integ", "log(1 + gasto en alimentos per cápita real)"),
    "prop_alimentos": ("concentradohogar", "alimentos, gasto_mon", "proporción del gasto en alimentos"),
    "prop_alim_fuera": ("concentradohogar", "ali_fuera, alimentos", "proporción del gasto en alimentos fuera del hogar"),
    "prestamos_recibidos": ("concentradohogar", "prestamos", "prestamos > 0"),
}


def procedencia(var: str, anio: int) -> tuple[str, str, str]:
    for p, (antes, despues) in CLAVES.items():
        cl = antes if anio <= 2018 else despues
        if var == p:
            return "ingresos", f"clave ∈ {{{cl}}}", "1 si la persona recibe ingresos con esas claves"
        if var == f"{p}_monto_tri":
            return "ingresos", f"ing_tri con clave ∈ {{{cl}}}", "suma trimestral de la persona"
        if var == f"hog_{p}":
            return "ingresos", f"clave ∈ {{{cl}}}", "1 si algún integrante del hogar la recibe"
    if var in DERIVADAS:
        return DERIVADAS[var]
    if var in DIRECTAS_POB:
        return "poblacion", var, "copia con coerción numérica"
    if var in DIRECTAS_CH:
        return "concentradohogar", var, "copia con coerción numérica"
    return "?", "?", "?"


def diccionario(df: pd.DataFrame, base: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    filas, matriz = [], []
    for c in df.columns:
        s = df[c]
        tipo, regla = inferir_tipo(c, s)
        x = s.dropna()
        num = pd.api.types.is_numeric_dtype(x) and not x.empty
        disp = df.groupby("anio")[c].apply(lambda v: float(v.notna().mean()))
        tabla, cols, transf = procedencia(c, 2024)
        ref = REF.get(c)
        filas.append({"base": base, "variable": c, "tipo_inferido": tipo, "regla": regla, "tipo_referencia": ref or "",
                      "coincide": "" if ref is None else ("sí" if ref == tipo else "no"),
                      "faltantes_pct": round(100 * float(s.isna().mean()), 2), "valores_distintos": int(x.nunique()),
                      "minimo": float(x.min()) if num else "", "maximo": float(x.max()) if num else "",
                      "media": round(float(x.mean()), 4) if num else "",
                      "anios_con_datos": ", ".join(str(a) for a, v in disp.items() if v > 0),
                      "tabla_origen": tabla, "columnas_origen": cols, "transformacion": transf})
        for a in YEARS:
            t, co, tr = procedencia(c, a)
            matriz.append({"base": base, "variable": c, "anio": a, "disponible_pct": round(100 * disp.get(a, 0.0), 2),
                           "archivo": f"{a}_{t.split(' ')[0]}/{t.split(' ')[0]}.csv" if t not in ("—", "?", "todas", "nombre del archivo") else t,
                           "columnas_origen": co, "transformacion": tr})
    return pd.DataFrame(filas), pd.DataFrame(matriz)


def main():
    per = pd.read_parquet(PERSONAS)
    hog = pd.read_parquet(HOGARES)
    d1, m1 = diccionario(per, "personas")
    d2, m2 = diccionario(hog, "hogares")
    dic = pd.concat([d1, d2], ignore_index=True)
    mat = pd.concat([m1, m2], ignore_index=True)
    dic.to_csv(f"{BASE}/datos/diccionario_datos.csv", index=False)
    mat.to_csv(f"{BASE}/datos/matriz_procedencia.csv", index=False)

    con_ref = dic[dic["coincide"] != ""]
    errores = con_ref[con_ref["coincide"] == "no"][["base", "variable", "tipo_inferido", "tipo_referencia", "regla"]]
    sin_proc = dic[dic["tabla_origen"] == "?"]["variable"].tolist()

    # conciliación con la estadística oficial (CONEVAL, carencia por acceso a la alimentación, personas, nacional)
    oficial = {2016: 20.1, 2018: 20.4}
    propio = {int(a): round(100 * float(np.average(g["ia_mod_sev"], weights=g["factor"])), 2)
              for a, g in per.groupby("anio")}
    conc = {str(a): {"framework_pct": propio[a], "oficial_pct": v, "diferencia_pp": round(propio[a] - v, 2)}
            for a, v in oficial.items()}

    out = {
        "variables": {"personas": int(per.shape[1]), "hogares": int(hog.shape[1])},
        "tipos_inferidos": dic["tipo_inferido"].value_counts().to_dict(),
        "deteccion_frente_a_referencia": {"evaluadas": int(len(con_ref)),
                                          "coinciden": int((con_ref["coincide"] == "sí").sum()),
                                          "exactitud": round(float((con_ref["coincide"] == "sí").mean()), 4),
                                          "no_coinciden": errores.to_dict(orient="records")},
        "variables_sin_procedencia": sin_proc,
        "fuentes_vinculadas": [
            {"fuente": "INEGI, ENIGH 2016-2024 (nueva serie): concentradohogar, hogares, poblacion, ingresos, trabajos",
             "llave": "folioviv, foliohog, numren", "uso": "base integrada"},
            {"fuente": "INEGI, INPC de agosto (verificado con notas del CEFP)", "llave": "anio (temporal)",
             "uso": "deflactor a pesos de agosto de 2024"},
            {"fuente": "Reglas de operación y montos de los programas (DOF, CEFP, prensa)",
             "llave": "clave de ingreso y año (programática)", "uso": "claves por año, elegibilidad y montos de los escenarios"},
            {"fuente": "Secretaría de Bienestar, Declaratoria de las Zonas de Atención Prioritaria para 2020 (DOF, 11 dic. 2019), "
                       "anexo A: categoría indígena municipal (A y B, municipios indígenas)",
             "llave": "ubica_geo homologada a cinco dígitos (territorial)",
             "uso": "regla de elegibilidad de 2020 en la falsación y en la validación ex ante–ex post"},
            {"fuente": "CDI (hoy INPI), Catálogo de localidades indígenas 2010: municipios indígenas (40% o más de población indígena)",
             "llave": "ubica_geo homologada a cinco dígitos (territorial)",
             "uso": "sensibilidad de la regla de elegibilidad de 2020"},
            {"fuente": "CONEVAL, medición de la pobreza 2016 y 2018", "llave": "anio y país",
             "uso": "conciliación de la inseguridad alimentaria"}],
        "conciliacion_coneval": conc,
        "fuente_conciliacion": "https://www.coneval.org.mx/Medicion/MP/Documents/Pobreza_18/Pobreza_2018_CONEVAL.pdf",
    }
    with open(f"{BASE}/resultados/oe1_ingesta.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "fuentes_vinculadas"}, ensure_ascii=False, indent=1, default=str))
    print(dic.groupby(["base", "tipo_inferido"]).size().to_string())


if __name__ == "__main__":
    main()
