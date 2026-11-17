"""Catálogo de localidades indígenas 2010 (CDI, hoy INPI): clasificación de los municipios.

Uso: python3 ingesta/inpi.py fuentes/catalogo_de_localidades_indigenas_2010.csv datos/inpi_municipios_2010.csv

Toma los totales municipales del catálogo y guarda, por clave de municipio de cinco dígitos, el tipo de
municipio y la población total e indígena de 2010. Las reglas de operación de la pensión de 2020 (numeral 3.2)
hacen elegibles a los 65 años a quienes viven en municipios o localidades A y B del INPI, es decir, en los
«municipios indígenas» del catálogo (40% o más de población indígena). La ENIGH pública no identifica la
localidad, así que el análisis usa la clasificación municipal.

Los municipios creados después de 2010 que aparecen en la ENIGH 2020 toman la clasificación del municipio del
que se separaron: Bacalar (23010) de Othón P. Blanco (23004) y Puerto Morelos (23011) de Benito Juárez (23005).
"""
from __future__ import annotations

import sys

import pandas as pd

NUEVOS = {"23010": ("Bacalar", "23004"), "23011": ("Puerto Morelos", "23005")}


def numero(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce").astype("Int64")


def main(entrada: str, salida: str) -> None:
    cat = pd.read_csv(entrada, encoding="latin1", dtype=str)
    m = cat[(cat["MUN"] != "000") & (cat["NOM_LOC"] == "Total Municipal")].copy()
    t = pd.DataFrame({
        "cve_mun": m["ENT"] + m["MUN"],
        "entidad": m["NOM_ENT"],
        "municipio": m["NOM_MUN"],
        "tipo_inpi": m["NOMTIPO"],
        "pob_total_2010": numero(m["POBTOT"]),
        "pob_indigena_2010": numero(m["POB_INDI"]),
        "origen": "catálogo 2010",
    })
    t["indigena_ab"] = (t["tipo_inpi"] == "Municipio indígena").astype(int)
    assert t["cve_mun"].is_unique
    nuevos = []
    for cve, (nombre, madre) in NUEVOS.items():
        f = t.loc[t["cve_mun"] == madre].iloc[0].copy()
        f["cve_mun"], f["municipio"] = cve, nombre
        f["pob_total_2010"] = f["pob_indigena_2010"] = pd.NA
        f["origen"] = f"creado después de 2010; clasificación de {madre}"
        nuevos.append(f)
    t = pd.concat([t, pd.DataFrame(nuevos)], ignore_index=True).sort_values("cve_mun").reset_index(drop=True)
    t.to_csv(salida, index=False)
    print(f"{len(t)} municipios; indígenas (A y B): {int(t['indigena_ab'].sum())}")
    print(t["tipo_inpi"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
