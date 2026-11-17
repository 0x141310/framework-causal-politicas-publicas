"""Categoría indígena de los municipios en 2020 (anexo A de la Declaratoria de Zonas de Atención Prioritaria 2020).

Uso: python3 ingesta/zap2020.py fuentes/anexo_a_zap_2020.pdf datos/categoria_indigena_2020.csv

Las reglas de operación de la pensión de 2020 (numeral 3.2) hacen elegibles a los 65 años a quienes viven en
municipios o localidades indígenas de tipo A y B. El anexo A de la Declaratoria de las Zonas de Atención
Prioritaria para 2020 (Diario Oficial de la Federación, 11 de diciembre de 2019) publica, para los 1,480
municipios de las zonas rurales, la categoría indígena vigente: A y B, «municipios indígenas»; C y D, «presencia
indígena»; E, «presencia indígena dispersa»; F, sin población indígena. Este script lee el PDF del anexo y guarda,
por clave de municipio de cinco dígitos, la entidad, el municipio, el grado de marginación de 2015, la categoría
y su descripción. Seis municipios aparecen «sin dato» (SD), entre ellos Puerto Morelos, creado en 2016, sin grado de
marginación de 2015. Un municipio que no aparece en el anexo, o que aparece sin dato, no se cuenta como de categoría A o B.
"""
from __future__ import annotations

import re
import sys

import pandas as pd
from pypdf import PdfReader

GRADOS = r"MUY ALTO|MUY BAJO|ALTO|MEDIO|BAJO|SIN DATO"
DESCRIPCIONES = r"MUNICIPIOS INDÍGENAS|PRESENCIA INDÍGENA DISPERSA|PRESENCIA INDÍGENA|SIN POBLACIÓN INDÍGENA|SIN DATO"
FILA = re.compile(
    rf"(?<!\d)(?P<ent>\d{{2}}) (?P<entidad>[^\d]+?) (?P<cve_mun>\d{{5}}) (?P<municipio>[^\d]+?) "
    rf"(?:(?P<marginacion>{GRADOS}) )?(?P<categoria>[A-F]|SD) (?P<descripcion>{DESCRIPCIONES}) (?P<nivel>{GRADOS})(?= |$)")
ENCABEZADO = re.compile(r"Página \d+ de \d+|LISTADO DE ZONAS DE ATENCIÓN PRIORITARIA RURALES 2020|1,480 MUNICIPIOS|ANEXO A|"
                        r"CLAVE DE ENTIDAD FEDERATIVA ENTIDAD FEDERATIVA CLAVE DE MUNICIPIO O ALCALDÍA MUNICIPIO O ALCALDÍA "
                        r"GRADO DE MARGINACIÓN 2015 CATEGORÍA INDÍGENA DESCRIPCIÓN DE CATEGORÍA INDÍGENA NIVEL DELICTIVO")


def leer(pdf: str) -> pd.DataFrame:
    texto = " ".join(p.extract_text() for p in PdfReader(pdf).pages)
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+", " ", ENCABEZADO.sub(" ", texto)).strip()
    filas = [m.groupdict() for m in FILA.finditer(texto)]
    t = pd.DataFrame(filas)
    for c in ("entidad", "municipio"):
        t[c] = t[c].str.strip()
    return t


def main(pdf: str, salida: str) -> None:
    t = leer(pdf)
    assert len(t) == 1480, f"se esperaban 1,480 municipios y se leyeron {len(t)}"
    assert t["cve_mun"].is_unique
    assert (t["cve_mun"].str[:2] == t["ent"]).all()
    t["indigena_ab"] = t["categoria"].isin(["A", "B"]).astype(int)
    t = t[["cve_mun", "entidad", "municipio", "marginacion", "categoria", "descripcion", "indigena_ab"]]
    t.sort_values("cve_mun").to_csv(salida, index=False)
    print(f"{len(t)} municipios; categoría A: {int((t.categoria == 'A').sum())}; B: {int((t.categoria == 'B').sum())}; "
          f"A y B: {int(t.indigena_ab.sum())}")
    print(t["categoria"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
