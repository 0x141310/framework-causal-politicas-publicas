"""Módulo 1 (mínimo): ingesta y armonización de la ENIGH 2016-2024 (nueva serie).

Vincula las tablas por llaves de hogar y persona (folioviv, foliohog, numren),
identifica programas sociales por clave de ingreso, construye indicadores de
inseguridad alimentaria y conserva procedencia y un reporte de calidad.

Claves identificadas empíricamente (perfil de edad y montos de pago; confirmar
con el catálogo de ingresos del descriptor de archivos de INEGI, §2.4.8):
  P104  Pensión para el Bienestar de las Personas Adultas Mayores (99% de 65+;
        pagos de 3,850 en 2022 y 6,000 en 2024)
  P045  otro registro de pensión para adultos mayores (94% de 65+; mismos montos)
  P101  Beca Benito Juárez de educación básica (familiar; titular adulto)
  P102  Beca Universal Benito Juárez de educación media superior (15-19 años)
  P038  otras becas del gobierno (mediana de edad 10 años)
Confirmadas con el catálogo de ingresos del descriptor 2024 (pág. 271): P104 pensión de adultos
mayores; P045 otros programas para adultos mayores; P032/P033 jubilaciones y pensiones contributivas.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

KEYS_H = ["folioviv", "foliohog"]
KEYS_P = ["folioviv", "foliohog", "numren"]
PROGRAMAS = {"pam": ["P104", "P045"], "pam_p104": ["P104"], "beca_basica": ["P101"], "beca_ems": ["P102"],
             "beca_otra_gob": ["P038"], "pension_contrib": ["P032", "P033"], "prospera": []}
# 2016 y 2018: el programa "65 y más" es P044 (pagos de 1,160 bimestrales; 99% de 65+) y PROSPERA es P042;
# las becas Benito Juárez (P101, P102) aún no existen. "pam_p104" = clave principal sin P045 en todos los años.
PROGRAMAS_2016_2018 = {"pam": ["P044", "P045"], "pam_p104": ["P044"], "beca_basica": [], "beca_ems": [],
                       "beca_otra_gob": ["P038"], "pension_contrib": ["P032", "P033"], "prospera": ["P042"]}


def programas(y: int) -> dict:
    return PROGRAMAS_2016_2018 if y in (2016, 2018) else PROGRAMAS
# Escala Mexicana de Seguridad Alimentaria (CONEVAL): seis reactivos de adultos y seis de menores
ADULTOS = ["acc_alim2", "acc_alim4", "acc_alim5", "acc_alim6", "acc_alim7", "acc_alim8"]
MENORES = ["acc_alim11", "acc_alim12", "acc_alim13", "acc_alim14", "acc_alim15", "acc_alim16"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")


def homologar_municipio(s: pd.Series) -> pd.Series:
    """Clave de entidad y municipio a cinco dígitos. En 2016 la ENIGH publica ubica_geo con nueve dígitos
    (los cuatro últimos son ceros); desde 2018, con cinco."""
    s = s.astype(str).str.strip()
    return s.where(s.str.len() <= 5, s.str.zfill(9).str[:5]).str.zfill(5)


def load_year(D: str, y: int, quality: dict) -> pd.DataFrame:
    rd = lambda t, **kw: pd.read_csv(f"{D}/{y}_{t}/{t}.csv", dtype={"folioviv": str, "ubica_geo": str, "clave": str},
                                     low_memory=False, **kw)
    ch = rd("concentradohogar", usecols=KEYS_H + ["ubica_geo", "tam_loc", "est_socio", "est_dis", "upm", "factor",
                                                  "sexo_jefe", "edad_jefe", "educa_jefe", "tot_integ", "menores",
                                                  "p65mas", "ing_cor", "ingtrab", "transfer", "jubilacion", "becas",
                                                  "bene_gob"])
    ch["ubica_geo"] = homologar_municipio(ch["ubica_geo"])
    ho = rd("hogares", usecols=KEYS_H + ADULTOS + MENORES + ["acc_alim1", "acc_alim3"])
    pcols = KEYS_P + ["parentesco", "sexo", "edad", "hablaind", "asis_esc", "nivel", "grado", "tipoesc", "tiene_b",
                      "nivelaprob", "gradoaprob", "disc_ver", "disc_camin"]
    hdr = set(pd.read_csv(f"{D}/{y}_poblacion/poblacion.csv", nrows=1).columns)
    po = rd("poblacion", usecols=[c for c in pcols if c in hdr])
    for c in pcols:
        if c not in po.columns:
            po[c] = np.nan  # variable no disponible en ese año
    ing = rd("ingresos", usecols=KEYS_P + ["clave", "ing_tri"])
    tr = rd("trabajos", usecols=KEYS_P + ["id_trabajo"])
    for d in (ch, ho, po, ing, tr):
        d["folioviv"] = d["folioviv"].str.zfill(10)
    q = quality.setdefault(str(y), {})
    q["filas"] = {"concentradohogar": len(ch), "hogares": len(ho), "poblacion": len(po), "ingresos": len(ing), "trabajos": len(tr)}
    q["duplicados_llave"] = {"concentradohogar": int(ch.duplicated(KEYS_H).sum()), "hogares": int(ho.duplicated(KEYS_H).sum()),
                             "poblacion": int(po.duplicated(KEYS_P).sum())}

    # --- hogar: inseguridad alimentaria (1 = sí, 2 = no; vacío = no aplica)
    for c in ADULTOS + MENORES + ["acc_alim1", "acc_alim3"]:
        ho[c] = _num(ho[c])
    ho["ia_adultos"] = (ho[ADULTOS] == 1).sum(axis=1)
    ho["ia_menores"] = (ho[MENORES] == 1).sum(axis=1)
    ho["con_menores_escala"] = ho[MENORES].notna().any(axis=1)
    tot = ho["ia_adultos"] + ho["ia_menores"]
    # umbrales CONEVAL: con menores 1-3 leve, 4-7 moderada, 8-12 severa; sin menores 1-2, 3-4, 5-6
    ho["ia_mod_sev"] = np.where(ho["con_menores_escala"], tot >= 4, ho["ia_adultos"] >= 3).astype(int)
    ho["ia_severa"] = np.where(ho["con_menores_escala"], tot >= 8, ho["ia_adultos"] >= 5).astype(int)
    ho["ia_adultos_alguna"] = (ho["ia_adultos"] >= 1).astype(int)
    q["faltantes_escala_adultos"] = int(ho[ADULTOS].isna().any(axis=1).sum())

    # --- programas por persona y por hogar
    ing["ing_tri"] = _num(ing["ing_tri"])
    for prog, claves in programas(y).items():
        sel = ing[ing["clave"].isin(claves)]
        per = sel.groupby(KEYS_P)["ing_tri"].sum().rename(f"{prog}_monto_tri").reset_index()
        po = po.merge(per, on=KEYS_P, how="left")
        po[prog] = po[f"{prog}_monto_tri"].notna().astype(int)
        hog = sel.groupby(KEYS_H).size().rename(f"hog_{prog}").reset_index()
        po = po.merge(hog, on=KEYS_H, how="left")
        po[f"hog_{prog}"] = po[f"hog_{prog}"].notna().astype(int)
    trab = tr.groupby(KEYS_P).size().rename("n_trabajos").reset_index()
    po = po.merge(trab, on=KEYS_P, how="left")
    po["trabaja"] = po["n_trabajos"].notna().astype(int)

    n0 = len(po)
    df = po.merge(ch, on=KEYS_H, how="left", validate="many_to_one").merge(
        ho[KEYS_H + ["ia_adultos", "ia_menores", "ia_mod_sev", "ia_severa", "ia_adultos_alguna"]], on=KEYS_H, how="left",
        validate="many_to_one")
    q["personas_sin_hogar_vinculado"] = int(df["factor"].isna().sum())
    q["perdidas_vinculacion"] = n0 - len(df)
    for c in ["sexo", "edad", "hablaind", "asis_esc", "nivelaprob", "gradoaprob", "tiene_b", "parentesco", "disc_ver",
              "disc_camin", "tipoesc", "nivel", "grado"]:
        df[c] = _num(df[c])
    df["anio"] = y
    df["entidad"] = df["folioviv"].str[:2]
    df["rural"] = (df["folioviv"].str[2] == "6").astype(int)
    df["mujer"] = (df["sexo"] == 2).astype(int)
    df["indigena"] = (df["hablaind"] == 1).astype(int)
    df["asiste"] = np.where(df["asis_esc"].isna(), np.nan, (df["asis_esc"] == 1).astype(float))
    df["ing_cor_pc"] = df["ing_cor"] / df["tot_integ"]
    df["ing_lab_pc"] = df["ingtrab"] / df["tot_integ"]
    df["log_ing_pc"] = np.log1p(df["ing_cor_pc"].clip(lower=0))
    df["log_ing_lab_pc"] = np.log1p(df["ing_lab_pc"].clip(lower=0))
    df["educ_baja"] = (df["nivelaprob"].fillna(0) <= 2).astype(int)  # hasta primaria
    df["fuente"] = f"INEGI ENIGH {y} NS: poblacion+concentradohogar+hogares+ingresos+trabajos"
    return df


def build(D: str, out: str, years=(2022, 2024)) -> dict:
    quality: dict = {}
    full = pd.concat([load_year(D, y, quality) for y in years], ignore_index=True)
    tag = f"{min(years)}_{max(years)}"
    full.to_parquet(f"{out}/enigh_personas_{tag}.parquet", index=False)
    qroo = full[full["entidad"] == "23"].reset_index(drop=True)
    qroo.to_parquet(f"{out}/enigh_qroo_personas_{tag}.parquet", index=False)
    # reporte de calidad para Quintana Roo
    cols = ["edad", "sexo", "hablaind", "asis_esc", "educa_jefe", "tot_integ", "ing_cor", "ia_mod_sev", "factor"]
    quality["qroo"] = {
        "personas": int(len(qroo)), "hogares": int(qroo.drop_duplicates(["anio"] + KEYS_H).shape[0]),
        "por_anio": qroo.groupby("anio").size().to_dict(),
        "completitud": {c: float(qroo[c].notna().mean()) for c in cols},
        "programas_personas": {p: int(qroo[p].sum()) for p in PROGRAMAS},
        "municipios": int(qroo["ubica_geo"].nunique()),
    }
    with open(f"{out}/calidad_ingesta_{tag}.json", "w") as fh:
        json.dump(quality, fh, ensure_ascii=False, indent=2, default=str)
    return quality


if __name__ == "__main__":
    yrs = tuple(int(a) for a in sys.argv[3].split(",")) if len(sys.argv) > 3 else (2022, 2024)
    q = build(sys.argv[1], sys.argv[2], yrs)
    print(json.dumps(q, ensure_ascii=False, indent=1, default=str))
