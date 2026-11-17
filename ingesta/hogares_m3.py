"""Base de hogares para el módulo 3.

Une concentradohogar, hogares y agregados de población de la ENIGH 2016-2024 con el
resultado de inseguridad alimentaria ya construido en `ingesta/enigh.py`.
Montos per cápita en pesos de agosto de 2024 (INPC de agosto, base 2Q jul 2018 = 100).

Uso: python3 -I ingesta/hogares_m3.py <dir_csv> <personas.parquet> <salida.parquet>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

KEYS_H = ["folioviv", "foliohog"]
YEARS = [2016, 2018, 2020, 2022, 2024]
INPC_AGO = {2016: 89.809333, 2018: 100.492, 2020: 107.867, 2022: 123.803, 2024: 136.013}
REGION = {  # referencia: norte
    "centro_occ": {"01", "06", "11", "14", "16", "18", "22", "24", "32"},
    "centro": {"09", "13", "15", "17", "21", "29"},
    "sur": {"04", "07", "12", "20", "23", "27", "30", "31"},
}
CH_COLS = ["ubica_geo", "tam_loc", "est_socio", "est_dis", "upm", "factor", "sexo_jefe", "edad_jefe", "educa_jefe",
           "tot_integ", "menores", "p65mas", "ocupados", "ing_cor", "ingtrab", "transfer", "jubilacion", "bene_gob",
           "remesas", "gasto_mon", "alimentos", "ali_fuera", "prestamos"]
HO_COLS = ["celular", "conex_inte", "num_auto", "num_van", "num_pickup", "num_pick", "num_refri", "num_lavad",
           "num_compu", "num_lap", "bene_licon"]

PREVIAS = ["tam_loc", "est_socio", "reg_centro_occ", "reg_centro", "reg_sur", "edad_jefe", "jefa_mujer", "educa_jefe",
           "tot_integ", "menores", "p65mas", "habla_indigena_hog"]
RECURSOS = ["ocupados_pc", "afiliacion_laboral", "auto", "refri", "lavadora", "computadora", "internet", "celular",
            "recibe_remesas", "recibe_programas", "recibe_jubilacion", "liconsa"]
INGRESO = ["log_ing_pc_real", "prop_ing_trabajo", "prop_transferencias"]
CONCURRENTES = ["log_gasto_pc_real", "log_gasto_alim_pc_real", "prop_alimentos", "prop_alim_fuera", "prestamos_recibidos"]
T_SET = PREVIAS + RECURSOS + INGRESO
G_SET = T_SET + CONCURRENTES


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")


def homologar_municipio(s: pd.Series) -> pd.Series:
    """Clave de entidad y municipio a cinco dígitos. En 2016 la ENIGH publica ubica_geo con nueve dígitos
    (los cuatro últimos son ceros); desde 2018, con cinco."""
    s = s.astype(str).str.strip()
    return s.where(s.str.len() <= 5, s.str.zfill(9).str[:5]).str.zfill(5)


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.where(b > 0, a / b.where(b > 0), 0.0), index=a.index).clip(0, 1)


def load_year(D: str, y: int) -> pd.DataFrame:
    def rd(t, cols):
        hdr = [c.lstrip("﻿") for c in pd.read_csv(f"{D}/{y}_{t}/{t}.csv", nrows=0).columns]
        use = KEYS_H + [c for c in cols if c in hdr]
        d = pd.read_csv(f"{D}/{y}_{t}/{t}.csv", dtype={"folioviv": str, "ubica_geo": str}, usecols=use, low_memory=False)
        d["folioviv"] = d["folioviv"].str.zfill(10)
        return d

    ch = rd("concentradohogar", CH_COLS)
    ch["ubica_geo"] = homologar_municipio(ch["ubica_geo"])
    ho = rd("hogares", HO_COLS)
    po = rd("poblacion", ["hablaind", "inscr_1"])
    for d, cols in ((ch, CH_COLS), (ho, HO_COLS)):
        for c in cols:
            if c in d.columns and c != "ubica_geo":
                d[c] = _num(d[c])
    po["hablaind"] = _num(po["hablaind"])
    po["inscr_1"] = _num(po["inscr_1"])
    pa = po.groupby(KEYS_H).agg(habla_indigena_hog=("hablaind", lambda s: int((s == 1).any())),
                                afiliacion_laboral=("inscr_1", lambda s: int((s == 1).any()))).reset_index()
    n0 = len(ch)
    h = ch.merge(ho, on=KEYS_H, how="left", validate="one_to_one").merge(pa, on=KEYS_H, how="left", validate="one_to_one")
    assert len(h) == n0
    h["anio"] = y
    h["entidad"] = h["folioviv"].str[:2]
    h["rural"] = (h["folioviv"].str[2] == "6").astype(int)
    for k, ents in REGION.items():
        h[f"reg_{k}"] = h["entidad"].isin(ents).astype(int)
    h["jefa_mujer"] = (h["sexo_jefe"] == 2).astype(int)
    z = lambda c: h[c].fillna(0) if c in h.columns else 0
    h["ocupados_pc"] = h["ocupados"] / h["tot_integ"]
    h["auto"] = ((z("num_auto") + z("num_van") + z("num_pickup") + z("num_pick")) > 0).astype(int)
    h["refri"] = (z("num_refri") > 0).astype(int)
    h["lavadora"] = (z("num_lavad") > 0).astype(int)
    h["computadora"] = ((z("num_compu") + z("num_lap")) > 0).astype(int)
    h["internet"] = (h["conex_inte"] == 1).astype(int)
    h["celular"] = (h["celular"] == 1).astype(int)
    h["recibe_remesas"] = (h["remesas"] > 0).astype(int)
    h["recibe_programas"] = (h["bene_gob"] > 0).astype(int)
    h["recibe_jubilacion"] = (h["jubilacion"] > 0).astype(int)
    h["liconsa"] = (h["bene_licon"] == 1).astype(int)
    defl = INPC_AGO[2024] / INPC_AGO[y]
    pc = lambda c: h[c].clip(lower=0) * defl / h["tot_integ"]
    h["ing_pc_real"] = pc("ing_cor")
    h["log_ing_pc_real"] = np.log1p(h["ing_pc_real"])
    h["prop_ing_trabajo"] = _ratio(h["ingtrab"].clip(lower=0), h["ing_cor"])
    h["prop_transferencias"] = _ratio(h["transfer"].clip(lower=0), h["ing_cor"])
    h["log_gasto_pc_real"] = np.log1p(pc("gasto_mon"))
    h["log_gasto_alim_pc_real"] = np.log1p(pc("alimentos"))
    h["prop_alimentos"] = _ratio(h["alimentos"].clip(lower=0), h["gasto_mon"])
    h["prop_alim_fuera"] = _ratio(h["ali_fuera"].clip(lower=0), h["alimentos"])
    h["prestamos_recibidos"] = (h["prestamos"] > 0).astype(int)
    return h


def main(D: str, personas: str, out: str) -> None:
    res = pd.read_parquet(personas, columns=["anio", "folioviv", "foliohog", "ia_mod_sev", "ia_severa", "ia_adultos"])
    res = res.groupby(["anio"] + KEYS_H, as_index=False).first()
    hs, calidad = [], {}
    for y in YEARS:
        h = load_year(D, y)
        n0 = len(h)
        h = h.merge(res[res.anio == y].drop(columns="anio"), on=KEYS_H, how="left", validate="one_to_one")
        calidad[str(y)] = {"hogares": n0, "sin_resultado": int(h["ia_mod_sev"].isna().sum()),
                           "faltantes": {c: int(h[c].isna().sum()) for c in G_SET if h[c].isna().any()}}
        hs.append(h)
    H = pd.concat(hs, ignore_index=True)
    keep = ["anio", "folioviv", "foliohog", "entidad", "ubica_geo", "rural", "est_dis", "upm", "factor",
            "ia_mod_sev", "ia_severa", "ia_adultos", "ing_pc_real", "sexo_jefe"] + G_SET
    H = H[keep]
    H.to_parquet(out, index=False)
    with open(out.replace(".parquet", "_calidad.json"), "w") as fh:
        json.dump(calidad, fh, ensure_ascii=False, indent=1)
    print(json.dumps(calidad, ensure_ascii=False))
    desc = H.groupby("anio")[G_SET].mean().T.round(3)
    print(desc.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
