"""Efecto de la pensión por tercil de riesgo predicho: ¿la pensión rinde más donde el riesgo es mayor?

Modelo de riesgo con variables predeterminadas de la persona y su localidad (sexo,
escolaridad, habla indígena, tamaño de localidad, estrato y región), entrenado con
personas de 50-80 años en 2016-2020 (resultado: inseguridad alimentaria moderada o
severa del hogar), aplicado a personas de 58-72 años en 2022-2024. Regresión discontinua
a los 65 (ancho ±7, núcleo triangular, 999 réplicas por hogar compartidas)
por tercil de riesgo, para el total y para mujeres. Validación: el riesgo no salta a los 65.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, ROOT)
from modulo_causal import rdd  # noqa: E402

DATA = ROOT + "/datos/enigh_personas_2016_2024.parquet"
OUT = ROOT + "/resultados"
YS = ["ia_mod_sev", "ia_adultos", "log_ing_pc"]
REG = {"centro_occ": {"01", "06", "11", "14", "16", "18", "22", "24", "32"}, "centro": {"09", "13", "15", "17", "21", "29"},
       "sur": {"04", "07", "12", "20", "23", "27", "30", "31"}}
FEATS = ["mujer", "nivelaprob", "habla_ind", "tam_loc", "est_socio", "reg_centro_occ", "reg_centro", "reg_sur"]


def ci(a):
    return [round(float(x), 4) for x in np.nanpercentile(a, [2.5, 97.5])]


def wq(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(cw, q)]) for q in qs]


def main():
    cols = ["anio", "edad", "mujer", "nivelaprob", "hablaind", "tam_loc", "est_socio", "entidad", "factor", "folioviv",
            "foliohog", "pam"] + YS
    df = pd.read_parquet(DATA, columns=cols)
    for c in ("tam_loc", "est_socio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["habla_ind"] = (df["hablaind"] == 1).astype(int)
    for k, e in REG.items():
        df[f"reg_{k}"] = df["entidad"].isin(e).astype(int)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    tr = df[df.anio.isin([2016, 2018, 2020]) & df.edad.between(50, 80)].dropna(subset=["ia_mod_sev"])
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=400, min_samples_leaf=100,
                                       l2_regularization=1.0, early_stopping=False, random_state=7)
    m.fit(tr[FEATS].to_numpy(float), tr["ia_mod_sev"].to_numpy(int))
    d = df[df.anio.isin([2022, 2024]) & df.edad.between(58, 72)].copy()
    d["riesgo"] = m.predict_proba(d[FEATS].to_numpy(float))[:, 1]
    c1, c2 = wq(d["riesgo"].to_numpy(), d["factor"].to_numpy(float), [1 / 3, 2 / 3])
    d["tercil"] = np.where(d["riesgo"] < c1, 1, np.where(d["riesgo"] < c2, 2, 3))
    R = {"n_entrenamiento": int(len(tr)), "cortes_terciles": [c1, c2],
         "prevalencia_por_tercil": {}, "validacion_riesgo_en_65": None, "rdd": {}}
    for t in (1, 2, 3):
        s = d[d.tercil == t]
        R["prevalencia_por_tercil"][t] = {"riesgo_medio": round(float(np.average(s.riesgo, weights=s.factor)), 4),
                                          "ia_observada": round(float(np.average(s.ia_mod_sev, weights=s.factor)), 4),
                                          "recibe_pension_66_72": round(float(np.average(s[s.edad >= 66].pam, weights=s[s.edad >= 66].factor)), 4)}
    v = rdd.estimate(d, "edad", 65, "riesgo", None, 7, "hogar_id", B=499, seed=65)
    R["validacion_riesgo_en_65"] = {"salto": round(v["forma_reducida"], 5), "ic95": [round(x, 5) for x in v["fr_ic95"]]}
    groups = {f"t{t}": d.tercil == t for t in (1, 2, 3)}
    groups.update({f"t{t}_mujeres": (d.tercil == t) & (d.mujer == 1) for t in (1, 2, 3)})
    groups["total"] = pd.Series(True, index=d.index)
    res = rdd.estimate_groups(d, "edad", 65, YS, "pam", 7, "hogar_id", groups, B=999, seed=1965)
    for g, r in res.items():
        R["rdd"][g] = {"n": r["n"], "primera_etapa": round(r["primera_etapa"], 4), "pe_ic95": [round(x, 4) for x in r["pe_ic95"]]}
        for y in YS:
            R["rdd"][g][y] = {"efecto_local": round(r[y]["efecto_local"], 4), "el_ic95": ci(r[y]["draws"]),
                              "forma_reducida": round(r[y]["forma_reducida"], 4), "fr_ic95": ci(r[y]["rf_draws"])}
    R["diferencias_t3_menos_t1"] = {}
    for suf in ("", "_mujeres"):
        a, b = res[f"t3{suf}"], res[f"t1{suf}"]
        R["diferencias_t3_menos_t1"]["total" if not suf else "mujeres"] = {
            y: {"efecto_local": round(a[y]["efecto_local"] - b[y]["efecto_local"], 4), "ic95": ci(a[y]["draws"] - b[y]["draws"]),
                "forma_reducida": round(a[y]["forma_reducida"] - b[y]["forma_reducida"], 4),
                "fr_ic95": ci(a[y]["rf_draws"] - b[y]["rf_draws"])} for y in YS}
    with open(f"{OUT}/riesgo_pension_m3.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
