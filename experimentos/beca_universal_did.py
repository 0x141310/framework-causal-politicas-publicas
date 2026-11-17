"""¿Beca universal o focalizada?

Diferencias en diferencias por evento: 15-17 años (media superior; beca universal desde 2019)
frente a 12-14 (secundaria; beca focalizada antes y después). Referencia 2018.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = ROOT + "/datos/enigh_personas_2016_2024.parquet"
OUT = ROOT + "/resultados"


def fit(d: pd.DataFrame, y: str) -> dict:
    d = d.dropna(subset=[y, "est_socio"]).copy()
    for t in (2016, 2020, 2022, 2024):
        d[f"g_{t}"] = d["grupo"] * (d["anio"] == t)
    d["g_post"] = d["grupo"] * d["anio"].isin([2022, 2024])
    f_ev = f"{y} ~ g_2016 + g_2020 + g_2022 + g_2024 + C(anio) + C(edad) + mujer + rural + C(est_socio)"
    m = smf.ols(f_ev, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["hogar_id"])[0]})
    out = {"n": int(m.nobs)}
    for t in (2016, 2020, 2022, 2024):
        k = f"g_{t}"
        out[str(t)] = [float(m.params[k]), float(m.conf_int().loc[k, 0]), float(m.conf_int().loc[k, 1])]
    # DiD agrupado: después (2022, 2024) frente a antes (2016, 2018), sin 2020
    dd = d[d["anio"] != 2020]
    # el grupo queda absorbido por los efectos fijos de edad
    m2 = smf.ols(f"{y} ~ g_post + C(anio) + C(edad) + mujer + rural + C(est_socio)", data=dd).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["hogar_id"])[0]})
    out["did_post"] = [float(m2.params["g_post"]), float(m2.conf_int().loc["g_post", 0]), float(m2.conf_int().loc["g_post", 1])]
    out["did_post_se"] = float(m2.bse["g_post"])
    return out


def means(d: pd.DataFrame, y: str) -> dict:
    g = d.dropna(subset=[y]).groupby(["anio", "grupo"]).apply(lambda s: np.average(s[y], weights=s["factor"]), include_groups=False)
    return {f"{a}_{'15_17' if b == 1 else '12_14'}": float(v) for (a, b), v in g.items()}


def main():
    cols = ["anio", "edad", "asiste", "mujer", "rural", "est_socio", "entidad", "factor", "folioviv", "foliohog",
            "hog_prospera", "hog_beca_basica", "hog_beca_ems", "hog_beca_otra_gob"]
    df = pd.read_parquet(DATA, columns=cols)
    df = df[df["edad"].between(12, 17)].copy()
    df["est_socio"] = pd.to_numeric(df["est_socio"], errors="coerce")
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["grupo"] = (df["edad"] >= 15).astype(int)
    df["apoyo_educ_hogar"] = ((df.hog_prospera == 1) | (df.hog_beca_basica == 1) | (df.hog_beca_ems == 1) |
                              (df.hog_beca_otra_gob == 1)).astype(int)
    R = {}
    for zona, d in (("pais", df), ("quintana_roo", df[df.entidad == "23"])):
        R[zona] = {
            "medias_asistencia": means(d, "asiste"), "medias_apoyo": means(d, "apoyo_educ_hogar"),
            "asistencia": fit(d, "asiste"), "primera_etapa_apoyo": fit(d, "apoyo_educ_hogar"),
        }
        a, f = R[zona]["asistencia"]["did_post"][0], R[zona]["primera_etapa_apoyo"]["did_post"][0]
        R[zona]["wald_did"] = a / f if abs(f) > 1e-6 else None
    # heterogeneidad por estrato (país)
    R["estrato"] = {}
    for lbl, m in (("bajo_1_2", df.est_socio.isin([1, 2])), ("alto_3_4", df.est_socio.isin([3, 4]))):
        d = df[m]
        R["estrato"][lbl] = {"asistencia": fit(d, "asiste"), "primera_etapa_apoyo": fit(d, "apoyo_educ_hogar"),
                             "medias_asistencia": means(d, "asiste"), "medias_apoyo": means(d, "apoyo_educ_hogar")}
    with open(f"{OUT}/beca_universal_did.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    for zona in ("pais", "quintana_roo"):
        print(f"\n== {zona}")
        print("  medias asistencia:", {k: round(v, 3) for k, v in R[zona]["medias_asistencia"].items()})
        print("  medias apoyo:", {k: round(v, 3) for k, v in R[zona]["medias_apoyo"].items()})
        for y in ("asistencia", "primera_etapa_apoyo"):
            r = R[zona][y]
            print(f"  {y:22s} n={r['n']}  " + "  ".join(f"{t}: {r[t][0]:+.4f} [{r[t][1]:+.4f},{r[t][2]:+.4f}]" for t in ("2016", "2020", "2022", "2024")),
                  f"| DiD post {r['did_post'][0]:+.4f} [{r['did_post'][1]:+.4f},{r['did_post'][2]:+.4f}]")
        print("  Wald-DiD:", R[zona]["wald_did"])
    for lbl, v in R["estrato"].items():
        print(f"\n== estrato {lbl}")
        for y in ("asistencia", "primera_etapa_apoyo"):
            r = v[y]
            print(f"  {y:22s} n={r['n']}  2016: {r['2016'][0]:+.4f} [{r['2016'][1]:+.4f},{r['2016'][2]:+.4f}] | DiD post {r['did_post'][0]:+.4f} [{r['did_post'][1]:+.4f},{r['did_post'][2]:+.4f}]")


if __name__ == "__main__":
    main()
