"""Falsación: la discontinuidad debe moverse con la regla.

2016-2018: programa "65 y más" (65+, sin pensión contributiva).
2020: pensión desde los 68; a los 65 en municipios o localidades indígenas de tipo A y B y para quienes la recibían
de 65 a 67 años en diciembre de 2019 (reglas de 2020, numeral 3.2). Aquí: municipios A y B de la categoría indígena
de 2020 (anexo A de la Declaratoria de Zonas de Atención Prioritaria 2020).
2022-2024: universal desde los 65.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ROOT)
from modulo_causal import rdd  # noqa: E402

DATA = ROOT + "/datos/enigh_personas_2016_2024.parquet"
OUT = ROOT + "/resultados"
YS = ["ia_mod_sev", "ia_adultos", "log_ing_pc"]
B = 999


def ci(d):
    return [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]


def municipios_indigenas(df: pd.DataFrame) -> dict[str, set]:
    """Municipios indígenas de la regla de 2020, con tres clasificaciones.

    Las reglas de operación de 2020 (numeral 3.2) hacen elegibles a los 65 años a quienes viven en municipios o
    localidades indígenas de tipo A y B. Principal: categoría indígena A y B vigente en 2020, del anexo A de la
    Declaratoria de Zonas de Atención Prioritaria 2020 (ingesta/zap2020.py). Sensibilidades: municipios indígenas
    del catálogo de localidades indígenas 2010 de la CDI, hoy INPI (40% o más de población indígena; ingesta/inpi.py),
    y municipios donde al menos 40% de las personas de 3 años o más habla una lengua indígena (ENIGH 2020,
    ponderado). La ENIGH pública no identifica la localidad ni el padrón de quienes ya recibían la pensión de 65 a
    67 años, así que la regla se asigna por municipio.
    """
    zap = pd.read_csv(ROOT + "/datos/categoria_indigena_2020.csv", dtype=str)
    cat = pd.read_csv(ROOT + "/datos/inpi_municipios_2010.csv", dtype=str)
    d20 = df[(df.anio == 2020) & (df.edad >= 3)]
    sh = d20.assign(ind=(d20.hablaind == 1).astype(float)).groupby("cve_mun").apply(
        lambda d: np.average(d["ind"], weights=d["factor"]), include_groups=False)
    return {"a_b_2020": set(zap.loc[zap["indigena_ab"] == "1", "cve_mun"]),
            "catalogo_2010": set(cat.loc[cat["indigena_ab"] == "1", "cve_mun"]),
            "aprox_enigh": set(sh[sh >= 0.4].index)}


def main():
    cols = ["anio", "edad", "pam", "pam_p104", "pam_monto_tri", "pension_contrib", "entidad", "ubica_geo", "hablaind",
            "factor", "folioviv", "foliohog", "ing_cor", "tot_integ"] + YS
    df = pd.read_parquet(DATA, columns=cols)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["cve_mun"] = df["ubica_geo"].astype(str).str[:5]
    clas = municipios_indigenas(df)
    mun_ind = clas["a_b_2020"]
    df["mun_ind"] = df["cve_mun"].isin(mun_ind)
    obs20 = set(df.loc[df.anio == 2020, "cve_mun"])
    R = {"municipios_observados_2020": len(obs20),
         "municipios_indigenas_observados_2020": {k: len(v & obs20) for k, v in clas.items()},
         "municipios_que_difieren_de_a_b_2020": {k: sorted((v ^ mun_ind) & obs20) for k, v in clas.items() if k != "a_b_2020"},
         "municipios_indigenas_qroo": sorted(m for m in mun_ind & obs20 if m.startswith("23"))}

    def groups(d):
        return {
            "2016_sin_contrib": (d.anio == 2016) & (d.pension_contrib == 0),
            "2018_sin_contrib": (d.anio == 2018) & (d.pension_contrib == 0),
            "2020_no_indigena": (d.anio == 2020) & ~d.mun_ind,
            "2020_no_ind_sin_hablantes": (d.anio == 2020) & ~d.mun_ind & (d.hablaind != 1),
            "2020_mun_indigena": (d.anio == 2020) & d.mun_ind,
            "2022_2024": d.anio.isin([2022, 2024]),
        }

    def run(d, label, treat="pam", donut=False, seed=13):
        out = {}
        for c in (65, 68):
            out[c] = rdd.estimate_groups(d, "edad", c, YS, treat, 7, "hogar_id", groups(d), B=B, seed=seed + c, donut=donut)
        return out

    res = {"pais": run(df, "pais"), "pais_dona": run(df, "pais_dona", donut=True),
           "pais_solo_clave_principal": run(df, "pais_p104", treat="pam_p104"),
           "quintana_roo": run(df[df.entidad == "23"], "qroo")}

    def table(r):
        rows = []
        for c, gs in r.items():
            for g, v in gs.items():
                row = {"umbral": c, "grupo": g, "n": v["n"], "primera_etapa": v["primera_etapa"], "pe_ic95": ci(v["fs_draws"])}
                for y in YS:
                    row[f"fr_{y}"] = v[y]["forma_reducida"]
                    row[f"fr_{y}_ic95"] = ci(v[y]["rf_draws"])
                rows.append(row)
        return rows

    R["saltos"] = {k: table(v) for k, v in res.items()}

    # F3: diferencia en discontinuidades (réplicas compartidas dentro de cada umbral)
    def didisc(r):
        o = {}
        a, b = r[68]["2020_no_indigena"], r[68]["2022_2024"]
        c_, d_ = r[65]["2022_2024"], r[65]["2020_no_indigena"]
        for lbl, (x1, x0) in {"a_los_68_(2020_menos_2022_24)": (a, b), "a_los_65_(2022_24_menos_2020)": (c_, d_)}.items():
            fs = x1["primera_etapa"] - x0["primera_etapa"]
            fs_d = x1["fs_draws"] - x0["fs_draws"]
            o[lbl] = {"primera_etapa": fs, "pe_ic95": ci(fs_d)}
            for y in YS:
                rf = x1[y]["forma_reducida"] - x0[y]["forma_reducida"]
                rf_d = x1[y]["rf_draws"] - x0[y]["rf_draws"]
                o[lbl][y] = {"forma_reducida": rf, "fr_ic95": ci(rf_d), "efecto_local": rf / fs, "el_ic95": ci(rf_d / fs_d)}
        return o

    R["diferencia_en_discontinuidades"] = {k: didisc(res[k]) for k in ("pais", "pais_dona", "pais_solo_clave_principal", "quintana_roo")}

    # F4: programa focalizado 2016-2018 a los 65 (sin pensión contributiva): efecto local
    R["focalizado_2016_2018"] = {g: {"primera_etapa": res["pais"][65][g]["primera_etapa"],
                                     **{y: {"efecto_local": res["pais"][65][g][y]["efecto_local"], "el_ic95": ci(res["pais"][65][g][y]["draws"])} for y in YS}}
                                 for g in ("2016_sin_contrib", "2018_sin_contrib")}

    # exploratorio: peso mecánico de la pensión en el ingreso del hogar de quienes la reciben cerca del umbral
    mech = {}
    for lbl, m in {"2016": (df.anio == 2016) & df.edad.between(65, 72), "2018": (df.anio == 2018) & df.edad.between(65, 72),
                   "2020": (df.anio == 2020) & df.edad.between(68, 75) & ~df.mun_ind,
                   "2022": (df.anio == 2022) & df.edad.between(65, 72), "2024": (df.anio == 2024) & df.edad.between(65, 72)}.items():
        s = df[m & (df.pam == 1) & (df.ing_cor > 0)]
        share = (s["pam_monto_tri"] / s["ing_cor"]).clip(0, 0.95)
        mech[lbl] = {"pension_mensual_mediana": float((s["pam_monto_tri"] / 3).median()),
                     "dlog_mecanico_medio": float(np.mean(-np.log1p(-share)))}
    R["exploratorio_peso_mecanico"] = mech

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not str(k).endswith("draws")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(f"{OUT}/falsacion_regla.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=1)

    print("Municipios indígenas observados en 2020:", R["municipios_indigenas_observados_2020"], "de",
          R["municipios_observados_2020"], "| difieren de A y B 2020:",
          {k: len(v) for k, v in R["municipios_que_difieren_de_a_b_2020"].items()}, "| QRoo:", R["municipios_indigenas_qroo"])
    for k in ("pais", "quintana_roo"):
        print(f"\n== SALTOS ({k}): primera etapa y forma reducida")
        for r in R["saltos"][k]:
            print(f"  {r['umbral']} {r['grupo']:26s} n={r['n']:6d} PE {r['primera_etapa']:+.3f} {np.round(r['pe_ic95'],3)} | "
                  f"IA {r['fr_ia_mod_sev']:+.4f} {np.round(r['fr_ia_mod_sev_ic95'],4)} | reac {r['fr_ia_adultos']:+.3f} {np.round(r['fr_ia_adultos_ic95'],3)} | "
                  f"log ing {r['fr_log_ing_pc']:+.3f} {np.round(r['fr_log_ing_pc_ic95'],3)}")
    for k, v in R["diferencia_en_discontinuidades"].items():
        print(f"\n== DIFERENCIA EN DISCONTINUIDADES ({k})")
        for lbl, o in v.items():
            print(f"  {lbl:34s} PE {o['primera_etapa']:+.3f} {np.round(o['pe_ic95'],3)}")
            for y in YS:
                print(f"     {y:12s} FR {o[y]['forma_reducida']:+.4f} {np.round(o[y]['fr_ic95'],4)} | LATE {o[y]['efecto_local']:+.4f} {np.round(o[y]['el_ic95'],4)}")
    print("\n== FOCALIZADO 2016-2018:", json.dumps(clean(R["focalizado_2016_2018"]), ensure_ascii=False))
    print("== PESO MECÁNICO:", json.dumps(mech, ensure_ascii=False))


if __name__ == "__main__":
    main()
