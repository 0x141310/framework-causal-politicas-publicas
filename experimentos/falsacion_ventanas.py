"""Falsación con ventanas que no cruzan el otro umbral, y efecto nacional sin la edad del umbral.

Corrige un problema de la falsación original (falsacion_regla.py): con ±7 años, la ventana a los 68 en 2022–2024
(61 a 75 años) contiene el umbral real de 65, y la ventana a los 65 en 2020 en municipios no indígenas (58 a 72)
contiene el umbral real de 68. Aquí cada ventana se trunca en el otro umbral:
  - 65 años, 2020 no indígenas: edades hasta 67.
  - 68 años, 2020 indígenas y 2022–2024: edades desde 65.
Además estima el efecto nacional 2022–2024 a los 65 años con y sin la edad de 65 (dona), con el estimador de la
Tabla IV (rdd.estimate, ±7 años, 999 réplicas).
Municipios indígenas de 2020: categoría A y B de 2020 (anexo A de la Declaratoria de Zonas de Atención Prioritaria
2020); como sensibilidades, el catálogo de localidades indígenas 2010 y la aproximación con hablantes de la ENIGH 2020.
Salida: resultados/falsacion_ventanas.json
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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


def falsacion(df: pd.DataFrame, col: str) -> dict:
    """Primeras etapas y diferencia en discontinuidades a los 65, con la clasificación municipal `col`."""
    ind = df[col]
    a20, a2224 = df.anio == 2020, df.anio.isin([2022, 2024])
    # 65 años: 2020 no indígenas truncado en 67
    d65 = df[a2224 | (a20 & ind) | (a20 & ~ind & (df.edad <= 67))]
    g65 = {"2020_no_indigena": (d65.anio == 2020) & ~d65[col], "2020_mun_indigena": (d65.anio == 2020) & d65[col],
           "2022_2024": d65.anio.isin([2022, 2024])}
    # 68 años: 2020 indígenas y 2022–2024 truncados desde 65
    d68 = df[(a20 & ~ind) | (a20 & ind & (df.edad >= 65)) | (a2224 & (df.edad >= 65))]
    g68 = {"2020_no_indigena": (d68.anio == 2020) & ~d68[col], "2020_mun_indigena": (d68.anio == 2020) & d68[col],
           "2022_2024": d68.anio.isin([2022, 2024])}
    res = {}
    for dona in (False, True):
        r65 = rdd.estimate_groups(d65, "edad", 65, YS, "pam", 7, "hogar_id", g65, B=B, seed=13 + 65, donut=dona)
        r68 = rdd.estimate_groups(d68, "edad", 68, YS, "pam", 7, "hogar_id", g68, B=B, seed=13 + 68, donut=dona)
        res["dona" if dona else "base"] = (r65, r68)

    def resumen(r):
        return {g: {"n": v["n"], "primera_etapa": v["primera_etapa"], "pe_ic95": ci(v["fs_draws"]),
                    **{y: {"forma_reducida": v[y]["forma_reducida"], "fr_ic95": ci(v[y]["rf_draws"]),
                           "efecto_local": v[y]["efecto_local"], "el_ic95": ci(v[y]["draws"])} for y in YS}}
                for g, v in r.items()}

    out = {}
    for k, (r65, r68) in res.items():
        out[f"primeras_etapas_{k}"] = {"65": resumen(r65), "68": resumen(r68)}
        x1, x0 = r65["2022_2024"], r65["2020_no_indigena"]
        fs = x1["primera_etapa"] - x0["primera_etapa"]
        fs_d = x1["fs_draws"] - x0["fs_draws"]
        o = {"primera_etapa": fs, "pe_ic95": ci(fs_d)}
        for y in YS:
            rf = x1[y]["forma_reducida"] - x0[y]["forma_reducida"]
            rf_d = x1[y]["rf_draws"] - x0[y]["rf_draws"]
            o[y] = {"forma_reducida": rf, "fr_ic95": ci(rf_d), "efecto_local": rf / fs, "el_ic95": ci(rf_d / fs_d)}
        out[f"diferencia_en_discontinuidades_65_{k}"] = o
    return out


def main():
    cols = ["anio", "edad", "pam", "ubica_geo", "hablaind", "factor", "folioviv", "foliohog"] + YS
    df = pd.read_parquet(DATA, columns=cols)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["cve_mun"] = df["ubica_geo"].astype(str).str[:5]
    clas = municipios_indigenas(df)
    obs20 = set(df.loc[df.anio == 2020, "cve_mun"])
    df["mun_ind"] = df["cve_mun"].isin(clas["a_b_2020"])
    df["mun_ind_2010"] = df["cve_mun"].isin(clas["catalogo_2010"])
    df["mun_ind_aprox"] = df["cve_mun"].isin(clas["aprox_enigh"])
    R = {"descripcion": __doc__.strip().split("\n")[0],
         "municipios_indigenas_observados_2020": {k: len(v & obs20) for k, v in clas.items()}}
    R.update(falsacion(df, "mun_ind"))
    R["sensibilidad_catalogo_2010"] = falsacion(df, "mun_ind_2010")
    R["sensibilidad_aproximacion_enigh"] = falsacion(df, "mun_ind_aprox")

    # efecto nacional 2022–2024 a los 65, con y sin la edad del umbral (mismo estimador y datos de la Tabla IV)
    n = pd.read_parquet(ROOT + "/datos/enigh_personas_2022_2024.parquet")
    n["hogar_id"] = n["anio"].astype(str) + "_" + n["folioviv"] + "_" + n["foliohog"].astype(str)
    R["nacional_2022_2024"] = {}
    for dona in (False, True):
        R["nacional_2022_2024"]["dona" if dona else "base"] = {
            y: {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                rdd.estimate(n, "edad", 65, y, "pam", 7, "hogar_id", donut=dona).items()} for y in YS}

    with open(f"{OUT}/falsacion_ventanas.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1, default=float)
    for etiqueta, base in (("A y B 2020", R), ("catálogo 2010", R["sensibilidad_catalogo_2010"]),
                           ("aproximación ENIGH", R["sensibilidad_aproximacion_enigh"])):
        for k in ("base", "dona"):
            pe = base[f"primeras_etapas_{k}"]
            print(etiqueta, k, "primeras etapas:", {c: {g: round(pe[c][g]["primera_etapa"] * 100, 1) for g in pe[c]} for c in pe})
            dd = base[f"diferencia_en_discontinuidades_65_{k}"]
            print(etiqueta, k, "DiD 65: IA", round(dd["ia_mod_sev"]["efecto_local"] * 100, 1),
                  [round(x * 100, 1) for x in dd["ia_mod_sev"]["el_ic95"]], "ingreso", round(dd["log_ing_pc"]["efecto_local"] * 100, 1),
                  [round(x * 100, 1) for x in dd["log_ing_pc"]["el_ic95"]], "FS", round(dd["primera_etapa"] * 100, 1))
    for k, v in R["nacional_2022_2024"].items():
        print("nacional", k, {y: (round(v[y]["efecto_local"] * 100, 2), [round(x * 100, 1) for x in v[y]["el_ic95"]],
                                  "FS", round(v[y]["primera_etapa"] * 100, 1), "n", v[y]["n"]) for y in YS})


if __name__ == "__main__":
    main()
