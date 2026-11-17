"""Utilidad: validación ex ante–ex post con la reforma de 2021.

Si en 2020 se hubiera usado el procedimiento de escenarios para anticipar el efecto de bajar la
edad de la pensión de 68 a 65 años, ¿habría acertado lo observado en 2022 y 2024?

Ex ante: regresión discontinua a los 68 años en la ENIGH 2020 (municipios que no son de categoría indígena A ni B en 2020),
escalada por monto con las curvas D1-D3. Ex post: regresión discontinua a los 65 años en 2022
y en 2024. Además, incorporación madura predicha (69-78 años en 2020) frente a observada
(66-75 años en 2024). Como sensibilidades, los municipios indígenas se toman del catálogo de localidades indígenas
2010 (clave `sensibilidad_catalogo_2010`) o se aproximan con los hablantes de lengua indígena de la ENIGH 2020
(clave `sensibilidad_aproximacion_enigh`).
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
INPC_AGO = {2020: 107.867, 2022: 123.803, 2024: 136.013}
MONTO = {2020: 2620.0, 2022: 3850.0, 2024: 6000.0}  # bimestral nominal (Excélsior 10 ene 2020; CEFP 063/2024)
Z_BONF = 2.64  # Bonferroni al 5% en seis contrastes


def real(y: int) -> float:
    return MONTO[y] * INPC_AGO[2024] / INPC_AGO[y]


CURVAS = {"D1": lambda r: r, "D2": lambda r: float(np.sqrt(r)), "D3": lambda r: 1.0}


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


def load() -> pd.DataFrame:
    cols = ["anio", "edad", "pam", "entidad", "ubica_geo", "hablaind", "factor", "folioviv", "foliohog"] + YS
    df = pd.read_parquet(DATA, columns=cols)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["cve_mun"] = df["ubica_geo"].astype(str).str[:5]
    clas = municipios_indigenas(df)
    df["mun_ind"] = df["cve_mun"].isin(clas["a_b_2020"])
    df["mun_ind_2010"] = df["cve_mun"].isin(clas["catalogo_2010"])
    df["mun_ind_aprox"] = df["cve_mun"].isin(clas["aprox_enigh"])
    return df


def cobertura(df: pd.DataFrame, mask: pd.Series, seed: int) -> dict:
    d = df[mask]
    g = d.groupby("hogar_id").apply(lambda x: pd.Series({"wp": float((x.factor * x.pam).sum()), "w": float(x.factor.sum())}),
                                    include_groups=False)
    wp, w = g["wp"].to_numpy(), g["w"].to_numpy()
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=B)
    draws = (cnt @ wp) / (cnt @ w)
    return {"punto": float(wp.sum() / w.sum()), "ic95": ci(draws), "n_personas": int(len(d)), "draws": draws}


def resumen(v):
    out = {"n": v["n"], "primera_etapa": float(v["primera_etapa"]), "pe_ic95": [float(x) for x in v["pe_ic95"]]}
    for y in YS:
        out[y] = {"efecto": float(v[y]["efecto_local"]), "ic95": ci(v[y]["draws"])}
    return out


def ex_ante_y_comparacion(df: pd.DataFrame, col: str, ex_post: dict) -> dict:
    """Ex ante a los 68 en 2020 (municipios no indígenas según `col`), comparación con lo observado e incorporación."""
    R = {}
    pre = (df.anio == 2020) & ~df[col]
    ex_ante = rdd.estimate_groups(df, "edad", 68, YS, "pam", 7, "hogar_id", {"2020_no_indigena": pre}, B=B, seed=7068)
    ex_ante_dona = rdd.estimate_groups(df, "edad", 68, YS, "pam", 7, "hogar_id", {"2020_no_indigena": pre}, B=B,
                                       seed=7069, donut=True)
    R["ex_ante_68_2020"] = resumen(ex_ante["2020_no_indigena"])
    R["ex_ante_68_2020_dona"] = resumen(ex_ante_dona["2020_no_indigena"])

    def comparar(base):
        comp = {}
        for cname, g in CURVAS.items():
            comp[cname] = {}
            z_all = []
            for yr in (2022, 2024):
                fac = g(real(yr) / real(2020))
                comp[cname][str(yr)] = {}
                for y in YS:
                    pred_pt = base[y]["efecto_local"] * fac
                    pred_d = base[y]["draws"] * fac
                    obs = ex_post[str(yr)][y]
                    dd = pred_d - obs["draws"]
                    delta = pred_pt - obs["efecto_local"]
                    ee = float(np.nanstd(dd, ddof=1))
                    lo, hi = ci(dd)
                    pooled = abs(float(ex_post["2022_2024"][y]["efecto_local"]))
                    z = delta / ee if ee > 0 else np.nan
                    z_all.append(z)
                    comp[cname][str(yr)][y] = {
                        "factor_monto": round(fac, 4), "predicho": float(pred_pt), "predicho_ic95": ci(pred_d),
                        "observado": float(obs["efecto_local"]), "observado_ic95": ci(obs["draws"]),
                        "delta": float(delta), "delta_ic95": [lo, hi], "z": float(z),
                        "consistente": bool(lo <= 0 <= hi),
                        "informativa": bool((hi - lo) / 2 < pooled), "efecto_observado_agrupado_abs": pooled}
            comp[cname]["rechazada"] = bool(np.nanmax(np.abs(z_all)) > Z_BONF)
            comp[cname]["suma_z2"] = float(np.nansum(np.square(z_all)))
            comp[cname]["max_abs_z"] = float(np.nanmax(np.abs(z_all)))
        mejor = min(CURVAS, key=lambda c: comp[c]["suma_z2"])
        return comp, mejor

    R["comparacion"], R["curva_mas_compatible"] = comparar(ex_ante["2020_no_indigena"])
    R["comparacion_dona"], R["curva_mas_compatible_dona"] = comparar(ex_ante_dona["2020_no_indigena"])

    c_pred = cobertura(df, pre & df.edad.between(69, 78), seed=7101)
    c_obs = cobertura(df, (df.anio == 2024) & df.edad.between(66, 75), seed=7102)
    dd = c_pred["draws"] - c_obs["draws"]
    R["incorporacion"] = {"predicha_69_78_2020": {k: v for k, v in c_pred.items() if k != "draws"},
                          "observada_66_75_2024": {k: v for k, v in c_obs.items() if k != "draws"},
                          "diferencia": float(c_pred["punto"] - c_obs["punto"]), "diferencia_ic95": ci(dd)}
    return R


def main():
    df = load()
    R = {"montos_bimestrales": MONTO, "montos_reales_2024": {y: round(real(y), 1) for y in MONTO},
         "cocientes": {y: round(real(y) / real(2020), 4) for y in (2022, 2024)},
         "clasificacion_municipal": "categoría indígena A y B de 2020 (anexo A, Declaratoria de Zonas de Atención Prioritaria 2020)"}
    post = {"2022": df.anio == 2022, "2024": df.anio == 2024, "2022_2024": df.anio.isin([2022, 2024])}
    ex_post = rdd.estimate_groups(df, "edad", 65, YS, "pam", 7, "hogar_id", post, B=B, seed=7065)
    R.update(ex_ante_y_comparacion(df, "mun_ind", ex_post))
    R["ex_post_65"] = {k: resumen(v) for k, v in ex_post.items()}
    for clave, col in (("sensibilidad_catalogo_2010", "mun_ind_2010"), ("sensibilidad_aproximacion_enigh", "mun_ind_aprox")):
        sens = ex_ante_y_comparacion(df, col, ex_post)
        R[clave] = {k: v for k, v in sens.items() if not k.startswith("comparacion_dona")}

    with open(f"{OUT}/reforma_2021_exante.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)

    print("montos reales:", R["montos_reales_2024"], "cocientes:", R["cocientes"])
    print("ex ante 68 (2020):", json.dumps(R["ex_ante_68_2020"], ensure_ascii=False))
    print("ex ante 68 dona:", json.dumps(R["ex_ante_68_2020_dona"], ensure_ascii=False))
    for k, v in R["ex_post_65"].items():
        print("ex post 65", k, json.dumps(v, ensure_ascii=False))
    for c in CURVAS:
        print(c, "rechazada:", R["comparacion"][c]["rechazada"], "suma z2:", round(R["comparacion"][c]["suma_z2"], 2),
              "max|z|:", round(R["comparacion"][c]["max_abs_z"], 2))
        for yr in ("2022", "2024"):
            for y in YS:
                r = R["comparacion"][c][yr][y]
                print(f"   {yr} {y}: pred {r['predicho']:.4f} obs {r['observado']:.4f} delta {r['delta']:.4f} "
                      f"[{r['delta_ic95'][0]:.4f}, {r['delta_ic95'][1]:.4f}] z {r['z']:.2f} cons {r['consistente']} inf {r['informativa']}")
    print("curva más compatible:", R["curva_mas_compatible"], "| con dona:", R["curva_mas_compatible_dona"])
    print("incorporación:", json.dumps(R["incorporacion"], ensure_ascii=False))
    for clave in ("sensibilidad_catalogo_2010", "sensibilidad_aproximacion_enigh"):
        S = R[clave]
        print(clave, "ex ante", json.dumps(S["ex_ante_68_2020"], ensure_ascii=False))
        print("   curva más compatible:", S["curva_mas_compatible"], "| rechazadas:",
              {c: S["comparacion"][c]["rechazada"] for c in CURVAS}, "| incorporación:",
              round(S["incorporacion"]["predicha_69_78_2020"]["punto"], 4), "vs",
              round(S["incorporacion"]["observada_66_75_2024"]["punto"], 4))


if __name__ == "__main__":
    main()
