"""Heterogeneidad, dosis, análisis tradicional frente a causal
y escenario de la Pensión Mujeres Bienestar (PMB) en Quintana Roo.
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
from modulo_causal import estimation as est  # noqa: E402
from modulo_causal import rdd  # noqa: E402

DATA = ROOT + "/datos/enigh_personas_2022_2024.parquet"
OUT = ROOT + "/resultados"
YS = ["ia_mod_sev", "ia_adultos", "log_ing_pc"]
B = 999
PAM_2024_BIM = 6000.0   # pensión de adultos mayores 2024, pesos bimestrales
PMB_2026_BIM = 3100.0   # Pensión Mujeres Bienestar 2026, pesos bimestrales (Excélsior, 20 mar 2026)


def summarize(late_draws, point):
    return {"efecto_local": float(point), "ic95": [float(x) for x in np.nanpercentile(late_draws, [2.5, 97.5])]}


def main():
    df = pd.read_parquet(DATA)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["est_socio"] = pd.to_numeric(df["est_socio"], errors="coerce")
    R: dict = {}

    # ---------------- E2.1 heterogeneidad y dosis (nacional, 2022+2024) ----------------
    g = {
        "todos": df["edad"].notna(),
        "mujeres": df["mujer"] == 1, "hombres": df["mujer"] == 0,
        "sin_pension_contrib": df["pension_contrib"] == 0, "con_pension_contrib": df["pension_contrib"] == 1,
        "mujeres_sin_contrib": (df["mujer"] == 1) & (df["pension_contrib"] == 0),
        "educ_hasta_primaria": df["educ_baja"] == 1, "educ_mayor": df["educ_baja"] == 0,
        "rural": df["rural"] == 1, "urbano": df["rural"] == 0,
    }
    het = rdd.estimate_groups(df, "edad", 65, YS, "pam", 7, "hogar_id", g, B=B, seed=11)
    R["heterogeneidad"] = {k: {"n": v["n"], "primera_etapa": v["primera_etapa"], "pe_ic95": v["pe_ic95"],
                               **{y: summarize(v[y]["draws"], v[y]["efecto_local"]) for y in YS}} for k, v in het.items()}
    # diferencias entre subgrupos con réplicas compartidas
    def diff(a, b, y):
        d = het[a][y]["draws"] - het[b][y]["draws"]
        return {"diferencia": float(het[a][y]["efecto_local"] - het[b][y]["efecto_local"]),
                "ic95": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}
    R["diferencias"] = {f"{a}-{b}": {y: diff(a, b, y) for y in YS}
                        for a, b in [("mujeres", "hombres"), ("sin_pension_contrib", "con_pension_contrib"),
                                     ("educ_hasta_primaria", "educ_mayor"), ("rural", "urbano")]}
    # dosis: efecto por año (montos 3,850 y 6,000 bimestrales)
    gy = {"a2022": df["anio"] == 2022, "a2024": df["anio"] == 2024}
    dose = rdd.estimate_groups(df, "edad", 65, YS, "pam", 7, "hogar_id", gy, B=B, seed=12, fe_col=None)
    R["dosis"] = {}
    for y in YS:
        r22, r24 = dose["a2022"][y], dose["a2024"][y]
        ratio_d = r22["draws"] / r24["draws"]
        R["dosis"][y] = {"2022": summarize(r22["draws"], r22["efecto_local"]), "2024": summarize(r24["draws"], r24["efecto_local"]),
                         "cociente_2022_2024": float(r22["efecto_local"] / r24["efecto_local"]),
                         "cociente_ic95": [float(x) for x in np.nanpercentile(ratio_d, [2.5, 97.5])],
                         "esperado_si_lineal": 3850 / 6000}

    # ---------------- E2.2 análisis tradicional frente a causal ----------------
    covs = ["mujer", "indigena", "rural", "est_socio", "educ_baja", "pension_contrib", "anio24"]
    df["anio24"] = (df["anio"] == 2024).astype(int)
    R["tradicional_vs_causal"] = {}
    for zona, d in (("nacional", df), ("quintana_roo", df[df["entidad"] == "23"])):
        s = d[d["edad"].between(65, 75)].dropna(subset=covs + YS)
        row = {}
        for y in YS:
            naive = est.linear_adjustment(s, "pam", y, [])
            adj = est.linear_adjustment(s, "pam", y, covs)
            row[y] = {"diferencia_simple": [naive["efecto"], *naive["ic95"]], "mco_ajustado": [adj["efecto"], *adj["ic95"]]}
        R["tradicional_vs_causal"][zona] = {"n": int(len(s)), **row}

    # ---------------- E2.3 escenario PMB en Quintana Roo ----------------
    q24 = df[(df["entidad"] == "23") & (df["anio"] == 2024)].copy()
    nat24 = df[df["anio"] == 2024]
    # umbral del 40% inferior del ingreso laboral per cápita (personas, país, 2024, ponderado)
    v = nat24["log_ing_lab_pc"].to_numpy(); ww = nat24["factor"].to_numpy(float)
    o_ = np.argsort(v); cw = np.cumsum(ww[o_]) / ww.sum()
    thr = float(v[o_][np.searchsorted(cw, 0.40)])
    w = q24["factor"]
    muj = q24[(q24["mujer"] == 1) & q24["edad"].between(60, 64)]
    muj65 = q24[(q24["mujer"] == 1) & (q24["edad"] >= 65)]
    tau = float(np.average(q24.loc[(q24.mujer == 1) & q24.edad.between(66, 75), "pam"],
                           weights=q24.loc[(q24.mujer == 1) & q24.edad.between(66, 75), "factor"]))
    N_A = float(muj["factor"].sum())
    N_B = float(muj.loc[muj.pension_contrib == 0, "factor"].sum())
    N_C = float(muj65.loc[muj65.pam == 1, "factor"].sum())
    base = {
        "n_muestra_mujeres_60_64": int(len(muj)), "N_mujeres_60_64": N_A, "N_sin_pension_contrib": N_B,
        "N_mujeres_65_con_pension": N_C, "tau_incorporacion": tau,
        "ia_base_60_64": float(np.average(muj["ia_mod_sev"], weights=muj["factor"])),
        "ia_adultos_base_60_64": float(np.average(muj["ia_adultos"], weights=muj["factor"])),
        "share_bottom40_A": float(np.average(muj["log_ing_lab_pc"] <= thr, weights=muj["factor"])),
        "share_bottom40_B": float(np.average(muj.loc[muj.pension_contrib == 0, "log_ing_lab_pc"] <= thr,
                                             weights=muj.loc[muj.pension_contrib == 0, "factor"])),
        "share_ia_A": float(np.average(muj["ia_mod_sev"], weights=muj["factor"])),
        "share_ia_B": float(np.average(muj.loc[muj.pension_contrib == 0, "ia_mod_sev"], weights=muj.loc[muj.pension_contrib == 0, "factor"])),
    }
    R["escenario_base"] = base
    LA = {y: het["mujeres"][y]["draws"] for y in YS}
    LB = {y: het["mujeres_sin_contrib"][y]["draws"] for y in YS}
    LA_pt = {y: het["mujeres"][y]["efecto_local"] for y in YS}
    LB_pt = {y: het["mujeres_sin_contrib"][y]["efecto_local"] for y in YS}

    def option(name, n_benef, bim, L, Lpt, budget=None):
        dose = bim / PAM_2024_BIM
        cost = n_benef * bim * 6 if budget is None else budget
        lifted = -L["ia_mod_sev"] * dose * n_benef
        items = -L["ia_adultos"] * dose * n_benef
        inc = np.expm1(L["log_ing_pc"] * dose)
        r = {"beneficiarias": n_benef, "monto_bimestral": bim, "dosis_relativa": dose, "costo_anual": cost,
             "hogares_salen_ia": {"esperado": float(-Lpt["ia_mod_sev"] * dose * n_benef),
                                  "ic95": [float(x) for x in np.nanpercentile(lifted, [2.5, 97.5])]},
             "reactivos_adultos_menos": {"esperado": float(-Lpt["ia_adultos"] * dose * n_benef),
                                         "ic95": [float(x) for x in np.nanpercentile(items, [2.5, 97.5])]},
             "ingreso_pc_cambio": {"esperado": float(np.expm1(Lpt["log_ing_pc"] * dose)),
                                   "ic95": [float(x) for x in np.nanpercentile(inc, [2.5, 97.5])]},
             "prob_reduce_ia": float(np.mean(lifted > 0)),
             "costo_por_hogar_que_sale": float(cost / (-Lpt["ia_mod_sev"] * dose * n_benef)) if Lpt["ia_mod_sev"] < 0 else None,
             "hogares_salen_por_millon": float(-Lpt["ia_mod_sev"] * dose * n_benef / cost * 1e6),
             "_lifted": lifted, "_items": items}
        return r

    nA = N_A * tau
    A = option("A", nA, PMB_2026_BIM, LA, LA_pt)
    Bopt = option("B", N_B * tau, PMB_2026_BIM, LB, LB_pt)
    bim_Bp = min(A["costo_anual"] / (N_B * tau) / 6, PAM_2024_BIM)
    Bp = option("Bp", N_B * tau, bim_Bp, LB, LB_pt, budget=min(A["costo_anual"], N_B * tau * bim_Bp * 6))
    bim_C = A["costo_anual"] / N_C / 6  # aumento bimestral por mujer de 65+ ya incorporada
    C = option("C", N_C, bim_C, LA, LA_pt, budget=A["costo_anual"])
    opts = {"A_universal": A, "B_focalizada": Bopt, "Bp_focalizada_mismo_presupuesto": Bp, "C_aumento_65mas": C}
    for k, o in opts.items():
        if k != "A_universal":
            # tolerancia relativa: bajo S2, C es algebraicamente igual a A en cada réplica y sin ella el
            # redondeo de punto flotante rompía empates (antes daba 7%)
            tol_h = 1e-9 * np.maximum(1.0, np.abs(A["_lifted"]))
            tol_i = 1e-9 * np.maximum(1.0, np.abs(A["_items"]))
            o["prob_supera_A_hogares"] = float(np.mean(o["_lifted"] > A["_lifted"] + tol_h))
            o["prob_empata_A_hogares"] = float(np.mean(np.abs(o["_lifted"] - A["_lifted"]) <= tol_h))
            o["prob_supera_A_reactivos"] = float(np.mean(o["_items"] > A["_items"] + tol_i))
    # sensibilidad: incorporación y monto real
    sens = {}
    for lbl, t_, real in (("tau_0.6", 0.6, 1.0), ("tau_1.0", 1.0, 1.0), ("monto_real_0.92", tau, 0.92)):
        o = option("A", N_A * t_, PMB_2026_BIM * real, LA, LA_pt)
        sens[lbl] = {"costo_anual": o["costo_anual"], "hogares_salen_ia": o["hogares_salen_ia"],
                     "reactivos_adultos_menos": o["reactivos_adultos_menos"]}
    R["escenario"] = {k: {kk: vv for kk, vv in o.items() if not kk.startswith("_")} for k, o in opts.items()}
    R["escenario_sensibilidad"] = sens

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k != "draws"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return None
        return o
    with open(f"{OUT}/escenario_pmb.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=1)

    # ---------------- impresión ----------------
    f = lambda r: f"{r['efecto_local']:+.4f} [{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}]"
    print("HETEROGENEIDAD (nacional, efecto local):")
    for k, v in R["heterogeneidad"].items():
        print(f"  {k:22s} n={v['n']:6d} PE={v['primera_etapa']:.3f} | IA {f(v['ia_mod_sev'])} | reactivos {f(v['ia_adultos'])} | log ing {f(v['log_ing_pc'])}")
    print("DIFERENCIAS:")
    for k, v in R["diferencias"].items():
        print(f"  {k:40s} " + " | ".join(f"{y} {v[y]['diferencia']:+.4f} [{v[y]['ic95'][0]:+.4f},{v[y]['ic95'][1]:+.4f}]" for y in YS))
    print("DOSIS:")
    for y, v in R["dosis"].items():
        print(f"  {y:12s} 2022 {f(v['2022'])} | 2024 {f(v['2024'])} | cociente {v['cociente_2022_2024']:.2f} [{v['cociente_ic95'][0]:.2f},{v['cociente_ic95'][1]:.2f}] esperado 0.64")
    print("TRADICIONAL vs CAUSAL:", json.dumps(clean(R["tradicional_vs_causal"]), ensure_ascii=False))
    print("BASE ESCENARIO:", json.dumps(base, ensure_ascii=False))
    for k, o in R["escenario"].items():
        print(f"  {k:34s} benef={o['beneficiarias']:8.0f} bim={o['monto_bimestral']:7.0f} costo={o['costo_anual']/1e6:7.1f} M | hogares salen IA {o['hogares_salen_ia']['esperado']:7.0f} {[round(x) for x in o['hogares_salen_ia']['ic95']]} | reactivos {o['reactivos_adultos_menos']['esperado']:7.0f} {[round(x) for x in o['reactivos_adultos_menos']['ic95']]} | ingreso {o['ingreso_pc_cambio']['esperado']:+.3f} | P(>A) {o.get('prob_supera_A_hogares')} {o.get('prob_supera_A_reactivos')}")
    print("SENSIBILIDAD:", json.dumps(sens, ensure_ascii=False))


if __name__ == "__main__":
    main()
