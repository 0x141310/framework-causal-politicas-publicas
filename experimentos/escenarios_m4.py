"""Módulo 4: comparador de escenarios de la Pensión Mujeres
Bienestar en Quintana Roo con presupuesto, capacidad y piso de equidad, más la variante estatal.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, ROOT)
from modulo_causal import rdd  # noqa: E402
from modulo_escenarios.comparador import Option, eb_shrink, evaluate, pareto, prob_best, summarize  # noqa: E402

DATA = ROOT + "/datos/enigh_personas_2016_2024.parquet"
OUT = ROOT + "/resultados"
YS = ["ia_mod_sev", "ia_adultos", "log_ing_pc"]
B = 999
TAU = 0.8986805040770941          # incorporación (escenario_pmb.py)
AMOUNTS = [3000.0, 3100.0, 4000.0, 5000.0, 6000.0]
CUT = (0.13747134048788534, 0.20840044046279527)  # cortes de terciles de riesgo (riesgo_pension_m3.py)
REG = {"centro_occ": {"01", "06", "11", "14", "16", "18", "22", "24", "32"}, "centro": {"09", "13", "15", "17", "21", "29"},
       "sur": {"04", "07", "12", "20", "23", "27", "30", "31"}}
RFEATS = ["mujer", "nivelaprob", "habla_ind", "tam_loc", "est_socio", "reg_centro_occ", "reg_centro", "reg_sur"]


def load():
    cols = ["anio", "edad", "mujer", "nivelaprob", "hablaind", "tam_loc", "est_socio", "entidad", "factor", "upm",
            "folioviv", "foliohog", "pam", "pension_contrib", "rural", "log_ing_lab_pc"] + YS
    df = pd.read_parquet(DATA, columns=cols)
    for c in ("tam_loc", "est_socio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["habla_ind"] = (df["hablaind"] == 1).astype(int)
    for k, e in REG.items():
        df[f"reg_{k}"] = df["entidad"].isin(e).astype(int)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["hog_habla"] = df.groupby("hogar_id")["habla_ind"].transform("max")
    return df


def risk_model(df):
    tr = df[df.anio.isin([2016, 2018, 2020]) & df.edad.between(50, 80)].dropna(subset=["ia_mod_sev"])
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=400, min_samples_leaf=100,
                                       l2_regularization=1.0, early_stopping=False, random_state=7)
    m.fit(tr[RFEATS].to_numpy(float), tr["ia_mod_sev"].to_numpy(int))
    return m


def tercile(r):
    return np.where(r < CUT[0], 1, np.where(r < CUT[1], 2, 3))


def main():
    t0 = time.time()
    df = load()
    m = risk_model(df)
    d = df[df.anio.isin([2022, 2024]) & df.edad.between(57, 72)].copy()
    d["riesgo"] = m.predict_proba(d[RFEATS].to_numpy(float))[:, 1]
    d["tercil"] = tercile(d["riesgo"].to_numpy())

    # ---------------- efectos: RDD a los 65, mujeres, réplicas compartidas ----------------
    dr = d[d.edad.between(58, 72)]
    W = dr.mujer == 1
    groups = {"mujeres": W, "t1": W & (dr.tercil == 1), "t2": W & (dr.tercil == 2), "t3": W & (dr.tercil == 3),
              "urbano": W & (dr.rural == 0), "rural": W & (dr.rural == 1), "m2024": W & (dr.anio == 2024)}
    eff = rdd.estimate_groups(dr, "edad", 65, YS, "pam", 7, "hogar_id", groups, B=B, seed=11)
    R = {"efectos": {}, "encogimiento": {}}
    for g, v in eff.items():
        R["efectos"][g] = {"n": v["n"], "primera_etapa": round(v["primera_etapa"], 4),
                           **{y: {"punto": round(v[y]["efecto_local"], 5),
                                  "ic95": [round(float(x), 5) for x in np.nanpercentile(v[y]["draws"], [2.5, 97.5])]} for y in YS}}
    print("efectos (mujeres):", {g: R["efectos"][g]["ia_mod_sev"] for g in groups}, flush=True)

    # ---------------- población: mujeres de 60 a 64 en Quintana Roo, 2024 ----------------
    nat24 = df[df.anio == 2024]
    v_, ww_ = nat24["log_ing_lab_pc"].to_numpy(), nat24["factor"].to_numpy(float)
    o_ = np.argsort(v_)
    thr = float(v_[o_][np.searchsorted(np.cumsum(ww_[o_]) / ww_.sum(), 0.40)])

    def population(lo, hi):
        p = d[(d.entidad == "23") & (d.anio == 2024) & (d.mujer == 1) & d.edad.between(lo, hi)].reset_index(drop=True)
        attrs = {"en_ia": p["ia_mod_sev"].to_numpy(float), "q40": (p["log_ing_lab_pc"] <= thr).to_numpy(float),
                 "rural_o_indigena": ((p["rural"] == 1) | (p["hog_habla"] == 1)).to_numpy(float),
                 "sin_contrib": (p["pension_contrib"] == 0).to_numpy(float), "tercil_alto": (p["tercil"] == 3).to_numpy(float)}
        codes, uniq = pd.factorize(p["upm"].astype(str))
        rng = np.random.default_rng(23)
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)
        w = p["factor"].to_numpy(float)
        return p, attrs, w, w[None, :] * cnt[:, codes]

    P, attrs, w, w_draws = population(60, 64)
    N_A = float(w.sum())
    R["poblacion"] = {"n_muestra": int(len(P)), "N": N_A, "umbral_q40_log_ing_lab_pc": thr,
                      "composicion": {k: round(float(np.average(a, weights=w)), 4) for k, a in attrs.items()},
                      "terciles": {int(t): round(float(w[P.tercil == t].sum() / N_A), 4) for t in (1, 2, 3)}}
    print("población:", R["poblacion"], flush=True)

    # ---------------- modelos de efecto por unidad ----------------
    def unit_effects(model, pop):
        tp, td = {}, {}
        for y in YS:
            if model in ("M0", "M3"):
                g = "mujeres" if model == "M0" else "m2024"
                tp[y] = np.full(len(pop), eff[g][y]["efecto_local"])
                td[y] = np.repeat(eff[g][y]["draws"][:, None], len(pop), axis=1)
            else:
                gs = ["t1", "t2", "t3"] if model == "M1" else ["urbano", "rural"]
                pt = np.array([eff[g][y]["efecto_local"] for g in gs])
                drw = np.column_stack([eff[g][y]["draws"] for g in gs])
                spt, sdr, info = eb_shrink(pt, drw)
                R["encogimiento"].setdefault(model, {})[y] = {"grupos": gs, "sin_encoger": pt.round(5).tolist(),
                                                              "encogido": spt.round(5).tolist(), **info}
                key = (pop["tercil"].to_numpy() - 1) if model == "M1" else pop["rural"].to_numpy().astype(int)
                tp[y] = spt[key]
                td[y] = sdr[:, key]
        return tp, td

    rules = {
        "U": dict(score=lambda p, th: p["edad"].to_numpy(float), elig=lambda p: np.ones(len(p), bool), foc=False),
        "NC": dict(score=lambda p, th: p["edad"].to_numpy(float), elig=lambda p: (p["pension_contrib"] == 0).to_numpy(), foc=True),
        "RR": dict(score=lambda p, th: p["riesgo"].to_numpy(float), elig=lambda p: np.ones(len(p), bool), foc=True),
        "RI": dict(score=lambda p, th: ((p["rural"] == 1) | (p["hog_habla"] == 1)).to_numpy(float) * 100 + p["edad"].to_numpy(float),
                   elig=lambda p: np.ones(len(p), bool), foc=True),
        "IMP": dict(score=lambda p, th: np.round(-th["ia_mod_sev"], 8) * 1e4 + p["edad"].to_numpy(float),
                    elig=lambda p: np.ones(len(p), bool), foc=True),
    }
    budget_R = N_A * TAU * 3100 * 6
    conds = {"R": dict(budget=budget_R, capacity=np.inf), "P": dict(budget=0.7 * budget_R, capacity=np.inf),
             "K": dict(budget=budget_R, capacity=0.5 * N_A)}
    R["condiciones"] = {k: {"presupuesto": v["budget"], "capacidad": (None if np.isinf(v["capacity"]) else v["capacity"])} for k, v in conds.items()}

    models, curves = ["M0", "M1", "M2", "M3"], ["D1", "D2", "D3"]
    eff_cache = {mo: unit_effects(mo, P) for mo in models}
    store = {}
    R["resultados"] = {}
    for c, cv in conds.items():
        for mo in models:
            tp, td = eff_cache[mo]
            for cu in curves:
                for admin in ((0.0, 0.05) if mo == "M0" else (0.0,)):
                    res = []
                    for rn, rr in rules.items():
                        for a in AMOUNTS:
                            opt = Option(rn, a, rr["score"](P, tp), rr["elig"](P), rr["foc"])
                            res.append(evaluate(opt, w=w, w_draws=w_draws, tau=TAU, budget=cv["budget"], capacity=cv["capacity"],
                                                admin=admin, theta_pt=tp, theta_dr=td, curve=cu, attrs=attrs))
                    store[(c, mo, cu, admin)] = res
    # piso de equidad: proporción del 40% inferior de la opción vigente (U, 3,100, condición R)
    ref = next(r for r in store[("R", "M0", "D1", 0.0)] if r["regla"] == "U" and r["monto"] == 3100.0)
    floor = ref["punto"]["prop_q40"]
    R["piso_equidad_q40"] = floor
    for key, res in store.items():
        feas = [r["punto"]["prop_q40"] >= floor - 1e-9 for r in res]
        pb = prob_best(res, feas)
        par = pareto([(r["punto"]["hogares_salen_ia"], r["punto"]["prop_q40"]) for r in res])
        rows = []
        for r, f, p_, pa in zip(res, feas, pb, par):
            s = summarize(r)
            s.update({"factible": f, "prob_mejor": round(p_, 4), "pareto": pa})
            rows.append(s)
        R["resultados"]["|".join(map(str, key))] = rows
    # robustez y recomendación por condición (costo de focalizar 0%)
    def cocientes(por_combinacion):
        """Cociente de cada opción factible frente a la mejor factible, en cada combinación de modelo y curva."""
        ratio = {}
        for (mo, cu), res in por_combinacion.items():
            feas = [r["punto"]["prop_q40"] >= floor - 1e-9 for r in res]
            best = max(r["punto"]["hogares_salen_ia"] for r, f in zip(res, feas) if f)
            for r, f in zip(res, feas):
                k = f"{r['regla']}-{int(r['monto'])}"
                ratio.setdefault(k, {})[f"{mo}-{cu}"] = (r["punto"]["hogares_salen_ia"] / best if best > 0 else np.nan) if f else None
        return ratio

    def robustez(ratio):
        rob = {}
        for k, dct in ratio.items():
            vals = list(dct.values())
            rob[k] = {"siempre_factible": all(v is not None for v in vals),
                      "peor_cociente": (round(float(min(vals)), 4) if all(v is not None for v in vals) else None),
                      "robusta": all(v is not None and v >= 0.9 for v in vals)}
        return rob

    def maximin(rob):
        cand = {k: v for k, v in rob.items() if v["siempre_factible"]}
        # maximin del cociente frente a la mejor; empate: menor costo unitario, luego regla U
        return max(cand, key=lambda k: (cand[k]["peor_cociente"], -float(k.split("-")[1]), k.startswith("U")))

    R["robustez"], R["recomendada"] = {}, {}
    cocientes_principal = {}
    for c in conds:
        ratio = cocientes({(mo, cu): store[(c, mo, cu, 0.0)] for mo in models for cu in curves})
        cocientes_principal[c] = ratio
        rob = robustez(ratio)
        R["robustez"][c] = rob
        bestk = maximin(rob)
        R["recomendada"][c] = {"opcion": bestk, **rob[bestk]}
    # Desempate entre robustas por equidad (regla fijada después del registro; se reporta como tal):
    # 1) robusta; 2) mayor cobertura de mujeres en inseguridad; 3) mayor peor cociente; 4) menor costo; 5) regla U.
    R["recomendada_equidad"] = {"regla": "posterior al registro: entre las robustas, mayor cobertura de mujeres en inseguridad"}
    for c in conds:
        rob = R["robustez"][c]
        ref_rows = store[(c, "M0", "D2", 0.0)]
        cov = {f"{r['regla']}-{int(r['monto'])}": r["punto"]["cobertura_mujeres_en_ia"] for r in ref_rows}
        cost = {f"{r['regla']}-{int(r['monto'])}": r["punto"]["costo"] for r in ref_rows}
        cand = [k for k, v in rob.items() if v["robusta"]]
        pick = max(cand, key=lambda k: (round(cov[k], 6), rob[k]["peor_cociente"], -cost[k], k.startswith("U")))
        comp, detail = {}, {}
        for mo in models:
            for cu in curves:
                res = store[(c, mo, cu, 0.0)]
                a = next(r for r in res if f"{r['regla']}-{int(r['monto'])}" == pick)
                v = next(r for r in res if r["regla"] == "U" and r["monto"] == 3100.0)
                dlt = a["reps"]["hogares_salen_ia"] - v["reps"]["hogares_salen_ia"]
                comp[f"{mo}-{cu}"] = {"dif_punto": a["punto"]["hogares_salen_ia"] - v["punto"]["hogares_salen_ia"],
                                      "ic95": [float(x) for x in np.nanpercentile(dlt, [2.5, 97.5])],
                                      "prob_mayor": float(np.mean(dlt > 1e-9)), "prob_menor": float(np.mean(dlt < -1e-9))}
                detail[f"{mo}-{cu}"] = {"elegida": summarize(a), "vigente_en_condicion": summarize(v)}
        R["recomendada_equidad"][c] = {"opcion": pick, "cobertura_ia": cov[pick], "frente_a_vigente": comp, "detalle": detail}
    # Referencia de la regla de 2026: las reglas de operación incorporan por fecha de registro y según la
    # disponibilidad presupuestal. La ENIGH no observa la fecha de registro; sin información de quién se registra
    # antes, el orden se modela sin prioridad sistemática: todas las elegibles empatan y reciben la misma fracción,
    # que es el valor esperado de un orden al azar. Se evalúa fuera de las 25 opciones, sólo como referencia.
    # La regla U (por edad) reproduce el escalonamiento de 2025 (63 y 64 años primero).
    R["referencia_registro"] = {}
    for c, cv in conds.items():
        for mo in models:
            tp, td = eff_cache[mo]
            for cu in curves:
                reg = evaluate(Option("REG", 3100.0, np.zeros(len(P)), np.ones(len(P), bool), False), w=w, w_draws=w_draws,
                               tau=TAU, budget=cv["budget"], capacity=cv["capacity"], admin=0.0, theta_pt=tp, theta_dr=td,
                               curve=cu, attrs=attrs)
                s = summarize(reg)
                s["factible"] = bool(reg["punto"]["prop_q40"] >= floor - 1e-9)
                comp = {}
                for etiqueta, k in (("edad_U3100", "U-3100"), ("maximin", R["recomendada"][c]["opcion"]),
                                    ("equidad", R["recomendada_equidad"][c]["opcion"])):
                    a = next(r for r in store[(c, mo, cu, 0.0)] if f"{r['regla']}-{int(r['monto'])}" == k)
                    dlt = a["reps"]["hogares_salen_ia"] - reg["reps"]["hogares_salen_ia"]
                    comp[etiqueta] = {"opcion": k, "dif_punto": a["punto"]["hogares_salen_ia"] - reg["punto"]["hogares_salen_ia"],
                                      "ic95": [float(x) for x in np.nanpercentile(dlt, [2.5, 97.5])],
                                      "cobertura_mujeres_en_ia": a["punto"]["cobertura_mujeres_en_ia"],
                                      "prop_q40": a["punto"]["prop_q40"]}
                s["frente_a"] = comp
                R["referencia_registro"][f"{c}|{mo}|{cu}"] = s
    # Sensibilidad de M2 a la definición de ruralidad. La principal es el ámbito rural de la ENIGH (tercer dígito de
    # folioviv igual a 6); la alternativa, el tamaño de localidad (tam_loc = 4, menos de 2,500 habitantes). Se
    # reestiman los efectos urbano y rural con las mismas réplicas (misma muestra y semilla) y se repite M2.
    tl = dr["tam_loc"] == 4
    eff_tl = rdd.estimate_groups(dr, "edad", 65, YS, "pam", 7, "hogar_id", {"urbano": W & ~tl, "rural": W & tl},
                                 B=B, seed=11)
    hh = df.drop_duplicates("hogar_id")
    dif = hh["rural"] != (hh["tam_loc"] == 4).astype(int)
    S = {"definicion_principal": "ámbito rural de la ENIGH: tercer dígito de folioviv igual a 6",
         "definicion_alternativa": "tam_loc = 4 (localidades de menos de 2,500 habitantes)",
         "hogares_que_difieren": {"n": int(dif.sum()), "de": int(len(hh)), "proporcion": round(float(dif.mean()), 4),
                                  "por_anio": {int(a): int(v) for a, v in dif.groupby(hh["anio"]).sum().items()}},
         "mujeres_muestra_rdd_que_difieren": int((W & (dr["rural"] != tl.astype(int))).sum()),
         "poblacion_objetivo_que_difiere": int((P["rural"] != (P["tam_loc"] == 4).astype(int)).sum()),
         "efectos": {g: {"n": v["n"], "primera_etapa": round(v["primera_etapa"], 4),
                         **{y: {"punto": round(v[y]["efecto_local"], 5),
                                "ic95": [round(float(x), 5) for x in np.nanpercentile(v[y]["draws"], [2.5, 97.5])]}
                            for y in YS}} for g, v in eff_tl.items()}}
    tp_tl, td_tl = {}, {}
    key_tl = (P["tam_loc"] == 4).to_numpy().astype(int)
    for y in YS:
        pt = np.array([eff_tl[g][y]["efecto_local"] for g in ("urbano", "rural")])
        drw = np.column_stack([eff_tl[g][y]["draws"] for g in ("urbano", "rural")])
        spt, sdr, _ = eb_shrink(pt, drw)
        tp_tl[y], td_tl[y] = spt[key_tl], sdr[:, key_tl]
    S["por_condicion"] = {}
    for c, cv in conds.items():
        combos = {}
        for mo in models:
            for cu in curves:
                if mo != "M2":
                    combos[(mo, cu)] = store[(c, mo, cu, 0.0)]
                    continue
                combos[(mo, cu)] = [evaluate(Option(rn, a, rr["score"](P, tp_tl), rr["elig"](P), rr["foc"]), w=w,
                                             w_draws=w_draws, tau=TAU, budget=cv["budget"], capacity=cv["capacity"],
                                             admin=0.0, theta_pt=tp_tl, theta_dr=td_tl, curve=cu, attrs=attrs)
                                    for rn, rr in rules.items() for a in AMOUNTS]
        ratio = cocientes(combos)
        rob = robustez(ratio)
        difs = [abs(ratio[k][f"M2-{cu}"] - cocientes_principal[c][k][f"M2-{cu}"]) for k in ratio for cu in curves
                if ratio[k][f"M2-{cu}"] is not None and cocientes_principal[c][k][f"M2-{cu}"] is not None]
        robustas = sorted(k for k, v in rob.items() if v["robusta"])
        S["por_condicion"][c] = {
            "robustas": robustas,
            "mismas_robustas": robustas == sorted(k for k, v in R["robustez"][c].items() if v["robusta"]),
            "recomendada_maximin": maximin(rob),
            "misma_recomendada": maximin(rob) == R["recomendada"][c]["opcion"],
            "mejor_M2": {cu: max((k for k in ratio if ratio[k][f"M2-{cu}"] is not None), key=lambda k: ratio[k][f"M2-{cu}"])
                         for cu in curves},
            "max_dif_cociente_M2": round(float(max(difs)), 6) if difs else None}
    R["sensibilidad_rural_tam_loc"] = S

    # curvas de monto para la regla U en la condición R (gráfica)
    R["monto_por_curva_R"] = {cu: [{"monto": r["monto"], "punto": r["punto"]["hogares_salen_ia"],
                                     "ic95": [float(x) for x in np.nanpercentile(r["reps"]["hogares_salen_ia"], [2.5, 97.5])]}
                                    for r in store[("R", "M0", cu, 0.0)] if r["regla"] == "U"] for cu in curves}
    # costo de focalización que empata (M0 y M1, D2): cociente de hogares frente a la mejor opción U
    R["costo_focalizar_empate"] = {}
    for c in conds:
        for mo in ("M0", "M1", "M2"):
            res = store[(c, mo, "D2", 0.0)]
            bu = max(r["punto"]["hogares_salen_ia"] for r in res if r["regla"] == "U")
            for rn in ("NC", "RR", "RI", "IMP"):
                br = max(r["punto"]["hogares_salen_ia"] for r in res if r["regla"] == rn)
                R["costo_focalizar_empate"][f"{c}|{mo}|{rn}"] = round(br / bu - 1, 4) if bu > 0 else None

    # ---------------- variante estatal (100 millones) ----------------
    R["variante_estatal"] = {}
    V_BUDGET = 100e6
    P2, attrs2, w2, w2_draws = population(57, 59)
    R["variante_estatal"]["poblacion_57_59"] = {"n_muestra": int(len(P2)), "N": float(w2.sum())}
    for mo in ("M0", "M1"):
        tp, td = eff_cache[mo]
        tp2, td2 = unit_effects(mo, P2)
        for cu in curves:
            v1 = evaluate(Option("V1_complemento", 1000.0, P["riesgo"].to_numpy(float), (P["tercil"] == 3).to_numpy(), True),
                          w=w, w_draws=w_draws, tau=TAU, budget=V_BUDGET, capacity=np.inf, admin=0.0, theta_pt=tp,
                          theta_dr=td, curve=cu, attrs=attrs, base_amount=3100.0)
            v2 = evaluate(Option("V2_adelanto_57_59", 3100.0, P2["edad"].to_numpy(float), np.ones(len(P2), bool), False),
                          w=w2, w_draws=w2_draws, tau=TAU, budget=V_BUDGET, capacity=np.inf, admin=0.0, theta_pt=tp2,
                          theta_dr=td2, curve=cu, attrs=attrs2)
            for v in (v1, v2):
                s = summarize(v)
                cost = v["punto"]["costo"]
                s["hogares_por_10M"] = round(v["punto"]["hogares_salen_ia"] / cost * 1e7, 3) if cost > 0 else None
                s["hogares_por_10M_ic95"] = [round(float(x), 3) for x in np.nanpercentile(v["reps"]["hogares_salen_ia"] / np.maximum(v["reps"]["costo"], 1) * 1e7, [2.5, 97.5])]
                R["variante_estatal"][f"{v['regla']}|{mo}|{cu}"] = s
    # referencia por 10 M de la PMB vigente (U 3,100, R)
    for cu in curves:
        r = next(x for x in store[("R", "M0", cu, 0.0)] if x["regla"] == "U" and x["monto"] == 3100.0)
        R["variante_estatal"][f"PMB_vigente|M0|{cu}"] = {"hogares_por_10M": round(r["punto"]["hogares_salen_ia"] / r["punto"]["costo"] * 1e7, 3),
                                                        "hogares_por_10M_ic95": [round(float(x), 3) for x in np.nanpercentile(r["reps"]["hogares_salen_ia"] / r["reps"]["costo"] * 1e7, [2.5, 97.5])]}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and (np.isinf(o) or np.isnan(o)):
            return None
        return o
    with open(f"{OUT}/escenarios_m4.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=1)
    print("piso q40:", round(floor, 4), "| recomendadas:", R["recomendada"], f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
