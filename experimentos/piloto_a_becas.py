"""Piloto A: beca Benito Juárez de educación básica (hogar) → asistencia escolar de 6 a 14 años.

Plan de análisis: módulo 2 completo con semilla 2026,
descubrimiento y estimación en mitades distintas, AIPW (ATE), refutaciones, control
negativo (dificultad para ver) y sensibilidad de Cinelli y Hazlett.
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
from modulo_causal import CausalSpec, Variable, pipeline  # noqa: E402

DATA = ROOT + "/datos/enigh_personas_2022_2024.parquet"
OUT = ROOT + "/resultados"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if hasattr(o, "edges") and hasattr(o, "names"):
        return [f"{a} {t} {b}" for a, t, b in o.edges()]
    return o


def main():
    df = pd.read_parquet(DATA)
    q = df[(df["entidad"] == "23") & (df["edad"].between(6, 14))].copy()
    for c in ("est_socio", "tam_loc", "educa_jefe"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["anio24"] = (q["anio"] == 2024).astype(int)
    q["dif_ver"] = q["disc_ver"].isin([1, 2, 3]).astype(int)  # alguna dificultad para ver
    q["beca"] = q["hog_beca_basica"]
    spec = CausalSpec([
        Variable("edad", "contexto", 0), Variable("mujer", "contexto", 0, "binaria"),
        Variable("indigena", "contexto", 0, "binaria"), Variable("rural", "contexto", 0, "binaria"),
        Variable("est_socio", "contexto", 0, "ordinal"), Variable("tam_loc", "contexto", 0, "ordinal"),
        Variable("anio24", "contexto", 0, "binaria"),
        Variable("educa_jefe", "previa", 1, "ordinal"), Variable("tot_integ", "previa", 1, "conteo"),
        Variable("menores", "previa", 1, "conteo"),
        Variable("log_ing_lab_pc", "previa", 2),
        Variable("beca", "intervencion", 3, "binaria"), Variable("asiste", "resultado", 4, "binaria")],
        treatment="beca", outcome="asiste", exogenous=["edad", "mujer", "anio24"])
    R = {"n": int(len(q))}
    # diagnóstico de positividad (pre-especificado)
    tab = pd.crosstab(q["beca"], q["asiste"], margins=True)
    R["positividad"] = {
        "tabla_beca_x_asiste": tab.to_dict(),
        "no_asisten_con_beca": int(((q.beca == 1) & (q.asiste == 0)).sum()),
        "no_asisten_sin_beca": int(((q.beca == 0) & (q.asiste == 0)).sum()),
        "tasa_asistencia_con_beca": float(q.loc[q.beca == 1, "asiste"].mean()),
        "tasa_asistencia_sin_beca": float(q.loc[q.beca == 0, "asiste"].mean()),
    }
    print("Positividad:", {k: v for k, v in R["positividad"].items() if k != "tabla_beca_x_asiste"})
    res = pipeline.run_module(q, spec, alpha=0.01, B=50, split="random", seed=2026, benchmarks=["log_ing_lab_pc", "educa_jefe"],
                              negative_controls=["dif_ver"], n_placebo=100, flexible=True)
    print(pipeline.print_report(res, spec))
    print(res["estabilidad_PC"].head(30).to_string())
    if res.get("sensibilidad") and "cotas" in res["sensibilidad"]:
        print(res["sensibilidad"]["cotas"].round(4).to_string())
    R["modulo"] = res
    # magnitud compatible (fijada de antemano: AIPW ATE) con todas las variables pretratamiento en la mitad de estimación
    from modulo_causal import estimation as est, refutation as ref
    from modulo_causal.pipeline import split_sample
    data = q[spec.names + ["dif_ver"]].dropna().reset_index(drop=True)
    _, d_est = split_sample(data, "random", seed=2026)
    pre = [v for v in spec.names if spec.tier(v) < spec.tier("beca")]
    mag = {"aipw": est.aipw(d_est, "beca", "asiste", pre, flexible=True, seed=2026),
           "mco": est.linear_adjustment(d_est, "beca", "asiste", pre),
           "ingenuo": est.linear_adjustment(d_est, "beca", "asiste", []),
           "control_negativo_dif_ver": ref.negative_control_outcome(d_est, "beca", "dif_ver", pre,
                                                                   lambda d, a, b, z: est.linear_adjustment(d, a, b, z)),
           "sensibilidad": {k: v for k, v in ref.sensitivity_summary(d_est, "beca", "asiste", pre).items() if k != "cotas"}}
    R["magnitud_pretratamiento"] = mag
    for k in ("aipw", "mco", "ingenuo"):
        r = mag[k]
        print(f"Magnitud {k:8s}: {r['efecto']:+.4f} IC95 [{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}] n={r['n']}")
    print("Control negativo:", {k: v for k, v in mag["control_negativo_dif_ver"].items()})
    print("Sensibilidad:", mag["sensibilidad"])
    # diagnóstico secundario: 15-17 con beca de media superior (asociación mecánica)
    j = df[(df["entidad"] == "23") & (df["edad"].between(15, 17))]
    R["ems_15_17"] = {
        "n": int(len(j)), "con_beca_ems": int(j["beca_ems"].sum()),
        "asistencia_con_beca": float(j.loc[j.beca_ems == 1, "asiste"].mean()),
        "asistencia_sin_beca": float(j.loc[j.beca_ems == 0, "asiste"].mean()),
        "no_asisten_con_beca": int(((j.beca_ems == 1) & (j.asiste == 0)).sum()),
    }
    print("Media superior 15-17:", R["ems_15_17"])
    with open(f"{OUT}/piloto_a_becas.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
