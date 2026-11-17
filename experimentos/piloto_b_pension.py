"""Piloto B: Pensión para el Bienestar de las Personas Adultas Mayores y bienestar del hogar.

Plan de análisis:
RDD difusa en la edad (umbral 65), Quintana Roo 2022+2024, ancho ±7, núcleo triangular,
efecto fijo de año, bootstrap por hogar (999, semilla 11).
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

DATA = ROOT + "/datos/enigh_personas_2022_2024.parquet"
OUT = ROOT + "/resultados"
PRIM = "ia_mod_sev"
SEC = ["log_ing_pc", "trabaja", "ia_adultos"]
COVS = ["mujer", "indigena", "rural", "est_socio"]


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    df = pd.read_parquet(DATA)
    df["hogar_id"] = df["anio"].astype(str) + "_" + df["folioviv"] + "_" + df["foliohog"].astype(str)
    df["est_socio"] = pd.to_numeric(df["est_socio"], errors="coerce")
    q = df[df["entidad"] == "23"].copy()
    R = {"muestra": "Quintana Roo 2022+2024"}

    # perfil por edad (primera etapa y resultados), descriptivo
    R["perfil_edad_qroo"] = rdd.by_age(q, "edad", ["pam", PRIM, "log_ing_pc", "trabaja"], 55, 75).round(4).to_dict("records")

    # análisis primario y secundarios
    est = lambda d, y, **kw: rdd.estimate(d, "edad", 65, y, "pam", kw.pop("bw", 7), "hogar_id", **kw)
    R["primario"] = est(q, PRIM)
    R["secundarios"] = {y: est(q, y) for y in SEC}
    # sensibilidad fijada de antemano
    R["sensibilidad"] = {
        "bw5": est(q, PRIM, bw=5), "bw10": est(q, PRIM, bw=10), "dona": est(q, PRIM, donut=True),
        "ponderado": est(q, PRIM, weights="factor"),
        "solo_P104": rdd.estimate(q, "edad", 65, PRIM, "pam_p104", 7, "hogar_id"),
    }
    # validez: covariables predeterminadas (forma reducida), conteos por edad, placebos
    R["balance"] = {c: rdd.estimate(q, "edad", 65, c, None, 7, "hogar_id") for c in COVS}
    counts = q.groupby("edad").size()
    R["conteos_por_edad"] = counts.loc[58:72].to_dict()
    R["placebo"] = rdd.placebo(q, "edad", 65, [60, 62, 68, 70], PRIM, "hogar_id")
    # evidencia externa: muestra nacional
    R["nacional"] = {"primario": est(df, PRIM), **{y: est(df, y) for y in SEC},
                     "balance": {c: rdd.estimate(df, "edad", 65, c, None, 7, "hogar_id") for c in COVS}}
    with open(f"{OUT}/piloto_b_pension.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=1)

    def line(lbl, r):
        s = f"{lbl:28s} n={r['n']:6d}  1ª etapa {r.get('primera_etapa', np.nan):+.3f} (F={r.get('F_primera_etapa', np.nan):6.1f})"
        s += f"  forma red. {r['forma_reducida']:+.4f} [{r['fr_ic95'][0]:+.4f},{r['fr_ic95'][1]:+.4f}]"
        if "efecto_local" in r:
            s += f"  LATE {r['efecto_local']:+.4f} [{r['el_ic95'][0]:+.4f},{r['el_ic95'][1]:+.4f}]"
        return s
    print(pd.DataFrame(R["perfil_edad_qroo"]).to_string(index=False))
    print(line("QRoo primario (IA mod/sev)", R["primario"]), " media 63-64:", round(R["primario"]["media_izq_cercana"], 3))
    for y, r in R["secundarios"].items():
        print(line(f"QRoo {y}", r))
    for k, r in R["sensibilidad"].items():
        print(line(f"QRoo sens {k}", r))
    for c, r in R["balance"].items():
        print(f"Balance {c:10s} salto {r['forma_reducida']:+.4f} [{r['fr_ic95'][0]:+.4f},{r['fr_ic95'][1]:+.4f}]")
    for p in R["placebo"]:
        print(f"Placebo {p['umbral_placebo']}: {p['salto']:+.4f} [{p['ic95'][0]:+.4f},{p['ic95'][1]:+.4f}] n={p['n']}")
    print("Conteos:", R["conteos_por_edad"])
    print(line("NACIONAL primario", R["nacional"]["primario"]), " media 63-64:", round(R["nacional"]["primario"]["media_izq_cercana"], 3))
    for y in SEC:
        print(line(f"NACIONAL {y}", R["nacional"][y]))
    for c, r in R["nacional"]["balance"].items():
        print(f"Nac. balance {c:10s} salto {r['forma_reducida']:+.4f} [{r['fr_ic95'][0]:+.4f},{r['fr_ic95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
