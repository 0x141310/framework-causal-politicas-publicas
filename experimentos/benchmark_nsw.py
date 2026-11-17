"""Validación con datos reales de respuesta conocida: NSW (LaLonde, 1986; Dehejia y Wahba, 1999).

El National Supported Work Demonstration asignó aleatoriamente un programa de
empleo; la diferencia de medias experimental es la referencia (verdad).
El framework recibe sólo datos observacionales (tratados del NSW frente a
controles de la CPS) y debe: descubrir la estructura, identificar el ajuste,
estimar el ATT, refutar y declarar sus límites (traslape, sensibilidad).
Es el análogo internacional del contraste PAE: programa de apoyo al empleo ->
ingresos laborales.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ROOT)
import causaldata  # noqa: E402

from modulo_causal import CausalSpec, Variable  # noqa: E402
from modulo_causal import discovery as disc  # noqa: E402
from modulo_causal import estimation as est  # noqa: E402
from modulo_causal import identification as ident  # noqa: E402
from modulo_causal import refutation as ref  # noqa: E402
from modulo_causal.graphs import CausalGraph  # noqa: E402

OUT = ROOT + "/resultados"


def prep(df):
    d = df.copy()
    d["u74"] = (d["re74"] == 0).astype(int)
    d["u75"] = (d["re75"] == 0).astype(int)
    for c in ("re74", "re75", "re78"):
        d[c] = d[c] / 1000.0  # miles de dólares
    return d.drop(columns=["data_id"])


def main():
    nsw = prep(causaldata.nsw_mixtape.load_pandas().data)
    cps = prep(causaldata.cps_mixtape.load_pandas().data)
    R = {}
    # --- verdad experimental
    t, c = nsw[nsw.treat == 1].re78, nsw[nsw.treat == 0].re78
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    R["experimental"] = {"efecto": diff, "ic95": (diff - 1.96 * se, diff + 1.96 * se), "n_t": len(t), "n_c": len(c)}

    # --- datos observacionales: tratados NSW + controles CPS
    obs = pd.concat([nsw[nsw.treat == 1], cps], ignore_index=True)
    R["n_obs"] = len(obs)
    spec = CausalSpec([
        Variable("age", "contexto", 0), Variable("black", "contexto", 0, "binaria"),
        Variable("hisp", "contexto", 0, "binaria"),
        Variable("educ", "previa", 1), Variable("nodegree", "previa", 1, "binaria"), Variable("marr", "previa", 1, "binaria"),
        Variable("re74", "previa", 2), Variable("u74", "previa", 2, "binaria"),
        Variable("re75", "previa", 3), Variable("u75", "previa", 3, "binaria"),
        Variable("treat", "intervencion", 4, "binaria"), Variable("re78", "resultado", 5)],
        treatment="treat", outcome="re78", exogenous=["age", "black", "hisp"],
        notes="Capas: demografía -> escolaridad y estado civil -> ingresos 1974 -> ingresos 1975 -> programa -> ingresos 1978")

    graphs = disc.run_all(obs[spec.names], spec, methods=("PC", "FCI", "GES", "LiNGAM"), alpha=0.01)
    R["arista_treat_re78"] = {m: g.type("treat", "re78") for m, g in graphs.items()}
    stab = disc.bootstrap_stability(obs[spec.names], spec, "PC", B=50, seed=1, alpha=0.01)
    tab = disc.stability_table(stab)
    tab.to_csv(f"{OUT}/nsw_estabilidad_pc.csv", index=False)
    cand: CausalGraph = stab["consenso"]
    R["grafo_candidato"] = [f"{a} {t} {b}" for a, t, b in cand.edges()]
    idf = ident.identify(cand, "treat", "re78", spec)
    R["identificacion"] = {k: v for k, v in idf.items() if not k.startswith("_")}
    fs = disc.bootstrap_stability(obs[spec.names], spec, "FCI", B=20, seed=2, alpha=0.01)
    R["fci"] = ident._pag_diagnostic(fs["consenso"], "treat", "re78")

    # conjunto de ajuste: el identificado si existe; además el conjunto completo pretratamiento
    pre = [v for v in spec.names if spec.tier(v) < spec.tier("treat")]
    Z_disc = idf.get("ajuste") or sorted(cand.possible_parents("treat"))
    R["ajuste_descubierto"] = Z_disc
    R["ajuste_pretratamiento"] = pre
    res = {}
    res["ingenuo"] = est.linear_adjustment(obs, "treat", "re78", [])
    res["MCO_descubierto"] = est.linear_adjustment(obs, "treat", "re78", Z_disc)
    res["MCO_pretratamiento"] = est.linear_adjustment(obs, "treat", "re78", pre)
    res["AIPW_ATT_descubierto"] = est.aipw_att(obs, "treat", "re78", Z_disc, flexible=True, seed=1)
    res["AIPW_ATT_pretratamiento"] = est.aipw_att(obs, "treat", "re78", pre, flexible=True, seed=1)
    # Recorte por traslape (Crump et al., 2009): controles con e(X) muy bajo no aportan
    res["AIPW_ATT_pretratamiento_recorte"] = est.aipw_att(obs, "treat", "re78", pre, flexible=True, seed=1, trim=0.1)
    R["estimaciones"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()
                             if not kk.startswith("_")}
                         for k, v in res.items()}
    truth = R["experimental"]["efecto"]
    for k, v in res.items():
        R["estimaciones"][k]["sesgo_vs_experimento"] = v["efecto"] - truth
        R["estimaciones"][k]["cubre_experimento"] = bool(v["ic95"][0] <= truth <= v["ic95"][1])

    # refutación y sensibilidad sobre el ajuste pretratamiento (MCO)
    lin = lambda d, a, b, z: est.linear_adjustment(d, a, b, z)
    R["refutaciones"] = [
        ref.placebo_treatment(obs, "treat", "re78", pre, lin, n=100, seed=3),
        ref.random_common_cause(obs, "treat", "re78", pre, lin, seed=3),
        ref.subset_stability(obs, "treat", "re78", pre, lin, seed=3),
        # control negativo: el programa (1976-77) no puede afectar ingresos de 1975
        ref.negative_control_outcome(obs, "treat", "re75", [v for v in pre if v not in ("re75", "u75")], lin),
    ]
    sens = ref.sensitivity_summary(obs, "treat", "re78", pre, benchmarks=["re75", "re74", "educ"])
    R["sensibilidad"] = {k: v for k, v in sens.items() if k != "cotas"}
    sens["cotas"].to_csv(f"{OUT}/nsw_cotas_sensibilidad.csv", index=False)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(f"{OUT}/nsw_benchmark.json", "w") as fh:
        json.dump(clean(R), fh, ensure_ascii=False, indent=2)

    print("Verdad experimental (miles USD):", round(truth, 3), np.round(R["experimental"]["ic95"], 3))
    print("Arista treat-re78:", R["arista_treat_re78"])
    print("Identificación:", R["identificacion"])
    print("FCI:", R["fci"])
    for k, v in res.items():
        print(f"{k:34s} {v['efecto']:8.3f}  IC95 [{v['ic95'][0]:.3f}, {v['ic95'][1]:.3f}]  sesgo {v['efecto']-truth:+.3f}"
              + (f"  ESS_ctrl={v.get('tamano_efectivo_controles', np.nan):.0f}" if 'tamano_efectivo_controles' in v else ""))
    print("Traslape:", res["AIPW_ATT_pretratamiento"]["traslape"])
    for r in R["refutaciones"]:
        print(r)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in R["sensibilidad"].items()})
    print(sens["cotas"].round(3).to_string())
    print(tab.head(25).to_string())


if __name__ == "__main__":
    main()
