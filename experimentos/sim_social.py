"""Pruebas controladas para un piloto social con microdatos (tipo ENIGH).

Modelo estructural (hogares con niñas, niños y jóvenes):
  rural, indígena -> escolaridad del jefe, tamaño del hogar -> ingreso per cápita
  -> beca (focalizada: más probable con menor ingreso, en zonas rurales e indígenas)
  -> asistencia escolar; además ingreso, escolaridad y ruralidad afectan la asistencia.
  U (no medido, p. ej. motivación o cercanía a la escuela) afecta beca y asistencia
  cuando 'latente' > 0.
La focalización en hogares desfavorecidos produce cancelación entre el efecto de
la beca y la confusión por ingreso: la correlación marginal beca–asistencia es
cercana a cero aunque el efecto exista.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import itertools
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, ROOT)
from modulo_causal import CausalSpec, Variable  # noqa: E402
from modulo_causal import discovery as disc  # noqa: E402
from modulo_causal import estimation as est  # noqa: E402
from modulo_causal import identification as ident  # noqa: E402
from modulo_causal import refutation as ref  # noqa: E402
from modulo_causal.pipeline import split_sample  # noqa: E402

SPEC = CausalSpec([
    Variable("rural", "contexto", 0, "binaria"), Variable("indigena", "contexto", 0, "binaria"),
    Variable("educ_jefe", "previa", 1), Variable("tam_hogar", "previa", 1, "conteo"),
    Variable("ingreso", "previa", 2), Variable("beca", "intervencion", 3, "binaria"),
    Variable("asistencia", "resultado", 4)], treatment="beca", outcome="asistencia")

TRUTH = nx.DiGraph([
    ("rural", "indigena"), ("rural", "educ_jefe"), ("indigena", "educ_jefe"), ("rural", "tam_hogar"),
    ("educ_jefe", "ingreso"), ("tam_hogar", "ingreso"), ("rural", "ingreso"), ("indigena", "ingreso"),
    ("ingreso", "beca"), ("rural", "beca"), ("indigena", "beca"), ("beca", "asistencia"),
    ("ingreso", "asistencia"), ("educ_jefe", "asistencia"), ("rural", "asistencia")])


def make(n, beta, latent, rng):
    rural = rng.binomial(1, .3, n)
    ind = rng.binomial(1, .15 + .2 * rural)
    educ = 9 - 2 * rural - 1.5 * ind + rng.normal(0, 3, n)
    tam = 4 + 0.8 * rural + rng.poisson(1.0, n)
    ing = 8 + 0.12 * educ - 0.10 * tam - 0.3 * rural - 0.2 * ind + rng.normal(0, .6, n)
    U = rng.normal(size=n)
    beca = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 - 1.2 * (ing - 8) + 0.6 * rural + 0.4 * ind + latent * U))))
    asist = 0.6 + beta * beca + 0.05 * (ing - 8) + 0.01 * educ - 0.04 * rural + 0.05 * latent * U + rng.normal(0, .15, n)
    return pd.DataFrame(dict(rural=rural, indigena=ind, educ_jefe=educ, tam_hogar=tam, ingreso=ing, beca=beca,
                             asistencia=asist))


def one_run(n, beta, latent, seed, alpha=0.01):
    rng = np.random.default_rng(seed)
    df = make(n, beta, latent, rng)
    d1, d2 = split_sample(df, "random", seed=seed)
    g = disc.run_pc(d1, SPEC, alpha=alpha)
    pc_edge = g.type("beca", "asistencia") == "-->"
    pa = sorted(g.possible_parents("beca") - {"asistencia"})
    te = ident.total_effect_test(d1, "beca", "asistencia", pa)
    has_path = nx.has_path(g.to_networkx(), "beca", "asistencia")
    restored = (not g.adjacent("beca", "asistencia")) and (not has_path) and te["p"] < alpha
    cand = g.with_edge("beca", "asistencia") if restored else g
    idf = ident.identify(cand, "beca", "asistencia", SPEC)
    f = disc.run_fci(d1, SPEC, alpha=alpha)
    fdiag = ident._pag_diagnostic(f, "beca", "asistencia")
    row = {"n": n, "beta": beta, "latente": latent, "seed": seed, "pc_detecta": pc_edge,
           "detecta_con_prueba_dirigida": pc_edge or restored or has_path, "estado": idf["estado"],
           "fci_estado": fdiag["estado"], "r_marginal": te["r_marginal"]}
    if idf["estado"] == "identificado":
        Z = idf["ajuste"]
        row["ajuste_valido_en_verdad"] = ident.is_valid_adjustment(TRUTH, "beca", "asistencia", set(Z))
        lin = est.linear_adjustment(d2, "beca", "asistencia", Z)
        aw = est.aipw(d2, "beca", "asistencia", Z, flexible=False, seed=seed)
        s = ref.sensitivity_summary(d2, "beca", "asistencia", Z)
        row.update(mco=lin["efecto"], mco_cubre=lin["ic95"][0] <= beta <= lin["ic95"][1],
                   mco_rechaza0=not (lin["ic95"][0] <= 0 <= lin["ic95"][1]),
                   aipw=aw["efecto"], aipw_cubre=aw["ic95"][0] <= beta <= aw["ic95"][1],
                   aipw_rechaza0=not (aw["ic95"][0] <= 0 <= aw["ic95"][1]), rv=s["RV_q1"])
    naive = est.linear_adjustment(d2, "beca", "asistencia", [])
    row["ingenuo"] = naive["efecto"]
    return row


def main(reps=50):
    t0 = time.time()
    rows = []
    for n, beta, latent in itertools.product((1000, 4000, 10000), (0.0, 0.06), (0.0, 1.0)):
        for r in range(reps):
            rows.append(one_run(n, beta, latent, seed=1_000_000 + 1000 * (n // 1000) + 100 * int(beta > 0) + 10 * int(latent) + r))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT + "/resultados/sim_social_replicas.csv", index=False)
    summarize(pd.read_csv(ROOT + "/resultados/sim_social_replicas.csv"))
    print(f"tiempo {time.time()-t0:.1f}s")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["beta_v"] = df["beta"]
    # conclusión final del pipeline: efecto identificado y su IC95 excluye cero
    df["concluye_efecto"] = (df["estado"] == "identificado") & (df["aipw_rechaza0"].fillna(0) > 0)
    g = df.groupby(["n", "beta", "latente"])
    summ = g.apply(lambda d: pd.Series({
        "r_marginal_media": d["r_marginal"].mean(),
        "PC_detecta": d["pc_detecta"].mean(),
        "PC+prueba_dirigida": d["detecta_con_prueba_dirigida"].mean(),
        "identificado": (d["estado"] == "identificado").mean(),
        "concluye_efecto": d["concluye_efecto"].mean(),  # potencia (β>0) o falso positivo (β=0)
        "ajuste_valido": d["ajuste_valido_en_verdad"].mean(),
        "sesgo_ingenuo": (d["ingenuo"] - d["beta_v"]).mean(),
        "sesgo_mco": (d["mco"] - d["beta_v"]).mean(),
        "sesgo_aipw": (d["aipw"] - d["beta_v"]).mean(),
        "rmse_aipw": np.sqrt(np.nanmean((d["aipw"] - d["beta_v"]) ** 2)),
        "cobertura_aipw": d["aipw_cubre"].mean(),
        "rechaza0_aipw": d["aipw_rechaza0"].mean(),
        "FCI_alerta_latente": d["fci_estado"].isin(["confusion_latente", "ambigua", "dirigida_no_visible"]).mean(),
        "RV_mediana": d["rv"].median(),
    }), include_groups=False).reset_index()
    summ.to_csv(ROOT + "/resultados/sim_social_resumen.csv", index=False)
    print(summ.round(3).to_string())
    return summ


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resumir":
        summarize(pd.read_csv(ROOT + "/resultados/sim_social_replicas.csv"))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
