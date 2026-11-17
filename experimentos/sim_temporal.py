"""Pruebas causales controladas del Anexo B (diseño temporal tipo PAE).

Modelo estructural trimestral (innovaciones N(0,1)):
  C_t = ρ C_{t-1} + e_C                     choque común (p. ej. turismo)
  A_t = ρ A_{t-1} + 0.5 C_t + e_A           actividad económica (ITAEE)
  S_t = ρ S_{t-1} + 0.4 U_{t-1} + 0.6 C_t + e_S   gasto (endógeno a la desocupación previa)
  U_t = ρ U_{t-1} - 0.3 A_{t-1} + β S_{t-1} + 0.5 C_{t-1} + e_U   desocupación
Condiciones: T ∈ {36, 80, 160}, ρ ∈ {0.3, 0.8}, β ∈ {0, −0.3}, confusor C
observado o omitido. 100 réplicas por condición con semillas registradas.

Estimación (g-computación lineal de una etapa): U_{t+1} ~ S_t + U_t + A_t + S_{t-1} (+ C_t),
IC95 por remuestreo de bloques de residuos (longitud 4, 500 réplicas) y HC3.
Descubrimiento: PC con capas temporales en una ventana (t-1, t, t+1).
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import itertools
import json
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, ROOT)
from modulo_causal import CausalGraph, CausalSpec, Variable  # noqa: E402
from modulo_causal import discovery as disc  # noqa: E402
from modulo_causal import estimation as est  # noqa: E402
from modulo_causal import identification as ident  # noqa: E402

BURN = 60


def simulate(T: int, rho: float, beta: float, rng) -> pd.DataFrame:
    n = T + BURN
    C, A, S, U = (np.zeros(n) for _ in range(4))
    e = rng.normal(size=(n, 4))
    for t in range(1, n):
        C[t] = rho * C[t - 1] + e[t, 0]
        A[t] = rho * A[t - 1] + 0.5 * C[t] + e[t, 1]
        S[t] = rho * S[t - 1] + 0.4 * U[t - 1] + 0.6 * C[t] + e[t, 2]
        U[t] = rho * U[t - 1] - 0.3 * A[t - 1] + beta * S[t - 1] + 0.5 * C[t - 1] + e[t, 3]
    sl = slice(BURN, BURN + T)  # T trimestres calendario; la ventana deja T-2 filas
    return pd.DataFrame({"C": C[sl], "A": A[sl], "S": S[sl], "U": U[sl]}).reset_index(drop=True)


def window(df: pd.DataFrame, observed: list[str]) -> pd.DataFrame:
    """Ventana desplegada (t-1, t, t+1): columnas X_m1, X_0, X_p1."""
    cols = {}
    for v in observed:
        cols[f"{v}_m1"] = df[v].shift(1)
        cols[f"{v}_0"] = df[v]
        cols[f"{v}_p1"] = df[v].shift(-1)
    return pd.DataFrame(cols).dropna().reset_index(drop=True)


def true_window_dag(beta: float, include_C: bool) -> nx.DiGraph:
    G = nx.DiGraph()
    for a, b in [("m1", "0"), ("0", "p1")]:
        G.add_edges_from([
            (f"C_{a}", f"C_{b}"), (f"A_{a}", f"A_{b}"), (f"S_{a}", f"S_{b}"), (f"U_{a}", f"U_{b}"),
            (f"U_{a}", f"S_{b}"), (f"A_{a}", f"U_{b}"), (f"C_{a}", f"U_{b}"),
        ])
        if beta != 0:
            G.add_edge(f"S_{a}", f"U_{b}")
    for s in ("m1", "0", "p1"):
        G.add_edges_from([(f"C_{s}", f"A_{s}"), (f"C_{s}", f"S_{s}")])
    if not include_C:
        G = G.copy()
    return G


def window_spec(observed: list[str]) -> CausalSpec:
    tier = {"m1": 0, "0": 1, "p1": 2}
    vs = [Variable(f"{v}_{s}", "intervencion" if (v, s) == ("S", "0") else
                   "resultado" if (v, s) == ("U", "p1") else "previa", tier[s])
          for s in ("m1", "0", "p1") for v in observed]
    return CausalSpec(vs, treatment="S_0", outcome="U_p1")


def run_condition(T, rho, beta, omit, reps=100, B=500, seed0=0, alpha=0.05):
    observed = ["A", "S", "U"] if omit else ["C", "A", "S", "U"]
    spec = window_spec(observed)
    adj = ["U_0", "A_0", "S_m1"] + ([] if omit else ["C_0"])
    truth_full = true_window_dag(beta, True)
    out = []
    for r in range(reps):
        seed = seed0 + r
        rng = np.random.default_rng(seed)
        W = window(simulate(T, rho, beta, rng), observed)
        # --- estimación del efecto S_t -> U_{t+1}
        lin = est.linear_adjustment(W, "S_0", "U_p1", adj)
        gb = est.gcomp_shift(W, "S_0", "U_p1", adj, factor=2.0, B=B, block=4, seed=seed)  # factor 2 => coef
        lo_b, hi_b = gb["coef_x_ic95"]
        # --- descubrimiento con capas temporales
        try:
            g = disc.run_pc(W[spec.names], spec, alpha=alpha)
            edge = g.type("S_0", "U_p1")
            pa_U = {a for a, t, b in g.edges() if b == "U_p1" and t == "-->"}
            true_pa = set(truth_full.predecessors("U_p1")) & set(spec.names)
            prec = len(pa_U & true_pa) / len(pa_U) if pa_U else np.nan
            rec = len(pa_U & true_pa) / len(true_pa) if true_pa else np.nan
        except Exception:
            edge, prec, rec = "error", np.nan, np.nan
        out.append({
            "T": T, "rho": rho, "beta": beta, "confusor_omitido": omit, "rep": r, "seed": seed,
            "n_efectivo": len(W), "beta_hat": lin["efecto"],
            "cubre_hc3": lin["ic95"][0] <= beta <= lin["ic95"][1], "ancho_hc3": lin["ic95"][1] - lin["ic95"][0],
            "cubre_boot": lo_b <= beta <= hi_b, "ancho_boot": hi_b - lo_b,
            "rechaza_cero_boot": not (lo_b <= 0 <= hi_b),
            "pc_arista_SU": edge == "-->", "pc_prec_padres_U": prec, "pc_rec_padres_U": rec,
        })
    return out


def main(reps=100, B=500):
    t0 = time.time()
    rows = []
    for T, rho, beta, omit in itertools.product((36, 80, 160), (0.3, 0.8), (0.0, -0.3), (False, True)):
        rows += run_condition(T, rho, beta, omit, reps=reps, B=B, seed0=10_000 * int(T) + int(100 * rho) + (7 if omit else 0) + (3 if beta else 0))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT + "/resultados/sim_temporal_replicas.csv", index=False)
    summ = summarize(df, reps)
    print(f"tiempo {time.time()-t0:.1f}s")
    return summ


def summarize(df: pd.DataFrame, reps: int = 100) -> pd.DataFrame:
    df = df.copy()
    df["beta_v"] = df["beta"]
    g = df.groupby(["T", "rho", "beta", "confusor_omitido"])
    summ = g.apply(lambda d: pd.Series({
        "n_efectivo": d["n_efectivo"].iloc[0],
        "sesgo": d["beta_hat"].mean() - d["beta_v"].iloc[0],
        "rmse": np.sqrt(np.mean((d["beta_hat"] - d["beta_v"].iloc[0]) ** 2)),
        "cobertura_boot": d["cubre_boot"].mean(),
        "cobertura_hc3": d["cubre_hc3"].mean(),
        "ancho_boot": d["ancho_boot"].mean(),
        "rechazo_cero": d["rechaza_cero_boot"].mean(),  # potencia si β≠0, error tipo I si β=0
        "pc_detecta_S_U": d["pc_arista_SU"].mean(),
        "pc_prec_padres_U": d["pc_prec_padres_U"].mean(),
        "pc_rec_padres_U": d["pc_rec_padres_U"].mean(),
    }), include_groups=False).reset_index()
    # error Monte Carlo de una proporción con 100 réplicas: ±1.96*sqrt(p(1-p)/100)
    summ["mc_ee_cobertura"] = np.sqrt(summ["cobertura_boot"] * (1 - summ["cobertura_boot"]) / reps)
    summ.to_csv(ROOT + "/resultados/sim_temporal_resumen.csv", index=False)
    print(summ.round(3).to_string())
    return summ


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resumir":
        summarize(pd.read_csv(ROOT + "/resultados/sim_temporal_replicas.csv"))
    else:
        main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
