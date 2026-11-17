"""Módulo 3: conjunto causal C por ventana de entrenamiento.

PC con capas (Fisher-z, α = 0.01) sobre 50 submuestras de 20,000 hogares estratificadas
por año; los indicadores de año entran como nodos de contexto exógenos. Consenso con
umbral de adyacencia 0.6. Tres ventanas: 2016-2018 (validación interna del origen 1),
2016-2020 (origen 1 y validación interna del origen 2) y 2016-2022 (origen 2).
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, ROOT)
from ingesta.hogares_m3 import INGRESO, PREVIAS, RECURSOS  # noqa: E402
from modulo_causal.discovery import run_pc  # noqa: E402
from modulo_causal.graphs import CausalGraph, consensus_graph, edge_frequencies  # noqa: E402
from modulo_causal.spec import CausalSpec, Variable  # noqa: E402

DATA = ROOT + "/datos/enigh_hogares_2016_2024.parquet"
OUT = ROOT + "/resultados"
WINDOWS = {"2016_2018": [2016, 2018], "2016_2020": [2016, 2018, 2020], "2016_2022": [2016, 2018, 2020, 2022]}
B, N_SUB, ALPHA, TAU = 50, 20_000, 0.01, 0.6
Y = "ia_mod_sev"


def spec_for(years: list[int]) -> CausalSpec:
    ctx = [f"anio_{y}" for y in years[1:]]
    V = [Variable(c, "contexto", 0, "binaria") for c in ctx]
    V += [Variable(c, "previa", 1) for c in PREVIAS] + [Variable(c, "contexto", 2) for c in RECURSOS]
    V += [Variable(c, "mediador", 3) for c in INGRESO] + [Variable(Y, "resultado", 4, "binaria")]
    return CausalSpec(V, treatment="log_ing_pc_real", outcome=Y, exogenous=ctx)


_H = None


def _init():
    global _H
    _H = pd.read_parquet(DATA)


def subsample(d: pd.DataFrame, years: list[int], rng: np.random.Generator) -> pd.DataFrame:
    sizes = d.groupby("anio").size()
    parts = []
    for y in years:
        n_y = int(round(N_SUB * sizes[y] / sizes.sum()))
        idx = rng.choice(np.flatnonzero(d["anio"].to_numpy() == y), size=n_y, replace=False)
        parts.append(d.iloc[idx])
    s = pd.concat(parts)
    for y in years[1:]:
        s[f"anio_{y}"] = (s["anio"] == y).astype(int)
    return s


def task(args):
    w, b = args
    years = WINDOWS[w]
    d = _H[_H["anio"].isin(years)]
    rng = np.random.default_rng([2024, list(WINDOWS).index(w), b])
    s = subsample(d, years, rng)
    spec = spec_for(years)
    t = time.time()
    g = run_pc(s, spec, alpha=ALPHA)
    return w, b, g.names, g.E, time.time() - t


def summarize(w: str, graphs: list[CausalGraph]) -> dict:
    cons = consensus_graph(graphs, threshold=TAU)
    freqs = edge_frequencies(graphs)
    years = WINDOWS[w]
    ctx = [f"anio_{y}" for y in years[1:]]
    adj_y = {r["a"] if r["b"] == Y else r["b"]: round(r["adyacencia"], 3) for r in freqs if Y in (r["a"], r["b"])}
    padres = sorted(p for p in cons.possible_parents(Y) if p not in ctx)
    G = cons.to_networkx()
    anc = sorted(a for a in nx.ancestors(G, Y) if a not in ctx)
    return {
        "B": len(graphs), "C": padres, "ancestros": anc, "causas_indirectas": sorted(set(anc) - set(padres)),
        "contexto_adyacente_al_resultado": sorted(p for p in cons.possible_parents(Y) if p in ctx),
        "frecuencia_adyacencia_resultado": dict(sorted(adj_y.items(), key=lambda kv: -kv[1])),
        "aristas_consenso": [list(e) for e in cons.edges()],
        "hijos_del_contexto": sorted({b for a, t, b in cons.edges() if a in ctx and b not in ctx} |
                                     {a for a, t, b in cons.edges() if b in ctx and a not in ctx}),
    }


def main():
    tasks = [(w, b) for w in WINDOWS for b in range(B)]
    res = {w: [] for w in WINDOWS}
    secs = []
    with Pool(2, initializer=_init) as pool:
        for i, (w, b, names, E, s) in enumerate(pool.imap_unordered(task, tasks)):
            res[w].append(CausalGraph(names, E, "PC"))
            secs.append(s)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(tasks)} corridas; media {np.mean(secs):.1f} s", flush=True)
    R = {"parametros": {"B": B, "n_submuestra": N_SUB, "alpha": ALPHA, "umbral": TAU}}
    for w, gs in res.items():
        R[w] = summarize(w, gs)
        print(f"\n== ventana {w}: C = {R[w]['C']}")
        print("   indirectas:", R[w]["causas_indirectas"])
        print("   contexto adyacente al resultado:", R[w]["contexto_adyacente_al_resultado"])
        print("   frecuencias:", R[w]["frecuencia_adyacencia_resultado"])
    with open(f"{OUT}/descubrimiento_m3.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
