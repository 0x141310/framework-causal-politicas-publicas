"""Descubrimiento causal con conocimiento previo y estabilidad por remuestreo.

Algoritmos (Sección III.A): PC -> CPDAG/PDAG, FCI -> PAG (admite confusión
latente), GES -> CPDAG (sin conocimiento previo en causal-learn; se reportan
las violaciones de capas), DirectLiNGAM -> DAG (lineal, no gaussiano).
La coincidencia entre algoritmos o remuestreos se interpreta como estabilidad,
no como demostración de causalidad.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from .graphs import CausalGraph, consensus_graph, edge_frequencies
from .spec import CausalSpec

warnings.filterwarnings("ignore")


def _matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    X = df[names].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("El descubrimiento requiere datos completos; imputar o filtrar antes")
    # estandarizar mejora la estabilidad numérica de Fisher-z y GES
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - X.mean(axis=0)) / sd


def run_pc(df, spec: CausalSpec, alpha: float = 0.01, test: str = "fisherz", names=None) -> CausalGraph:
    from causallearn.search.ConstraintBased.PC import pc

    names = names or spec.names
    cg = pc(_matrix(df, names), alpha=alpha, indep_test=test, stable=True,
            background_knowledge=spec.background_knowledge(names), show_progress=False, node_names=names)
    return CausalGraph.from_causallearn(cg.G, names, f"PC(α={alpha})")


def run_fci(df, spec: CausalSpec, alpha: float = 0.01, test: str = "fisherz", names=None) -> CausalGraph:
    from causallearn.search.ConstraintBased.FCI import fci

    names = names or spec.names
    G, _ = fci(_matrix(df, names), independence_test_method=test, alpha=alpha,
               background_knowledge=spec.background_knowledge(names), show_progress=False, node_names=names)
    return CausalGraph.from_causallearn(G, names, f"FCI(α={alpha})")


def run_ges(df, spec: CausalSpec, names=None) -> CausalGraph:
    from causallearn.search.ScoreBased.GES import ges

    names = names or spec.names
    rec = ges(_matrix(df, names), score_func="local_score_BIC", node_names=names)
    g = CausalGraph.from_causallearn(rec["G"], names, "GES(BIC)")
    g.info["violaciones_capas"] = [(a, b) for a, b in g.directed_edges() if spec.is_forbidden(a, b)]
    return g


def run_lingam(df, spec: CausalSpec, names=None, threshold: float = 0.05) -> CausalGraph:
    import lingam

    names = names or spec.names
    X = _matrix(df, names)
    model = lingam.DirectLiNGAM(prior_knowledge=spec.lingam_prior(names))
    model.fit(X)
    B = model.adjacency_matrix_.copy()
    B[np.abs(B) < threshold] = 0.0
    g = CausalGraph.from_adjacency_weights(B, names, method="DirectLiNGAM")
    g.info["B"] = B
    g.info["orden_causal"] = [names[k] for k in model.causal_order_]
    return g


def _quiet(fn: Callable) -> Callable:
    """causal-learn imprime mensajes de orientación aun con verbose=False."""
    def wrapper(*a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)
    wrapper.__name__ = fn.__name__
    return wrapper


run_pc, run_fci, run_ges, run_lingam = map(_quiet, (run_pc, run_fci, run_ges, run_lingam))
ALGORITHMS: dict[str, Callable] = {"PC": run_pc, "FCI": run_fci, "GES": run_ges, "LiNGAM": run_lingam}


def _resample(df: pd.DataFrame, rng: np.random.Generator, block: int | None) -> pd.DataFrame:
    n = len(df)
    if not block or block <= 1:
        return df.iloc[rng.integers(0, n, n)].reset_index(drop=True)
    # bootstrap de bloques móviles para series temporales
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return df.iloc[idx].reset_index(drop=True)


def bootstrap_stability(df, spec: CausalSpec, method: str = "PC", B: int = 100, block: int | None = None,
                        seed: int = 0, threshold: float = 0.6, **kw) -> dict:
    """Ejecuta un algoritmo sobre B remuestreos y resume la estabilidad de aristas."""
    rng = np.random.default_rng(seed)
    fn = ALGORITHMS[method]
    graphs = []
    for _ in range(B):
        try:
            graphs.append(fn(_resample(df, rng, block), spec, **kw))
        except Exception as exc:  # p. ej. matriz singular en un remuestreo
            warnings.warn(f"Remuestreo fallido en {method}: {exc}")
    freqs = edge_frequencies(graphs)
    return {
        "method": method,
        "B": len(graphs),
        "frecuencias": freqs,
        "consenso": consensus_graph(graphs, threshold=threshold),
        "grafos": graphs,
    }


def stability_table(stab: dict, min_adj: float = 0.2) -> pd.DataFrame:
    rows = []
    for r in stab["frecuencias"]:
        if r["adyacencia"] < min_adj:
            continue
        types = {k: v for k, v in r.items() if k not in ("a", "b", "adyacencia")}
        best = max(types.items(), key=lambda kv: kv[1])
        rows.append({
            "a": r["a"], "b": r["b"], "adyacencia": round(r["adyacencia"], 3),
            "orientacion_modal": best[0], "freq_orientacion": round(best[1], 3),
            **{k: round(v, 3) for k, v in types.items()},
        })
    return pd.DataFrame(rows)


def run_all(df, spec: CausalSpec, methods=("PC", "FCI", "GES", "LiNGAM"), alpha: float = 0.01) -> dict[str, CausalGraph]:
    out = {}
    for m in methods:
        kw = {"alpha": alpha} if m in ("PC", "FCI") else {}
        out[m] = ALGORITHMS[m](df, spec, **kw)
    return out
