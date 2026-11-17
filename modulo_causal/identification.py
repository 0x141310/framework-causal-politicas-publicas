"""Identificación: conjuntos de ajuste, clases de equivalencia e implicaciones comprobables.

- Criterio de ajuste generalizado (Shpitser, VanderWeele y Robins, 2010) para
  una intervención X y un resultado Y en un DAG.
- Conjunto de ajuste óptimo O(X, Y) (Henckel, Perković y Maathuis, 2022).
- Enumeración de los DAG compatibles con un PDAG/CPDAG y efectos posibles
  (idea de IDA, Maathuis et al., 2009): si todos los miembros dan el mismo
  conjunto de ajuste, el efecto es identificable con ese grafo; si no, se
  reporta el rango de efectos posibles.
- Visibilidad de X -> Y en un PAG (Zhang, 2008): condición suficiente para
  descartar un confusor latente entre X y Y.
- Implicaciones comprobables (Markov local) con corrección de Benjamini-Hochberg.
"""
from __future__ import annotations

from itertools import combinations, product

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .graphs import ARROW, CIRCLE, NONE, TAIL, CausalGraph
from .spec import CausalSpec


# ------------------------------------------------------------- DAG: ajuste
def causal_nodes(G: nx.DiGraph, x: str, y: str) -> set[str]:
    """Nodos en caminos causales propios x -> ... -> y (sin x)."""
    if y not in nx.descendants(G, x):
        return set()
    return (nx.descendants(G, x) & (nx.ancestors(G, y) | {y}))


def forbidden_set(G: nx.DiGraph, x: str, y: str) -> set[str]:
    cn = causal_nodes(G, x, y)
    forb = {x}
    for c in cn:
        forb |= nx.descendants(G, c) | {c}
    return forb


def is_valid_adjustment(G: nx.DiGraph, x: str, y: str, Z: set[str]) -> bool:
    Z = set(Z)
    if Z & forbidden_set(G, x, y):
        return False
    cn = causal_nodes(G, x, y)
    H = G.copy()
    # grafo de puerta trasera propio: quitar primeras aristas de caminos causales
    for c in list(H.successors(x)):
        if c in cn:
            H.remove_edge(x, c)
    return nx.is_d_separator(H, {x}, {y}, Z)


def optimal_adjustment_set(G: nx.DiGraph, x: str, y: str) -> set[str] | None:
    cn = causal_nodes(G, x, y)
    if not cn:
        return None  # no hay camino causal: efecto total nulo
    pa = set()
    for c in cn:
        pa |= set(G.predecessors(c))
    O = pa - forbidden_set(G, x, y)
    return O


def parent_adjustment_set(G: nx.DiGraph, x: str, y: str) -> set[str]:
    return set(G.predecessors(x))


# ----------------------------------------------------- PDAG: enumeración
def _v_structures(G: nx.DiGraph, skeleton: set[frozenset] | None = None) -> set[tuple]:
    """v-estructuras a -> c <- b con a y b no adyacentes en el esqueleto completo."""
    vs = set()
    for c in G.nodes:
        pa = sorted(G.predecessors(c))
        for a, b in combinations(pa, 2):
            adj = (frozenset((a, b)) in skeleton) if skeleton is not None else (G.has_edge(a, b) or G.has_edge(b, a))
            if not adj:
                vs.add((a, c, b))
    return vs


def enumerate_dags(g: CausalGraph, spec: CausalSpec | None = None, max_undirected: int = 14) -> list[nx.DiGraph] | None:
    """DAG compatibles con un PDAG (mismo esqueleto, sin v-estructuras nuevas,
    acíclicos y coherentes con el conocimiento previo). None si hay demasiadas
    aristas no dirigidas o si el grafo tiene marcas de PAG."""
    directed, undirected = [], []
    for a, t, b in g.edges():
        if t == "-->":
            directed.append((a, b))
        elif t == "---":
            undirected.append((a, b))
        else:
            return None  # marcas de PAG o bidireccionales: no es un PDAG
    if len(undirected) > max_undirected:
        return None
    base = nx.DiGraph()
    base.add_nodes_from(g.names)
    base.add_edges_from(directed)
    skel = g.skeleton()
    ref_v = _v_structures(base, skel)
    out = []
    for bits in product((0, 1), repeat=len(undirected)):
        D = base.copy()
        ok = True
        for (a, b), s in zip(undirected, bits):
            u, v = (a, b) if s == 0 else (b, a)
            if spec is not None and spec.is_forbidden(u, v):
                ok = False
                break
            D.add_edge(u, v)
        if not ok or not nx.is_directed_acyclic_graph(D):
            continue
        if _v_structures(D, skel) != ref_v:
            continue
        out.append(D)
    return out


def total_effect_test(df: pd.DataFrame, x: str, y: str, parents_x: list[str]) -> dict:
    """Prueba dirigida de existencia de efecto total.

    Bajo la condición de Markov y suficiencia causal, si y no desciende de x
    entonces x ⟂ y | Pa(x). Rechazar esa independencia es evidencia de que y
    desciende de x (o de confusión latente). Protege contra la eliminación de
    la arista por cancelación entre efecto y confusión (infidelidad casi
    exacta), típica de programas focalizados en población desfavorecida.
    """
    r, p = partial_corr_test(df, x, y, list(parents_x))
    r0, p0 = partial_corr_test(df, x, y, [])
    return {"dado": list(parents_x), "r_parcial": r, "p": p, "r_marginal": r0, "p_marginal": p0}


def identify(g: CausalGraph, x: str, y: str, spec: CausalSpec | None = None) -> dict:
    """Dictamen de identificación del efecto total de x sobre y con el grafo g."""
    res = {"x": x, "y": y, "metodo_grafo": g.method}
    marks = {t for _, t, _ in g.edges()}
    if marks & {"o->", "o-o", "<->", "--o", "<-o"}:
        res["tipo"] = "PAG"
        res.update(_pag_diagnostic(g, x, y))
        return res
    dags = [g.to_networkx()] if g.is_dag() else enumerate_dags(g, spec)
    if dags is None:
        res.update(estado="no_evaluable", motivo="demasiadas aristas no dirigidas para enumerar la clase")
        return res
    if not dags:
        res.update(estado="incoherente", motivo="ningún DAG compatible con el grafo y el conocimiento previo")
        return res
    sets = []
    for D in dags:
        O = optimal_adjustment_set(D, x, y)
        sets.append(None if O is None else frozenset(O))
    distinct = set(sets)
    res["n_dags"] = len(dags)
    res["conjuntos_ajuste"] = [None if s is None else sorted(s) for s in distinct]
    if distinct == {None}:
        res.update(estado="sin_camino_causal", motivo=f"en todos los DAG compatibles {x} no es ancestro de {y}")
    elif len(distinct) == 1:
        res.update(estado="identificado", ajuste=sorted(next(iter(distinct))))
    else:
        res.update(estado="no_identificado_por_clase",
                   motivo="los DAG de la clase de equivalencia implican conjuntos de ajuste distintos; se reporta el rango de efectos posibles (IDA)")
    res["_dags"] = dags
    res["_sets"] = sets
    return res


def _pag_diagnostic(g: CausalGraph, x: str, y: str) -> dict:
    t = g.type(x, y)
    out = {"arista_xy": t}
    if t is None:
        out.update(estado="sin_adyacencia", motivo="FCI no encuentra adyacencia directa; puede existir efecto indirecto")
        return out
    if t == "-->":
        # visibilidad (condición suficiente): existe v no adyacente a y con v *-> x
        vis = False
        for v in g.names:
            if v in (x, y) or g.adjacent(v, y) or not g.adjacent(v, x):
                continue
            i, j = g.idx(v), g.idx(x)
            if g.E[i, j] == ARROW:  # marca de punta en x
                vis = True
                break
        out["visible"] = vis
        out.update(estado="dirigida_visible" if vis else "dirigida_no_visible",
                   motivo=("x -> y es visible en el PAG estimado: si el PAG es correcto no hay confusor latente "
                           "entre x y y; con muestra finita una confusión débil puede pasar inadvertida" if vis else
                           "x -> y no es visible: no se puede descartar un confusor latente"))
    elif t == "<->":
        out.update(estado="confusion_latente", motivo="FCI indica un confusor latente entre x y y")
    elif t in ("o->",):
        out.update(estado="ambigua", motivo="o-> admite x -> y o un confusor latente (x <-> y)")
    else:
        out.update(estado="ambigua", motivo=f"marca {t} no permite orientar el efecto")
    return out


# ------------------------------------------- implicaciones comprobables
def partial_corr_test(df: pd.DataFrame, a: str, b: str, Z: list[str]) -> tuple[float, float]:
    """Correlación parcial y valor p de Fisher-z."""
    cols = [a, b] + list(Z)
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    n = len(X)
    if Z:
        Zm = np.column_stack([np.ones(n), X[:, 2:]])
        beta, *_ = np.linalg.lstsq(Zm, X[:, :2], rcond=None)
        R = X[:, :2] - Zm @ beta
    else:
        R = X[:, :2]
    r = np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - len(Z) - 3, 1))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(r), float(p)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def testable_implications(G: nx.DiGraph, df: pd.DataFrame, q: float = 0.05, max_tests: int = 500) -> pd.DataFrame:
    """Markov local: cada nodo es independiente de sus no descendientes dados sus padres."""
    rows = []
    for v in G.nodes:
        pa = sorted(G.predecessors(v))
        nondesc = set(G.nodes) - nx.descendants(G, v) - {v} - set(pa)
        for w in sorted(nondesc):
            if len(rows) >= max_tests:
                break
            if any(r["a"] == w and r["b"] == v for r in rows):
                continue  # el par ya se contrastó desde el otro extremo
            r, p = partial_corr_test(df, v, w, pa)
            rows.append({"a": v, "b": w, "dado": ",".join(pa), "r_parcial": r, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bh"] = benjamini_hochberg(out["p"].to_numpy(), q)
        out["rechaza"] = out["p_bh"] < q
    return out
