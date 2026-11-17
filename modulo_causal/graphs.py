"""Representación común de grafos causales (DAG, CPDAG/PDAG y PAG).

Convención: E[i, j] es la marca en el extremo j de la arista i–j.
  TAIL = -1 (cola), ARROW = 1 (punta), CIRCLE = 2 (círculo, PAG), 0 = sin arista.
  i -> j  : E[i, j] = ARROW, E[j, i] = TAIL
  i -- j  : E[i, j] = E[j, i] = TAIL
  i <-> j : E[i, j] = E[j, i] = ARROW   (confusión latente en PAG)
  i o-> j : E[i, j] = ARROW, E[j, i] = CIRCLE
causal-learn usa la convención transpuesta (G.graph[a, b] = marca en a).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np

TAIL, ARROW, CIRCLE, NONE = -1, 1, 2, 0
_SYMBOL = {TAIL: "-", ARROW: ">", CIRCLE: "o"}


def edge_type(mi: int, mj: int) -> str | None:
    """Tipo de arista dado (marca en i, marca en j), leída como i ? j."""
    if mi == NONE and mj == NONE:
        return None
    left = {TAIL: "-", ARROW: "<", CIRCLE: "o"}[mi]
    right = {TAIL: "-", ARROW: ">", CIRCLE: "o"}[mj]
    return f"{left}-{right}"  # p. ej. '-->' se escribe '-->'


def _norm(t: str | None) -> str | None:
    if t is None:
        return None
    return {"--->": "-->"}.get(t, t)


@dataclass
class CausalGraph:
    names: list[str]
    E: np.ndarray
    method: str = ""
    info: dict = field(default_factory=dict)

    # --------------------------------------------------------- constructores
    @classmethod
    def from_causallearn(cls, G, names: list[str], method: str = "") -> "CausalGraph":
        E = np.asarray(G.graph).T.copy().astype(int)
        return cls(list(names), E, method)

    @classmethod
    def from_dag(cls, dag: nx.DiGraph, names: list[str] | None = None, method: str = "verdad") -> "CausalGraph":
        names = names or list(dag.nodes)
        idx = {n: k for k, n in enumerate(names)}
        E = np.zeros((len(names), len(names)), dtype=int)
        for a, b in dag.edges:
            if a in idx and b in idx:
                E[idx[a], idx[b]] = ARROW
                E[idx[b], idx[a]] = TAIL
        return cls(names, E, method)

    @classmethod
    def from_adjacency_weights(cls, B: np.ndarray, names: list[str], tol: float = 1e-8, method: str = "") -> "CausalGraph":
        """B[i, j] != 0 significa j -> i (convención de LiNGAM)."""
        p = len(names)
        E = np.zeros((p, p), dtype=int)
        for i in range(p):
            for j in range(p):
                if abs(B[i, j]) > tol:
                    E[j, i] = ARROW
                    E[i, j] = TAIL
        return cls(list(names), E, method)

    # ---------------------------------------------------------------- lectura
    def idx(self, n: str) -> int:
        return self.names.index(n)

    def type(self, a: str, b: str) -> str | None:
        i, j = self.idx(a), self.idx(b)
        return edge_type(self.E[j, i], self.E[i, j])

    def adjacent(self, a: str, b: str) -> bool:
        i, j = self.idx(a), self.idx(b)
        return self.E[i, j] != NONE or self.E[j, i] != NONE

    def edges(self) -> list[tuple[str, str, str]]:
        out = []
        for i, j in combinations(range(len(self.names)), 2):
            t = edge_type(self.E[j, i], self.E[i, j])
            if t is None:
                continue
            a, b = self.names[i], self.names[j]
            # normalizar para que las flechas apunten a la derecha cuando se pueda
            if t in ("<--", "<-o"):
                a, b = b, a
                t = {"<--": "-->", "<-o": "o->"}[t]
            out.append((a, t, b))
        return out

    def directed_edges(self) -> set[tuple[str, str]]:
        return {(a, b) for a, t, b in self.edges() if t == "-->"}

    def skeleton(self) -> set[frozenset]:
        return {frozenset((a, b)) for a, _, b in self.edges()}

    def is_dag(self) -> bool:
        return all(t == "-->" for _, t, _ in self.edges()) and nx.is_directed_acyclic_graph(self.to_networkx())

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.names)
        g.add_edges_from(self.directed_edges())
        return g

    def parents(self, n: str) -> set[str]:
        return {a for a, b in self.directed_edges() if b == n}

    def possible_parents(self, n: str) -> set[str]:
        """Padres dirigidos más vecinos con arista no dirigida o con círculo."""
        out = set()
        for a, t, b in self.edges():
            if b == n and t in ("-->", "---", "o->", "o-o"):
                out.add(a)
            elif a == n and t in ("---", "o-o"):
                out.add(b)
        return out

    def with_edge(self, a: str, b: str, kind: str = "-->") -> "CausalGraph":
        E = self.E.copy()
        i, j = self.idx(a), self.idx(b)
        mark = {"-": TAIL, "<": ARROW, ">": ARROW, "o": CIRCLE}
        E[j, i], E[i, j] = mark[kind[0]], mark[kind[-1]]
        return CausalGraph(list(self.names), E, self.method + f" + {a}{kind}{b}", dict(self.info))

    def __str__(self) -> str:
        lines = [f"Grafo ({self.method}): {len(self.edges())} aristas"]
        for a, t, b in self.edges():
            lines.append(f"  {a} {t} {b}")
        return "\n".join(lines)


# ----------------------------------------------------------------- métricas
def compare_to_truth(est: CausalGraph, truth: CausalGraph) -> dict:
    """Precisión y exhaustividad de adyacencias y orientaciones."""
    s_est, s_true = est.skeleton(), truth.skeleton()
    tp_adj = len(s_est & s_true)
    adj_prec = tp_adj / len(s_est) if s_est else np.nan
    adj_rec = tp_adj / len(s_true) if s_true else np.nan
    d_est, d_true = est.directed_edges(), truth.directed_edges()
    tp_or = len(d_est & d_true)
    or_prec = tp_or / len(d_est) if d_est else np.nan
    or_rec = tp_or / len(d_true) if d_true else np.nan
    # SHD simplificada: adyacencias faltantes/extra + orientaciones erróneas
    shd = len(s_est ^ s_true)
    for pair in s_est & s_true:
        a, b = tuple(pair)
        if est.type(a, b) != truth.type(a, b):
            shd += 1
    return {
        "adj_precision": adj_prec,
        "adj_recall": adj_rec,
        "orient_precision": or_prec,
        "orient_recall": or_rec,
        "shd": shd,
        "n_edges_est": len(s_est),
    }


# --------------------------------------------------------------- consenso
def edge_frequencies(graphs: list[CausalGraph]) -> list[dict]:
    """Frecuencia de cada tipo de arista a través de remuestreos o algoritmos."""
    names = graphs[0].names
    out = []
    n = len(graphs)
    for a, b in combinations(names, 2):
        c = Counter()
        for g in graphs:
            c[g.type(a, b)] += 1
        adj = 1 - c[None] / n
        if adj == 0:
            continue
        row = {"a": a, "b": b, "adyacencia": adj}
        for t, k in c.items():
            if t is not None:
                row[t] = k / n
        out.append(row)
    out.sort(key=lambda r: -r["adyacencia"])
    return out


def consensus_graph(graphs: list[CausalGraph], threshold: float = 0.6, orient_threshold: float = 0.5) -> CausalGraph:
    """Grafo con las aristas cuya frecuencia de adyacencia supera el umbral.

    La orientación se asigna si una orientación concreta aparece en al menos
    `orient_threshold` de los grafos donde la arista existe; en otro caso queda
    no dirigida (o con círculos si proviene de PAG).
    """
    names = graphs[0].names
    p = len(names)
    E = np.zeros((p, p), dtype=int)
    freqs = edge_frequencies(graphs)
    for r in freqs:
        if r["adyacencia"] < threshold:
            continue
        i, j = names.index(r["a"]), names.index(r["b"])
        adj = r["adyacencia"]
        cand = {t: v / adj for t, v in r.items() if t not in ("a", "b", "adyacencia")}
        best, share = max(cand.items(), key=lambda kv: kv[1])
        if share < orient_threshold:
            best = "---"
        left, right = best[0], best[-1]
        mark = {"-": TAIL, "<": ARROW, ">": ARROW, "o": CIRCLE}
        E[j, i] = mark[left]  # marca en a
        E[i, j] = mark[right]  # marca en b
    return CausalGraph(names, E, method=f"consenso(τ={threshold})")
