"""Módulo 4: comparador de escenarios de una etapa con restricciones (Sección IV.D del protocolo).

Cada opción es una regla de prioridad más un monto. La asignación sigue la regla hasta
agotar presupuesto, capacidad o elegibles; los empates reciben la misma fracción. Los
resultados se evalúan en réplicas que combinan la incertidumbre del efecto (réplicas
bootstrap del estimador causal) y la de la población (bootstrap de UPM), de modo que cada
opción sale con valor esperado, IC95 y probabilidad de ser la mejor entre las factibles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def allocate(score: np.ndarray, weight: np.ndarray, eligible: np.ndarray, limit: float) -> np.ndarray:
    """Fracción incorporada de cada unidad siguiendo `score` de mayor a menor.

    `weight` es el número de personas que representa cada unidad (factor × incorporación);
    `limit` es el máximo de personas que caben (presupuesto o capacidad). Las unidades con
    el mismo puntaje forman un bloque que recibe la misma fracción.
    """
    frac = np.zeros(len(weight))
    idx = np.flatnonzero(eligible & (weight > 0))
    if len(idx) == 0 or limit <= 0:
        return frac
    s = score[idx]
    order = np.argsort(-s, kind="mergesort")
    idx, s = idx[order], s[order]
    remaining = float(limit)
    i = 0
    while i < len(idx) and remaining > 1e-9:
        j = i + 1
        while j < len(idx) and s[j] == s[i]:
            j += 1
        grp = idx[i:j]
        wsum = float(weight[grp].sum())
        f = min(1.0, remaining / wsum)
        frac[grp] = f
        remaining -= f * wsum
        i = j
    return frac


def dose_factor(curve: str, amount: float, ref: float = 6000.0) -> float:
    """Multiplicador del efecto según el monto (curvas D1, D2 y D3)."""
    if curve == "D1":
        return amount / ref
    if curve == "D2":
        return float(np.sqrt(amount / ref))
    if curve == "D3":
        return 1.0
    raise ValueError(curve)


def eb_shrink(point: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encogimiento empírico de Bayes de G efectos de subgrupo hacia su media ponderada.

    point: (G,), draws: (B, G). La varianza entre grupos se estima por momentos y se
    mantiene fija al encoger cada réplica, con los mismos pesos.
    """
    se2 = np.nanvar(draws, axis=0, ddof=1)
    w = 1 / se2
    center = float(np.sum(w * point) / np.sum(w))
    G = len(point)
    tau2 = max(0.0, float(np.sum((point - center) ** 2) / (G - 1) - np.mean(se2)))
    keep = tau2 / (tau2 + se2)  # 1 - B_g
    center_d = (draws * w).sum(axis=1, keepdims=True) / w.sum()
    pt = center + keep * (point - center)
    dr = center_d + keep * (draws - center_d)
    return pt, dr, {"tau2": tau2, "centro": center, "fraccion_conservada": keep.tolist()}


@dataclass
class Option:
    regla: str
    monto: float
    score: np.ndarray
    eligible: np.ndarray
    focalizada: bool = False
    nota: str = ""
    extra: dict = field(default_factory=dict)


def evaluate(opt: Option, *, w: np.ndarray, w_draws: np.ndarray, tau: float, budget: float, capacity: float,
             admin: float, theta_pt: dict, theta_dr: dict, curve: str, attrs: dict, base_amount: float = 0.0,
             ref: float = 6000.0) -> dict:
    """Evalúa una opción en el punto y en cada réplica.

    theta_pt[y]: efecto por unidad (N,) en la escala del resultado y; theta_dr[y]: (B, N).
    attrs: indicadores por unidad para las métricas de equidad (0/1).
    base_amount > 0 indica un complemento sobre un monto existente (efecto incremental).
    """
    unit_cost = opt.monto * 6 * (1 + (admin if opt.focalizada else 0.0))
    if base_amount > 0:
        mult = dose_factor(curve, base_amount + opt.monto, ref) - dose_factor(curve, base_amount, ref)
    else:
        mult = dose_factor(curve, opt.monto, ref)

    def one(weights, th):
        we = weights * tau
        limit = min(budget / unit_cost, capacity)
        frac = allocate(opt.score, we, opt.eligible, limit)
        nb = we * frac
        n = nb.sum()
        out = {"beneficiarias": n, "costo": n * unit_cost,
               "hogares_salen_ia": float(np.sum(nb * -th["ia_mod_sev"] * mult)),
               "reactivos_menos": float(np.sum(nb * -th["ia_adultos"] * mult)),
               "ingreso_pct": float(np.sum(nb * np.expm1(th["log_ing_pc"] * mult)) / n) if n > 0 else 0.0}
        for k, a in attrs.items():
            out[f"prop_{k}"] = float(np.sum(nb * a) / n) if n > 0 else np.nan
        tot_ia = float(np.sum(weights * attrs["en_ia"]))
        out["cobertura_mujeres_en_ia"] = float(np.sum(weights * frac * tau * attrs["en_ia"]) / tot_ia) if tot_ia > 0 else np.nan
        return out

    point = one(w, theta_pt)
    B = w_draws.shape[0]
    reps = {k: np.empty(B) for k in ("hogares_salen_ia", "reactivos_menos", "ingreso_pct", "costo", "beneficiarias")}
    for b in range(B):
        r = one(w_draws[b], {y: theta_dr[y][b] for y in theta_dr})
        for k in reps:
            reps[k][b] = r[k]
    return {"regla": opt.regla, "monto": opt.monto, "focalizada": opt.focalizada, "nota": opt.nota,
            "costo_unitario_anual": unit_cost, "multiplicador_dosis": mult, "punto": point, "reps": reps}


def summarize(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k not in ("reps", "punto")}
    out["punto"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["punto"].items()}
    out["ic95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in res["reps"].items()}
    return out


def prob_best(results: list[dict], feasible: list[bool], key: str = "hogares_salen_ia",
              tol: float = 1e-9) -> list[float]:
    """Proporción de réplicas en que cada opción factible logra el mayor valor de `key`.

    Si varias opciones empatan en una réplica (diferencia relativa menor que `tol`), la réplica se reparte entre
    ellas por partes iguales; así, dos opciones que eligen a los mismos hogares reciben la misma probabilidad.
    """
    idx = [i for i, f in enumerate(feasible) if f]
    if not idx:
        return [0.0] * len(results)
    M = np.column_stack([results[i]["reps"][key] for i in idx])
    mx = M.max(axis=1, keepdims=True)
    gana = M >= mx - tol * np.maximum(1.0, np.abs(mx))
    p = (gana / gana.sum(axis=1, keepdims=True)).mean(axis=0)
    out = [0.0] * len(results)
    for k, i in enumerate(idx):
        out[i] = float(p[k])
    return out


def pareto(points: list[tuple[float, float]]) -> list[bool]:
    """Opciones no dominadas al maximizar las dos coordenadas."""
    flags = []
    for i, (a, b) in enumerate(points):
        dom = any((c >= a and d >= b) and (c > a or d > b) for j, (c, d) in enumerate(points) if j != i)
        flags.append(not dom)
    return flags
