"""Pruebas de refutación y análisis de sensibilidad.

Refutación (al estilo de DoWhy; Sharma y Kiciman, 2020):
  - tratamiento placebo (permutación): el efecto debe desaparecer;
  - causa común aleatoria: el efecto no debe moverse;
  - submuestras: el efecto debe ser estable;
  - resultado de control negativo (si se declara uno).
Sensibilidad a confusión no medida:
  - valor de robustez RV (Cinelli y Hazlett, 2020): fuerza mínima (R² parcial
    con intervención y resultado) que un confusor debería tener para anular el
    efecto (q = 1) o su significancia (α);
  - cotas con covariables de referencia: "un confusor k veces tan fuerte como Z_j";
  - valor E (VanderWeele y Ding, 2017) para razones de riesgo.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import ols


def placebo_treatment(df, x, y, Z, estimator: Callable, n: int = 50, seed: int = 0, block: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    real = estimator(df, x, y, Z)["efecto"]
    eff = []
    for _ in range(n):
        d = df.copy()
        if block:
            # permutación por bloques para preservar autocorrelación
            k = int(np.ceil(len(d) / block))
            order = rng.permutation(k)
            idx = np.concatenate([np.arange(b * block, min((b + 1) * block, len(d))) for b in order])
            d[x] = d[x].to_numpy()[idx]
        else:
            d[x] = rng.permutation(d[x].to_numpy())
        eff.append(estimator(d, x, y, Z)["efecto"])
    eff = np.array(eff)
    p = (np.sum(np.abs(eff) >= abs(real)) + 1) / (n + 1)
    return {"prueba": "tratamiento placebo", "efecto_real": real, "placebo_media": float(eff.mean()),
            "placebo_p95_abs": float(np.percentile(np.abs(eff), 95)), "p_permutacion": float(p),
            "supera": bool(p < 0.05)}


def random_common_cause(df, x, y, Z, estimator: Callable, n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real = estimator(df, x, y, Z)["efecto"]
    eff = []
    for _ in range(n):
        d = df.copy()
        d["_ruido"] = rng.normal(size=len(d))
        eff.append(estimator(d, x, y, list(Z) + ["_ruido"])["efecto"])
    eff = np.array(eff)
    rel = float(np.max(np.abs(eff - real)) / (abs(real) + 1e-12))
    return {"prueba": "causa común aleatoria", "efecto_real": real, "media": float(eff.mean()),
            "cambio_rel_max": rel, "supera": bool(rel < 0.10)}


def subset_stability(df, x, y, Z, estimator: Callable, frac: float = 0.8, n: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real = estimator(df, x, y, Z)
    eff = []
    for _ in range(n):
        idx = rng.choice(len(df), int(frac * len(df)), replace=False)
        eff.append(estimator(df.iloc[np.sort(idx)].reset_index(drop=True), x, y, Z)["efecto"])
    eff = np.array(eff)
    same_sign = float(np.mean(np.sign(eff) == np.sign(real["efecto"])))
    return {"prueba": f"submuestras ({int(frac*100)}%)", "efecto_real": real["efecto"],
            "p5_p95": np.percentile(eff, [5, 95]).tolist(), "mismo_signo": same_sign, "supera": bool(same_sign >= 0.9)}


def negative_control_outcome(df, x, y_nc, Z, estimator: Callable) -> dict:
    r = estimator(df, x, y_nc, Z)
    lo, hi = r["ic95"]
    return {"prueba": f"control negativo ({y_nc})", "efecto": r["efecto"], "ic95": r["ic95"],
            "supera": bool(lo <= 0 <= hi)}


# ------------------------------------------------------------ sensibilidad
def robustness_value(t: float, dof: int, q: float = 1.0, alpha: float | None = None) -> float:
    """RV_q (alpha=None) o RV_{q,α} de Cinelli y Hazlett (2020)."""
    fq = q * abs(t) / np.sqrt(dof)
    if alpha is None:
        return float(0.5 * (np.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2))
    fcrit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
    fqa = fq - fcrit
    if fqa <= 0:
        return 0.0
    rv = 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    if fq > 1 / fcrit:  # fórmula extrema
        rv = (fq ** 2 - fcrit ** 2) / (1 + fq ** 2)
    return float(rv)


def partial_r2(t: float, dof: int) -> float:
    return float(t ** 2 / (t ** 2 + dof))


def ovb_bounds(df, x, y, Z: list[str], benchmarks: list[str], k: tuple[float, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Cotas de sesgo por variable omitida usando covariables observadas como referencia."""
    out_fit = ols(df, y, [x] + list(Z))
    trt_fit = ols(df, x, list(Z))
    est, se, dof = out_fit["coef"][x], out_fit["se_clasico"][x], out_fit["dof"]
    rows = []
    for bm in benchmarks:
        r2dxj = partial_r2(trt_fit["t_clasico"][bm], trt_fit["dof"])
        r2yxj = partial_r2(out_fit["t_clasico"][bm], dof)
        for kd in k:
            ky = kd
            r2dz = kd * r2dxj / (1 - r2dxj)
            if r2dz >= 1:
                continue
            r2zxj = kd * r2dxj ** 2 / ((1 - kd * r2dxj) * (1 - r2dxj))
            if r2zxj >= 1:
                continue
            r2yz = min(((np.sqrt(ky) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj)), 1.0)
            bias = se * np.sqrt(dof) * np.sqrt(r2yz * r2dz / (1 - r2dz))
            adj = est - np.sign(est) * bias
            adj_se = np.sqrt(max((1 - r2yz) / (1 - r2dz), 0)) * se * np.sqrt(dof / (dof - 1))
            zc = stats.t.ppf(0.975, dof - 1)
            rows.append({"referencia": bm, "k": kd, "R2_D~U": r2dz, "R2_Y~U": r2yz,
                         "efecto_ajustado": adj, "ic95_ajustado": (adj - zc * adj_se, adj + zc * adj_se),
                         "anula_signo": bool(np.sign(adj) != np.sign(est) or adj == 0)})
    return pd.DataFrame(rows)


def sensitivity_summary(df, x, y, Z: list[str], benchmarks: list[str] | None = None) -> dict:
    fit = ols(df, y, [x] + list(Z))
    t, dof = fit["t_clasico"][x], fit["dof"]
    res = {
        "efecto_mco": fit["coef"][x], "t": t, "gl": dof,
        "R2_parcial_Y~D|X": partial_r2(t, dof),
        "RV_q1": robustness_value(t, dof, 1.0),
        "RV_q1_a05": robustness_value(t, dof, 1.0, 0.05),
    }
    if benchmarks:
        res["cotas"] = ovb_bounds(df, x, y, Z, benchmarks)
    return res


def e_value(rr: float) -> float:
    """Valor E para una razón de riesgo (VanderWeele y Ding, 2017)."""
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))
