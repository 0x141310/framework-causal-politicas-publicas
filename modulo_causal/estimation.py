"""Estimación de efectos identificados.

- Ajuste lineal (MCO con errores HC3): efecto bajo linealidad; base del
  análisis de sensibilidad de Cinelli y Hazlett.
- AIPW con ajuste cruzado (doble robustez; Robins et al., 1994; Chernozhukov
  et al., 2018) para intervenciones binarias, con diagnóstico de traslape.
- DML parcialmente lineal para intervenciones continuas.
- g-computación de una etapa para contrastes de dosis (p. ej. +10% de gasto),
  con remuestreo de bloques de residuos para series temporales (Anexo B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float) if cols else np.empty((len(df), 0))
    return np.column_stack([np.ones(len(df)), X])


def ols(df: pd.DataFrame, y: str, regressors: list[str]) -> dict:
    """MCO con errores robustos HC3. Devuelve coeficientes, ee, t y gl."""
    X = _design(df, regressors)
    Y = df[y].to_numpy(dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    e = Y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / np.clip(1 - h, 1e-8, None)) ** 2
    V = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.diag(V))
    se_classic = np.sqrt(np.diag(XtX_inv) * (e @ e) / (n - k))
    names = ["const"] + list(regressors)
    return {
        "coef": dict(zip(names, beta)), "se": dict(zip(names, se)),
        "se_clasico": dict(zip(names, se_classic)),
        "t": dict(zip(names, beta / se)), "t_clasico": dict(zip(names, beta / se_classic)),
        "dof": n - k, "n": n, "resid": e, "fitted": X @ beta, "X": X, "names": names,
        "r2": 1 - (e @ e) / np.sum((Y - Y.mean()) ** 2),
    }


def linear_adjustment(df: pd.DataFrame, x: str, y: str, Z: list[str]) -> dict:
    fit = ols(df, y, [x] + list(Z))
    b, se = fit["coef"][x], fit["se"][x]
    z = stats.t.ppf(0.975, fit["dof"])
    return {
        "estimador": "ajuste lineal (MCO, HC3)", "efecto": b, "ee": se,
        "ic95": (b - z * se, b + z * se), "p": 2 * (1 - stats.t.cdf(abs(b / se), fit["dof"])),
        "t_clasico": fit["t_clasico"][x], "dof": fit["dof"], "n": fit["n"], "ajuste": list(Z), "_fit": fit,
    }


def _default_models(binary_outcome: bool, flexible: bool, seed: int = 0):
    if flexible:
        kw = dict(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)  # semilla registrada
        m_out = HistGradientBoostingClassifier(**kw) if binary_outcome else HistGradientBoostingRegressor(**kw)
        m_ps = HistGradientBoostingClassifier(**kw)
    else:
        m_out = LogisticRegression(max_iter=2000) if binary_outcome else LinearRegression()
        m_ps = LogisticRegression(max_iter=2000)
    return m_out, m_ps


def aipw(df: pd.DataFrame, x: str, y: str, Z: list[str], folds: int = 5, flexible: bool = True,
         trim: float = 0.01, seed: int = 0) -> dict:
    """Efecto promedio (ATE) de una intervención binaria con AIPW y ajuste cruzado."""
    D = df[x].to_numpy(dtype=int)
    Y = df[y].to_numpy(dtype=float)
    W = df[Z].to_numpy(dtype=float) if Z else np.zeros((len(df), 1))
    binary_y = set(np.unique(Y)) <= {0.0, 1.0}
    m_out, m_ps = _default_models(binary_y, flexible, seed)
    n = len(df)
    mu0, mu1, ps = np.zeros(n), np.zeros(n), np.zeros(n)
    kf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(W, D):
        p_model = clone(m_ps).fit(W[tr], D[tr])
        ps[te] = p_model.predict_proba(W[te])[:, 1]
        for d, arr in ((0, mu0), (1, mu1)):
            idx = tr[D[tr] == d]
            mdl = clone(m_out).fit(W[idx], Y[idx])
            arr[te] = mdl.predict_proba(W[te])[:, 1] if binary_y else mdl.predict(W[te])
    keep = (ps > trim) & (ps < 1 - trim)
    psk = ps[keep]
    phi = (mu1 - mu0 + D * (Y - mu1) / np.clip(ps, 1e-6, 1) - (1 - D) * (Y - mu0) / np.clip(1 - ps, 1e-6, 1))[keep]
    ate = phi.mean()
    se = phi.std(ddof=1) / np.sqrt(len(phi))
    return {
        "estimador": f"AIPW ajuste cruzado ({'boosting' if flexible else 'paramétrico'})",
        "efecto": ate, "ee": se, "ic95": (ate - 1.96 * se, ate + 1.96 * se),
        "p": 2 * (1 - stats.norm.cdf(abs(ate / se))), "n": int(keep.sum()), "n_recortadas": int((~keep).sum()),
        "traslape": {
            "ps_tratados_p1_p99": np.percentile(ps[D == 1], [1, 99]).round(3).tolist(),
            "ps_control_p1_p99": np.percentile(ps[D == 0], [1, 99]).round(3).tolist(),
            "ps_min_max": [float(ps.min()), float(ps.max())],
        },
        "ajuste": list(Z), "_ps": ps,
    }


def aipw_att(df: pd.DataFrame, x: str, y: str, Z: list[str], folds: int = 5, flexible: bool = True,
             trim: float = 0.0, seed: int = 0) -> dict:
    """Efecto promedio en los tratados (ATT), doblemente robusto con ajuste cruzado.

    τ = Σ_i [D_i (Y_i − μ0(X_i)) − (1 − D_i) e(X_i)/(1 − e(X_i)) (Y_i − μ0(X_i))] / Σ_i D_i
    """
    D = df[x].to_numpy(dtype=int)
    Y = df[y].to_numpy(dtype=float)
    W = df[Z].to_numpy(dtype=float)
    binary_y = set(np.unique(Y)) <= {0.0, 1.0}
    m_out, m_ps = _default_models(binary_y, flexible, seed)
    n = len(df)
    mu0, ps = np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(W, D):
        ps[te] = clone(m_ps).fit(W[tr], D[tr]).predict_proba(W[te])[:, 1]
        idx = tr[D[tr] == 0]
        mdl = clone(m_out).fit(W[idx], Y[idx])
        mu0[te] = mdl.predict_proba(W[te])[:, 1] if binary_y else mdl.predict(W[te])
    keep = ps < 1 - trim if trim > 0 else np.ones(n, bool)
    ps_c = np.clip(ps, 1e-6, 1 - 1e-6)
    p1 = D[keep].mean()
    psi = (D * (Y - mu0) - (1 - D) * ps_c / (1 - ps_c) * (Y - mu0))[keep]
    att = psi.sum() / D[keep].sum()
    # varianza por función de influencia: (psi_i - D_i * att) / p1
    phi = (psi - D[keep] * att) / p1
    se = phi.std(ddof=1) / np.sqrt(keep.sum())
    w_ctrl = (ps_c / (1 - ps_c))[(D == 0) & keep]
    ess = w_ctrl.sum() ** 2 / (w_ctrl ** 2).sum()
    return {"estimador": f"AIPW-ATT ajuste cruzado ({'boosting' if flexible else 'paramétrico'})", "efecto": att,
            "ee": se, "ic95": (att - 1.96 * se, att + 1.96 * se), "p": 2 * (1 - stats.norm.cdf(abs(att / se))),
            "n": int(keep.sum()), "tamano_efectivo_controles": float(ess),
            "traslape": {"ps_tratados_p1_p50_p99": np.percentile(ps[D == 1], [1, 50, 99]).round(3).tolist(),
                         "ps_control_p50_p99_p999": np.percentile(ps[D == 0], [50, 99, 99.9]).round(4).tolist()},
            "ajuste": list(Z), "_ps": ps}


def dml_plr(df: pd.DataFrame, x: str, y: str, Z: list[str], folds: int = 5, seed: int = 0) -> dict:
    """DML parcialmente lineal: Y = θ X + g(Z) + e, X = m(Z) + v."""
    X = df[x].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    W = df[Z].to_numpy(dtype=float)
    ry, rx = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(W):
        ry[te] = Y[te] - HistGradientBoostingRegressor(max_depth=3, max_iter=300, random_state=seed).fit(W[tr], Y[tr]).predict(W[te])
        rx[te] = X[te] - HistGradientBoostingRegressor(max_depth=3, max_iter=300, random_state=seed).fit(W[tr], X[tr]).predict(W[te])
    theta = (rx @ ry) / (rx @ rx)
    psi = (ry - theta * rx) * rx
    se = np.sqrt(np.mean(psi ** 2) / np.mean(rx ** 2) ** 2 / len(df))
    return {"estimador": "DML parcialmente lineal", "efecto": theta, "ee": se,
            "ic95": (theta - 1.96 * se, theta + 1.96 * se), "p": 2 * (1 - stats.norm.cdf(abs(theta / se))),
            "n": len(df), "ajuste": list(Z)}


def gcomp_shift(df: pd.DataFrame, x: str, y: str, Z: list[str], factor: float = 1.10, B: int = 500,
                block: int | None = 4, seed: int = 0) -> dict:
    """g-computación de una etapa con modelo lineal: E[Y(factor·x) − Y(x)].

    Para series temporales usa remuestreo de bloques de residuos condicionado
    al modelo (Anexo B). Con block=None usa remuestreo i.i.d. de residuos.
    """
    fit = ols(df, y, [x] + list(Z))
    X, e = fit["X"], fit["resid"]
    j = fit["names"].index(x)
    xv = df[x].to_numpy(dtype=float)

    def contrast(beta):
        return float(np.mean(beta[j] * (factor - 1) * xv))

    est = contrast(np.array(list(fit["coef"].values())))
    rng = np.random.default_rng(seed)
    n = len(e)
    if block and block > 1:
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, (B, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    else:
        idx = rng.integers(0, n, (B, n))
    Ystar = fit["fitted"][None, :] + e[idx]  # B x n
    betas = np.linalg.pinv(X) @ Ystar.T  # k x B
    boots = betas[j] * (factor - 1) * xv.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"estimador": f"g-computación lineal, contraste ×{factor}", "efecto": est,
            "ic95": (float(lo), float(hi)), "ee": float(np.std(boots, ddof=1)), "n": n,
            "coef_x": fit["coef"][x], "coef_x_ic95": tuple(np.percentile(betas[j], [2.5, 97.5])),
            "ajuste": list(Z), "B": B, "bloque": block}
