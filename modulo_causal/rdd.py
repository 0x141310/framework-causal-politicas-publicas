"""Regresión discontinua (nítida o difusa) con variable de asignación discreta.

Identificación (Hahn, Todd y Van der Klaauw, 2001): si los resultados potenciales
son continuos en el umbral, el salto en el resultado dividido entre el salto en
la probabilidad de recibir la intervención identifica el efecto local para
quienes la reciben por cruzar el umbral (cumplidores).
Con edad en años enteros hay pocos puntos de soporte (Lee y Card, 2008); por eso
se reportan sensibilidad de ancho de banda, dona y umbrales placebo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _design(x: np.ndarray, c: float, fe: np.ndarray | None):
    r = x - c
    D = (x >= c).astype(float)
    cols = [np.ones_like(r), D, r, D * r]
    if fe is not None and fe.shape[1]:
        cols += [fe[:, k] for k in range(fe.shape[1])]
    return np.column_stack(cols)


def _wls_jump(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    XtW = X.T * w
    try:
        beta = np.linalg.solve(XtW @ X, XtW @ y)  # ecuaciones normales (k pequeño)
    except np.linalg.LinAlgError:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return float(beta[1])


def prepare(df: pd.DataFrame, running: str, cutoff: float, bw: float, donut: bool = False,
            weights: str | None = None, fe_col: str | None = "anio") -> pd.DataFrame:
    d = df[(df[running] >= cutoff - bw) & (df[running] <= cutoff + bw)].copy()
    if donut:
        d = d[d[running] != cutoff]
    h = bw + 1.0  # los extremos del ancho de banda conservan peso positivo
    d["_k"] = np.clip(1 - np.abs(d[running] - cutoff) / h, 0, None)
    if weights:
        d["_k"] = d["_k"] * d[weights]
    d["_fe"] = 0.0
    return d


def _fe_matrix(d: pd.DataFrame, fe_col: str | None) -> np.ndarray | None:
    if not fe_col or d[fe_col].nunique() < 2:
        return None
    dums = pd.get_dummies(d[fe_col], drop_first=True, dtype=float)
    return dums.to_numpy()


def estimate(df: pd.DataFrame, running: str, cutoff: float, outcome: str, treat: str | None, bw: float,
             cluster: str, B: int = 999, seed: int = 11, donut: bool = False, weights: str | None = None,
             fe_col: str | None = "anio") -> dict:
    d = prepare(df, running, cutoff, bw, donut, weights, fe_col)
    d = d.dropna(subset=[outcome] + ([treat] if treat else []))
    x = d[running].to_numpy(float)
    fe = _fe_matrix(d, fe_col)
    X = _design(x, cutoff, fe)
    w = d["_k"].to_numpy(float)
    y = d[outcome].to_numpy(float)
    rf = _wls_jump(X, y, w)
    fs = _wls_jump(X, d[treat].to_numpy(float), w) if treat else np.nan
    late = rf / fs if treat else rf
    # bootstrap de conglomerados (hogares): remuestrear G hogares con reemplazo equivale
    # a ponderar cada hogar por su conteo multinomial
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(d[cluster])
    G = len(uniq)
    t_arr = d[treat].to_numpy(float) if treat else None
    rfs, fss, lates = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G))
        wb = w * cnt[codes]
        rfs[b] = _wls_jump(X, y, wb)
        if treat:
            fss[b] = _wls_jump(X, t_arr, wb)
            lates[b] = rfs[b] / fss[b] if abs(fss[b]) > 1e-9 else np.nan
    out = {
        "outcome": outcome, "bw": bw, "donut": donut, "ponderado": bool(weights), "n": int(len(d)), "hogares": G,
        "n_izq": int((x < cutoff).sum()), "n_der": int((x >= cutoff).sum()),
        "forma_reducida": rf, "fr_ic95": tuple(np.nanpercentile(rfs, [2.5, 97.5])), "fr_ee": float(np.nanstd(rfs, ddof=1)),
        "media_izq_cercana": float(d.loc[(x < cutoff) & (x >= cutoff - 2), outcome].mean()),
    }
    if treat:
        se_fs = float(np.nanstd(fss, ddof=1))
        out.update({
            "primera_etapa": fs, "pe_ic95": tuple(np.nanpercentile(fss, [2.5, 97.5])), "pe_ee": se_fs,
            "F_primera_etapa": (fs / se_fs) ** 2 if se_fs > 0 else np.nan,
            "efecto_local": late, "el_ic95": tuple(np.nanpercentile(lates, [2.5, 97.5])),
            "el_ee": float(np.nanstd(lates, ddof=1)),
        })
    return out


def estimate_groups(df: pd.DataFrame, running: str, cutoff: float, outcomes: list[str], treat: str, bw: float,
                    cluster: str, groups: dict[str, pd.Series], B: int = 999, seed: int = 11,
                    fe_col: str | None = "anio", donut: bool = False) -> dict:
    """Efectos locales por subgrupo y resultado con réplicas bootstrap compartidas.

    Todos los subgrupos usan los mismos conteos multinomiales por hogar en cada
    réplica, de modo que las diferencias entre subgrupos y las probabilidades
    de que una opción supere a otra conservan su correlación.
    """
    d0 = prepare(df, running, cutoff, bw, donut, None, fe_col)
    codes_all, uniq = pd.factorize(d0[cluster])
    G = len(uniq)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=B)  # B x G
    out = {}
    for gname, mask in groups.items():
        m = mask.reindex(d0.index).fillna(False).to_numpy(bool)
        dg = d0[m]
        codes = codes_all[m]
        x = dg[running].to_numpy(float)
        fe = _fe_matrix(dg, fe_col)
        X = _design(x, cutoff, fe)
        w = dg["_k"].to_numpy(float)
        t = dg[treat].to_numpy(float)
        fs = _wls_jump(X, t, w)
        fs_d = np.array([_wls_jump(X, t, w * counts[b, codes]) for b in range(B)])
        res = {"n": int(m.sum()), "primera_etapa": fs, "pe_ic95": tuple(np.percentile(fs_d, [2.5, 97.5])),
               "fs_draws": fs_d}
        for y in outcomes:
            yy = dg[y].to_numpy(float)
            ok = ~np.isnan(yy)
            rf = _wls_jump(X[ok], yy[ok], w[ok])
            rf_d = np.array([_wls_jump(X[ok], yy[ok], (w * counts[b, codes])[ok]) for b in range(B)])
            late_d = rf_d / fs_d
            res[y] = {"efecto_local": rf / fs, "el_ic95": tuple(np.nanpercentile(late_d, [2.5, 97.5])),
                      "el_ee": float(np.nanstd(late_d, ddof=1)), "draws": late_d,
                      "forma_reducida": rf, "fr_ic95": tuple(np.nanpercentile(rf_d, [2.5, 97.5])), "rf_draws": rf_d}
        out[gname] = res
    return out


def by_age(df: pd.DataFrame, running: str, cols: list[str], lo: float, hi: float) -> pd.DataFrame:
    d = df[(df[running] >= lo) & (df[running] <= hi)]
    g = d.groupby(running)
    out = g[cols].mean()
    out["n"] = g.size()
    return out.reset_index()


def placebo(df: pd.DataFrame, running: str, true_cut: float, fake_cuts: list[float], outcome: str, cluster: str,
            bw: float = 4, B: int = 499, seed: int = 11) -> list[dict]:
    res = []
    for fc in fake_cuts:
        side = df[df[running] < true_cut] if fc < true_cut else df[df[running] >= true_cut]
        r = estimate(side, running, fc, outcome, None, bw, cluster, B=B, seed=seed)
        res.append({"umbral_placebo": fc, "salto": r["forma_reducida"], "ic95": r["fr_ic95"], "n": r["n"]})
    return res
