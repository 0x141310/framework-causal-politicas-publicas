"""Módulo 3: explicación SHAP y papel causal de cada variable.

TreeSHAP para árboles (escala log-odds) y SHAP lineal para la logística con
independencia: phi_j = beta_j (x_j - E[x_j]) en el espacio transformado, sumando las
bases spline de cada variable. Global: media de |phi| en los hogares de Quintana Roo
del año de prueba y en una muestra nacional de 5,000. Local: hogares de riesgo alto,
mediano y bajo de Quintana Roo (origen 2). Estabilidad: Spearman entre orígenes y
entre Quintana Roo y el país.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.insert(0, ROOT)
from ingesta.hogares_m3 import CONCURRENTES, T_SET  # noqa: E402

DATA = ROOT + "/datos/enigh_hogares_2016_2024.parquet"
RES = ROOT + "/resultados"
ORIGINS = {"o1": {"train": [2016, 2018, 2020], "test": 2022, "final": "2016_2020"},
           "o2": {"train": [2016, 2018, 2020, 2022], "test": 2024, "final": "2016_2022"}}


def phi_of(cls, m, X: pd.DataFrame, Xbg: pd.DataFrame, feats):
    if cls == "arbol":
        e = shap.TreeExplainer(m)
        phi = np.asarray(e.shap_values(X.to_numpy(float)))
        base = float(np.ravel(e.expected_value)[0])
        return pd.DataFrame(phi, columns=feats, index=X.index), base
    pre, lr = m.named_steps["pre"], m.named_steps["lr"]
    Xt, mu = pre.transform(X), pre.transform(Xbg).mean(axis=0)
    names = pre.get_feature_names_out()
    contrib = (Xt - mu) * lr.coef_[0]
    owner = []
    for n in names:
        tr, col = n.split("__", 1)
        owner.append(col.rsplit("_sp_", 1)[0] if tr == "spl" else col)
    out = pd.DataFrame(contrib, columns=names, index=X.index).T.groupby(np.array(owner)).sum().T[feats]
    base = float(lr.intercept_[0] + mu @ lr.coef_[0])
    return out, base


def roles(disc_w: dict, feats: list[str]) -> dict:
    C, anc = set(disc_w["C"]), set(disc_w["ancestros"])
    out = {}
    for f in feats:
        if f in CONCURRENTES:
            out[f] = "concurrente"
        elif f in C:
            out[f] = "causa directa"
        elif f in anc:
            out[f] = "causa indirecta"
        else:
            out[f] = "asociada sin ruta dirigida"
    return out


def main():
    H = pd.read_parquet(DATA)
    P = json.load(open(f"{RES}/prediccion_m3.json"))
    D = json.load(open(f"{RES}/descubrimiento_m3.json"))
    R = {"global": {}, "local": {}, "estabilidad": {}, "papel": {}, "masa_por_papel": {}}
    imp_store = {}
    for o, spec in ORIGINS.items():
        models = joblib.load(f"{RES}/modelos_m3_{o}.joblib")
        test = H[H.anio == spec["test"]]
        tq = test[test.entidad == "23"]
        tn = test.sample(5000, random_state=0)
        bg = H[H.anio.isin(spec["train"])].sample(20000, random_state=1)
        R["global"][o], R["papel"][o], R["masa_por_papel"][o] = {}, {}, {}
        for s in ("C", "T", "G"):
            cls = P["seleccion"][o][s]["elegido"]
            _, m, feats = models[f"{s}_{cls}"]
            rl = roles(D[spec["final"]], feats)
            R["papel"][o][s] = rl
            res = {"clase": cls}
            for zone, X in (("quintana_roo", tq), ("pais", tn)):
                phi, base = phi_of(cls, m, X[feats], bg[feats], feats)
                imp = phi.abs().mean().sort_values(ascending=False)
                res[zone] = {"base_logodds": round(base, 4), "n": int(len(X)),
                             "importancia": {k: round(float(v), 5) for k, v in imp.items()},
                             "signo_medio": {k: round(float(phi[k].mean()), 5) for k in imp.index}}
                imp_store[(o, s, zone)] = imp
                if zone == "quintana_roo":
                    tot = imp.sum()
                    mass = pd.Series({k: v for k, v in imp.items()}).groupby(pd.Series(rl)).sum() / tot
                    R["masa_por_papel"][o][s] = {k: round(float(v), 4) for k, v in mass.items()}
                    if o == "o2" and s in ("C", "T"):
                        p = 1 / (1 + np.exp(-(phi.sum(axis=1) + base)))
                        order = p.sort_values()
                        picks = {"alto": order.index[-1], "mediano": order.index[len(order) // 2], "bajo": order.index[0]}
                        R["local"].setdefault(s, {})
                        for lab, ix in picks.items():
                            row = phi.loc[ix].sort_values(key=np.abs, ascending=False).head(6)
                            R["local"][s][lab] = {"p": round(float(p.loc[ix]), 4), "base_logodds": round(base, 4),
                                                  "observado": int(X.loc[ix, "ia_mod_sev"]),
                                                  "contribuciones": [{"variable": k, "valor": round(float(X.loc[ix, k]), 3),
                                                                      "phi": round(float(v), 4)} for k, v in row.items()]}
            R["global"][o][s] = res
            print(f"[{o}] {s} ({cls}) QRoo top:", list(res["quintana_roo"]["importancia"].items())[:6], flush=True)
    for s in ("C", "T", "G"):
        a, b = imp_store[("o1", s, "quintana_roo")], imp_store[("o2", s, "quintana_roo")]
        idx = sorted(set(a.index) | set(b.index))
        r_o = spearmanr(a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)).statistic
        R["estabilidad"][s] = {"origenes_qroo": round(float(r_o), 3)}
        for o in ORIGINS:
            q, n = imp_store[(o, s, "quintana_roo")], imp_store[(o, s, "pais")]
            R["estabilidad"][s][f"qroo_vs_pais_{o}"] = round(float(spearmanr(q, n.reindex(q.index)).statistic), 3)
    with open(f"{RES}/shap_m3.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"estabilidad": R["estabilidad"], "masa_por_papel": R["masa_por_papel"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
