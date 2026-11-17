"""Módulo 3: predicción del riesgo de inseguridad alimentaria.

Origen móvil con ventana expansiva (horizonte de una encuesta):
  origen 1: entrenamiento 2016-2020 -> prueba 2022 (validación interna: 2016-2018 -> 2020)
  origen 2: entrenamiento 2016-2022 -> prueba 2024 (validación interna: 2016-2020 -> 2022)
Conjuntos C (padres descubiertos en la ventana), T (27 previas) y G (T + 5 concurrentes);
logística L2 y árboles de profundidad acotada con 8 configuraciones cada uno; referencias
de promedio histórico y persistencia por entidad x ámbito. Métricas ponderadas por el
factor de expansión, IC95 por bootstrap de UPM con réplicas compartidas.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, ROOT)
from ingesta.hogares_m3 import CONCURRENTES, G_SET, T_SET  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT + "/datos/enigh_hogares_2016_2024.parquet"
DISC = os.environ.get("M3_DISC", ROOT + "/resultados/descubrimiento_m3.json")
OUT = os.environ.get("M3_OUT", ROOT + "/resultados")
FRAC = float(os.environ.get("M3_FRAC", "1.0"))  # sólo para pruebas de humo
Y = "ia_mod_sev"
QROO = "23"
B = int(os.environ.get("M3_B", "500"))
ORIGINS = {
    "o1": {"train": [2016, 2018, 2020], "test": 2022, "inner": "2016_2018", "final": "2016_2020"},
    "o2": {"train": [2016, 2018, 2020, 2022], "test": 2024, "inner": "2016_2020", "final": "2016_2022"},
}
CONT = ["tam_loc", "est_socio", "edad_jefe", "educa_jefe", "tot_integ", "menores", "p65mas", "ocupados_pc",
        "log_ing_pc_real", "prop_ing_trabajo", "prop_transferencias", "log_gasto_pc_real", "log_gasto_alim_pc_real",
        "prop_alimentos", "prop_alim_fuera"]
LOGIT_GRID = [{"C": c, "splines": s} for s in (False, True) for c in (0.01, 0.1, 1.0, 10.0)]
TREE_GRID = [{"learning_rate": lr, "max_depth": d} for lr in (0.05, 0.1) for d in (3, 6)]
TREE_ITERS = (150, 400)


# ----------------------------------------------------------------------------- modelos
def make_logit(cfg: dict, feats: list[str]) -> Pipeline:
    cont = [f for f in feats if f in CONT]
    rest = [f for f in feats if f not in cont]
    if cfg["splines"] and cont:
        tr = [("spl", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]), cont)]
        if rest:
            tr.append(("z", StandardScaler(), rest))
        pre = ColumnTransformer(tr)
    else:
        pre = ColumnTransformer([("z", StandardScaler(), feats)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=cfg["C"], max_iter=3000))])


def make_tree(cfg: dict, iters: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"], max_iter=iters,
                                          min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
                                          random_state=7)


def wbrier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def fit_select(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str]) -> dict:
    """Ajusta la rejilla en entrenamiento interno y elige por Brier ponderado en validación."""
    X, y = tr[feats].to_numpy(float), tr[Y].to_numpy(int)
    Xv, yv, wv = va[feats].to_numpy(float), va[Y].to_numpy(int), va["factor"].to_numpy(float)
    out = {"logit": [], "arbol": []}
    for cfg in LOGIT_GRID:
        m = make_logit(cfg, feats).fit(pd.DataFrame(X, columns=feats), y)
        p = m.predict_proba(pd.DataFrame(Xv, columns=feats))[:, 1]
        out["logit"].append({"cfg": cfg, "brier_val": wbrier(p, yv, wv), "p_val": p})
    for cfg in TREE_GRID:
        m = make_tree(cfg, max(TREE_ITERS)).fit(X, y)
        for k, p in enumerate(m.staged_predict_proba(Xv), start=1):
            if k in TREE_ITERS:
                c = dict(cfg, max_iter=k)
                out["arbol"].append({"cfg": c, "brier_val": wbrier(p[:, 1], yv, wv), "p_val": p[:, 1]})
    sel = {}
    for cls, lst in out.items():
        best = min(lst, key=lambda r: r["brier_val"])
        sel[cls] = {"cfg": best["cfg"], "brier_val": best["brier_val"], "p_val": best["p_val"],
                    "rejilla": [{"cfg": r["cfg"], "brier_val": round(r["brier_val"], 6)} for r in lst]}
    sel["elegido"] = min(("logit", "arbol"), key=lambda c: sel[c]["brier_val"])
    return sel


def refit(cls: str, cfg: dict, d: pd.DataFrame, feats: list[str]):
    if cls == "logit":
        return make_logit(cfg, feats).fit(d[feats], d[Y].to_numpy(int))
    c = {k: v for k, v in cfg.items() if k != "max_iter"}
    return make_tree(c, cfg["max_iter"]).fit(d[feats].to_numpy(float), d[Y].to_numpy(int))


def predict(cls: str, m, d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = d[feats] if cls == "logit" else d[feats].to_numpy(float)
    return m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- referencias
def cell_prev(d: pd.DataFrame) -> pd.Series:
    g = d.assign(wy=d["factor"] * d[Y]).groupby(["entidad", "rural"])
    return g["wy"].sum() / g["factor"].sum()


def baselines(train: pd.DataFrame, target: pd.DataFrame) -> dict:
    years = sorted(train["anio"].unique())
    per_year = pd.concat({y: cell_prev(train[train.anio == y]) for y in years}, axis=1)
    hist = per_year.mean(axis=1)
    pers = per_year[years[-1]]
    key = pd.MultiIndex.from_frame(target[["entidad", "rural"]])
    nat_hist = float(np.mean([np.average(train.loc[train.anio == y, Y], weights=train.loc[train.anio == y, "factor"]) for y in years]))
    nat_pers = float(np.average(train.loc[train.anio == years[-1], Y], weights=train.loc[train.anio == years[-1], "factor"]))
    return {"promedio_historico": hist.reindex(key).fillna(nat_hist).to_numpy(),
            "persistencia": pers.reindex(key).fillna(nat_pers).to_numpy()}


# ----------------------------------------------------------------------------- métricas
class Evaluator:
    """Métricas ponderadas con réplicas bootstrap de UPM compartidas entre modelos."""

    def __init__(self, d: pd.DataFrame, seed: int):
        self.y = d[Y].to_numpy(float)
        self.w = d["factor"].to_numpy(float)
        self.psu, uniq = pd.factorize(d["anio"].astype(str) + "_" + d["upm"].astype(str))
        self.ent_codes, self.ents = pd.factorize(d["entidad"])
        self.qroo = (d["entidad"] == QROO).to_numpy()
        G = len(uniq)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float32)

    @staticmethod
    def _prep(p):
        """Orden y grupos de empates de p, fijos entre réplicas."""
        order = np.argsort(p, kind="mergesort")
        _, grp = np.unique(p[order], return_inverse=True)
        return order, grp, order[::-1]

    @staticmethod
    def _auc(st, y, w):
        order, grp, _ = st
        ys, ws = y[order], w[order]
        wn = np.bincount(grp, ws * (1 - ys))
        wp = np.bincount(grp, ws * ys)
        below = np.cumsum(wn) - wn
        return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))

    @staticmethod
    def _recall(st, y, w, share=0.2):
        desc = st[2]
        cw = np.cumsum(w[desc])
        sel = desc[cw <= share * cw[-1]]
        cap = np.sum(w[sel] * y[sel])
        return float(cap / np.sum(w * y)), float(cap / np.sum(w[sel]))

    def _one(self, p, y, w, ent_codes, n_ent, qmask, st=None):
        st = st if st is not None else self._prep(p)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        sw = w.sum()
        out = {"brier": float(np.sum(w * (p - y) ** 2) / sw),
               "logloss": float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / sw),
               "auc": self._auc(st, y, w), "prev_pred": float(np.sum(w * p) / sw), "prev_obs": float(np.sum(w * y) / sw)}
        out["recall20"], out["precision20"] = self._recall(st, y, w)
        if ent_codes is not None:
            sw_e = np.bincount(ent_codes, w, n_ent)
            pe = np.bincount(ent_codes, w * p, n_ent) / sw_e
            oe = np.bincount(ent_codes, w * y, n_ent) / sw_e
            out["mae_estatal_pp"] = float(np.mean(np.abs(pe - oe)) * 100)
        if qmask is not None and qmask.any():
            out["error_qroo_pp"] = float((np.sum(w[qmask] * p[qmask]) / w[qmask].sum() -
                                          np.sum(w[qmask] * y[qmask]) / w[qmask].sum()) * 100)
        return out

    def evaluate(self, p: np.ndarray, mask: np.ndarray | None = None, boot: bool = True) -> dict:
        m = np.ones_like(self.y, bool) if mask is None else mask
        y, w, pp, psu = self.y[m], self.w[m], p[m], self.psu[m]
        nat = mask is None
        ent = self.ent_codes[m] if nat else None
        q = self.qroo[m] if nat else None
        st = self._prep(pp)
        point = self._one(pp, y, w, ent, len(self.ents), q, st)
        point["cal_pendiente"] = cal_slope(pp, y, w)
        if not boot:
            return {"punto": point}
        reps = {k: np.empty(B) for k in point if k != "cal_pendiente"}
        for b in range(B):
            wb = w * self.counts[b, psu]
            r = self._one(pp, y, wb, ent, len(self.ents), q, st)
            for k in reps:
                reps[k][b] = r[k]
        return {"punto": point, "reps": reps}


def cal_slope(p, y, w) -> float:
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y.astype(int), sample_weight=w)
    return float(m.coef_[0, 0])


def ci(a):
    return [float(x) for x in np.nanpercentile(a, [2.5, 97.5])]


def summ(ev: dict) -> dict:
    o = {k: round(v, 6) for k, v in ev["punto"].items()}
    if "reps" in ev:
        o["ic95"] = {k: [round(x, 6) for x in ci(v)] for k, v in ev["reps"].items()}
    return o


def diff(a: dict, b: dict, keys=("brier", "auc", "logloss", "recall20", "mae_estatal_pp")) -> dict:
    o = {}
    for k in keys:
        if k in a["reps"] and k in b["reps"]:
            d = a["reps"][k] - b["reps"][k]
            o[k] = {"dif": round(a["punto"][k] - b["punto"][k], 6), "ic95": [round(x, 6) for x in ci(d)]}
    return o


def state_table(p, d: pd.DataFrame) -> pd.DataFrame:
    g = d.assign(wp=d["factor"] * p, wy=d["factor"] * d[Y]).groupby("entidad")
    return pd.DataFrame({"pred": g["wp"].sum() / g["factor"].sum(), "obs": g["wy"].sum() / g["factor"].sum()})


# ----------------------------------------------------------------------------- principal
def main():
    t0 = time.time()
    H = pd.read_parquet(DATA)
    if FRAC < 1:
        H = H.sample(frac=FRAC, random_state=0)
    disc = json.load(open(DISC))
    R = {"B": B, "conjuntos": {}, "seleccion": {}, "validacion": {},
         "prueba": {}, "diferencias": {}, "estados": {}, "cambio_prevalencia": {}, "S1": {}, "S2": {}, "S3": {}, "S4": {}}
    preds_out = []
    reps_store = {}
    for o, spec in ORIGINS.items():
        train = H[H.anio.isin(spec["train"])].reset_index(drop=True)
        test = H[H.anio == spec["test"]].reset_index(drop=True)
        inner_tr = train[train.anio < spec["train"][-1]].reset_index(drop=True)
        inner_va = train[train.anio == spec["train"][-1]].reset_index(drop=True)
        sets_inner = {"C": disc[spec["inner"]]["C"], "T": T_SET, "G": G_SET}
        sets_final = {"C": disc[spec["final"]]["C"], "T": T_SET, "G": G_SET}
        R["conjuntos"][o] = {"C_validacion_interna": sets_inner["C"], "C_final": sets_final["C"]}
        ev_va = Evaluator(inner_va, seed=101 + int(o[1]))
        ev_te = Evaluator(test, seed=201 + int(o[1]))
        R["seleccion"][o], R["validacion"][o], R["prueba"][o] = {}, {"pais": {}}, {"pais": {}, "quintana_roo": {}}
        evs_va, evs_te, p_test, models = {}, {}, {}, {}
        for s in ("C", "T", "G"):
            sel = fit_select(inner_tr, inner_va, sets_inner[s])
            R["seleccion"][o][s] = {cls: {"cfg": sel[cls]["cfg"], "brier_val": round(sel[cls]["brier_val"], 6),
                                          "rejilla": sel[cls]["rejilla"]} for cls in ("logit", "arbol")}
            R["seleccion"][o][s]["elegido"] = sel["elegido"]
            for cls in ("logit", "arbol"):
                name = f"{s}_{cls}"
                evs_va[name] = ev_va.evaluate(sel[cls]["p_val"])
                m = refit(cls, sel[cls]["cfg"], train, sets_final[s])
                models[name] = (cls, m, sets_final[s])
                p_test[name] = predict(cls, m, test, sets_final[s])
                evs_te[name] = ev_te.evaluate(p_test[name])
                evs_te[name + "@qroo"] = ev_te.evaluate(p_test[name], mask=ev_te.qroo)
            ch = sel["elegido"]
            for tag, src in (("", s + "_" + ch),):
                evs_va[s] = evs_va[src]
                evs_te[s] = evs_te[src]
                evs_te[s + "@qroo"] = evs_te[src + "@qroo"]
                p_test[s] = p_test[src]
            # errores estatales en validación interna (para intervalos conformales)
            st_va = state_table(sel[ch]["p_val"], inner_va)
            q = np.sort(np.abs(st_va["pred"] - st_va["obs"]).to_numpy())[int(np.ceil(0.9 * (len(st_va) + 1))) - 1]
            st_te = state_table(p_test[s], test)
            cover = ((st_te["obs"] >= st_te["pred"] - q) & (st_te["obs"] <= st_te["pred"] + q))
            R["estados"].setdefault(o, {})[s] = {
                "q90_pp": round(q * 100, 3), "cobertura": round(float(cover.mean()), 3),
                "qroo": {"pred": round(float(st_te.loc[QROO, "pred"]), 4), "obs": round(float(st_te.loc[QROO, "obs"]), 4),
                         "intervalo": [round(float(st_te.loc[QROO, "pred"] - q), 4), round(float(st_te.loc[QROO, "pred"] + q), 4)]},
                "tabla": {e: [round(float(r.pred), 4), round(float(r.obs), 4)] for e, r in st_te.iterrows()}}
            print(f"[{o}] {s}: elegido {ch} {sel[ch]['cfg']} | val Brier {sel[ch]['brier_val']:.5f} | "
                  f"prueba Brier {evs_te[s]['punto']['brier']:.5f} AUC {evs_te[s]['punto']['auc']:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        # referencias
        bl = baselines(train, test)
        for k, p in bl.items():
            p_test[k] = p
            evs_te[k] = ev_te.evaluate(p)
            evs_te[k + "@qroo"] = ev_te.evaluate(p, mask=ev_te.qroo)
            st = state_table(p, test)
            R["estados"][o][k] = {"qroo": {"pred": round(float(st.loc[QROO, "pred"]), 4), "obs": round(float(st.loc[QROO, "obs"]), 4)}}
        for name, ev in evs_va.items():
            R["validacion"][o]["pais"][name] = summ(ev)
        for name, ev in evs_te.items():
            zone = "quintana_roo" if name.endswith("@qroo") else "pais"
            R["prueba"][o][zone][name.replace("@qroo", "")] = summ(ev)
        # habilidad frente al promedio histórico: 1 - Brier / Brier de referencia
        R.setdefault("habilidad", {})[o] = {"pais": {}, "quintana_roo": {}}
        for name, ev in evs_te.items():
            zone = "quintana_roo" if name.endswith("@qroo") else "pais"
            ref = evs_te["promedio_historico" + ("@qroo" if zone == "quintana_roo" else "")]
            sk = 1 - ev["reps"]["brier"] / ref["reps"]["brier"]
            R["habilidad"][o][zone][name.replace("@qroo", "")] = {
                "punto": round(1 - ev["punto"]["brier"] / ref["punto"]["brier"], 5), "ic95": [round(x, 5) for x in ci(sk)]}
        # diferencias pareadas
        D = {}
        for a, b in (("C", "T"), ("G", "T"), ("C", "G"), ("T", "promedio_historico"), ("T", "persistencia"),
                     ("C", "promedio_historico"), ("C", "persistencia"), ("G", "persistencia"),
                     ("T_arbol", "T_logit"), ("C_arbol", "C_logit")):
            D[f"{a}-{b}"] = {"pais": diff(evs_te[a], evs_te[b]), "quintana_roo": diff(evs_te[a + "@qroo"], evs_te[b + "@qroo"])}
        for a, b in (("C", "T"), ("G", "T")):
            dv = evs_va[a]["reps"]["brier"] - evs_va[b]["reps"]["brier"]
            dt = evs_te[a]["reps"]["brier"] - evs_te[b]["reps"]["brier"]
            D[f"{a}-{b}"]["validacion_pais"] = diff(evs_va[a], evs_va[b], keys=("brier", "auc", "recall20"))
            D[f"{a}-{b}"]["cambio_brecha_brier_prueba_menos_val"] = {
                "dif": round(float(np.mean(dt) - np.mean(dv)), 6), "ic95": [round(x, 6) for x in ci(dt - dv)]}
        R["diferencias"][o] = D
        reps_store[o] = {k: v["reps"] for k, v in evs_te.items()}
        # H4: cambio de prevalencia explicado por covariables (modelo final fijo)
        last = train[train.anio == spec["train"][-1]]
        R["cambio_prevalencia"][o] = {}
        for s in ("C", "T", "G"):
            cls, m, feats = models[f"{s}_{R['seleccion'][o][s]['elegido']}"]
            p_last = predict(cls, m, last, feats)
            obs_last = float(np.average(last[Y], weights=last["factor"]))
            obs_te = float(np.average(test[Y], weights=test["factor"]))
            pred_last = float(np.average(p_last, weights=last["factor"]))
            pred_te = float(np.average(p_test[s], weights=test["factor"]))
            R["cambio_prevalencia"][o][s] = {"obs_ultimo_entrenamiento": round(obs_last, 4), "obs_prueba": round(obs_te, 4),
                                             "pred_ultimo_entrenamiento": round(pred_last, 4), "pred_prueba": round(pred_te, 4),
                                             "cambio_obs_pp": round((obs_te - obs_last) * 100, 2),
                                             "cambio_pred_pp": round((pred_te - pred_last) * 100, 2),
                                             "fraccion_explicada": round((pred_te - pred_last) / (obs_te - obs_last), 3)
                                             if abs(obs_te - obs_last) > 1e-9 else None}
        # S2: recalibración del intercepto con la prevalencia nacional observada
        R["S2"][o] = {}
        target = float(np.average(test[Y], weights=test["factor"]))
        for s in ("C", "T", "G"):
            lp = np.log(np.clip(p_test[s], 1e-6, 1 - 1e-6) / (1 - np.clip(p_test[s], 1e-6, 1 - 1e-6)))
            lo, hi = -3.0, 3.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if np.average(1 / (1 + np.exp(-(lp + mid))), weights=test["factor"]) > target:
                    hi = mid
                else:
                    lo = mid
            pr = 1 / (1 + np.exp(-(lp + (lo + hi) / 2)))
            e = ev_te.evaluate(pr, boot=False)["punto"]
            R["S2"][o][s] = {"desplazamiento_logit": round((lo + hi) / 2, 4), "brier": round(e["brier"], 6),
                             "mae_estatal_pp": round(e["mae_estatal_pp"], 3), "error_qroo_pp": round(e["error_qroo_pp"], 3),
                             "cal_pendiente": round(e["cal_pendiente"], 4)}
        # S1: horizonte de dos encuestas (sólo origen 1 -> 2024)
        if o == "o1":
            t24 = H[H.anio == 2024].reset_index(drop=True)
            ev24 = Evaluator(t24, seed=301)
            R["S1"] = {"pais": {}, "quintana_roo": {}}
            for s in ("C", "T", "G"):
                cls, m, feats = models[f"{s}_{R['seleccion'][o][s]['elegido']}"]
                p24 = predict(cls, m, t24, feats)
                R["S1"]["pais"][s] = summ(ev24.evaluate(p24))
                R["S1"]["quintana_roo"][s] = summ(ev24.evaluate(p24, mask=ev24.qroo))
        # S4: equidad en Quintana Roo (modelos elegidos)
        tq = test[test.entidad == QROO]
        R["S4"][o] = {}
        for grp, col, vals in (("ambito", "rural", {"urbano": 0, "rural": 1}), ("jefatura", "jefa_mujer", {"hombre": 0, "mujer": 1}),
                               ("lengua", "habla_indigena_hog", {"no_habla": 0, "habla": 1})):
            for lab, v in vals.items():
                msk = ev_te.qroo & (test[col].to_numpy() == v)
                for s in ("C", "T", "G"):
                    e = ev_te.evaluate(p_test[s], mask=msk)
                    R["S4"][o].setdefault(f"{grp}:{lab}", {"n": int(msk.sum())})[s] = summ(e)
        # S3: modelo local (sólo Quintana Roo) con la misma rejilla
        trq, teq = train[train.entidad == QROO].reset_index(drop=True), test[test.entidad == QROO].reset_index(drop=True)
        itr, iva = trq[trq.anio < spec["train"][-1]].reset_index(drop=True), trq[trq.anio == spec["train"][-1]].reset_index(drop=True)
        evq = Evaluator(teq, seed=401 + int(o[1]))
        R["S3"][o] = {}
        for s in ("C", "T", "G"):
            sel = fit_select(itr, iva, sets_inner[s])
            ch = sel["elegido"]
            m = refit(ch, sel[ch]["cfg"], trq, sets_final[s])
            pl = predict(ch, m, teq, sets_final[s])
            pn = p_test[s][(test.entidad == QROO).to_numpy()]
            el, en = evq.evaluate(pl), evq.evaluate(pn)
            R["S3"][o][s] = {"local": summ(el), "nacional": summ(en), "elegido_local": ch, "cfg_local": sel[ch]["cfg"],
                             "local-nacional": diff(el, en, keys=("brier", "auc", "recall20"))}
        # guardar predicciones y modelos
        keep = test[["anio", "folioviv", "foliohog", "entidad", "rural", "factor", Y]].copy()
        for k, p in p_test.items():
            keep[f"p_{k}"] = p
        keep["origen"] = o
        preds_out.append(keep)
        joblib.dump({k: v for k, v in models.items()}, f"{OUT}/modelos_m3_{o}.joblib")
        print(f"[{o}] terminado ({time.time() - t0:.0f} s)", flush=True)
    # estimando primario: promedio de los dos orígenes (país), réplicas por año independientes
    R["promedio_origenes"] = {"pais": {}}
    for name in ("C", "T", "G", "promedio_historico", "persistencia", "C_logit", "C_arbol", "T_logit", "T_arbol", "G_logit", "G_arbol"):
        avg = {k: (reps_store["o1"][name][k] + reps_store["o2"][name][k]) / 2 for k in reps_store["o1"][name]}
        pt = {k: (R["prueba"]["o1"]["pais"][name][k] + R["prueba"]["o2"]["pais"][name][k]) / 2 for k in avg}
        R["promedio_origenes"]["pais"][name] = {"punto": {k: round(v, 6) for k, v in pt.items()},
                                                "ic95": {k: [round(x, 6) for x in ci(v)] for k, v in avg.items()}}
        reps_store.setdefault("prom", {})[name] = avg
    R["promedio_origenes"]["diferencias"] = {}
    for a, b in (("C", "T"), ("G", "T"), ("T", "promedio_historico"), ("T", "persistencia"), ("C", "persistencia")):
        R["promedio_origenes"]["diferencias"][f"{a}-{b}"] = {
            k: {"dif": round(R["promedio_origenes"]["pais"][a]["punto"][k] - R["promedio_origenes"]["pais"][b]["punto"][k], 6),
                "ic95": [round(x, 6) for x in ci(reps_store["prom"][a][k] - reps_store["prom"][b][k])]}
            for k in ("brier", "auc", "logloss", "recall20", "mae_estatal_pp")}
    pd.concat(preds_out).to_parquet(f"{OUT}/predicciones_m3.parquet", index=False)
    with open(f"{OUT}/prediccion_m3.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(f"listo ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
