"""Exploratorio: ¿qué variables mueven la prevalencia predicha?

Con el modelo final de cada origen, la diferencia entre años del promedio ponderado de
los valores SHAP de cada variable (escala log-odds) descompone el cambio del logit medio
predicho entre el último año de entrenamiento y el año de prueba.
"""

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))

import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, ROOT)
DATA = ROOT + "/datos/enigh_hogares_2016_2024.parquet"
RES = ROOT + "/resultados"
H = pd.read_parquet(DATA)
P = json.load(open(f"{RES}/prediccion_m3.json"))
R = {"nota": "exploratorio"}
for o, (y0, y1) in {"o1": (2020, 2022), "o2": (2022, 2024)}.items():
    models = joblib.load(f"{RES}/modelos_m3_{o}.joblib")
    R[o] = {}
    for s in ("C", "T"):
        cls = P["seleccion"][o][s]["elegido"]
        _, m, feats = models[f"{s}_{cls}"]
        e = shap.TreeExplainer(m)
        means = {}
        for y in (y0, y1):
            d = H[H.anio == y].sample(25000, random_state=y)
            phi = np.asarray(e.shap_values(d[feats].to_numpy(float)))
            means[y] = pd.Series(np.average(phi, axis=0, weights=d["factor"]), index=feats)
        dlt = (means[y1] - means[y0]).sort_values()
        R[o][s] = {"cambio_logit_total": round(float(dlt.sum()), 4),
                   "por_variable": {k: round(float(v), 4) for k, v in dlt.items()},
                   "participacion": {k: round(float(v / dlt.sum()), 3) for k, v in dlt.items()}}
        print(o, s, "total", round(float(dlt.sum()), 4), {k: round(float(v), 4) for k, v in dlt.head(6).items()}, "| sube:", {k: round(float(v), 4) for k, v in dlt.tail(3).items()})
json.dump(R, open(f"{RES}/descomposicion_m3.json", "w"), ensure_ascii=False, indent=1)
