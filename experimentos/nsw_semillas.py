"""NSW con controles observacionales: variación del AIPW-ATT entre 10 semillas de ajuste cruzado.

Conjunto óptimo descubierto por el pipeline (nsw_pipeline.log) frente a todas las variables pretratamiento.
Salida: resultados/nsw_semillas.csv
"""
import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))

import sys  # noqa: E402

sys.path.insert(0, ROOT)
import causaldata  # noqa: E402
import pandas as pd  # noqa: E402
from benchmark_nsw import prep  # noqa: E402
from modulo_causal import estimation as est  # noqa: E402

OPTIMO = ["age", "black", "educ", "re74", "re75", "u75"]
PRETRATAMIENTO = ["age", "black", "hisp", "educ", "nodegree", "marr", "re74", "u74", "re75", "u75"]


def main():
    nsw = prep(causaldata.nsw_mixtape.load_pandas().data)
    cps = prep(causaldata.cps_mixtape.load_pandas().data)
    obs = pd.concat([nsw[nsw.treat == 1], cps], ignore_index=True)
    filas = []
    for s in range(10):
        o = est.aipw_att(obs, "treat", "re78", OPTIMO, flexible=True, seed=s)
        p = est.aipw_att(obs, "treat", "re78", PRETRATAMIENTO, flexible=True, seed=s)
        filas.append({"seed": s, "O": o["efecto"], "O_lo": o["ic95"][0], "O_hi": o["ic95"][1],
                      "pre": p["efecto"], "pre_lo": p["ic95"][0], "pre_hi": p["ic95"][1]})
    d = pd.DataFrame(filas)
    d.to_csv(ROOT + "/resultados/nsw_semillas.csv", index=False)
    print(d.round(3).to_string(index=False))
    print("mediana con el conjunto óptimo:", round(d.O.median(), 3), "rango:", round(d.O.min(), 3), "a", round(d.O.max(), 3))


if __name__ == "__main__":
    main()
