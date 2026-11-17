"""Robustez de la definición de la pensión: clave federal P104 sola frente a P104 + P045.

La intervención principal (`pam`) suma P104 (Pensión para el Bienestar de las Personas Adultas Mayores) y P045
(otros programas para adultos mayores). Aquí se repiten, con el estimador y los datos de la Tabla IV (ENIGH 2022 y
2024, país, ±7 años, kernel triangular, efectos fijos de año, bootstrap por hogar con 999 réplicas), el efecto
nacional y el de las mujeres —el que alimenta los escenarios— con `pam_p104` como primera etapa. La forma reducida
no cambia; sólo cambia la primera etapa. También describe a quienes sólo reportan P045.
Salida: resultados/robustez_p104.json
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ROOT)
from modulo_causal import rdd  # noqa: E402

YS = ["ia_mod_sev", "ia_adultos", "log_ing_pc"]


def resumen(r: dict) -> dict:
    return {k: (list(map(float, v)) if isinstance(v, (tuple, list, np.ndarray)) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
            for k, v in r.items() if k not in ("draws", "rf_draws", "fs_draws")}


def main():
    n = pd.read_parquet(ROOT + "/datos/enigh_personas_2022_2024.parquet")
    n["hogar_id"] = n["anio"].astype(str) + "_" + n["folioviv"] + "_" + n["foliohog"].astype(str)
    R = {"descripcion": __doc__.strip().split("\n")[0]}

    # quiénes reportan P045 sin P104 (personas de 65 a 72 años)
    m = n[(n.edad >= 65) & (n.edad <= 72)]
    solo45 = m[(m.pam == 1) & (m.pam_p104 == 0)]
    con104 = m[m.pam_p104 == 1]
    R["descripcion_p045"] = {
        "personas_65_72": int(len(m)),
        "recibe_p104_o_p045": float(m.pam.mean()),
        "recibe_p104": float(m.pam_p104.mean()),
        "solo_p045_entre_receptores": float(len(solo45) / max((m.pam == 1).sum(), 1)),
        "monto_trimestral_mediano_solo_p045": {int(a): float(g.pam_monto_tri.median()) for a, g in solo45.groupby("anio")},
        "monto_trimestral_mediano_p104": {int(a): float(g.pam_p104_monto_tri.median()) for a, g in con104.groupby("anio")},
        "solo_p045_por_entidad_top5": {str(k): int(v) for k, v in solo45.entidad.value_counts().head(5).items()},
    }

    grupos = {"todos": n, "mujeres": n[n.mujer == 1]}
    R["efectos"] = {}
    for g, d in grupos.items():
        R["efectos"][g] = {}
        for t in ("pam", "pam_p104"):
            R["efectos"][g][t] = {y: resumen(rdd.estimate(d, "edad", 65, y, t, 7, "hogar_id")) for y in YS}

    with open(ROOT + "/resultados/robustez_p104.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1, default=float)

    print(json.dumps(R["descripcion_p045"], ensure_ascii=False, indent=1))
    for g in grupos:
        for t in ("pam", "pam_p104"):
            e = R["efectos"][g][t]
            ia, ing = e["ia_mod_sev"], e["log_ing_pc"]
            pct = lambda v: round((np.exp(v) - 1) * 100, 1)
            print(g, t, "PE", round(ia["primera_etapa"] * 100, 1),
                  "IA", round(ia["efecto_local"] * 100, 1), [round(x * 100, 1) for x in ia["el_ic95"]],
                  "reactivos", round(e["ia_adultos"]["efecto_local"], 2), [round(x, 2) for x in e["ia_adultos"]["el_ic95"]],
                  "ingreso %", pct(ing["efecto_local"]), [pct(x) for x in ing["el_ic95"]])


if __name__ == "__main__":
    main()
