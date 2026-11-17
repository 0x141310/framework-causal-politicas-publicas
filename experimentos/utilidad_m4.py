"""Utilidad: pérdida de seguir la regla vigente.

Con los resultados guardados del comparador (escenarios_m4.py, sin recalcular efectos), compara en cada
condición y en cada combinación de modelo de efecto y curva de dosis los hogares que salen de
inseguridad alimentaria con la regla vigente (3,100 pesos con el orden de registro de las reglas de 2026,
modelado sin prioridad), con la variante por edad (escalonamiento de 2025) y con la opción elegida
por el framework, frente a la mejor opción factible.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import json

import numpy as np

RES = ROOT + "/resultados"


def main():
    e = json.load(open(f"{RES}/escenarios_m4.json"))
    elegidas = {c: e["recomendada_equidad"][c]["opcion"] for c in ("R", "P", "K")}
    maximin = {c: e["recomendada"][c]["opcion"] for c in ("R", "P", "K")}
    out = {"vigente": "REG-3100: 3,100 pesos con orden de registro (reglas de 2026), sin prioridad",
           "variante_edad": "U-3100: 3,100 pesos con prioridad por edad (escalonamiento de 2025)",
           "elegidas": elegidas, "maximin": maximin,
           "condiciones": {}}
    for c in ("R", "P", "K"):
        filas = []
        for m in ("M0", "M1", "M2", "M3"):
            for d in ("D1", "D2", "D3"):
                rows = e["resultados"][f"{c}|{m}|{d}|0.0"]
                key = {f"{r['regla']}-{int(r['monto'])}": r for r in rows}
                best = max(r["punto"]["hogares_salen_ia"] for r in rows if r["factible"])
                fila = {"modelo": m, "curva": d, "mejor_factible": best}
                for etiqueta, k in (("edad", "U-3100"), ("elegida", elegidas[c]), ("maximin", maximin[c])):
                    r = key[k]
                    h = r["punto"]["hogares_salen_ia"]
                    fila[etiqueta] = {"opcion": k, "hogares": h, "cociente": h / best if best > 0 else None,
                                      "perdida_hogares": best - h, "factible": bool(r["factible"]),
                                      "prop_q40": r["punto"]["prop_q40"],
                                      "cobertura_mujeres_en_ia": r["punto"]["cobertura_mujeres_en_ia"]}
                # regla de 2026: orden de registro, modelado sin prioridad sistemática (referencia fuera de las 25)
                r = e["referencia_registro"][f"{c}|{m}|{d}"]
                h = r["punto"]["hogares_salen_ia"]
                fila["vigente"] = {"opcion": "REG-3100", "hogares": h, "cociente": h / best if best > 0 else None,
                                    "perdida_hogares": best - h, "factible": bool(r["factible"]),
                                    "prop_q40": r["punto"]["prop_q40"],
                                    "cobertura_mujeres_en_ia": r["punto"]["cobertura_mujeres_en_ia"]}
                filas.append(fila)

        def resumen(et):
            cs = [f[et]["cociente"] for f in filas]
            return {"opcion": filas[0][et]["opcion"], "peor_cociente": float(min(cs)), "cociente_medio": float(np.mean(cs)),
                    "perdida_media_hogares": float(np.mean([f[et]["perdida_hogares"] for f in filas])),
                    "perdida_max_hogares": float(max(f[et]["perdida_hogares"] for f in filas)),
                    "combinaciones_infactibles": int(sum(not f[et]["factible"] for f in filas)),
                    "cobertura_mujeres_en_ia_M0_D2": next(f[et]["cobertura_mujeres_en_ia"] for f in filas
                                                          if f["modelo"] == "M0" and f["curva"] == "D2"),
                    "prop_q40": filas[0][et]["prop_q40"]}

        out["condiciones"][c] = {"vigente": resumen("vigente"), "edad": resumen("edad"), "elegida": resumen("elegida"),
                                 "maximin": resumen("maximin"), "detalle": filas}
    with open(f"{RES}/utilidad_m4.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for c, v in out["condiciones"].items():
        print(c, {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v[k].items()}
                  for k in ("vigente", "edad", "elegida", "maximin")})


if __name__ == "__main__":
    main()
