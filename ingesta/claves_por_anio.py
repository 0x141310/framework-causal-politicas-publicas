"""Perfil de edad y montos por clave de ingreso (2016, 2018, 2020) para identificar programas."""
import sys
import pandas as pd
D = sys.argv[1]
for y in (2016, 2018, 2020):
    po = pd.read_csv(f"{D}/{y}_poblacion/poblacion.csv", dtype={"folioviv": str}, usecols=["folioviv", "foliohog", "numren", "edad"], low_memory=False)
    ing = pd.read_csv(f"{D}/{y}_ingresos/ingresos.csv", dtype={"folioviv": str, "clave": str}, low_memory=False)
    m = ing.merge(po, on=["folioviv", "foliohog", "numren"], how="left")
    ks = [f"ing_{k}" for k in range(1, 7)]
    prog = m[m.clave.str.match(r"P0(3[2-9]|4\d)|P1\d\d")]
    rows = []
    for c, s in prog.groupby("clave"):
        vals = pd.to_numeric(s[ks].stack(), errors="coerce"); vals = vals[vals > 0]
        top = vals.round(0).value_counts().head(3).index.tolist()
        rows.append({"clave": c, "n": len(s), "edad_med": s.edad.median(), "p10": s.edad.quantile(.1), "p90": s.edad.quantile(.9),
                     "pct65": round((s.edad >= 65).mean(), 2), "pct68": round((s.edad >= 68).mean(), 2), "pagos_frecuentes": top})
    print(f"\n### {y}"); print(pd.DataFrame(rows).to_string(index=False))
