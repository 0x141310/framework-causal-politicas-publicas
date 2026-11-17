import sys
import pandas as pd
D = sys.argv[1]
for y in (2022, 2024):
    ing = pd.read_csv(f"{D}/{y}_ingresos/ingresos.csv", dtype={"folioviv": str, "clave": str}, low_memory=False)
    for c in ("P104", "P045", "P101", "P102", "P038"):
        s = ing[ing.clave == c]
        ks = [f"ing_{k}" for k in range(1, 7)]
        vals = pd.to_numeric(s[ks].stack(), errors="coerce")
        vals = vals[vals > 0]
        nz = (s[ks].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).sum(axis=1)
        print(y, c, "n=", len(s), "| valores mensuales más frecuentes:", vals.round(0).value_counts().head(6).to_dict(),
              "| meses con pago (mediana):", nz.median(), "| ing_tri p25/p50/p75:", s.ing_tri.quantile([.25, .5, .75]).round(0).tolist())
