"""Exploración inicial de la ENIGH: tamaños de muestra y perfil de las claves de ingreso por edad."""
import sys
import pandas as pd
D = sys.argv[1]
for y in (2022, 2024):
    ch = pd.read_csv(f"{D}/{y}_concentradohogar/concentradohogar.csv", dtype={"folioviv": str, "ubica_geo": str}, low_memory=False)
    ch["ent"] = ch["folioviv"].str.zfill(10).str[:2]
    po = pd.read_csv(f"{D}/{y}_poblacion/poblacion.csv", dtype={"folioviv": str}, low_memory=False,
                     usecols=["folioviv", "foliohog", "numren", "edad", "sexo", "asis_esc", "tiene_b", "otorg_b", "forma_b", "nivel", "grado", "nivelaprob", "gradoaprob", "tipoesc"])
    po["ent"] = po["folioviv"].str.zfill(10).str[:2]
    q = ch[ch.ent == "23"]; pq = po[po.ent == "23"]
    print(f"\n### {y}: hogares {len(ch):,} (QRoo {len(q):,}); personas {len(po):,} (QRoo {len(pq):,}); factor QRoo hogares {q.factor.sum():,.0f}")
    print("  QRoo personas 60-70:", ((pq.edad >= 60) & (pq.edad <= 70)).sum(), " 55-75:", ((pq.edad >= 55) & (pq.edad <= 75)).sum(), " 6-17:", ((pq.edad >= 6) & (pq.edad <= 17)).sum())
    print("  tiene_b valores:", po.tiene_b.value_counts(dropna=False).head(6).to_dict(), " otorg_b:", po.otorg_b.value_counts(dropna=False).head(8).to_dict())
    print("  asis_esc valores:", po.asis_esc.value_counts(dropna=False).to_dict())
    print("  tiene_b por asis_esc:\n", pd.crosstab(po.asis_esc.fillna(-1), po.tiene_b.fillna(-1)))
    ing = pd.read_csv(f"{D}/{y}_ingresos/ingresos.csv", dtype={"folioviv": str, "clave": str}, low_memory=False)
    m = ing.merge(po[["folioviv", "foliohog", "numren", "edad"]], on=["folioviv", "foliohog", "numren"], how="left")
    m["mensual"] = m["ing_tri"] / 3
    prog = m[m.clave.str.match(r"P0(3[2-9]|4\d|5\d)|P1\d\d")]
    tab = prog.groupby("clave").agg(n=("edad", "size"), edad_med=("edad", "median"), p10=("edad", lambda s: s.quantile(.1)),
                                    p90=("edad", lambda s: s.quantile(.9)), pct65=("edad", lambda s: (s >= 65).mean()),
                                    mensual_med=("mensual", "median"))
    print(tab.round(2).to_string())
