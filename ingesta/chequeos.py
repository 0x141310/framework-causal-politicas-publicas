import sys
import numpy as np, pandas as pd
D, P = sys.argv[1], sys.argv[2]
for y in (2022, 2024):
    ho = pd.read_csv(f"{D}/{y}_hogares/hogares.csv", dtype={"folioviv": str}, low_memory=False,
                     usecols=["folioviv", "foliohog"] + [f"acc_alim{k}" for k in range(1, 17)])
    for c in ho.columns[2:]:
        ho[c] = pd.to_numeric(ho[c].astype(str).str.strip().replace({"": np.nan}), errors="coerce")
    miss = ho["acc_alim4"].isna()
    print(y, "faltan reactivos de adultos:", miss.mean().round(3),
          "| entre ellos, respuestas a reactivos 1-3:", ho.loc[miss, ["acc_alim1", "acc_alim2", "acc_alim3"]].apply(lambda s: s.value_counts(dropna=False).to_dict()).to_dict())
df = pd.read_parquet(P)
for y in (2022, 2024):
    d = df[df.anio == y]
    w = d["factor"]
    print(y, "IA moderada o severa (personas, ponderado): nacional %.1f%%" % (100 * np.average(d.ia_mod_sev, weights=w)),
          "| Quintana Roo %.1f%%" % (100 * np.average(d[d.entidad == "23"].ia_mod_sev, weights=d[d.entidad == "23"].factor)),
          "| PAM entre 65+ QRoo: %.1f%%" % (100 * np.average(d[(d.entidad == "23") & (d.edad >= 65)].pam, weights=d[(d.entidad == "23") & (d.edad >= 65)].factor)))
