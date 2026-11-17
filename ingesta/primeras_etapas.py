"""Diseño (sin resultados): recepción de la pensión por edad y año; cobertura de becas por edad y año."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["anio", "edad", "pam", "pam_p104", "pension_contrib", "entidad", "ubica_geo",
                                           "hablaind", "factor", "prospera", "beca_ems", "beca_otra_gob", "hog_prospera",
                                           "hog_beca_basica", "hog_beca_ems", "asiste", "folioviv", "foliohog"])
w = lambda d, c: np.average(d[c], weights=d["factor"]) if len(d) else np.nan
print("Recepción de pensión (ponderada) por edad, país:")
t = df[df.edad.between(60, 72)].groupby(["anio", "edad"]).apply(lambda d: w(d, "pam"), include_groups=False).unstack(0)
print((100 * t).round(1).to_string())
print("\n2016/2018 sin pensión contributiva:")
t2 = df[df.edad.between(60, 70) & df.anio.isin([2016, 2018]) & (df.pension_contrib == 0)].groupby(["anio", "edad"]).apply(lambda d: w(d, "pam"), include_groups=False).unstack(0)
print((100 * t2).round(1).to_string())
# municipios indígenas (2020): categoría A y B de 2020 (anexo A de la Declaratoria de Zonas de Atención Prioritaria
# 2020); como comparación, municipios con 40% o más de
# hablantes de lengua indígena de 3 años o más en la ENIGH 2020 (ponderado)
cat = pd.read_csv(Path(__file__).resolve().parents[1] / "datos" / "categoria_indigena_2020.csv", dtype=str)
ab2020 = set(cat.loc[cat["indigena_ab"] == "1", "cve_mun"])
df["cve_mun"] = df["ubica_geo"].astype(str).str[:5]
d20 = df[(df.anio == 2020) & (df.edad >= 3)].copy()
d20["ind"] = (d20["hablaind"] == 1).astype(float)
sh = d20.groupby("cve_mun").apply(lambda d: np.average(d["ind"], weights=d["factor"]), include_groups=False)
print("\nMunicipios 2020: A y B observados", len(ab2020 & set(sh.index)), "| con >=40% hablantes", int((sh >= 0.4).sum()), "| de", len(sh))
m = df[(df.anio == 2020)].copy(); m["mun_ind"] = m["cve_mun"].isin(ab2020)
t3 = m[m.edad.between(62, 70)].groupby(["mun_ind", "edad"]).apply(lambda d: w(d, "pam"), include_groups=False).unstack(0)
print("2020 recepción por edad, municipio indígena A o B (True/False):"); print((100 * t3).round(1).to_string())
print("\nCobertura de beca/apoyo educativo por edad y año (persona con P042, P102 o P038; hogar con P101):")
df["beca_persona"] = ((df.prospera == 1) | (df.beca_ems == 1) | (df.beca_otra_gob == 1)).astype(int)
t4 = df[df.edad.between(12, 19)].groupby(["anio", "edad"]).apply(lambda d: w(d, "beca_persona"), include_groups=False).unstack(0)
print((100 * t4).round(1).to_string())
