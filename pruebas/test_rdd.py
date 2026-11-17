"""El estimador de regresión discontinua recupera un efecto conocido en datos simulados."""
import numpy as np
import pandas as pd
import pytest

from modulo_causal import rdd


def simular(n=40_000, efecto=0.30, seed=7):
    """Edad discreta de 58 a 72 años; la probabilidad de recibir el programa salta de 2% a 60% a los 65."""
    rng = np.random.default_rng(seed)
    edad = rng.integers(58, 73, n)
    anio = rng.choice([2022, 2024], n)
    trata = (rng.random(n) < np.where(edad >= 65, 0.60, 0.02)).astype(float)
    y = 1.0 + 0.02 * (edad - 65) + 0.10 * (anio == 2024) + efecto * trata + rng.normal(0, 0.5, n)
    return pd.DataFrame({"edad": edad, "anio": anio, "trata": trata, "y": y, "hogar": np.arange(n) // 2})


def test_recupera_el_efecto_local():
    r = rdd.estimate(simular(), "edad", 65, "y", "trata", 7, "hogar", B=199, seed=3)
    assert r["primera_etapa"] == pytest.approx(0.58, abs=0.03)
    assert r["efecto_local"] == pytest.approx(0.30, abs=0.08)
    lo, hi = r["el_ic95"]
    assert lo < 0.30 < hi


def test_sin_efecto_el_intervalo_incluye_cero():
    r = rdd.estimate(simular(efecto=0.0, seed=11), "edad", 65, "y", "trata", 7, "hogar", B=199, seed=5)
    lo, hi = r["el_ic95"]
    assert lo < 0 < hi


def test_la_dona_excluye_la_edad_del_umbral():
    d = simular(seed=13)
    r = rdd.estimate(d, "edad", 65, "y", "trata", 7, "hogar", B=49, seed=1, donut=True)
    assert r["n"] == int((d["edad"] != 65).sum())
