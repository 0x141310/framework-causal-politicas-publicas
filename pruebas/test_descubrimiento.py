"""PC y LiNGAM recuperan la arista intervención → resultado en un modelo lineal simulado con capas temporales."""
import numpy as np
import pandas as pd
import pytest

from modulo_causal.discovery import run_lingam, run_pc
from modulo_causal.spec import CausalSpec, Variable


@pytest.fixture(scope="module")
def caso():
    rng = np.random.default_rng(5)
    n = 3000
    z = rng.uniform(-1, 1, n)                       # covariable previa
    x = 0.8 * z + rng.uniform(-1, 1, n)              # intervención
    y = 0.6 * x + 0.5 * z + rng.uniform(-1, 1, n)    # resultado
    spec = CausalSpec(variables=[Variable("z", "previa", 0), Variable("x", "intervencion", 1),
                                 Variable("y", "resultado", 2)], treatment="x", outcome="y")
    return pd.DataFrame({"z": z, "x": x, "y": y}), spec


def test_pc_respeta_las_capas_y_encuentra_la_arista(caso):
    df, spec = caso
    g = run_pc(df, spec, alpha=0.01)
    assert ("x", "y") in g.directed_edges()
    assert ("y", "x") not in g.directed_edges()


def test_lingam_encuentra_la_arista(caso):
    df, spec = caso
    g = run_lingam(df, spec)
    assert ("x", "y") in g.directed_edges()
