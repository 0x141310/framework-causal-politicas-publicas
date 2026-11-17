"""Los datos procesados incluidos reproducen las cifras de integración y conciliación de la tesis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATOS = Path(__file__).resolve().parents[1] / "datos"
PERSONAS = DATOS / "enigh_personas_2016_2024.parquet"

pytestmark = pytest.mark.skipif(not PERSONAS.exists(), reason="faltan los datos procesados en datos/")


@pytest.fixture(scope="module")
def personas():
    return pd.read_parquet(PERSONAS, columns=["anio", "folioviv", "foliohog", "numren", "entidad",
                                             "ia_mod_sev", "factor"])


def test_integracion_sin_perdidas_ni_duplicados(personas):
    assert len(personas) == 1_461_036
    assert not personas.duplicated(["anio", "folioviv", "foliohog", "numren"]).any()
    assert int((personas["entidad"] == "23").sum()) == 35_639


@pytest.mark.parametrize("anio, cifra", [(2016, 19.95), (2018, 20.29)])
def test_conciliacion_con_coneval(personas, anio, cifra):
    g = personas[personas["anio"] == anio]
    assert round(100 * float(np.average(g["ia_mod_sev"], weights=g["factor"])), 2) == cifra


def test_categoria_indigena_2020():
    # anexo A de la Declaratoria de Zonas de Atención Prioritaria 2020: 1,480 municipios, 623 de categoría A o B
    archivo = DATOS / "categoria_indigena_2020.csv"
    if not archivo.exists():
        pytest.skip("falta datos/categoria_indigena_2020.csv")
    t = pd.read_csv(archivo, dtype=str).set_index("cve_mun")
    assert len(t) == 1480
    assert (t["categoria"] == "A").sum() == 457 and (t["categoria"] == "B").sum() == 166
    assert t.loc["23010", "categoria"] == "B"  # Bacalar
    assert t.loc["23002", "categoria"] == "A"  # Felipe Carrillo Puerto

