"""Pruebas del generador de casos y de la calificación de la evaluación del descubrimiento causal."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calificar import METODOS, contenido_zip, etapa1, etapa2, evaluar_grafo  # noqa: E402
from generar_casos import generar  # noqa: E402

OPC = dict(familias=("lineal_gaussiana", "confusion_latente", "mixta_binaria"), tamanos=(300,), replicas=2)


def test_misma_frase_mismos_archivos():
    a = generar("prueba", **OPC)
    b = generar("prueba", **OPC)
    c = generar("otra frase", **OPC)
    assert [hashlib.sha256(x).hexdigest() for x in a] == [hashlib.sha256(x).hexdigest() for x in b]
    assert hashlib.sha256(a[0]).hexdigest() != hashlib.sha256(c[0]).hexdigest()


def test_capas_y_pares_coherentes():
    z, parejas, clave = generar("capas", **OPC)
    clave, parejas = json.loads(clave), json.loads(parejas)
    zf = zipfile.ZipFile(io.BytesIO(z))
    for c in clave["casos"]:
        capas = c["capas"]
        assert all(capas[a] <= capas[b] for a, b in c["aristas"])  # ninguna arista contra el orden de capas
        x, y = parejas[c["caso"]]["x"], parejas[c["caso"]]["y"]
        assert capas[x] <= capas[y]
        df = pd.read_csv(io.BytesIO(zf.read(f"entregar/{c['caso']}.csv")))
        assert list(df.columns) == sorted(df.columns) and len(df) == c["n"]
        assert not df.isna().any().any()


def test_calificacion_del_grafo_verdadero():
    _, _, clave = generar("perfecto", **OPC)
    c = json.loads(clave)["casos"][0]
    perfecto = [[a, "-->", b] for a, b in c["aristas"]]
    r = evaluar_grafo(perfecto, c, "B")
    assert r["recuperadas"] == r["verdaderas"] == len(c["aristas"])
    assert r.get("falsas", 0) == 0 and r.get("dir_correcta", 0) == len(c["aristas"])
    invertido = [[b, "-->", a] for a, b in c["aristas"]]
    r2 = evaluar_grafo(invertido, c, "A")
    assert r2.get("dir_invertida", 0) == len(c["aristas"])


def test_fallos_se_califican_y_se_reportan():
    _, _, clave = generar("fallos", **OPC)
    c = json.loads(clave)["casos"][0]
    vacio = {m: [] for m in METODOS}
    pred1 = {"casos": {c["caso"]: {"A": {**vacio, "errores": {"GES": "ValueError: prueba"}}, "B": dict(vacio)}}}
    r = etapa1({"casos": [c]}, pred1)
    ges = [a for a in r["agregados"] if (a["condicion"], a["metodo"], a["familia"], a["n"]) == ("A", "GES", "todos", "todos")][0]
    assert ges["errores"] == 1 and ges["exhaustividad"] == 0
    par = c["pareja"]
    pred2 = {"casos": {c["caso"]: {"x": par["x"], "y": par["y"], "error": "RuntimeError: prueba", "afirma_efecto": False,
                                   "dictamen": "error", "arista_restituida": False, "identificacion": {}}}}
    assert etapa2({"casos": [c]}, pred2)["resumen"] == {"error": 1}


def test_comprobacion_de_datos_por_contenido():
    z, _, _ = generar("contenido", **OPC)
    otro = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(z)) as a, zipfile.ZipFile(otro, "w", zipfile.ZIP_STORED) as b:
        for n in a.namelist():
            b.writestr(n, a.read(n))
    assert hashlib.sha256(otro.getvalue()).hexdigest() != hashlib.sha256(z).hexdigest()  # otros bytes comprimidos
    assert contenido_zip(otro.getvalue()) == contenido_zip(z)  # mismos archivos
    assert contenido_zip(generar("otra frase", **OPC)[0]) != contenido_zip(z)
