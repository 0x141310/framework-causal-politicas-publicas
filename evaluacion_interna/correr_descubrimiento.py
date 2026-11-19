#!/usr/bin/env python3
"""Corre la versión congelada del framework (1.0.0) sobre los casos sintéticos y guarda sus predicciones.

Etapa 1, sin pares: para cada caso y condición (A, sin orden temporal; B, con las capas de capas.json) guarda el grafo de
consenso de PC que el módulo causal usa antes de la prueba dirigida. Usa la misma mitad de descubrimiento y los mismos
parámetros que run_module: partición aleatoria 50/50 con semilla 0, α = 0.01, 50 remuestreos y umbral de 0.6. Como
resultados secundarios guarda PC, FCI, GES y DirectLiNGAM en una corrida y el consenso de FCI (25 remuestreos).

Etapa 2, con parejas.json: corre el módulo causal completo (run_module, condición B) para el par de cada caso y guarda
la prueba dirigida, la identificación, el dictamen y las estimaciones.

    python3 correr_descubrimiento.py etapa1 --datos entregar.zip --salida resultados
    python3 correr_descubrimiento.py etapa2 --datos entregar.zip --parejas parejas.json --salida resultados

Cada etapa escribe predicciones_etapaN.json y su huella SHA-256 en huella_etapaN.txt, e imprime la huella antes de que
se lea el archivo siguiente. Antes de correr, el programa comprueba que los archivos del módulo causal coinciden con
MANIFEST.sha256 de la versión 1.0.0.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
FRAMEWORK = Path(os.environ.get("FRAMEWORK_DIR", AQUI.parent))
sys.path.insert(0, str(FRAMEWORK))

PARAMETROS = {"alpha": 0.01, "remuestreos_PC": 50, "remuestreos_FCI": 25, "umbral_consenso": 0.6,
              "umbral_orientacion": 0.5, "semilla": 0, "particion": "aleatoria 50/50 con semilla 0 (split_sample)",
              "metodos_una_corrida": ["PC", "FCI", "GES", "LiNGAM"], "prueba_independencia": "fisherz"}
NIVELES_AFIRMA = ("efecto_identificado_bajo_supuestos", "identificado_fragil")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def limpio(o):
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [limpio(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def verificar_version() -> dict:
    """Compara los archivos del módulo causal con MANIFEST.sha256 de la versión 1.0.0."""
    esperado = {}
    for ln in (FRAMEWORK / "MANIFEST.sha256").read_text().splitlines():
        if ln.strip():
            h, ruta = ln.split(maxsplit=1)
            if ruta.startswith("modulo_causal/"):
                esperado[ruta] = h
    actual = {r: sha256_bytes((FRAMEWORK / r).read_bytes()) for r in esperado}
    distintos = [r for r in esperado if actual[r] != esperado[r]]
    if not esperado or distintos:
        raise SystemExit(f"El módulo causal no coincide con MANIFEST.sha256: {distintos or 'manifiesto sin módulo'}")
    import causallearn
    import lingam
    import networkx
    import sklearn
    try:
        from importlib.metadata import version
        cl = version("causal-learn")
    except Exception:  # noqa: BLE001
        cl = getattr(causallearn, "__version__", "?")
    return {"modulo_causal_sha256": actual, "coincide_con_manifiesto": True,
            "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "causal-learn": cl, "lingam": lingam.__version__, "scikit-learn": sklearn.__version__,
                        "networkx": networkx.__version__, "nucleos": os.cpu_count()}}


def leer_casos(ruta: Path):
    """Lee los CSV y capas.json desde entregar.zip o desde una carpeta."""
    archivos = {}
    if ruta.suffix == ".zip":
        with zipfile.ZipFile(ruta) as z:
            for n in z.namelist():
                archivos[Path(n).name] = z.read(n)
    else:
        for f in ruta.iterdir():
            archivos[f.name] = f.read_bytes()
    capas = json.loads(archivos["capas.json"])
    casos = {Path(n).stem: pd.read_csv(io.BytesIO(b)) for n, b in sorted(archivos.items()) if n.endswith(".csv")}
    huella = sha256_bytes(ruta.read_bytes()) if ruta.is_file() else None
    return casos, capas, huella


def especificacion(df: pd.DataFrame, capas: dict | None = None, x: str | None = None, y: str | None = None):
    from modulo_causal import CausalSpec, Variable

    nombres = list(df.columns)
    capa = lambda v: int((capas or {}).get(v, 0))  # noqa: E731
    if x is None:  # marcador: el descubrimiento sólo usa capas y restricciones, no la intervención ni el resultado
        por_capa = sorted(nombres, key=capa)
        x, y = por_capa[0], por_capa[-1]
    variables = []
    for v in nombres:
        tipo = "binaria" if df[v].isin([0, 1]).all() else "continua"
        rol = "intervencion" if v == x else "resultado" if v == y else "previa"
        variables.append(Variable(v, rol, capa(v), tipo))
    return CausalSpec(variables, treatment=x, outcome=y)


def aristas(g) -> list:
    return sorted([a, t, b] for a, t, b in g.edges())


def seguro(errores: dict, clave: str, fn):
    """Si el framework falla en un caso, registra el error y devuelve None: el caso se califica como fallo
    (sin conexiones propuestas) y la corrida sigue con los demás."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        errores[clave] = f"{type(e).__name__}: {e}"[:300]
        return None


def etapa1_caso(tarea):
    nombre, df, capas = tarea
    from modulo_causal import discovery as disc
    from modulo_causal.pipeline import split_sample

    datos = df.dropna().reset_index(drop=True)
    d_disc, _ = split_sample(datos, "random", seed=PARAMETROS["semilla"])
    out = {"n": len(datos), "n_descubrimiento": len(d_disc)}
    for cond, cap in (("A", None), ("B", capas)):
        spec = especificacion(datos, cap)
        names = spec.names
        t0 = time.time()
        errores = {}
        una = {}
        for m in PARAMETROS["metodos_una_corrida"]:  # mismo orden y mismos argumentos que run_all con los cuatro métodos
            g = seguro(errores, m, lambda m=m: disc.run_all(d_disc[names], spec, methods=(m,), alpha=PARAMETROS["alpha"])[m])
            una[m] = aristas(g) if g is not None else []
        pc = seguro(errores, "consenso_PC", lambda: disc.bootstrap_stability(
            d_disc[names], spec, "PC", B=PARAMETROS["remuestreos_PC"], seed=PARAMETROS["semilla"],
            threshold=PARAMETROS["umbral_consenso"], alpha=PARAMETROS["alpha"]))
        fci = seguro(errores, "consenso_FCI", lambda: disc.bootstrap_stability(
            d_disc[names], spec, "FCI", B=PARAMETROS["remuestreos_FCI"], seed=PARAMETROS["semilla"] + 1,
            threshold=PARAMETROS["umbral_consenso"], alpha=PARAMETROS["alpha"]))
        out[cond] = {
            "consenso_PC": aristas(pc["consenso"]) if pc else [],
            "frecuencias_PC": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                               for r in pc["frecuencias"]] if pc else [],
            "remuestreos_PC_validos": pc["B"] if pc else 0,
            "consenso_FCI": aristas(fci["consenso"]) if fci else [],
            **una,
            "errores": errores,
            "segundos": round(time.time() - t0, 1),
        }
    return nombre, out


def etapa2_caso(tarea):
    nombre, df, capas, par, previo = tarea
    from modulo_causal.pipeline import run_module

    datos = df.dropna().reset_index(drop=True)
    spec = especificacion(datos, capas, par["x"], par["y"])
    t0 = time.time()
    errores = {}
    R = seguro(errores, "run_module", lambda: run_module(datos, spec, alpha=PARAMETROS["alpha"], B=PARAMETROS["remuestreos_PC"],
                                                         threshold=PARAMETROS["umbral_consenso"], seed=PARAMETROS["semilla"]))
    if R is None:  # fallo del framework: no afirma nada y se reporta aparte
        return nombre, {"x": par["x"], "y": par["y"], "error": errores["run_module"], "prueba_dirigida": None,
                        "arista_restituida": False, "grafo_candidato": [], "grafo_igual_al_de_la_etapa1": None,
                        "identificacion": {}, "fci_diagnostico": None, "ida": None, "estimacion_lineal": None,
                        "estimacion_principal": None, "dictamen": "error", "afirma_efecto": False,
                        "dictamen_texto": "El módulo causal se detuvo con un error.", "segundos": round(time.time() - t0, 1)}
    est = R.get("estimaciones") or []
    resumen = lambda e: {k: e.get(k) for k in ("estimador", "efecto", "ee", "ic95", "p")} if e else None  # noqa: E731
    candidato = aristas(R["grafo_candidato"])
    sin_restituir = [e for e in candidato if not ("arista_restituida" in R and e[0] == par["x"] and e[2] == par["y"])]
    return nombre, {
        "x": par["x"], "y": par["y"],
        "prueba_dirigida": R["prueba_efecto_total"],
        "arista_restituida": "arista_restituida" in R,
        "grafo_candidato": candidato,
        "grafo_igual_al_de_la_etapa1": (sin_restituir == previo) if previo is not None else None,
        "identificacion": {k: v for k, v in R["identificacion"].items() if k in ("estado", "motivo", "ajuste", "n_dags", "tipo")},
        "fci_diagnostico": (R.get("fci_diagnostico") or {}).get("estado"),
        "ida": R.get("ida"),
        "estimacion_lineal": resumen(est[0]) if est else None,
        "estimacion_principal": resumen(est[-1]) if est else None,
        "dictamen": R["dictamen"]["nivel"],
        "afirma_efecto": R["dictamen"]["nivel"] in NIVELES_AFIRMA,
        "dictamen_texto": R["dictamen"]["texto"],
        "segundos": round(time.time() - t0, 1),
    }


def guardar(salida: Path, etapa: int, obj: dict) -> str:
    b = (json.dumps(limpio(obj), ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    nombre = f"predicciones_etapa{etapa}.json"
    (salida / nombre).write_bytes(b)
    h = sha256_bytes(b)
    (salida / f"huella_etapa{etapa}.txt").write_text(f"{h}  {nombre}\n")
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["etapa1", "etapa2"])
    ap.add_argument("--datos", required=True, type=Path, help="entregar.zip o la carpeta con los CSV y capas.json")
    ap.add_argument("--parejas", type=Path, help="parejas.json (sólo etapa 2)")
    ap.add_argument("--salida", type=Path, default=Path("resultados"))
    ap.add_argument("--procesos", type=int, default=os.cpu_count())
    ap.add_argument("--casos", nargs="*", help="sólo estos casos (para pruebas)")
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    version = verificar_version()
    casos, capas, huella_datos = leer_casos(a.datos)
    if a.casos:
        casos = {k: v for k, v in casos.items() if k in a.casos}
    inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"version_framework": "1.0.0 (commit 439bdf4510810755edd8c4737fefd583114f13a6)", **version,
            "parametros": PARAMETROS, "datos_sha256": huella_datos, "inicio_utc": inicio,
            "programa_sha256": sha256_bytes(Path(__file__).read_bytes())}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.procesos) as ex:
        if a.etapa == "etapa1":
            res = dict(ex.map(etapa1_caso, [(k, v, capas[k]) for k, v in casos.items()]))
            etapa = 1
        else:
            if not a.parejas:
                raise SystemExit("La etapa 2 requiere --parejas parejas.json")
            parejas_b = a.parejas.read_bytes()
            parejas = json.loads(parejas_b)
            meta["parejas_sha256"] = sha256_bytes(parejas_b)
            previo = {}
            f1 = a.salida / "predicciones_etapa1.json"
            if f1.exists():
                p1 = json.loads(f1.read_text())
                previo = {k: v["B"]["consenso_PC"] for k, v in p1["casos"].items()}
                meta["etapa1_sha256"] = sha256_bytes(f1.read_bytes())
            res = dict(ex.map(etapa2_caso, [(k, v, capas[k], parejas[k], previo.get(k)) for k, v in casos.items()]))
            etapa = 2
    meta["fin_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["minutos"] = round((time.time() - t0) / 60, 1)
    h = guardar(a.salida, etapa, {"meta": meta, "casos": dict(sorted(res.items()))})
    print(f"Etapa {etapa}: {len(res)} casos en {meta['minutos']} min")
    print(f"Huella SHA-256 de predicciones_etapa{etapa}.json: {h}")


if __name__ == "__main__":
    main()
