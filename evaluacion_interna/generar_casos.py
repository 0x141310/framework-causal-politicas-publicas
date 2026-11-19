#!/usr/bin/env python3
"""Genera los casos sintéticos para evaluar el descubrimiento causal del framework (versión 1.0.0).

Requiere Python 3.10 o posterior, numpy y pandas.

    python3 generar_casos.py --frase <frase> --salida casos

Archivos en la carpeta de salida:
  entregar.zip   un CSV por caso con variables anónimas (V01, V02, ...) y capas.json con el orden temporal por capas.
                 Es lo único que recibe la etapa 1.
  parejas.json   un par (X, Y) por caso para la etapa 2.
  clave.json     la verdad de cada caso y la frase, para calificar.
Al terminar imprime la huella SHA-256 de los tres archivos; con ellas puede comprobarse después que parejas.json y
clave.json no cambiaron. La misma frase y las mismas opciones reproducen los mismos archivos. En la evaluación interna,
evaluacion_interna.sh usa como frase el valor randomness de una ronda de drand y aparta parejas.json y clave.json antes
de la etapa 1.

Diseño por omisión: 5 familias de mecanismos × 2 tamaños de muestra (1,000 y 5,000) × 4 réplicas = 40 casos. Cada
caso tiene un grafo aleatorio distinto, de 8 a 12 variables observadas, con a lo sumo 3 padres por variable:
  lineal_gaussiana     relaciones lineales con ruido normal.
  lineal_no_gaussiana  relaciones lineales con ruido uniforme, de Laplace o exponencial.
  no_lineal            relaciones tanh, umbral, seno, cuadrática o cúbica, con ruido normal.
  mixta_binaria        cerca de 40% de las variables son binarias (umbral sobre una combinación lineal más ruido).
  confusion_latente    como la lineal gaussiana, más una o dos variables no observadas que afectan a dos variables.
En la mitad de los casos una variable queda aislada (sin relación con ninguna otra). En una cuarta parte de los casos
lineales, el par de la segunda etapa tiene un efecto directo casi cancelado por una causa común observada.
Las opciones --familias, --n, --replicas y --pesos cambian el diseño.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0"
FAMILIAS = ("lineal_gaussiana", "lineal_no_gaussiana", "no_lineal", "mixta_binaria", "confusion_latente")
LINEALES = ("lineal_gaussiana", "lineal_no_gaussiana", "confusion_latente")
FECHA_ZIP = (2026, 1, 1, 0, 0, 0)  # fija, para que el zip sea reproducible


def semilla(frase: str) -> int:
    return int.from_bytes(hashlib.sha256(frase.encode("utf-8")).digest()[:8], "big")


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def a_json(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


# ------------------------------------------------------------------ estructura
def grafo(rng, p, densidad=2.8, max_padres=3):
    """DAG aleatorio: A[i, j] = 1 si i -> j. Devuelve la matriz y un orden topológico."""
    orden = [int(v) for v in rng.permutation(p)]
    prob = min(0.9, densidad / (p - 1))
    A = np.zeros((p, p), dtype=int)
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < prob:
                A[orden[a], orden[b]] = 1
    for j in range(p):
        padres = np.flatnonzero(A[:, j])
        if len(padres) > max_padres:
            A[rng.choice(padres, len(padres) - max_padres, replace=False), j] = 0
    return A, orden


def cierre(A):
    """anc[i, j] = True si hay un camino dirigido i -> ... -> j."""
    p = A.shape[0]
    R = A.astype(bool).copy()
    for k in range(p):
        R = R | (R[:, [k]] & R[[k], :])
    return R


def peso(rng, lo, hi):
    return float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]))


# ------------------------------------------------------------------- un caso
def caso(rng, nombre, familia, n, pesos=(0.5, 1.2)):
    p = int(rng.integers(8, 13))
    A, orden = grafo(rng, p)
    pos = {v: k for k, v in enumerate(orden)}

    # motivo de cancelación (sólo familias lineales, una cuarta parte de los casos)
    cancel = None
    if familia in ("lineal_gaussiana", "lineal_no_gaussiana") and rng.random() < 0.25:
        i, j, k = sorted(int(x) for x in rng.choice(p, 3, replace=False))
        C, X, Y = orden[i], orden[j], orden[k]
        A[:, C] = 0
        A[:, X] = 0
        A[C, X] = 1
        A[:, Y] = 0
        A[X, Y] = 1
        A[C, Y] = 1
        cancel = {"C": C, "X": X, "Y": Y, "delta": 0.15}

    # variable aislada en la mitad de los casos (nunca una del motivo de cancelación)
    aislada = None
    if rng.random() < 0.5:
        libres = [v for v in range(p) if not cancel or v not in (cancel["C"], cancel["X"], cancel["Y"])]
        aislada = int(rng.choice(libres))
        A[aislada, :] = 0
        A[:, aislada] = 0

    # confusores no observados
    latentes = []
    if familia == "confusion_latente":
        anc = cierre(A)
        candidatos = [(a, b) for a in range(p) for b in range(a + 1, p)
                      if a != aislada and b != aislada and not A[a, b] and not A[b, a] and not anc[a, b] and not anc[b, a]]
        nlat = 2 if rng.random() < 0.5 else 1
        usados = set()
        for idx in rng.permutation(len(candidatos)):
            a, b = candidatos[int(idx)]
            if a in usados or b in usados:
                continue
            latentes.append({"hijos": [int(a), int(b)], "pesos": [abs(peso(rng, 0.6, 1.0)), abs(peso(rng, 0.6, 1.0)) * float(rng.choice([-1, 1]))]})
            usados |= {a, b}
            if len(latentes) == nlat:
                break

    # mecanismos
    W = np.zeros((p, p))
    for i, j in zip(*np.nonzero(A)):
        W[i, j] = peso(rng, *pesos)
    ruido = {}
    for j in range(p):
        if familia == "lineal_no_gaussiana":
            ruido[j] = str(rng.choice(["uniforme", "laplace", "exponencial"]))
        else:
            ruido[j] = "normal"
    escala = {j: float(rng.uniform(0.6, 1.2)) for j in range(p)}
    funcs = {}
    if familia == "no_lineal":
        for i, j in zip(*np.nonzero(A)):
            funcs[(int(i), int(j))] = str(rng.choice(["tanh", "umbral", "seno", "cuadratica", "cubica"],
                                                     p=[0.3, 0.25, 0.15, 0.15, 0.15]))
    binarias = set()
    prevalencia = {}
    if familia == "mixta_binaria":
        binarias = {int(v) for v in rng.choice([v for v in range(p) if v != aislada], max(2, round(0.4 * p)), replace=False)}
        prevalencia = {v: float(rng.uniform(0.2, 0.5)) for v in sorted(binarias)}

    def g(f, x):
        return {"tanh": np.tanh(1.5 * x), "umbral": np.maximum(x, 0) - 0.4, "seno": np.sin(1.5 * x),
                "cuadratica": x ** 2 - 1, "cubica": x ** 3 / 3}[f]

    def e(j):
        t, s = ruido[j], escala[j]
        if t == "uniforme":
            return s * rng.uniform(-np.sqrt(3), np.sqrt(3), n)
        if t == "laplace":
            return s * rng.laplace(0, 1 / np.sqrt(2), n)
        if t == "exponencial":
            return s * (rng.exponential(1.0, n) - 1.0)
        return s * rng.standard_normal(n)

    L = [rng.standard_normal(n) for _ in latentes]
    Z = np.zeros((n, p))       # variables estandarizadas (las que usan los hijos)
    salida = np.zeros((n, p))  # datos observados (binarias en 0/1)
    sd = np.ones(p)
    for j in orden:
        s = np.zeros(n)
        for i in np.flatnonzero(A[:, j]):
            w = W[i, j]
            if cancel and (i, j) == (cancel["C"], cancel["Y"]):
                a_cx, b_xy = W[cancel["C"], cancel["X"]], W[cancel["X"], cancel["Y"]]
                sx = np.sqrt(a_cx ** 2 + escala[cancel["X"]] ** 2)
                w = -(1 - cancel["delta"]) * b_xy * sx / a_cx
                W[i, j] = w
            s += w * (g(funcs[(int(i), int(j))], Z[:, i]) if funcs else Z[:, i])
        for k, lat in enumerate(latentes):
            if j in lat["hijos"]:
                s += lat["pesos"][lat["hijos"].index(j)] * L[k]
        if j in binarias:
            raw = s + escala[j] * rng.logistic(0, 1, n)
            x = (raw > np.quantile(raw, 1 - prevalencia[j])).astype(float)
        else:
            x = s + e(j)
        sd[j] = x.std()
        Z[:, j] = (x - x.mean()) / sd[j]
        salida[:, j] = x if j in binarias else Z[:, j]  # binarias en 0/1; continuas estandarizadas

    # efectos totales en unidades estandarizadas (sólo familias lineales y continuas)
    efecto_total = None
    if familia in LINEALES:
        Bm = W / sd[np.newaxis, :]
        efecto_total = np.linalg.inv(np.eye(p) - Bm) - np.eye(p)

    # capas: tres bloques contiguos del orden topológico (ninguna arista va de una capa posterior a una anterior)
    c1 = int(rng.integers(2, p - 3))
    c2 = int(rng.integers(c1 + 2, p - 1)) if c1 + 2 < p - 1 else c1 + 1
    capa = {v: (0 if pos[v] < c1 else 1 if pos[v] < c2 else 2) for v in range(p)}

    # nombres anónimos y orden de columnas sin relación con el orden causal
    perm = [int(x) for x in rng.permutation(p)]
    nom = {v: f"V{perm[v] + 1:02d}" for v in range(p)}
    cols = sorted(range(p), key=lambda v: nom[v])
    datos = pd.DataFrame({nom[v]: np.round(salida[:, v], 6) for v in cols})

    # par de la segunda etapa
    anc = cierre(A)
    comun_obs = lambda a, b: bool(np.any(anc[:, a] & anc[:, b]))  # noqa: E731
    comun_lat = lambda a, b: any(set(l["hijos"]) == {a, b} for l in latentes)  # noqa: E731
    ordenado = lambda a, b: (a, b) if capa[a] <= capa[b] else (b, a)  # noqa: E731
    obs = [v for v in range(p) if v != aislada]
    directos = [(int(i), int(j)) for i, j in zip(*np.nonzero(A))]
    mediados = [(a, b) for a in range(p) for b in range(p) if anc[a, b] and not A[a, b]]
    nulos = [ordenado(a, b) for a in obs for b in obs if a < b and not anc[a, b] and not anc[b, a]]
    nulos_lat = [x for x in nulos if comun_lat(*x)]
    nulos_obs = [x for x in nulos if comun_obs(*x) and not comun_lat(*x)]
    nulos_ind = [x for x in nulos if not comun_obs(*x) and not comun_lat(*x)]
    grupos = {"directo": directos, "mediado": mediados, "nulo_latente": nulos_lat,
              "nulo_causa_comun": nulos_obs, "nulo_independiente": nulos_ind}
    if cancel:
        par, tipo = (cancel["X"], cancel["Y"]), "directo_cancelado"
    else:
        # mitad con efecto (directo o mediado) y mitad sin efecto (con causa común o independiente)
        r = rng.random()
        if r < 0.25:
            preferencia = ["directo", "mediado"]
        elif r < 0.5:
            preferencia = ["mediado", "directo"]
        elif familia == "confusion_latente" and rng.random() < 0.5:
            preferencia = ["nulo_latente", "nulo_causa_comun", "nulo_independiente"]
        elif rng.random() < 0.6:
            preferencia = ["nulo_causa_comun", "nulo_independiente"]
        else:
            preferencia = ["nulo_independiente", "nulo_causa_comun"]
        respaldo = ["directo", "mediado", "nulo_causa_comun", "nulo_independiente"]
        tipo = next(t for t in preferencia + respaldo if grupos[t])
        opciones = grupos[tipo]
        par = opciones[int(rng.integers(len(opciones)))]
    x, y = par
    clave = {
        "caso": nombre, "familia": familia, "n": n, "p": p,
        "variables": [nom[v] for v in cols],
        "aristas": sorted([nom[int(i)], nom[int(j)]] for i, j in zip(*np.nonzero(A))),
        "pesos_estandarizados": {f"{nom[int(i)]}->{nom[int(j)]}": round(float(W[i, j] / sd[j]), 6) for i, j in zip(*np.nonzero(A))},
        "funciones": {f"{nom[i]}->{nom[j]}": f for (i, j), f in funcs.items()},
        "ruido": {nom[j]: ruido[j] for j in range(p) if j not in binarias},
        "binarias": sorted(nom[v] for v in binarias),
        "aislada": nom[aislada] if aislada is not None else None,
        "latentes": [{"hijos": [nom[a] for a in l["hijos"]], "pesos": [round(w, 6) for w in l["pesos"]]} for l in latentes],
        "cancelacion": ({"causa_comun": nom[cancel["C"]], "x": nom[cancel["X"]], "y": nom[cancel["Y"]], "delta": cancel["delta"]}
                        if cancel else None),
        "capas": {nom[v]: capa[v] for v in cols},
        "pareja": {"x": nom[x], "y": nom[y], "tipo": tipo, "efecto": tipo.startswith("directo") or tipo == "mediado",
                   "efecto_total_estandarizado": (round(float(efecto_total[x, y]), 6) if efecto_total is not None else None)},
    }
    return datos, {nom[v]: capa[v] for v in cols}, {"x": nom[x], "y": nom[y]}, clave


# --------------------------------------------------------------------- main
def generar(frase, familias=FAMILIAS, tamanos=(1000, 5000), replicas=4, pesos=(0.5, 1.2)):
    rng = np.random.default_rng(semilla(frase))
    plan = [(f, n) for f in familias for n in tamanos for _ in range(replicas)]
    plan = [plan[int(k)] for k in rng.permutation(len(plan))]  # el número de caso no revela la familia
    archivos, capas, parejas, claves = {}, {}, {}, []
    for k, (fam, n) in enumerate(plan, 1):
        nombre = f"caso_{k:02d}"
        datos, capa, par, clave = caso(rng, nombre, fam, n, pesos)
        archivos[f"{nombre}.csv"] = datos.to_csv(index=False).encode("utf-8")
        capas[nombre], parejas[nombre] = capa, par
        claves.append(clave)
    leeme = ("Casos sintéticos para la evaluación interna del descubrimiento causal.\n"
             "Un CSV por caso; las variables son anónimas y el orden de las columnas no sigue el orden causal.\n"
             "capas.json da, para la condición B, la capa temporal de cada variable: ninguna variable de una capa\n"
             "posterior causa a una de una capa anterior. La condición A no usa esa información.\n")
    archivos["capas.json"] = a_json(capas)
    archivos["LEEME.txt"] = leeme.encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre in sorted(archivos):
            info = zipfile.ZipInfo(f"entregar/{nombre}", date_time=FECHA_ZIP)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, archivos[nombre])
    opciones = {"familias": list(familias), "tamanos": list(tamanos), "replicas": replicas, "pesos": list(pesos)}
    clave = {"version_generador": VERSION, "frase": frase, "opciones": opciones, "casos": claves}
    return buf.getvalue(), a_json(parejas), a_json(clave)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frase", required=True, help="frase que genera los casos")
    ap.add_argument("--salida", default="casos")
    ap.add_argument("--familias", nargs="+", default=list(FAMILIAS), choices=FAMILIAS)
    ap.add_argument("--n", nargs="+", type=int, default=[1000, 5000], help="tamaños de muestra")
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--pesos", nargs=2, type=float, default=[0.5, 1.2], help="rango del valor absoluto de los pesos")
    a = ap.parse_args()
    zip_bytes, parejas, clave = generar(a.frase, a.familias, a.n, a.replicas, tuple(a.pesos))
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    (out / "entregar.zip").write_bytes(zip_bytes)
    (out / "parejas.json").write_bytes(parejas)
    (out / "clave.json").write_bytes(clave)
    print(f"Casos generados: {len(json.loads(clave)['casos'])}")
    print(f"  entregar.zip  {sha256(zip_bytes)}")
    print(f"  parejas.json  {sha256(parejas)}")
    print(f"  clave.json    {sha256(clave)}")


if __name__ == "__main__":
    main()
