#!/usr/bin/env python3
"""Califica las predicciones del descubrimiento con la clave de los casos.

    python3 calificar.py --clave clave.json --resultados resultados [--parejas parejas.json] [--datos entregar.zip]
                         [--huella-parejas H] [--huella-clave H]

Comprueba primero las huellas: las de parejas.json y clave.json contra las que se imprimieron al generar los casos, y
las de las predicciones contra huella_etapaN.txt. Si se da --datos, vuelve a generar los casos con la frase y las
opciones de la clave y comprueba que los archivos de entregar.zip coinciden, uno por uno, con los generados.

Etapa 1 (salida principal: consenso de PC antes de la prueba dirigida; secundarias: PC, FCI, GES y LiNGAM en una
corrida y consenso de FCI). Por caso y condición cuenta:
  conexiones verdaderas recuperadas, omitidas y propuestas por error (estas últimas según su origen: confusión no
  observada, camino indirecto, causa común observada u otra);
  sobre las conexiones verdaderas recuperadas, la dirección: correcta, invertida o abierta (sin dirección, bidirigida o
  con círculos); en la condición B se separan las direcciones que ya fijaba el orden por capas;
  pares sin relación (sin camino en ningún sentido y sin causa común) en los que no se propuso conexión.
Etapa 2: por tipo de par, si el dictamen afirma un efecto (identificado o identificado frágil) o se abstiene, si la
prueba dirigida agregó una conexión verdadera o falsa y, en las familias lineales, si el IC95 cubre el efecto verdadero.
Escribe calificacion.json y calificacion.md en la carpeta de resultados.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
METODOS = ["consenso_PC", "PC", "FCI", "GES", "LiNGAM", "consenso_FCI"]
INVERSO = {"-->": "<--", "<--": "-->", "o->": "<-o", "<-o": "o->", "--o": "o--", "o--": "--o"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def contenido_zip(b: bytes) -> dict:
    """Huella de cada archivo del zip, por nombre: no depende de cómo se comprimió."""
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        return {n: sha(z.read(n)) for n in sorted(z.namelist())}


def tipo_entre(est: dict, u: str, v: str):
    """Tipo de la conexión estimada leída como u ? v (None si no hay conexión)."""
    e = est.get(frozenset((u, v)))
    if e is None:
        return None
    a, t, b = e
    return t if (a, b) == (u, v) else INVERSO.get(t, t)


def ancestros(verdad: set, nombres: list) -> dict:
    hijos = defaultdict(set)
    for a, b in verdad:
        hijos[a].add(b)
    desc = {}
    for v in nombres:
        vistos, pila = set(), [v]
        while pila:
            for h in hijos[pila.pop()]:
                if h not in vistos:
                    vistos.add(h)
                    pila.append(h)
        desc[v] = vistos
    return desc  # desc[a] = descendientes de a


def evaluar_grafo(aristas_est: list, clave: dict, condicion: str) -> dict:
    nombres = clave["variables"]
    verdad = {(a, b) for a, b in clave["aristas"]}
    adj_v = {frozenset(e) for e in verdad}
    est = {frozenset((a, b)): (a, t, b) for a, t, b in aristas_est}
    desc = ancestros(verdad, nombres)
    latentes = {frozenset(l["hijos"]) for l in clave["latentes"]}
    capas = clave["capas"]
    r = Counter()
    r["verdaderas"] = len(adj_v)
    r["propuestas"] = len(est)
    for u, v in verdad:
        t = tipo_entre(est, u, v)
        if t is None:
            r["omitidas"] += 1
            continue
        r["recuperadas"] += 1
        cat = "correcta" if t == "-->" else "invertida" if t == "<--" else "abierta"
        r[f"dir_{cat}"] += 1
        if condicion == "B" and capas[u] < capas[v]:
            r[f"dir_fijada_{cat}"] += 1
        else:
            r[f"dir_aprendida_{cat}"] += 1
    for par in set(est) - adj_v:
        a, b = tuple(par)
        if par in latentes:
            r["falsa_confusion_latente"] += 1
            if est[par][1] == "<->":
                r["confusion_senalada"] += 1
        elif b in desc[a] or a in desc[b]:
            r["falsa_camino_indirecto"] += 1
        elif desc_comun(a, b, desc):
            r["falsa_causa_comun"] += 1
        else:
            r["falsa_otra"] += 1
        r["falsas"] += 1
    for par in latentes:
        r["pares_confundidos"] += 1
    sin_relacion = [(a, b) for i, a in enumerate(nombres) for b in nombres[i + 1:]
                    if b not in desc[a] and a not in desc[b] and not desc_comun(a, b, desc) and frozenset((a, b)) not in latentes]
    r["pares_sin_relacion"] = len(sin_relacion)
    r["pares_sin_relacion_sin_conexion"] = sum(frozenset(p) not in est for p in sin_relacion)
    return dict(r)


def desc_comun(a, b, desc) -> bool:
    return any(a in d and b in d for v, d in desc.items() if v not in (a, b))


def tasas(c: Counter) -> dict:
    div = lambda x, y: round(x / y, 3) if y else None  # noqa: E731
    dirigidas = c["dir_correcta"] + c["dir_invertida"]
    return {
        "casos": c["casos"], "conexiones_verdaderas": c["verdaderas"], "propuestas": c["propuestas"],
        "precision": div(c["recuperadas"], c["recuperadas"] + c["falsas"]),
        "exhaustividad": div(c["recuperadas"], c["verdaderas"]),
        "dir_correcta": div(c["dir_correcta"], c["recuperadas"]), "dir_invertida": div(c["dir_invertida"], c["recuperadas"]),
        "dir_abierta": div(c["dir_abierta"], c["recuperadas"]),
        "acierto_entre_orientadas": div(c["dir_correcta"], dirigidas),
        "dir_aprendida_correcta": div(c["dir_aprendida_correcta"], c["dir_aprendida_correcta"] + c["dir_aprendida_invertida"] + c["dir_aprendida_abierta"]),
        "dir_fijada_correcta": div(c["dir_fijada_correcta"], c["dir_fijada_correcta"] + c["dir_fijada_invertida"] + c["dir_fijada_abierta"]),
        "falsas": c["falsas"], "falsas_confusion_latente": c["falsa_confusion_latente"],
        "falsas_camino_indirecto": c["falsa_camino_indirecto"], "falsas_causa_comun": c["falsa_causa_comun"],
        "falsas_otra": c["falsa_otra"],
        "confusion_senalada": div(c["confusion_senalada"], c["pares_confundidos"]),
        "sin_relacion_sin_conexion": div(c["pares_sin_relacion_sin_conexion"], c["pares_sin_relacion"]),
        "errores": c["errores"],
    }


def etapa1(clave: dict, pred: dict) -> dict:
    agregados = defaultdict(Counter)
    por_caso = {}
    for c in clave["casos"]:
        p = pred["casos"][c["caso"]]
        por_caso[c["caso"]] = {}
        for cond in ("A", "B"):
            for m in METODOS:
                r = evaluar_grafo(p[cond][m], c, cond)
                if m in p[cond].get("errores", {}):  # fallo del framework: cuenta como grafo vacío y se reporta
                    r["errores"] = 1
                por_caso[c["caso"]][f"{cond}_{m}"] = r
                for clave_ag in [(cond, m, "todos", "todos"), (cond, m, c["familia"], "todos"), (cond, m, c["familia"], c["n"]),
                                 (cond, m, "todos", c["n"])]:
                    agregados[clave_ag].update(r)
                    agregados[clave_ag]["casos"] += 1
    return {"por_caso": por_caso,
            "agregados": [{"condicion": k[0], "metodo": k[1], "familia": k[2], "n": k[3], **tasas(v)} for k, v in sorted(
                agregados.items(), key=lambda kv: (kv[0][0], METODOS.index(kv[0][1]), str(kv[0][2]), str(kv[0][3])))]}


def etapa2(clave: dict, pred: dict) -> dict:
    filas = []
    for c in clave["casos"]:
        p = pred["casos"][c["caso"]]
        par = c["pareja"]
        assert (p["x"], p["y"]) == (par["x"], par["y"]), c["caso"]
        verdad = par["efecto"]
        afirma = p["afirma_efecto"]
        cat = ("error" if p.get("error") else "acierto" if afirma and verdad else "afirmacion_falsa" if afirma
               else "omision" if verdad else "abstencion_correcta")
        fila = {"caso": c["caso"], "familia": c["familia"], "n": c["n"], "tipo": par["tipo"], "efecto_verdadero": verdad,
                "dictamen": p["dictamen"], "resultado": cat, "arista_restituida": p["arista_restituida"],
                "restitucion": (None if not p["arista_restituida"] else "verdadera" if verdad else "falsa"),
                "identificacion": p["identificacion"].get("estado")}
        verdadero = par.get("efecto_total_estandarizado")
        e = p.get("estimacion_principal")
        if verdadero is not None and e and e.get("ic95"):
            lo, hi = e["ic95"]
            fila.update(efecto_total=verdadero, estimado=e["efecto"], cubre=lo <= verdadero <= hi)
        if p.get("ida"):  # efectos posibles cuando la clase de equivalencia no identifica el efecto
            fila.update(ida_min=p["ida"]["min"], ida_max=p["ida"]["max"], efecto_total=verdadero)
        filas.append(fila)
    por_tipo = defaultdict(Counter)
    for f in filas:
        por_tipo[f["tipo"]][f["resultado"]] += 1
        por_tipo[f["tipo"]][f["dictamen"]] += 1
        por_tipo[f["tipo"]]["casos"] += 1
        if f["restitucion"]:
            por_tipo[f["tipo"]][f"restituida_{f['restitucion']}"] += 1
    cob = [f["cubre"] for f in filas if "cubre" in f]
    return {"por_caso": filas, "por_tipo": {k: dict(v) for k, v in por_tipo.items()},
            "resumen": dict(Counter(f["resultado"] for f in filas)),
            "restituciones": dict(Counter(f["restitucion"] for f in filas if f["restitucion"])),
            "cobertura_ic95_lineales": {"casos": len(cob), "cubre": sum(cob)}}


def tabla(filas, columnas, encabezados):
    out = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    for f in filas:
        out.append("| " + " | ".join("—" if f.get(c) is None else (f"{100 * f[c]:.0f}%" if isinstance(f[c], float) and c not in ("n",) else str(f[c])) for c in columnas) + " |")
    return "\n".join(out)


def informe(cal: dict) -> str:
    L = ["# Evaluación del descubrimiento causal: calificación", ""]
    L += [f"- {k}: {v}" for k, v in cal["verificacion"].items()] + [""]
    ag = cal["etapa1"]["agregados"]
    cols = ["precision", "exhaustividad", "dir_correcta", "dir_invertida", "dir_abierta", "falsas", "falsas_confusion_latente",
            "falsas_camino_indirecto", "sin_relacion_sin_conexion"]
    enc = ["Precisión", "Exhaustividad", "Dirección correcta", "Invertida", "Abierta", "Conexiones falsas", "…por confusión no observada",
           "…por camino indirecto", "Pares sin relación sin conexión"]
    for cond, titulo in (("A", "Condición A, sin orden temporal"), ("B", "Condición B, con orden por capas")):
        filas = [dict(r, etiqueta=f"{r['familia']} · {r['n']}") for r in ag if r["condicion"] == cond and r["metodo"] == "consenso_PC"
                 and (r["n"] == "todos" or r["familia"] == "todos")]
        L += [f"## Etapa 1, consenso de PC (salida principal). {titulo}", "", tabla(filas, ["etiqueta"] + cols, ["Grupo"] + enc), ""]
    filas = [dict(r, etiqueta=f"{r['condicion']} · {r['metodo']}") for r in ag if r["familia"] == "todos" and r["n"] == "todos"]
    fallos = [f"{r['condicion']} · {r['metodo']}: {r['errores']}" for r in filas if r.get("errores")]
    L += ["Casos en que el framework se detuvo con un error (se califican como grafo vacío): "
          + ("; ".join(fallos) if fallos else "ninguno") + ".", ""]
    L += ["## Etapa 1, todos los métodos (secundario)", "",
          tabla(filas, ["etiqueta", "precision", "exhaustividad", "dir_correcta", "dir_abierta", "acierto_entre_orientadas",
                        "dir_aprendida_correcta", "dir_fijada_correcta", "confusion_senalada"],
                ["Condición · método", "Precisión", "Exhaustividad", "Dirección correcta", "Abierta", "Acierto entre orientadas",
                 "Aprendidas correctas", "Fijadas por capas correctas", "Confusión señalada (<->)"]), ""]
    e2 = cal.get("etapa2")
    if e2:
        tipos = sorted(e2["por_tipo"])
        filas = [{"tipo": t, **{k: e2["por_tipo"][t].get(k, 0) for k in ("casos", "acierto", "omision", "afirmacion_falsa",
                                                                        "abstencion_correcta", "error", "restituida_verdadera",
                                                                        "restituida_falsa")}}
                 for t in tipos]
        L += ["## Etapa 2, módulo causal completo con el par revelado (condición B)", "",
              tabla(filas, ["tipo", "casos", "acierto", "omision", "afirmacion_falsa", "abstencion_correcta", "error",
                            "restituida_verdadera", "restituida_falsa"],
                    ["Tipo de par", "Casos", "Afirma un efecto real", "Omite un efecto real", "Afirma un efecto inexistente",
                     "Se abstiene sin efecto", "Error del módulo", "Prueba dirigida agrega conexión verdadera", "…falsa"]), "",
              f"Cobertura del IC95 en familias lineales: {e2['cobertura_ic95_lineales']['cubre']} de "
              f"{e2['cobertura_ic95_lineales']['casos']} casos con estimación.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clave", type=Path, required=True)
    ap.add_argument("--resultados", type=Path, required=True)
    ap.add_argument("--parejas", type=Path)
    ap.add_argument("--datos", type=Path, help="entregar.zip, para comprobarlo contra la frase de la clave")
    ap.add_argument("--huella-parejas")
    ap.add_argument("--huella-clave")
    a = ap.parse_args()
    clave_b = a.clave.read_bytes()
    clave = json.loads(clave_b)
    ver = {}
    if a.huella_clave:
        ver["clave.json coincide con la huella impresa al generar"] = sha(clave_b) == a.huella_clave
    if a.parejas and a.huella_parejas:
        ver["parejas.json coincide con la huella impresa al generar"] = sha(a.parejas.read_bytes()) == a.huella_parejas
    preds = {}
    for etapa in (1, 2):
        f = a.resultados / f"predicciones_etapa{etapa}.json"
        if f.exists():
            b = f.read_bytes()
            h = (a.resultados / f"huella_etapa{etapa}.txt").read_text().split()[0]
            ver[f"predicciones de la etapa {etapa} coinciden con su huella"] = sha(b) == h
            preds[etapa] = json.loads(b)
    if a.datos:
        sys.path.insert(0, str(AQUI))
        from generar_casos import generar
        o = clave["opciones"]
        z, _, _ = generar(clave["frase"], tuple(o["familias"]), tuple(o["tamanos"]), o["replicas"], tuple(o["pesos"]))
        ver["los archivos de entregar.zip se reproducen con la frase de la clave"] = contenido_zip(z) == contenido_zip(a.datos.read_bytes())
    if 2 in preds:
        sin_error = [v for v in preds[2]["casos"].values() if not v.get("error")]
        iguales = [v.get("grafo_igual_al_de_la_etapa1") for v in sin_error]
        ver["el grafo de la etapa 2 (antes de la prueba dirigida) es el de la etapa 1"] = all(x is True for x in iguales)
        ver["casos de la etapa 2 que terminaron sin error"] = f"{len(sin_error)} de {len(preds[2]['casos'])}"
    cal = {"verificacion": ver, "etapa1": etapa1(clave, preds[1])}
    if 2 in preds:
        cal["etapa2"] = etapa2(clave, preds[2])
    (a.resultados / "calificacion.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1) + "\n")
    md = informe(cal)
    (a.resultados / "calificacion.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
