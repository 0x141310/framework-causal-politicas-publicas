"""Compara los archivos de una ejecución nueva con los de la versión publicada.

Uso: python comparar_resultados.py <carpeta publicada> <carpeta nueva>

Compara cada JSON, CSV y parquet de la carpeta publicada con el archivo del mismo nombre en la
carpeta nueva: los números, con tolerancia relativa de 1e-6 (absoluta de 1e-9); los textos, tal
cual. No compara los registros de ejecución (*.log) ni los tiempos que guardan algunos JSON.
ejecutar_todo.sh lo corre al final contra la copia que guarda en publicado/.

Termina con código 0 si todo coincide (idéntico o dentro de la tolerancia) y con código 1 si algún
archivo difiere o falta; con código 2 si una de las carpetas no existe o la publicada está vacía.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-6, 1e-9
OMITIR = ("segundos",)  # tiempos de ejecución
TIPOS = (".json", ".csv", ".parquet")


def hojas(obj, ruta=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from hojas(v, f"{ruta}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from hojas(v, f"{ruta}[{i}]")
    else:
        yield ruta, obj


def es_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def rel(a: float, b: float) -> float:
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 and math.isfinite(m) else 0.0


def comparar_json(a, b) -> tuple[list[str], float]:
    """Diferencias encontradas y diferencia relativa máxima entre los números que coinciden."""
    ha, hb = dict(hojas(a)), dict(hojas(b))
    difs, peor = [], 0.0
    for k in sorted(set(ha) | set(hb)):
        if any(o in k.lower() for o in OMITIR):
            continue
        if k not in ha or k not in hb:
            difs.append(f"{k}: sólo en la versión {'nueva' if k in hb else 'publicada'}")
            continue
        va, vb = ha[k], hb[k]
        if es_num(va) and es_num(vb):
            if math.isnan(va) and math.isnan(vb):
                continue
            if math.isclose(va, vb, rel_tol=RTOL, abs_tol=ATOL):
                peor = max(peor, rel(va, vb))
            else:
                difs.append(f"{k}: {va} -> {vb}")
        elif va != vb:
            difs.append(f"{k}: {str(va)[:40]!r} -> {str(vb)[:40]!r}")
    return difs, peor


def comparar_tabla(a: pd.DataFrame, b: pd.DataFrame) -> tuple[list[str], float]:
    difs, peor = [], 0.0
    nuevas = [str(c) for c in b.columns if c not in a.columns]
    faltan = [str(c) for c in a.columns if c not in b.columns]
    if nuevas:
        difs.append(f"columnas nuevas: {', '.join(nuevas[:5])}")
    if faltan:
        difs.append(f"faltan columnas: {', '.join(faltan[:5])}")
    if len(a) != len(b):
        return difs + [f"filas: {len(a)} -> {len(b)}"], peor
    for c in a.columns:
        if c not in b.columns:
            continue
        sa, sb = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            xa = sa.to_numpy(dtype=float, na_value=np.nan)
            xb = sb.to_numpy(dtype=float, na_value=np.nan)
            ok = np.isclose(xa, xb, rtol=RTOL, atol=ATOL, equal_nan=True)
            if not ok.all():
                i = int(np.flatnonzero(~ok)[0])
                difs.append(f"columna {c}: {int((~ok).sum())} valores distintos (fila {i}: {xa[i]} -> {xb[i]})")
                continue
            m = np.maximum(np.abs(xa), np.abs(xb))
            d = np.abs(xa - xb)
            usar = np.isfinite(m) & (m > 0) & np.isfinite(d)
            if usar.any():
                peor = max(peor, float(np.max(d[usar] / m[usar])))
        elif not sa.astype(str).equals(sb.astype(str)):
            difs.append(f"columna {c}: textos distintos")
    return difs, peor


def comparar(f: Path, g: Path) -> tuple[list[str], float]:
    if f.suffix == ".json":
        return comparar_json(json.loads(f.read_text(encoding="utf-8")), json.loads(g.read_text(encoding="utf-8")))
    leer = pd.read_csv if f.suffix == ".csv" else pd.read_parquet
    return comparar_tabla(leer(f), leer(g))


def verificar(pub: Path, nueva: Path) -> dict:
    """Compara las dos carpetas, imprime el detalle y devuelve cuántos archivos hay en cada estado."""
    cuenta = {"idéntico": 0, "coincide": 0, "difiere": 0, "falta": 0}
    print(f"== {nueva} frente a {pub}")
    for f in sorted(p for p in pub.iterdir() if p.suffix in TIPOS):
        g = nueva / f.name
        detalle = ""
        if not g.exists():
            estado, detalle = "falta", "no se generó"
        elif f.read_bytes() == g.read_bytes():
            estado = "idéntico"
        else:
            try:
                difs, peor = comparar(f, g)
            except Exception as e:  # noqa: BLE001
                difs, peor = [f"no se pudo comparar: {e!r}"[:200]], 0.0
            if difs:
                estado = "difiere"
                detalle = f"{len(difs)} diferencia{'s' if len(difs) > 1 else ''}; {difs[0]}"
            else:
                estado = "coincide"
                detalle = f"diferencia relativa máxima {peor:.1e}" if peor > 0 else "mismos valores"
        cuenta[estado] += 1
        print(f"  {estado:9s} {f.name}" + (f"  ({detalle})" if detalle else ""))
    for p in sorted(nueva.iterdir()):
        if p.suffix in TIPOS and not (pub / p.name).exists():
            print(f"  {'nuevo':9s} {p.name}  (no está en la versión publicada)")
    print(f"   idénticos: {cuenta['idéntico']}; dentro de la tolerancia: {cuenta['coincide']}; "
          f"distintos: {cuenta['difiere']}; faltantes: {cuenta['falta']} (no se comparan registros ni tiempos)")
    return cuenta


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pub, nueva = Path(sys.argv[1]), Path(sys.argv[2])
    for carpeta in (pub, nueva):
        if not carpeta.is_dir():
            print(f"error: no existe la carpeta {carpeta}")
            return 2
    if not any(p.suffix in TIPOS for p in pub.iterdir()):
        print(f"error: la carpeta publicada {pub} no tiene archivos que comparar")
        return 2
    cuenta = verificar(pub, nueva)
    return 1 if cuenta["difiere"] or cuenta["falta"] else 0


if __name__ == "__main__":
    sys.exit(main())
