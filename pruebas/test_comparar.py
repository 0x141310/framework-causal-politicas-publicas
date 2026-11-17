"""Verificador de la reproducción: señala archivos distintos o faltantes con un código de salida distinto de cero."""
import json
import sys

import pytest

import comparar_resultados as cr


def escribir(carpeta, nombre, datos):
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / nombre).write_text(json.dumps(datos), encoding="utf-8")


def correr(monkeypatch, pub, nueva):
    monkeypatch.setattr(sys, "argv", ["comparar_resultados.py", str(pub), str(nueva)])
    return cr.main()


@pytest.fixture
def publicado(tmp_path):
    escribir(tmp_path / "pub", "a.json", {"x": 1.0, "segundos": 10})
    escribir(tmp_path / "pub", "b.json", {"y": [1, 2, 3]})
    return tmp_path / "pub"


def test_idénticos_o_dentro_de_la_tolerancia(monkeypatch, tmp_path, publicado):
    escribir(tmp_path / "nueva", "a.json", {"x": 1.0 + 1e-9, "segundos": 99})  # los tiempos no se comparan
    escribir(tmp_path / "nueva", "b.json", {"y": [1, 2, 3]})
    escribir(tmp_path / "nueva", "c.json", {"z": 0})  # un archivo nuevo no es error
    assert correr(monkeypatch, publicado, tmp_path / "nueva") == 0


def test_un_valor_distinto_es_error(monkeypatch, tmp_path, publicado):
    escribir(tmp_path / "nueva", "a.json", {"x": 1.1, "segundos": 10})
    escribir(tmp_path / "nueva", "b.json", {"y": [1, 2, 3]})
    assert correr(monkeypatch, publicado, tmp_path / "nueva") == 1


def test_un_archivo_faltante_es_error(monkeypatch, tmp_path, publicado):
    escribir(tmp_path / "nueva", "a.json", {"x": 1.0, "segundos": 10})
    assert correr(monkeypatch, publicado, tmp_path / "nueva") == 1


def test_carpeta_nueva_vacia_o_inexistente(monkeypatch, tmp_path, publicado):
    (tmp_path / "vacia").mkdir()
    assert correr(monkeypatch, publicado, tmp_path / "vacia") == 1  # todo falta
    assert correr(monkeypatch, publicado, tmp_path / "no_existe") == 2
