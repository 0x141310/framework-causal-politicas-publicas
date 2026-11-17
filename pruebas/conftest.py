"""Configuración de pytest: la raíz del repositorio en la ruta de importación."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
