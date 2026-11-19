"""Verifica una ronda de la cadena default de drand (pedersen-bls-chained) con la clave pública de la red.

    python3 verificar_drand.py ronda.json
"""
import hashlib
import json
import sys

from py_ecc.bls import G2Basic

CLAVE_PUBLICA = bytes.fromhex("868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31")
GENESIS, PERIODO = 1595431050, 30

r = json.load(open(sys.argv[1]))
firma = bytes.fromhex(r["signature"])
previa = bytes.fromhex(r["previous_signature"])
mensaje = hashlib.sha256(previa + int(r["round"]).to_bytes(8, "big")).digest()
print("ronda", r["round"], "emitida (s Unix)", GENESIS + (int(r["round"]) - 1) * PERIODO)
print("randomness = SHA-256(firma):", hashlib.sha256(firma).hexdigest() == r["randomness"])
print("firma BLS válida con la clave pública de la red:", G2Basic.Verify(CLAVE_PUBLICA, mensaje, firma))
