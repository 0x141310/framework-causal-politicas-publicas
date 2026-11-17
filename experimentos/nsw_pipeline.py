"""NSW observacional a través del pipeline completo del Módulo 2 (ATT)."""

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))

import sys
sys.path.insert(0, ROOT)
import causaldata
import pandas as pd
from modulo_causal import CausalSpec, Variable, pipeline
from benchmark_nsw import prep

nsw = prep(causaldata.nsw_mixtape.load_pandas().data)
cps = prep(causaldata.cps_mixtape.load_pandas().data)
obs = pd.concat([nsw[nsw.treat == 1], cps], ignore_index=True)
spec = CausalSpec([
    Variable("age", "contexto", 0), Variable("black", "contexto", 0, "binaria"), Variable("hisp", "contexto", 0, "binaria"),
    Variable("educ", "previa", 1), Variable("nodegree", "previa", 1, "binaria"), Variable("marr", "previa", 1, "binaria"),
    Variable("re74", "previa", 2), Variable("u74", "previa", 2, "binaria"),
    Variable("re75", "previa", 3), Variable("u75", "previa", 3, "binaria"),
    Variable("treat", "intervencion", 4, "binaria"), Variable("re78", "resultado", 5)],
    treatment="treat", outcome="re78", exogenous=["age", "black", "hisp"])
R = pipeline.run_module(obs, spec, alpha=0.01, B=30, split=None, estimand="ATT", benchmarks=["re75", "re74", "educ"],
                        n_placebo=100, seed=7)
print(pipeline.print_report(R, spec))
print("Discrepancia:", R.get("discrepancia_ajuste"))
