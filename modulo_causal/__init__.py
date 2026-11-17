"""Módulo 2 del framework: descubrimiento y estimación causal.

Flujo: CausalSpec (roles y capas) -> discovery (PC, FCI, GES, LiNGAM con
estabilidad por remuestreo) -> identification (criterio de ajuste, clase de
equivalencia, diagnóstico de confusión latente, implicaciones comprobables)
-> estimation (MCO, AIPW, DML, g-computación) -> refutation (placebo,
causa común aleatoria, submuestras, sensibilidad) -> dictamen.
"""
from .spec import CausalSpec, Variable
from .graphs import CausalGraph, compare_to_truth, consensus_graph, edge_frequencies
from . import discovery, identification, estimation, refutation, pipeline

__all__ = ["CausalSpec", "Variable", "CausalGraph", "compare_to_truth", "consensus_graph", "edge_frequencies",
           "discovery", "identification", "estimation", "refutation", "pipeline"]
