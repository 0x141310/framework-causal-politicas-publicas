"""Especificación causal previa al descubrimiento (Módulo 2, Sección IV.B).

Registra el papel de cada variable (intervención, resultado, covariable previa,
contexto, mediador), su orden temporal (capa) y las restricciones del dominio.
Las capas se traducen en conocimiento previo: ninguna arista puede ir de una
capa posterior a una anterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np

ROLES = {"intervencion", "resultado", "previa", "contexto", "mediador", "posterior"}
KINDS = {"continua", "binaria", "ordinal", "conteo"}


@dataclass
class Variable:
    name: str
    role: str
    tier: int
    kind: str = "continua"
    description: str = ""
    source: str = ""

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Rol desconocido para {self.name}: {self.role}")
        if self.kind not in KINDS:
            raise ValueError(f"Tipo desconocido para {self.name}: {self.kind}")
        if self.tier < 0:
            raise ValueError("La capa temporal debe ser un entero no negativo")


@dataclass
class CausalSpec:
    variables: list[Variable]
    treatment: str
    outcome: str
    forbidden: list[tuple[str, str]] = field(default_factory=list)
    required: list[tuple[str, str]] = field(default_factory=list)
    exogenous: list[str] = field(default_factory=list)  # sin padres posibles
    notes: str = ""

    def __post_init__(self):
        names = self.names
        for v in (self.treatment, self.outcome):
            if v not in names:
                raise ValueError(f"{v} no está en la especificación")
        if self.tier(self.treatment) > self.tier(self.outcome):
            raise ValueError("La intervención no puede ser posterior al resultado")

    # ------------------------------------------------------------------ util
    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def var(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def tier(self, name: str) -> int:
        return self.var(name).tier

    def subset(self, names: list[str]) -> "CausalSpec":
        keep = [v for v in self.variables if v.name in names]
        return CausalSpec(
            variables=keep,
            treatment=self.treatment,
            outcome=self.outcome,
            forbidden=[e for e in self.forbidden if e[0] in names and e[1] in names],
            required=[e for e in self.required if e[0] in names and e[1] in names],
            exogenous=[x for x in self.exogenous if x in names],
            notes=self.notes,
        )

    def is_forbidden(self, a: str, b: str) -> bool:
        """¿Está prohibida la arista a -> b?"""
        if (a, b) in self.forbidden:
            return True
        if b in self.exogenous:
            return True
        return self.tier(a) > self.tier(b)

    # ------------------------------------------------- conocimiento previo
    def background_knowledge(self, names: list[str] | None = None):
        """Conocimiento previo para causal-learn (PC y FCI)."""
        from causallearn.graph.GraphNode import GraphNode
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

        names = names or self.names
        bk = BackgroundKnowledge()
        nodes = {n: GraphNode(n) for n in names}
        for n in names:
            bk.add_node_to_tier(nodes[n], int(self.tier(n)))
        for a, b in self.forbidden:
            if a in nodes and b in nodes:
                bk.add_forbidden_by_node(nodes[a], nodes[b])
        for x in self.exogenous:
            if x in nodes:
                for n in names:
                    if n != x:
                        bk.add_forbidden_by_node(nodes[n], nodes[x])
        for a, b in self.required:
            if a in nodes and b in nodes:
                bk.add_required_by_node(nodes[a], nodes[b])
        return bk

    def lingam_prior(self, names: list[str] | None = None) -> np.ndarray:
        """Matriz de conocimiento previo para DirectLiNGAM.

        pk[i, j] = 0 significa que no existe camino dirigido de x_j a x_i;
        -1 significa desconocido; 1 significa camino requerido.
        """
        names = names or self.names
        p = len(names)
        pk = -np.ones((p, p), dtype=int)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                # camino b -> a imposible si b es posterior a a
                if self.tier(b) > self.tier(a) or a in self.exogenous:
                    pk[i, j] = 0
        for a, b in self.required:
            if a in names and b in names:
                pk[names.index(b), names.index(a)] = 1
        return pk

    def to_dict(self) -> dict:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "variables": [v.__dict__ for v in self.variables],
            "forbidden": self.forbidden,
            "required": self.required,
            "exogenous": self.exogenous,
            "notes": self.notes,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
