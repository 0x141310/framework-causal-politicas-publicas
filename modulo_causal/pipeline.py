"""Orquestación del Módulo 2: especificación -> descubrimiento -> identificación
-> estimación -> refutación -> dictamen.

Para no usar dos veces los mismos datos, la estructura se descubre en una
partición y el efecto se estima en otra (división aleatoria o temporal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import discovery as disc
from . import estimation as est
from . import identification as ident
from . import refutation as ref
from .graphs import CausalGraph
from .spec import CausalSpec


def split_sample(df: pd.DataFrame, how="random", frac: float = 0.5, seed: int = 0, time_col: str | None = None,
                 cut=None):
    if how is None:
        return df, df
    if how == "random":
        rng = np.random.default_rng(seed)
        m = rng.random(len(df)) < frac
        return df[m].reset_index(drop=True), df[~m].reset_index(drop=True)
    if how == "time":
        return df[df[time_col] <= cut].reset_index(drop=True), df[df[time_col] > cut].reset_index(drop=True)
    raise ValueError(how)


def _is_binary(s: pd.Series) -> bool:
    return set(pd.unique(s.dropna())) <= {0, 1}


def estimate_all(df, x, y, Z, block=None, flexible=True, estimand: str = "ATE") -> list[dict]:
    out = [est.linear_adjustment(df, x, y, Z)]
    if _is_binary(df[x]) and Z:
        out.append(est.aipw_att(df, x, y, Z, flexible=flexible) if estimand == "ATT"
                   else est.aipw(df, x, y, Z, flexible=flexible))
    elif Z:
        out.append(est.dml_plr(df, x, y, Z))
    return out


def ida_range(df, x, y, sets: list) -> dict:
    """Efectos posibles (lineales) para cada conjunto de ajuste de la clase."""
    effs = []
    for s in {frozenset(s) if s is not None else None for s in sets}:
        if s is None:
            effs.append(0.0)
        else:
            effs.append(est.linear_adjustment(df, x, y, sorted(s))["efecto"])
    return {"efectos_posibles": sorted(effs), "min": min(effs), "max": max(effs)}


def verdict(identif: dict, fci_diag: dict | None, main: dict | None, refs: list[dict], sens: dict | None) -> dict:
    """Dictamen con niveles explícitos de evidencia."""
    if identif["estado"] == "sin_camino_causal":
        return {"nivel": "sin_efecto_estructural", "texto": "El grafo candidato no contiene un camino causal de la intervención al resultado."}
    if identif["estado"] != "identificado":
        return {"nivel": "no_identificable", "texto": f"No identificable con el grafo candidato: {identif.get('motivo', '')}"}
    latent = fci_diag is not None and fci_diag.get("estado") in ("confusion_latente", "ambigua", "dirigida_no_visible")
    lo, hi = main["ic95"]
    distinct = not (lo <= 0 <= hi)
    passed = all(r.get("supera", True) for r in refs if r["prueba"] != "tratamiento placebo")
    placebo = next((r for r in refs if r["prueba"] == "tratamiento placebo"), None)
    parts = []
    if distinct and passed and (placebo is None or placebo["supera"]):
        nivel = "efecto_identificado_bajo_supuestos"
        parts.append("Efecto identificado por ajuste de puerta trasera, distinto de cero y estable ante refutaciones.")
    elif not distinct:
        nivel = "identificado_no_distinguible_de_cero"
        parts.append("Efecto identificado, pero su intervalo incluye cero.")
    else:
        nivel = "identificado_fragil"
        parts.append("Efecto identificado, pero no supera todas las refutaciones.")
    if latent:
        parts.append(f"FCI no descarta confusión latente ({fci_diag.get('motivo')}); la conclusión depende de suficiencia causal.")
    if fci_diag is not None and fci_diag.get("estado") == "sin_adyacencia":
        parts.append("FCI no conserva la arista intervención–resultado: la evidencia estructural es débil y el efecto descansa en la prueba dirigida.")
    if sens:
        parts.append(f"Un confusor no medido tendría que explicar al menos {100*sens['RV_q1']:.1f}% de la varianza residual de la intervención y del resultado para anular el efecto ({100*sens['RV_q1_a05']:.1f}% para volverlo no significativo al 5%).")
        cotas = sens.get("cotas")
        lin_sig = sens["RV_q1_a05"] > 0
        if not lin_sig:
            parts.append("La estimación lineal de referencia ya no es significativa al 5%; las cotas de sensibilidad se reportan sin dictamen adicional.")
        if cotas is not None and len(cotas) and lin_sig:
            k1 = cotas[cotas["k"] == 1]
            weak = [r["referencia"] for _, r in k1.iterrows() if r["ic95_ajustado"][0] <= 0 <= r["ic95_ajustado"][1] or r["anula_signo"]]
            if weak:
                parts.append("Un confusor tan fuerte como " + ", ".join(weak) +
                             " bastaría para que el intervalo incluya cero: el efecto es frágil ante confusión comparable a la observada.")
                if nivel == "efecto_identificado_bajo_supuestos":
                    nivel = "identificado_fragil"
        if sens["RV_q1_a05"] < 0.02 and nivel == "efecto_identificado_bajo_supuestos":
            nivel = "identificado_fragil"
    return {"nivel": nivel, "texto": " ".join(parts), "confusion_latente_posible": bool(latent)}


def run_module(df: pd.DataFrame, spec: CausalSpec, *, alpha: float = 0.01, B: int = 50, threshold: float = 0.6,
               split="random", split_kw: dict | None = None, expert_dag=None, benchmarks: list[str] | None = None,
               methods=("PC", "FCI", "GES", "LiNGAM"), block: int | None = None, flexible: bool = True,
               negative_controls: list[str] | None = None, seed: int = 0, n_placebo: int = 30,
               estimand: str = "ATE") -> dict:
    x, y = spec.treatment, spec.outcome
    names = spec.names
    data = df[names + [c for c in (negative_controls or []) if c not in names]].dropna().reset_index(drop=True)
    d_disc, d_est = split_sample(data, split, seed=seed, **(split_kw or {}))
    R: dict = {"n_total": len(data), "n_descubrimiento": len(d_disc), "n_estimacion": len(d_est)}

    # 2. descubrimiento
    graphs = disc.run_all(d_disc[names], spec, methods=methods, alpha=alpha)
    R["grafos"] = graphs
    stab_pc = disc.bootstrap_stability(d_disc[names], spec, "PC", B=B, block=block, seed=seed, threshold=threshold, alpha=alpha)
    R["estabilidad_PC"] = disc.stability_table(stab_pc)
    candidate: CausalGraph = stab_pc["consenso"]
    # prueba dirigida de efecto total: x ⟂ y | Pa(x) bajo ausencia de efecto
    pa_x = sorted(candidate.possible_parents(x) - {y})
    te = ident.total_effect_test(d_disc, x, y, pa_x)
    R["prueba_efecto_total"] = te
    import networkx as nx
    has_path = nx.has_path(candidate.to_networkx(), x, y)
    if not candidate.adjacent(x, y) and not has_path and te["p"] < alpha and not spec.is_forbidden(x, y):
        candidate = candidate.with_edge(x, y, "-->")
        R["arista_restituida"] = (
            f"{x} -> {y} restituida: {x} y {y} son dependientes dado Pa({x}) (p={te['p']:.2g}) aunque PC eliminó la arista; "
            f"correlación marginal {te['r_marginal']:.3f} frente a parcial {te['r_parcial']:.3f} (posible cancelación por confusión)")
    R["grafo_candidato"] = candidate
    if "FCI" in methods:
        stab_fci = disc.bootstrap_stability(d_disc[names], spec, "FCI", B=max(B // 2, 10), block=block, seed=seed + 1,
                                            threshold=threshold, alpha=alpha)
        R["estabilidad_FCI"] = disc.stability_table(stab_fci)
        R["fci_diagnostico"] = ident._pag_diagnostic(stab_fci["consenso"], x, y)
    # acuerdo entre algoritmos sobre la arista x-y
    R["arista_xy_por_metodo"] = {m: g.type(x, y) for m, g in graphs.items()}

    # 3. identificación
    identif = ident.identify(candidate, x, y, spec)
    R["identificacion"] = {k: v for k, v in identif.items() if not k.startswith("_")}
    if expert_dag is not None:
        R["identificacion_experto"] = {
            "ajuste_optimo": sorted(ident.optimal_adjustment_set(expert_dag, x, y) or []),
            "implicaciones": ident.testable_implications(expert_dag, d_est[names]),
        }
    if identif["estado"] in ("identificado",) and "_dags" in identif:
        R["implicaciones_candidato"] = ident.testable_implications(identif["_dags"][0], d_est[names])

    # 4. estimación en la partición de estimación
    main, refs, sens = None, [], None
    pre = [v for v in names if spec.tier(v) < spec.tier(x)]
    if identif["estado"] == "identificado":
        Z = identif["ajuste"]
        R["estimaciones"] = estimate_all(d_est, x, y, Z, flexible=flexible, estimand=estimand)
        main = R["estimaciones"][-1]
        # contraste con el conjunto pretratamiento completo (criterio de causa disyuntiva):
        # si difiere, el grafo descubierto podría omitir confusores
        if sorted(pre) != sorted(Z):
            R["estimaciones_pretratamiento"] = estimate_all(d_est, x, y, pre, flexible=flexible, estimand=estimand)
            alt = R["estimaciones_pretratamiento"][-1]
            lo1, hi1 = main["ic95"]
            lo2, hi2 = alt["ic95"]
            R["discrepancia_ajuste"] = {
                "efecto_descubierto": main["efecto"], "efecto_pretratamiento": alt["efecto"],
                "ic_se_traslapan": bool(lo1 <= hi2 and lo2 <= hi1),
                "dif_relativa": abs(main["efecto"] - alt["efecto"]) / (abs(alt["efecto"]) + 1e-12),
            }
        lin = lambda d, a, b, z: est.linear_adjustment(d, a, b, z)
        refs = [
            ref.placebo_treatment(d_est, x, y, Z, lin, n=n_placebo, seed=seed, block=block),
            ref.random_common_cause(d_est, x, y, Z, lin, seed=seed),
            ref.subset_stability(d_est, x, y, Z, lin, seed=seed),
        ]
        for nc in negative_controls or []:
            refs.append(ref.negative_control_outcome(d_est, x, nc, Z, lin))
        bm = [b for b in (benchmarks or []) if b in Z]
        sens = ref.sensitivity_summary(d_est, x, y, Z, benchmarks=bm or None)
    elif identif["estado"] == "no_identificado_por_clase":
        R["ida"] = ida_range(d_est, x, y, identif["_sets"])
    R["refutaciones"] = refs
    R["sensibilidad"] = sens
    R["dictamen"] = verdict(identif, R.get("fci_diagnostico"), main, refs, sens) if identif["estado"] == "identificado" \
        else verdict(identif, None, None, [], None)
    disc_ = R.get("discrepancia_ajuste")
    if disc_ and (not disc_["ic_se_traslapan"] or disc_["dif_relativa"] > 0.5):
        R["dictamen"]["texto"] += (f" Advertencia: el efecto con el ajuste descubierto ({disc_['efecto_descubierto']:.3f}) difiere del "
                                   f"obtenido con todas las variables pretratamiento ({disc_['efecto_pretratamiento']:.3f}); "
                                   "el grafo descubierto podría omitir confusores.")
        if R["dictamen"]["nivel"] == "efecto_identificado_bajo_supuestos":
            R["dictamen"]["nivel"] = "identificado_fragil"
    return R


def print_report(R: dict, spec: CausalSpec) -> str:
    L = []
    L.append(f"Intervención: {spec.treatment}  →  Resultado: {spec.outcome}")
    L.append(f"n = {R['n_total']} (descubrimiento {R['n_descubrimiento']}, estimación {R['n_estimacion']})")
    L.append("\nArista intervención–resultado por algoritmo: " + ", ".join(f"{m}: {t}" for m, t in R["arista_xy_por_metodo"].items()))
    te = R.get("prueba_efecto_total")
    if te:
        L.append(f"Prueba dirigida de efecto total: r_parcial={te['r_parcial']:.3f} (p={te['p']:.2g}) dado {te['dado']}; r_marginal={te['r_marginal']:.3f}")
    if "arista_restituida" in R:
        L.append("  " + R["arista_restituida"])
    L.append("\n" + str(R["grafo_candidato"]))
    L.append("\nIdentificación: " + str(R["identificacion"]))
    if "fci_diagnostico" in R:
        L.append("FCI: " + str(R["fci_diagnostico"]))
    for e in R.get("estimaciones", []):
        L.append(f"  {e['estimador']}: {e['efecto']:.4f}  IC95 [{e['ic95'][0]:.4f}, {e['ic95'][1]:.4f}]  n={e['n']}")
    for e in R.get("estimaciones_pretratamiento", []):
        L.append(f"  [pretratamiento completo] {e['estimador']}: {e['efecto']:.4f}  IC95 [{e['ic95'][0]:.4f}, {e['ic95'][1]:.4f}]")
    if "ida" in R:
        L.append("  IDA (efectos posibles): " + str(np.round(R["ida"]["efectos_posibles"], 4)))
    for r in R["refutaciones"]:
        L.append("  Refutación: " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    if R["sensibilidad"]:
        s = R["sensibilidad"]
        L.append(f"  Sensibilidad: RV={s['RV_q1']:.3f}, RV(α=0.05)={s['RV_q1_a05']:.3f}, R²parcial={s['R2_parcial_Y~D|X']:.4f}")
    L.append("\nDICTAMEN: " + R["dictamen"]["nivel"] + " — " + R["dictamen"]["texto"])
    return "\n".join(L)
