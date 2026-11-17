"""Extensión experimental de PPO en un entorno sintético.

No usa datos reales ni recomienda políticas. Pregunta: con transiciones conocidas de varios
años, ¿aprende PPO una regla de incorporación mejor que la regla miope de una etapa, y qué
pasa cuando se evalúa en un mundo distinto al de entrenamiento?

Entorno: 2,000 mujeres sintéticas de 55 a 64 años, 6 años. Elegibles de 60 a 64; a los 65
salen. Probabilidad anual de inseguridad q_t = r + rho * q_{t-1} - delta * inscrita (campo
medio, sin ruido de Bernoulli). Inscripción permanente, 150 registros por año y 500 lugares.
Mundos: A (delta constante), B (delta baja con el riesgo, como E1), C (delta sube con el riesgo).
Acción anual: pesos de prioridad sobre riesgo, edad y ruralidad.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _Path

ROOT = _os.environ.get("FRAMEWORK_DIR", str(_Path(__file__).resolve().parents[1]))


import itertools
import json
import time
from multiprocessing import Pool

import numpy as np

OUT = ROOT + "/resultados"
N, T, K, BMAX, RHO = 2000, 6, 150, 500, 0.3
EVAL_SEEDS = list(range(10_000, 10_200))  # evaluación de todas las reglas
BUSQUEDA_SEEDS = list(range(20_000, 20_040))  # búsqueda de los pesos del óptimo de la clase (distintas de las de evaluación)
WORLDS = ("A", "B", "C")
TIMESTEPS = 60_000
SEMILLAS_PPO = tuple(range(1, 11))  # 10 semillas de entrenamiento (Anexo B)


def effect(world: str, r: np.ndarray) -> np.ndarray:
    if world == "A":
        return np.full_like(r, 0.025)
    if world == "B":
        return 0.05 * np.exp(-10 * r)
    if world == "C":
        return 0.32 * r
    raise ValueError(world)


class Sim:
    """Simulador sin dependencias de gymnasium (también lo usa el entorno de PPO)."""

    def __init__(self, world: str):
        self.world = world

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self.age = rng.integers(55, 65, N)
        self.rural = (rng.random(N) < 0.34).astype(float)
        self.r = np.clip(rng.beta(1.2, 14, N) * np.where(self.rural == 1, 1.3, 1.0), 0.002, 0.6)
        self.delta = effect(self.world, self.r)
        self.q = self.r / (1 - RHO)
        self.enrolled = np.zeros(N, bool)
        self.t = 0
        self.zr = (self.r - self.r.mean()) / self.r.std()
        self.tiebreak = rng.random(N)  # desempate neutral para la regla miope
        return self.obs()

    def obs(self) -> np.ndarray:
        elig = (self.age >= 60) & (self.age <= 64) & ~self.enrolled
        n_e = max(int(elig.sum()), 1)
        return np.array([self.t / T, self.enrolled.sum() / BMAX, float(self.zr[elig].mean()) if elig.any() else 0.0,
                         float(((self.age >= 63) & elig).sum() / n_e)], dtype=np.float32)

    def step_scores(self, score: np.ndarray) -> float:
        elig = (self.age >= 60) & (self.age <= 64) & ~self.enrolled
        slots = int(min(K, BMAX - self.enrolled.sum(), elig.sum()))
        if slots > 0:
            idx = np.flatnonzero(elig)
            top = idx[np.argsort(-score[idx], kind="mergesort")[:slots]]
            self.enrolled[top] = True
        active = self.enrolled & (self.age <= 64)
        self.q = np.clip(self.r + RHO * self.q - self.delta * active, 0.001, 0.999)
        reward = -100.0 * float(self.q[self.age <= 64].sum()) / N
        self.age = self.age + 1
        self.enrolled &= self.age <= 64
        self.t += 1
        return reward

    def weights_to_score(self, a: np.ndarray) -> np.ndarray:
        za = (self.age - 62) / 2.0
        return a[0] * self.zr + a[1] * za + a[2] * self.rural


def run(world: str, policy, seeds=EVAL_SEEDS, assumed_world: str | None = None) -> np.ndarray:
    """Retorno total por episodio. policy(sim, obs, mundo_supuesto) -> puntajes por mujer, o None (nadie)."""
    sim = Sim(world)
    out = []
    for s in seeds:
        obs = sim.reset(s)
        tot = 0.0
        for _ in range(T):
            sc = policy(sim, obs, assumed_world or world)
            tot += _no_enroll_step(sim) if sc is None else sim.step_scores(sc)
            obs = sim.obs()
        out.append(tot)
    return np.array(out)


def _no_enroll_step(sim: Sim) -> float:
    active = sim.enrolled & (sim.age <= 64)
    sim.q = np.clip(sim.r + RHO * sim.q - sim.delta * active, 0.001, 0.999)
    reward = -100.0 * float(sim.q[sim.age <= 64].sum()) / N
    sim.age = sim.age + 1
    sim.enrolled &= sim.age <= 64
    sim.t += 1
    return reward


POLICIES = {
    "nadie": lambda sim, obs, w: None,
    "miope_una_etapa": lambda sim, obs, w: np.minimum(effect(w, sim.r), sim.r + RHO * sim.q - 0.001) + 1e-9 * sim.tiebreak,
    "por_edad": lambda sim, obs, w: sim.age.astype(float) + 1e-6 * sim.zr,
    "por_riesgo": lambda sim, obs, w: sim.r.copy(),
    "al_azar": lambda sim, obs, w: np.random.default_rng(int(sim.t * 7919 + int(sim.r.sum() * 1e6))).random(N),
}


def static_policy(a):
    a = np.asarray(a, float)
    return lambda sim, obs, w: sim.weights_to_score(a)


def grid_best(world: str) -> dict:
    vals = (-1.0, -0.5, 0.0, 0.5, 1.0)
    best, best_a = -np.inf, None
    for a in itertools.product(vals, vals, vals):
        if not any(a):
            continue
        m = run(world, static_policy(a), seeds=BUSQUEDA_SEEDS).mean()
        if m > best:
            best, best_a = m, a
    return {"pesos": list(best_a), "retorno_40": float(best)}


def train_ppo(args):
    world, seed = args
    import gymnasium as gym
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO
    torch.set_num_threads(1)

    class Env(gym.Env):
        def __init__(self):
            self.sim = Sim(world)
            self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
            self.observation_space = spaces.Box(-10.0, 10.0, (4,), np.float32)
            self.k = seed * 1_000_000

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.k += 1
            return self.sim.reset(self.k), {}

        def step(self, action):
            r = self.sim.step_scores(self.sim.weights_to_score(np.asarray(action, float)))
            return self.sim.obs(), r, self.sim.t >= T, False, {}

    t0 = time.time()
    model = PPO("MlpPolicy", Env(), n_steps=600, batch_size=120, n_epochs=10, gamma=1.0, gae_lambda=0.95,
                learning_rate=3e-4, policy_kwargs={"net_arch": [32, 32]}, seed=seed, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)

    def pol(sim, obs, w):
        a, _ = model.predict(obs, deterministic=True)
        return sim.weights_to_score(np.asarray(a, float))
    evals = {w: run(w, pol).tolist() for w in WORLDS}
    # acciones promedio por año en el mundo de entrenamiento (para leer la regla aprendida)
    sim = Sim(world)
    acts = []
    for s in EVAL_SEEDS[:20]:
        obs = sim.reset(s)
        row = []
        for _ in range(T):
            a, _ = model.predict(obs, deterministic=True)
            row.append(np.asarray(a, float).tolist())
            sim.step_scores(sim.weights_to_score(np.asarray(a, float)))
            obs = sim.obs()
        acts.append(row)
    return {"mundo_entrenamiento": world, "semilla": seed, "segundos": time.time() - t0, "eval": evals,
            "acciones_medias_por_anio": np.mean(np.array(acts), axis=0).round(3).tolist()}


def main():
    t0 = time.time()
    R = {"parametros": {"N": N, "T": T, "K": K, "BMAX": BMAX, "RHO": RHO,
                   "episodios_eval": len(EVAL_SEEDS), "episodios_busqueda": len(BUSQUEDA_SEEDS), "pasos_ppo": TIMESTEPS}}
    base = {w: run(w, POLICIES["nadie"]) for w in WORLDS}
    R["heuristicas"] = {}
    for w in WORLDS:
        R["heuristicas"][w] = {}
        for name, pol in POLICIES.items():
            if name == "nadie":
                continue
            for assumed in (WORLDS if name == "miope_una_etapa" else (w,)):
                ret = run(w, pol, assumed_world=assumed)
                key = name if assumed == w else f"{name}_supone_{assumed}"
                R["heuristicas"][w][key] = {"evitados_media": float((ret - base[w]).mean() * N / 100),
                                            "evitados_ic95": [float(x) for x in np.percentile((ret - base[w]) * N / 100, [2.5, 97.5])]}
    print("heurísticas listas", round(time.time() - t0), flush=True)
    R["optimo_clase"] = {}
    for w in WORLDS:
        g = grid_best(w)
        ret = run(w, static_policy(g["pesos"]))
        R["optimo_clase"][w] = {"pesos": g["pesos"], "evitados_media": float((ret - base[w]).mean() * N / 100)}
    print("óptimo de la clase:", R["optimo_clase"], round(time.time() - t0), flush=True)
    jobs = [(w, s) for w in WORLDS for s in SEMILLAS_PPO]
    with Pool(2) as p:
        ppo = p.map(train_ppo, jobs)
    R["ppo"] = []
    for r in ppo:
        row = {k: v for k, v in r.items() if k != "eval"}
        row["evitados_media"] = {w: float((np.array(r["eval"][w]) - base[w]).mean() * N / 100) for w in WORLDS}
        R["ppo"].append(row)
        print(r["mundo_entrenamiento"], r["semilla"], {w: round(v, 1) for w, v in row["evitados_media"].items()},
              "acciones año 1 y 6:", row["acciones_medias_por_anio"][0], row["acciones_medias_por_anio"][-1], flush=True)
    with open(f"{OUT}/ppo_m4.json", "w") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps({w: R["heuristicas"][w] for w in WORLDS}, ensure_ascii=False))
    print(f"listo ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
