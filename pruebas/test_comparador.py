"""Comparador de escenarios: límites de presupuesto y capacidad, empates, curvas de dosis y probabilidad de ser la mejor."""
import numpy as np
import pytest

from modulo_escenarios.comparador import Option, allocate, dose_factor, eb_shrink, evaluate, prob_best


def test_asigna_por_puntaje_hasta_el_limite():
    frac = allocate(np.array([4.0, 3.0, 2.0, 1.0]), np.ones(4), np.ones(4, bool), 2.5)
    assert frac.tolist() == [1.0, 1.0, 0.5, 0.0]


def test_empates_reciben_la_misma_fraccion():
    # así se modela la regla de 2026 (orden de registro): sin prioridad, todas las elegibles empatan
    frac = allocate(np.zeros(5), np.full(5, 2.0), np.ones(5, bool), 4.0)
    assert np.allclose(frac, 0.4)


def test_no_asigna_a_quien_no_es_elegible():
    frac = allocate(np.array([5.0, 4.0, 3.0]), np.ones(3), np.array([False, True, True]), 10.0)
    assert frac.tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("curva, monto, esperado",
                         [("D1", 3000, 0.5), ("D1", 6000, 1.0), ("D2", 1500, 0.5), ("D3", 3000, 1.0)])
def test_curvas_de_dosis(curva, monto, esperado):
    assert dose_factor(curva, monto) == pytest.approx(esperado)


def poblacion(n=6, B=20, seed=0):
    rng = np.random.default_rng(seed)
    w = np.full(n, 100.0)
    w_draws = w[None, :] * rng.integers(1, 3, size=(B, n))
    th = {"ia_mod_sev": np.full(n, -0.03), "ia_adultos": np.full(n, -0.1), "log_ing_pc": np.full(n, 0.1)}
    th_dr = {k: np.repeat(v[None, :], B, axis=0) for k, v in th.items()}
    attrs = {"en_ia": np.array([1, 0, 1, 0, 0, 1], float), "q40": np.array([1, 1, 0, 0, 1, 0], float)}
    return w, w_draws, th, th_dr, attrs


def test_respeta_presupuesto_y_capacidad():
    w, w_draws, th, th_dr, attrs = poblacion()
    opcion = Option("U", 3000.0, np.arange(6, dtype=float), np.ones(6, bool))
    presupuesto = 250 * 3000 * 6  # alcanza para 250 personas durante un año
    kw = dict(w=w, w_draws=w_draws, tau=1.0, admin=0.0, theta_pt=th, theta_dr=th_dr, curve="D1", attrs=attrs)
    r = evaluate(opcion, budget=presupuesto, capacity=np.inf, **kw)
    assert r["punto"]["beneficiarias"] == pytest.approx(250)
    assert r["punto"]["costo"] <= presupuesto + 1e-6
    assert np.all(r["reps"]["costo"] <= presupuesto + 1e-6)
    r = evaluate(opcion, budget=presupuesto, capacity=120.0, **kw)
    assert r["punto"]["beneficiarias"] == pytest.approx(120)


def test_sin_prioridad_conserva_la_composicion_de_la_poblacion():
    w, w_draws, th, th_dr, attrs = poblacion()
    opcion = Option("REG", 3100.0, np.zeros(6), np.ones(6, bool))
    r = evaluate(opcion, w=w, w_draws=w_draws, tau=0.9, budget=np.inf, capacity=200.0, admin=0.0,
                 theta_pt=th, theta_dr=th_dr, curve="D2", attrs=attrs)
    assert r["punto"]["beneficiarias"] == pytest.approx(200)
    assert r["punto"]["prop_q40"] == pytest.approx(attrs["q40"].mean())


def test_encogimiento_sin_varianza_entre_grupos():
    rng = np.random.default_rng(1)
    puntos = np.array([0.1, -0.1, 0.0])
    pt, _, info = eb_shrink(puntos, rng.normal(0.0, 1.0, size=(500, 3)) + puntos)
    assert info["tau2"] == 0.0
    assert np.allclose(pt, pt[0])


def test_prob_mejor_reparte_los_empates():
    # A y B eligen a los mismos hogares (mismos valores en cada réplica): deben recibir la misma probabilidad
    a = np.array([5.0, 7.0, 6.0, 9.0])
    c = np.array([6.0, 1.0, 1.0, 1.0])
    res = [{"reps": {"hogares_salen_ia": a}}, {"reps": {"hogares_salen_ia": a.copy()}},
           {"reps": {"hogares_salen_ia": c}}, {"reps": {"hogares_salen_ia": np.full(4, 99.0)}}]
    p = prob_best(res, [True, True, True, False])
    assert p[0] == pytest.approx(p[1]) == pytest.approx(3 / 8)
    assert p[2] == pytest.approx(1 / 4)
    assert p[3] == 0.0  # no factible
    assert sum(p) == pytest.approx(1.0)
