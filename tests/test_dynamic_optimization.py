import math

import numpy as np

from dynamic_optimization import (
    OptimizationConstraint,
    OptimizationProblem,
    OptimizationVariable,
    build_state_from_decision_vector,
    candidates_to_csv,
    evaluate_candidate,
    generate_global_candidates,
    json_safe,
    run_optimization,
)


def _base_problem(mode: str = "maximize", objective: str = "total_aero_load_n", target=None):
    return OptimizationProblem(
        objective_mode=mode,
        objective_name=objective,
        objective_target=target,
        variables=[
            OptimizationVariable("front_heave", -10.0, 10.0, 0.0, enabled=True),
        ],
        constraints=[],
        fixed_inputs={"base_state": {"hf": 0.0, "hr": 0.0, "rf": 0.0, "rr": 0.0}},
        search_method="grid_refine",
        max_global_points=15,
        n_best_candidates=3,
    )


def test_decision_vector_to_state_mapping():
    state = build_state_from_decision_vector(
        {"front_heave": 1.5, "rear_roll": -2.2},
        {"hf": 0.0, "hr": 4.0, "rf": 7.0, "rr": 8.0},
    )
    assert state["hf"] == 1.5
    assert state["hr"] == 4.0
    assert state["rf"] == 7.0
    assert state["rr"] == -2.2


def test_objective_mode_maximize_minimize_target():
    evaluator = lambda sim: {"total_aero_load_n": float(sim["hf"]) + 10.0}
    decision = {"front_heave": 2.0}

    p_max = _base_problem(mode="maximize")
    c_max = evaluate_candidate(p_max, decision, evaluator)
    assert math.isclose(c_max.objective_value_raw, 12.0, rel_tol=1e-9)
    assert math.isclose(c_max.total_cost, -12.0, rel_tol=1e-9)

    p_min = _base_problem(mode="minimize")
    c_min = evaluate_candidate(p_min, decision, evaluator)
    assert math.isclose(c_min.total_cost, 12.0, rel_tol=1e-9)

    p_tgt = _base_problem(mode="target", target=10.0)
    c_tgt = evaluate_candidate(p_tgt, decision, evaluator)
    assert math.isclose(c_tgt.total_cost, 4.0, rel_tol=1e-9)


def test_penalty_calculation_eq_ge_le():
    problem = _base_problem(mode="maximize")
    problem.constraints = [
        OptimizationConstraint("aero_balance_front_pct", "eq", 44.0, 2.0),
        OptimizationConstraint("drag_force_n", "le", 120.0, 3.0),
        OptimizationConstraint("total_aero_load_n", "ge", 80.0, 5.0),
    ]

    def evaluator(_):
        return {
            "total_aero_load_n": 70.0,  # ge violation: 10
            "aero_balance_front_pct": 46.0,  # eq error: 2
            "drag_force_n": 130.0,  # le violation: 10
        }

    cand = evaluate_candidate(problem, {"front_heave": 0.0}, evaluator)
    expected_penalty = 2.0 * (2.0 ** 2) + 3.0 * (10.0 ** 2) + 5.0 * (10.0 ** 2)
    assert math.isclose(cand.penalty_value, expected_penalty, rel_tol=1e-9)
    assert not cand.is_feasible_like


def test_global_search_candidate_generation_modes():
    p2 = OptimizationProblem(
        objective_mode="maximize",
        objective_name="total_aero_load_n",
        objective_target=None,
        variables=[
            OptimizationVariable("front_heave", -10, 10, 0, True),
            OptimizationVariable("rear_heave", -10, 10, 0, True),
        ],
        constraints=[],
        fixed_inputs={},
        search_method="auto",
        max_global_points=49,
        n_best_candidates=3,
    )
    method2, cands2 = generate_global_candidates(p2, [v for v in p2.variables if v.enabled])
    assert method2 == "grid_refine"
    assert len(cands2) > 0

    p4 = OptimizationProblem(
        objective_mode="maximize",
        objective_name="total_aero_load_n",
        objective_target=None,
        variables=[
            OptimizationVariable("front_heave", -10, 10, 0, True),
            OptimizationVariable("rear_heave", -10, 10, 0, True),
            OptimizationVariable("front_roll", -5, 5, 0, True),
            OptimizationVariable("rear_roll", -5, 5, 0, True),
        ],
        constraints=[],
        fixed_inputs={},
        search_method="auto",
        max_global_points=120,
        n_best_candidates=4,
    )
    method4, cands4 = generate_global_candidates(p4, [v for v in p4.variables if v.enabled])
    assert method4 == "sample_refine"
    assert len(cands4) == 120


def test_best_candidate_selection_and_refinement():
    # Minimum at hf = 2.5.
    def evaluator(sim):
        x = float(sim["hf"])
        drag = (x - 2.5) ** 2 + 1.0
        return {
            "drag_force_n": drag,
            "total_aero_load_n": 100.0 - drag,
            "aero_balance_front_pct": 44.0,
            "hRideF": x,
            "hRideR": 0.0,
        }

    problem = OptimizationProblem(
        objective_mode="minimize",
        objective_name="drag_force_n",
        objective_target=None,
        variables=[OptimizationVariable("front_heave", -6.0, 6.0, 0.0, True)],
        constraints=[OptimizationConstraint("aero_balance_front_pct", "eq", 44.0, 10.0)],
        fixed_inputs={},
        search_method="grid_refine",
        max_global_points=21,
        n_best_candidates=3,
    )
    result = run_optimization(problem, evaluator)
    best_hf = float(result.best_candidate.decision_vector["front_heave"])
    assert abs(best_hf - 2.5) < 0.5
    assert result.global_stage_count >= 10
    assert result.refinement_stage_count >= 1


def test_json_serialization_safety_and_csv():
    problem = _base_problem()
    cand = evaluate_candidate(
        problem,
        {"front_heave": 0.0},
        lambda _sim: {
            "total_aero_load_n": np.float64(123.4),
            "aero_balance_front_pct": np.nan,
            "drag_force_n": 10.0,
        },
    )
    payload = json_safe(cand)
    assert payload["objective_value_raw"] == 123.4
    assert payload["outputs"]["aero_balance_front_pct"] is None

    csv_text = candidates_to_csv([cand])
    assert "objective_value_raw" in csv_text
    assert "decision__front_heave" in csv_text


def test_end_to_end_target_with_constraints():
    # Best around hf=3, hr=1 with target balance=44.
    def evaluator(sim):
        hf = float(sim["hf"])
        hr = float(sim["hr"])
        total_aero = 250.0 - (hf - 3.0) ** 2 - (hr - 1.0) ** 2
        balance = 44.0 + 0.2 * (hf - 3.0) - 0.1 * (hr - 1.0)
        drag = 80.0 + 0.6 * (hf - 1.5) ** 2 + 0.4 * (hr - 0.5) ** 2
        return {
            "total_aero_load_n": total_aero,
            "aero_balance_front_pct": balance,
            "drag_force_n": drag,
            "hRideF": hf,
            "hRideR": hr,
            "front_total_load_n": 3000 + 20 * hf,
            "rear_total_load_n": 3000 + 20 * hr,
            "total_load_fl_n": 1500,
            "total_load_fr_n": 1500,
            "total_load_rl_n": 1500,
            "total_load_rr_n": 1500,
            "front_axle_lateral_force_n": 2200,
            "rear_axle_lateral_force_n": 2100,
            "total_lateral_force_n": 4300,
            "front_axle_longitudinal_force_n": 1100,
            "rear_axle_longitudinal_force_n": 1000,
            "total_longitudinal_force_n": 2100,
        }

    problem = OptimizationProblem(
        objective_mode="maximize",
        objective_name="total_aero_load_n",
        objective_target=None,
        variables=[
            OptimizationVariable("front_heave", -5.0, 6.0, 0.0, True),
            OptimizationVariable("rear_heave", -5.0, 6.0, 0.0, True),
        ],
        constraints=[
            OptimizationConstraint("aero_balance_front_pct", "eq", 44.0, 80.0),
            OptimizationConstraint("drag_force_n", "le", 95.0, 5.0),
        ],
        fixed_inputs={},
        search_method="auto",
        max_global_points=121,
        n_best_candidates=5,
    )
    result = run_optimization(problem, evaluator)
    best = result.best_candidate.decision_vector
    assert abs(float(best["front_heave"]) - 3.0) < 1.0
    assert abs(float(best["rear_heave"]) - 1.0) < 1.0
    assert isinstance(result.diagnostics.get("refinement_infos"), list)
