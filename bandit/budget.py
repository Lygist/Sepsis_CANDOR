import os, json
import numpy as np
from dataclasses import dataclass
from trajectory_classes import Trajectory
from two_state import (
    generate_dataset_of_trajectories,
    generate_annotations,
    run_vanilla_is,
    run_is_plus,
    run_dr,
    calculate_true_policy_value,
    calculate_policy_value_rmse_and_ci
)


MASTER_SEED = 20250928
RNG_MASTER = np.random.default_rng(MASTER_SEED)

NUM_DATASETS = 100
TRAJECTORIES_PER_DATASET = 100
BASE_OUTPUT_DIR = "results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

FIXED_RATIO = 10
FIXED_BUDGET = 300


@dataclass(frozen=True)
class EnvConfig:
    state_distribution: np.ndarray
    true_reward_means: np.ndarray
    true_reward_stds: np.ndarray
    behavior_policy: np.ndarray
    evaluation_policy: np.ndarray
    expert_bias: np.ndarray
    expert_std: np.ndarray
    llm_bias: np.ndarray
    llm_std: np.ndarray


@dataclass(frozen=True)
class GridConfig:
    budgets_to_test: list[float]
    cost_ratios_to_test: list[float]
    expert_allocations: list[float]


@dataclass(frozen=True)
class RunOnceResult:
    sweep_axis: str
    sweep_values: list[float]
    allocations: list[float]
    # per-estimator arrays of shape (n_alloc, n_sweep)
    estimates: dict[str, np.ndarray]   # keys like {"IS+", "CANDOR"}


@dataclass(frozen=True)
class RMSESummary:
    sweep_axis: str
    sweep_values: list[float]
    allocations: list[float]
    rmse: dict[str, np.ndarray]        # (n_alloc, n_sweep)
    half_width: dict[str, np.ndarray]  # (n_alloc, n_sweep)


def count_annotations(
    budget: float,
    cost_ratio: float,
    expert_percent: float,
    llm_cost: float = 1.0
) -> tuple[int, int]:
    # Max number of annotations is 2x factual data points for 2-action bandit
    # Note: it may lead to multiple annotations from different sources at the same non-factual spot
    # Simply average them for now, but we need a more rigorous aggregating rule
    max_annotations = 2 * TRAJECTORIES_PER_DATASET
    expert_budget = (expert_percent / 100) * budget
    llm_budget = budget - expert_budget
    expert_cost = cost_ratio * llm_cost
    # Calculate the number of annotations with the new cap
    num_expert = int(expert_budget / expert_cost)
    num_llm = int(llm_budget / llm_cost)
    # Prioritize expert annotations
    num_expert_annotations = min(num_expert, max_annotations)
    remaining_capacity = max_annotations - num_expert_annotations
    num_llm_annotations = min(num_llm, remaining_capacity)

    return num_expert_annotations, num_llm_annotations


def run_once(
    env: EnvConfig,
    factual_dataset: list[Trajectory],
    sweep_axis: str,
    sweep_values: list[float],
    fixed_cost_ratio: float | None,
    fixed_budget: float | None,
    allocations: list[float]
) -> RunOnceResult:
    """
    Use RNG_MASTER to draw seeds deterministically in a fixed order:
      - once for factual (done outside this function),
      - once per (budget, ratio, alloc) combo here.
    For each combo, create rng_ann = default_rng(seed) and generate annotations ONCE.
    Both estimators consume the same (factual, annotations) → deterministic, fair comparison.
    """
    assert sweep_axis in {"budget", "ratio"}, \
        f"sweep_axis must be 'budget' or 'ratio', got {sweep_axis!r}"
    if sweep_axis == "budget" and fixed_cost_ratio is None:
        raise ValueError("fixed_cost_ratio is required for budget sweep.")
    if sweep_axis == "ratio" and fixed_budget is None:
        raise ValueError("fixed_budget is required for ratio sweep.")

    n_a, n_s = len(allocations), len(sweep_values)
    out_cis = np.empty((n_a, n_s), float)
    out_candor = np.empty((n_a, n_s), float)

    for ai, alloc in enumerate(allocations):
        for si, s in enumerate(sweep_values):
            budget = s if sweep_axis == "budget" else int(fixed_budget)
            cost_ratio = int(fixed_cost_ratio) if sweep_axis == "budget" else s

            # counts derived from (budget, ratio, alloc)
            n_exp, n_llm = count_annotations(budget, cost_ratio, alloc)

            # draw one seed from the master for this combo and split into two child RNGs
            combo_seed = int(RNG_MASTER.integers(0, 2**31 - 1))
            rng_combo = np.random.default_rng(combo_seed)
            expert_seed = int(rng_combo.integers(0, 2 ** 31 - 1))
            llm_seed = int(rng_combo.integers(0, 2 ** 31 - 1))
            rng_expert = np.random.default_rng(expert_seed)
            rng_llm = np.random.default_rng(llm_seed)

            expert_annotations = generate_annotations(
                factual_dataset, n_exp, env.true_reward_means + env.expert_bias, env.expert_std, rng=rng_expert
            )
            llm_annotations = generate_annotations(
                factual_dataset, n_llm, env.true_reward_means + env.llm_bias, env.llm_std, rng=rng_llm
            )
            annotations = np.stack((expert_annotations, llm_annotations), axis=0)

            out_cis[ai, si] = run_is_plus(
                env.evaluation_policy, env.behavior_policy, factual_dataset, annotations
            )
            out_candor[ai, si] = run_dr(
                env.evaluation_policy, env.behavior_policy, factual_dataset, annotations
            )

    return RunOnceResult(
        sweep_axis=sweep_axis,
        sweep_values=sweep_values,
        allocations=allocations,
        estimates={"C-IS": out_cis, "CANDOR": out_candor},
    )


def aggregate_rmse_over_datasets(results: list[RunOnceResult], true_value: float) -> RMSESummary:
    if not results:
        raise ValueError("Empty results.")
    ax = results[0].sweep_axis
    sv = results[0].sweep_values
    al = results[0].allocations
    names = list(results[0].estimates.keys())

    # basic alignment checks
    for r in results[1:]:
        if r.sweep_axis != ax or r.sweep_values != sv or r.allocations != al:
            raise ValueError("RunOnceResult alignment mismatch.")
        if list(r.estimates.keys()) != names:
            raise ValueError("Estimator set mismatch.")

    n_a, n_s = len(al), len(sv)
    rmse = {k: np.empty((n_a, n_s), float) for k in names}
    half_width = {k: np.empty((n_a, n_s), float) for k in names}

    for name in names:
        # stack per dataset: (n_ds, n_alloc, n_sweep)
        stack = np.stack([r.estimates[name] for r in results], axis=0)
        r_, h_ = calculate_policy_value_rmse_and_ci(stack, true_value)
        rmse[name] = r_
        half_width[name] = h_

    return RMSESummary(ax, sv, al, rmse, half_width)


def _summary_to_dict(summary: RMSESummary) -> dict:
    return {
        "sweep_axis": summary.sweep_axis,                                 # "budget" or "ratio"
        "sweep_values": summary.sweep_values,
        "allocations": summary.allocations,
        "rmse": {k: v.tolist() for k, v in summary.rmse.items()},         # keys are IS^+ and CANDOR
        "half_width": {k: v.tolist() for k, v in summary.half_width.items()}
    }


def main() -> None:
    env = EnvConfig(
        state_distribution=np.array([0.5, 0.5]),
        true_reward_means=np.array([[3., 5.], [1.5, -1.5]]),
        true_reward_stds=np.array([[8., 6.], [6., 8.]]),
        behavior_policy=np.array([[0.8, 0.2], [0.2, 0.8]]),
        evaluation_policy=np.array([[0.15, 0.85], [0.85, 0.15]]),
        expert_bias=np.array([[0.1, 0.1], [0.1, 0.1]]),
        expert_std=np.array([[2.0, 2.0], [2.0, 2.0]]),
        llm_bias=np.array([[0.8, 0.8], [0.8, 0.8]]),
        llm_std=np.array([[0.2, 0.2], [0.2, 0.2]])
    )

    grid = GridConfig(
        budgets_to_test=[100, 200, 300, 400, 500],
        cost_ratios_to_test=[5, 7.5, 10, 12.5, 15],
        expert_allocations=[0, 20, 40, 60, 80, 100],
    )

    true_evaluation_policy_value = calculate_true_policy_value(
        env.evaluation_policy, env.state_distribution, env.true_reward_means)

    baseline_is: list[float] = []
    baseline_dr: list[float] = []
    results_budget: list[RunOnceResult] = []
    results_ratio: list[RunOnceResult] = []

    for ds_idx in range(NUM_DATASETS):
        seed_factual = int(RNG_MASTER.integers(0, 2 ** 31 - 1))
        rng_factual = np.random.default_rng(seed_factual)
        factual_dataset = generate_dataset_of_trajectories(
            env.state_distribution, env.true_reward_means,
            env.true_reward_stds, env.behavior_policy, rng_factual,
            num_trajectories=TRAJECTORIES_PER_DATASET)

        is_estimate = run_vanilla_is(env.evaluation_policy, env.behavior_policy, factual_dataset)
        baseline_is.append(is_estimate)
        dr_estimate = run_dr(env.evaluation_policy, env.behavior_policy, factual_dataset)
        baseline_dr.append(dr_estimate)

        r_budget = run_once(
            env=env, factual_dataset=factual_dataset,
            sweep_axis="budget", sweep_values=grid.budgets_to_test,
            fixed_cost_ratio=FIXED_RATIO, fixed_budget=None,
            allocations=grid.expert_allocations
        )
        results_budget.append(r_budget)

        r_ratio = run_once(
            env=env, factual_dataset=factual_dataset,
            sweep_axis="ratio", sweep_values=grid.cost_ratios_to_test,
            fixed_cost_ratio=None, fixed_budget=FIXED_BUDGET,
            allocations=grid.expert_allocations
        )
        results_ratio.append(r_ratio)

    # Aggregate RMSE + CI across datasets
    is_rmse, is_ci = calculate_policy_value_rmse_and_ci(np.array(baseline_is), true_evaluation_policy_value)
    dr_rmse, dr_ci = calculate_policy_value_rmse_and_ci(np.array(baseline_dr), true_evaluation_policy_value)
    summary_budget = aggregate_rmse_over_datasets(results_budget, true_evaluation_policy_value)
    summary_ratio = aggregate_rmse_over_datasets(results_ratio, true_evaluation_policy_value)

    payload = {
        "ENV": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in env.__dict__.items()
        },
        "TRUTH": true_evaluation_policy_value,
        "IS": {
            "rmse": float(is_rmse),
            "half_width": float(is_ci),
        },
        "DR": {
            "rmse": float(dr_rmse),
            "half_width": float(dr_ci),
        },
        "SUMMARY_BUDGET": _summary_to_dict(summary_budget),
        "SUMMARY_RATIO": _summary_to_dict(summary_ratio),
        "FIXED_RATIO": FIXED_RATIO, "FIXED_BUDGET": FIXED_BUDGET
    }

    with open(os.path.join(BASE_OUTPUT_DIR, "budget.json"), "w") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    main()
