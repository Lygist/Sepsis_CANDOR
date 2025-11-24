import numpy as np
from two_state import (
    generate_dataset_of_trajectories,
    generate_annotations,
    run_vanilla_is,
    run_is_plus,
    run_dr,
    calculate_true_policy_value
)

# ======================
# Experiment controls
# ======================
MASTER_SEED = 20251124
RNG_MASTER = np.random.default_rng(MASTER_SEED)

NUM_RUNS = 100
TRAJ_PER_RUN = 400
TRAJ_LEN = 1  # bandit

ANNOTATION_BUDGETS = [40, 80, 120, 160, 200]

ALPHA_VALS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# ======================
# Environment / policies
# ======================
STATE_PROBS = np.array([0.33, 0.33, 0.34]) # 3-state, 5-action

REWARD_MEANS = abs(np.random.normal(3,1,(3,5)))

REWARD_STDS = np.random.lognormal(1,0.5,(3,5))

PI_B = np.array([
    [0.8, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.8, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.8]
])

PI_E = np.array([
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.6, 0.1, 0.1, 0.1, 0.1]
])

TRUE_VALUE = calculate_true_policy_value(PI_E, STATE_PROBS, REWARD_MEANS)

# ======================
# Imperfect annotation parameters (global)
# ======================
IMP_ANNOT_BIAS = -0.75      # additive mean bias for imperfect annotations
IMP_ANNOT_STD_ADD = 0.3    # additional std added for imperfect annotations


def _eval_with_annotations(rng, dataset, budget, means, stds):
    """
    Generate 'budget' annotations using (means, stds) and evaluate:
      - C-IS (IS+): run_is_plus with annotations=[ann]
      - CANDOR:     run_dr with annotations having a leading 'num_sets' dim

    Returns: (cis_value, candor_value) as floats.
    """
    ann = generate_annotations(
        factual_dataset=dataset,
        num_annotations=budget,
        annotated_reward_means=means,
        annotated_reward_stds=stds,
        rng=rng,
    )[np.newaxis, ...]  # shape (1, batch, T, A)

    cis_val = float(run_is_plus(PI_E, PI_B, dataset, annotations=ann))
    candor_val = float(run_dr(PI_E, PI_B, dataset, annotations=ann))

    return cis_val, candor_val


def run_once(rng):
    """
    One Monte Carlo replicate using the provided child RNG.
    Returns:
        is_val: float
        dr_val: float
        cis_perfect: (len(ANNOTATION_BUDGETS),) float
        cis_imperfect: (len(ANNOTATION_BUDGETS),) float
        candor_perfect: (len(ANNOTATION_BUDGETS),) float
        candor_imperfect: (len(ANNOTATION_BUDGETS),) float

    UPDATE: Modified the shape of cis (NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS))
    """
    # Generate dataset under behavior policy using this run's RNG
    dataset = generate_dataset_of_trajectories(
        state_distribution=STATE_PROBS,
        reward_means=REWARD_MEANS,
        reward_stds=REWARD_STDS,
        policy=PI_B,
        trajectory_len=TRAJ_LEN,
        num_trajectories=TRAJ_PER_RUN,
        rng=rng,
    )

    # Baselines (no annotations)
    is_val = float(run_vanilla_is(PI_E, PI_B, dataset))
    dr_val = float(run_dr(PI_E, PI_B, dataset, annotations=None))

    # Imperfect annotation distribution (biased means, inflated stds)
    imperfect_means = REWARD_MEANS + IMP_ANNOT_BIAS
    imperfect_stds  = REWARD_STDS + IMP_ANNOT_STD_ADD

    cis_perfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    cis_imperfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)

    candor_perfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_imperfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)

    m_idx=0
    for m in ANNOTATION_BUDGETS:
        # Perfect annotations
        ann_perfect = generate_annotations(
            factual_dataset=dataset,
            num_annotations=m,
            annotated_reward_means=REWARD_MEANS,
            annotated_reward_stds=REWARD_STDS,
            rng=rng,
        )[np.newaxis, ...]
        # Imperfect annotations
        ann_imperfect = generate_annotations(
            factual_dataset=dataset,
            num_annotations=m,
            annotated_reward_means=imperfect_means,
            annotated_reward_stds=imperfect_stds,
            rng=rng,
        )[np.newaxis, ...]

        # Loop over Alphas
        for a_idx, alpha in enumerate(ALPHA_VALS):
            # --- IS+ (C-IS) ---
            cis_perfect[a_idx, m_idx] = float(run_is_plus(
                PI_E, PI_B, dataset, annotations=ann_perfect, alpha=alpha
            ))
            cis_imperfect[a_idx, m_idx] = float(run_is_plus(
                PI_E, PI_B, dataset, annotations=ann_imperfect, alpha=alpha
            ))

            # --- CANDOR (Weighted DR) ---
            candor_perfect[a_idx, m_idx] = float(run_dr(
                PI_E, PI_B, dataset, annotations=ann_perfect, alpha=alpha
            ))
            candor_imperfect[a_idx, m_idx] = float(run_dr(
                PI_E, PI_B, dataset, annotations=ann_imperfect, alpha=alpha
            ))

        m_idx = m_idx + 1

    return (is_val, dr_val,
            cis_perfect, cis_imperfect,
            candor_perfect, candor_imperfect)


def main():
    is_runs = np.zeros(NUM_RUNS, dtype=float)
    dr_runs = np.zeros(NUM_RUNS, dtype=float)
    cis_perfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    cis_imperfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_perfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_imperfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)

    for r in range(NUM_RUNS):
        rng = np.random.default_rng(RNG_MASTER.integers(0, 2**31 - 1))  # independent RNG per run
        is_v, dr_v, cis_p, cis_i, candor_p, candor_i = run_once(rng)
        is_runs[r] = is_v
        dr_runs[r] = dr_v
        cis_perfect_runs[r, :, :] = cis_p
        cis_imperfect_runs[r, :, :] = cis_i
        candor_perfect_runs[r, :, :] = candor_p
        candor_imperfect_runs[r, :, :] = candor_i

    np.savez(
        "boxplot_data_alpha_3s5a.npz",
        true_value=TRUE_VALUE,
        budgets=np.array(ANNOTATION_BUDGETS),
        alphas=ALPHA_VALS,
        IS_estimates=is_runs,
        DR_estimates=dr_runs,
        CIS_perfect_estimates=cis_perfect_runs,
        CIS_imperfect_estimates=cis_imperfect_runs,
        CANDOR_perfect_estimates=candor_perfect_runs,
        CANDOR_imperfect_estimates=candor_imperfect_runs,
    )
    print(
        "Saved boxplot_data.npz with shapes:",
        "\n  IS:", is_runs.shape,
        "\n  DR:", dr_runs.shape,
        "\n  CIS perfect:", cis_perfect_runs.shape,
        "\n  CIS imperfect:", cis_imperfect_runs.shape,
        "\n  CANDOR perfect:", candor_perfect_runs.shape,
        "\n  CANDOR imperfect:", candor_imperfect_runs.shape,
        "\n  budgets:", ANNOTATION_BUDGETS,
        "\n  alphas:", ALPHA_VALS,
        "\n  TRUE_VALUE:", TRUE_VALUE,
    )


if __name__ == "__main__":
    main()
