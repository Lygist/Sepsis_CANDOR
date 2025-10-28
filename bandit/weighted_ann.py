# weighted_ann.py
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
MASTER_SEED = 20251028  # Updated seed
RNG_MASTER = np.random.default_rng(MASTER_SEED)

NUM_RUNS = 100
TRAJ_PER_RUN = 100
TRAJ_LEN = 1  # bandit

ANNOTATION_BUDGETS = [20, 40, 60, 80, 100]

# --- New parameters for weighted annotations ---

# 9 weight combinations for (Doctor, Nurse, AI)
WEIGHT_COMBINATIONS = np.array([
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1],
    [0.1, 0.3, 0.6],
    [0.1, 0.6, 0.3],
    [1 / 3, 1 / 3, 1 / 3],  # Equal weighting
    [0.3, 0.1, 0.6],
    [0.6, 0.3, 0.1],
    [0.6, 0.1, 0.3],
    [0.8, 0.1, 0.1]
])
NUM_WEIGHTS = len(WEIGHT_COMBINATIONS)

# Fixed alpha for IS+
ALPHA_CIS = 0.5

# ======================
# Environment / policies
# ======================
STATE_PROBS = np.array([0.5, 0.5])  # 2-state, 2-action
REWARD_MEANS = np.array([[3.0, 5.0],
                         [1.5, -1.5]])
REWARD_STDS = np.array([[8.0, 6.0],
                        [6.0, 8.0]])

PI_B = np.array([[0.80, 0.20],
                 [0.20, 0.80]])
PI_E = np.array([[0.15, 0.85],
                 [0.85, 0.15]])

TRUE_VALUE = calculate_true_policy_value(PI_E, STATE_PROBS, REWARD_MEANS)

# ======================
# Imperfect annotation parameters (3 sources)
# ======================
# Bias: Doctor < Nurse < AI
DR_BIAS = 0.2
NU_BIAS = 0.5
AI_BIAS = 1.0

# Std (Variance): AI < Nurse < Doctor
AI_STD_ADD = 0.1
NU_STD_ADD = 0.3
DR_STD_ADD = 0.6

# Pre-calculate the means and stds for each source
DR_MEANS = REWARD_MEANS + DR_BIAS
DR_STDS = REWARD_STDS + DR_STD_ADD

NU_MEANS = REWARD_MEANS + NU_BIAS
NU_STDS = REWARD_STDS + NU_STD_ADD

AI_MEANS = REWARD_MEANS + AI_BIAS
AI_STDS = REWARD_STDS + AI_STD_ADD


def run_once(rng):
    """
    One Monte Carlo replicate using the provided child RNG.

    Generates 3 separate sets of annotations (Dr, Nu, AI) and then
    evaluates CANDOR and IS+ for each of the 9 weight combinations.

    Returns:
        is_val: float
        dr_val: float
        cis_weighted: (NUM_WEIGHTS, len(ANNOTATION_BUDGETS)) float
        candor_weighted: (NUM_WEIGHTS, len(ANNOTATION_BUDGETS)) float
    """
    # Generate dataset under behavior policy
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

    # Initialize result arrays for this run
    cis_weighted = np.zeros((NUM_WEIGHTS, len(ANNOTATION_BUDGETS)), dtype=float)
    candor_weighted = np.zeros((NUM_WEIGHTS, len(ANNOTATION_BUDGETS)), dtype=float)

    # Loop over annotation budgets
    for m_idx, m_budget in enumerate(ANNOTATION_BUDGETS):

        # --- Generate 3 separate annotation sets for this budget ---
        # Note: We use the same 'rng' and 'm_budget'. This means they
        # will annotate the *same* (s,a) pairs, but with different
        # reward values drawn from their respective biased distributions.

        # Source 1: Doctor
        ann_dr = generate_annotations(
            factual_dataset=dataset,
            num_annotations=m_budget,
            annotated_reward_means=DR_MEANS,
            annotated_reward_stds=DR_STDS,
            rng=rng,
        )

        # Source 2: Nurse
        ann_nu = generate_annotations(
            factual_dataset=dataset,
            num_annotations=m_budget,
            annotated_reward_means=NU_MEANS,
            annotated_reward_stds=NU_STDS,
            rng=rng,
        )

        # Source 3: AI
        ann_ai = generate_annotations(
            factual_dataset=dataset,
            num_annotations=m_budget,
            annotated_reward_means=AI_MEANS,
            annotated_reward_stds=AI_STDS,
            rng=rng,
        )

        # Stack all sources into a single array (M, batch, T, A)
        # M = 3 (Dr, Nu, AI)
        all_annotations = np.stack([ann_dr, ann_nu, ann_ai], axis=0)

        # --- Evaluate for each weight combination ---
        for w_idx, weights in enumerate(WEIGHT_COMBINATIONS):
            # CANDOR: Pass all annotations + weights
            candor_weighted[w_idx, m_idx] = float(run_dr(
                PI_E, PI_B, dataset,
                annotations=all_annotations,
                source_weights=weights
            ))

            # IS+ (C-IS): Pass all annotations + weights + fixed alpha
            cis_weighted[w_idx, m_idx] = float(run_is_plus(
                PI_E, PI_B, dataset,
                annotations=all_annotations,
                alpha=ALPHA_CIS,
                source_weights=weights
            ))

    return (is_val, dr_val, cis_weighted, candor_weighted)


def main():
    # --- Initialize arrays to store results from all runs ---
    is_runs = np.zeros(NUM_RUNS, dtype=float)
    dr_runs = np.zeros(NUM_RUNS, dtype=float)

    # Shape: (NUM_RUNS, NUM_WEIGHTS, NUM_BUDGETS)
    cis_weighted_runs = np.zeros(
        (NUM_RUNS, NUM_WEIGHTS, len(ANNOTATION_BUDGETS)),
        dtype=float
    )
    candor_weighted_runs = np.zeros(
        (NUM_RUNS, NUM_WEIGHTS, len(ANNOTATION_BUDGETS)),
        dtype=float
    )

    print(f"Starting {NUM_RUNS} Monte Carlo runs...")
    for r in range(NUM_RUNS):
        # Create a new independent RNG for each run
        rng = np.random.default_rng(RNG_MASTER.integers(0, 2 ** 31 - 1))

        is_v, dr_v, cis_w, candor_w = run_once(rng)

        is_runs[r] = is_v
        dr_runs[r] = dr_v
        cis_weighted_runs[r, :, :] = cis_w
        candor_weighted_runs[r, :, :] = candor_w

        if (r + 1) % 10 == 0:
            print(f"  Completed run {r + 1}/{NUM_RUNS}")

    # --- Save all results to a .npz file ---
    output_filename = "weighted_ann.npz"
    np.savez(
        output_filename,
        true_value=TRUE_VALUE,
        budgets=np.array(ANNOTATION_BUDGETS),
        weight_combinations=WEIGHT_COMBINATIONS,
        IS_estimates=is_runs,
        DR_estimates=dr_runs,
        CIS_weighted_estimates=cis_weighted_runs,
        CANDOR_weighted_estimates=candor_weighted_runs,
    )

    print(
        f"\nSaved results to {output_filename} with shapes:",
        f"\n  true_value: {TRUE_VALUE}",
        f"\n  budgets: {np.array(ANNOTATION_BUDGETS).shape}",
        f"\n  weight_combinations: {WEIGHT_COMBINATIONS.shape}",
        f"\n  IS_estimates: {is_runs.shape}",
        f"\n  DR_estimates: {dr_runs.shape}",
        f"\n  CIS_weighted_estimates: {cis_weighted_runs.shape}",
        f"\n  CANDOR_weighted_estimates: {candor_weighted_runs.shape}",
    )


if __name__ == "__main__":
    main()