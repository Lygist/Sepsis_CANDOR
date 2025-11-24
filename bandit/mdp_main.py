import numpy as np
import gym
import torch
from stable_baselines3 import DQN

from env import TabularMDPEnv
from sequential import (
    generate_sequential_dataset_of_trajectories,
    calculate_true_seq_policy_value,
    generate_q_annotations,
    run_pdis,
    run_sequential_cis,
    run_sequential_candor,
    calculate_q_table
)

# ======================
# Experiment Environment
# ======================
# 3 states, 5 actions, 4 steps
MASTER_SEED = 20251124
RNG_MASTER = np.random.default_rng(MASTER_SEED)

NUM_RUNS = 100
TRAJ_PER_RUN = 400

# Budget: Number of annotations per dataset (Global budget)
# Note: With 400 trajectories and Horizon 4, total steps = 1600.
# Max possible CF actions = 1600 * 4 = 6400.
# Budgets are relatively sparse.
ANNOTATION_BUDGETS = [400, 800, 1600, 2400, 3200]

ALPHA_VALS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

NUM_STATES = 3
NUM_ACTIONS = 5
HORIZON = 4
GAMMA = 1.0

_rng_env = np.random.default_rng(MASTER_SEED)
TRANSITION_PROBS = _rng_env.dirichlet(alpha=np.ones(NUM_STATES), size=(NUM_STATES, NUM_ACTIONS))
REWARD_MEANS = np.abs(_rng_env.normal(loc=3.0, scale=1.0, size=(NUM_STATES, NUM_ACTIONS)))
REWARD_STDS = _rng_env.lognormal(mean=0.5, sigma=0.2, size=(NUM_STATES, NUM_ACTIONS))
INITIAL_STATE_DIST = np.array([0.4, 0.3, 0.3])

PI_B = np.array([
    [0.6, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.6]
])

PI_E = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.6],
    [0.6, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.6, 0.1, 0.1, 0.1]
])

TRUE_VALUE = calculate_true_seq_policy_value(
    PI_E, INITIAL_STATE_DIST, TRANSITION_PROBS, REWARD_MEANS, GAMMA, HORIZON
)

# ======================
# Imperfect Annotation Parameters
# ======================
IMP_ANNOT_BIAS = -1
IMP_ANNOT_STD_ADD = 0.5


# ======================
# Train Annotators
# ======================
def train_dqn_annotator(biased=False, total_timesteps=20000, seed=42):
    train_env = TabularMDPEnv(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        horizon=HORIZON,
        transition_probs=TRANSITION_PROBS,
        reward_means=REWARD_MEANS.copy(),
        reward_stds=REWARD_STDS.copy(),
        initial_state_dist=INITIAL_STATE_DIST,
        seed=seed
    )

    if biased:
        train_env.set_biased_rewards(IMP_ANNOT_BIAS, IMP_ANNOT_STD_ADD)

    model = DQN("MlpPolicy", train_env, verbose=0, learning_rate=1e-3, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model


# # Train Global Experts
# print("Initializing: Training Perfect Annotator (DQN)...")
# DQN_TRUE = train_dqn_annotator(biased=False, seed=MASTER_SEED)
#
# print("Initializing: Training Imperfect Annotator (DQN)...")
# DQN_BIASED = train_dqn_annotator(biased=True, seed=MASTER_SEED)

Q_TABLE_PERFECT = calculate_q_table(
    PI_E, TRANSITION_PROBS, REWARD_MEANS, GAMMA, HORIZON
)

# 2. Imperfect Q-Table (Target for Imperfect Annotations)
# Calculated using exact dynamics but Biased Rewards
REWARD_MEANS_BIASED = REWARD_MEANS + IMP_ANNOT_BIAS
# Note: In DP, std doesn't affect Q-values, only mean does.
Q_TABLE_IMPERFECT = calculate_q_table(
    PI_E, TRANSITION_PROBS, REWARD_MEANS_BIASED, GAMMA, HORIZON
)

def run_once(rng):
    """
    One Monte Carlo replicate.
    """
    # 1. Generate Dataset
    dataset = generate_sequential_dataset_of_trajectories(
        NUM_STATES, NUM_ACTIONS, HORIZON, INITIAL_STATE_DIST,
        TRANSITION_PROBS, REWARD_MEANS, REWARD_STDS, PI_B,
        num_trajectories=TRAJ_PER_RUN, rng=rng
    )

    # 2. Baselines (No Annotations)
    # PDIS
    pdis_val = run_pdis(PI_E, PI_B, dataset, gamma=GAMMA)

    # DR
    n_samples = len(dataset)
    nan_anns = np.full((n_samples, HORIZON, NUM_ACTIONS), np.nan)
    dr_val = run_sequential_candor(
        PI_E, PI_B, dataset, nan_anns, n_fold=2, alpha=0.5, gamma=GAMMA
    )

    # 3. Initialize Arrays for Annotated Methods
    cpdis_perfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    cpdis_imperfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_perfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_imperfect = np.zeros((len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)

    # 4. Loop Budgets
    for m_idx, budget in enumerate(ANNOTATION_BUDGETS):
        # Generate Annotations
        # ann_perfect = generate_q_annotations(dataset, DQN_TRUE, budget, rng)
        # ann_imperfect = generate_q_annotations(dataset, DQN_BIASED, budget, rng)
        ann_perfect = generate_q_annotations(dataset, Q_TABLE_PERFECT, budget, rng)
        ann_imperfect = generate_q_annotations(dataset, Q_TABLE_IMPERFECT, budget, rng)

        # Loop Alphas
        for a_idx, alpha in enumerate(ALPHA_VALS):
            # --- C-PDIS ---
            cpdis_perfect[a_idx, m_idx] = run_sequential_cis(
                PI_E, PI_B, dataset, ann_perfect, alpha=alpha, gamma=GAMMA
            )
            cpdis_imperfect[a_idx, m_idx] = run_sequential_cis(
                PI_E, PI_B, dataset, ann_imperfect, alpha=alpha, gamma=GAMMA
            )

            # --- CANDOR ---
            candor_perfect[a_idx, m_idx] = run_sequential_candor(
                PI_E, PI_B, dataset, ann_perfect, n_fold=2, alpha=alpha, gamma=GAMMA
            )
            candor_imperfect[a_idx, m_idx] = run_sequential_candor(
                PI_E, PI_B, dataset, ann_imperfect, n_fold=2, alpha=alpha, gamma=GAMMA
            )

    return (pdis_val, dr_val,
            cpdis_perfect, cpdis_imperfect,
            candor_perfect, candor_imperfect)


def main():
    print(f"Starting Experiment: {NUM_RUNS} runs, {TRAJ_PER_RUN} traj/run.")
    print(f"True Value: {TRUE_VALUE:.5f}")

    # Storage
    pdis_runs = np.zeros(NUM_RUNS, dtype=float)
    dr_runs = np.zeros(NUM_RUNS, dtype=float)

    # Shape: (Runs, Alphas, Budgets)
    cpdis_perfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    cpdis_imperfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_perfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)
    candor_imperfect_runs = np.zeros((NUM_RUNS, len(ALPHA_VALS), len(ANNOTATION_BUDGETS)), dtype=float)

    for r in range(NUM_RUNS):
        # Independent RNG per run
        rng_seed = int(RNG_MASTER.integers(0, 2 ** 31 - 1))
        rng = np.random.default_rng(rng_seed)

        if (r + 1) % 10 == 0:
            print(f"Processing Run {r + 1}/{NUM_RUNS}...")

        (pdis, dr,
         cpdis_p, cpdis_i,
         candor_p, candor_i) = run_once(rng)

        pdis_runs[r] = pdis
        dr_runs[r] = dr
        cpdis_perfect_runs[r] = cpdis_p
        cpdis_imperfect_runs[r] = cpdis_i
        candor_perfect_runs[r] = candor_p
        candor_imperfect_runs[r] = candor_i

    # Save Results
    filename = "boxplot_data_mdp.npz"
    np.savez(
        filename,
        true_value=TRUE_VALUE,
        budgets=np.array(ANNOTATION_BUDGETS),
        alphas=np.array(ALPHA_VALS),
        PDIS_estimates=pdis_runs,
        DR_estimates=dr_runs,
        CPDIS_perfect_estimates=cpdis_perfect_runs,
        CPDIS_imperfect_estimates=cpdis_imperfect_runs,
        CANDOR_perfect_estimates=candor_perfect_runs,
        CANDOR_imperfect_estimates=candor_imperfect_runs,
    )

    print("\nExperiment Complete.")
    print(f"Saved results to {filename} with shapes:")
    print(f"  PDIS: {pdis_runs.shape}")
    print(f"  DR: {dr_runs.shape}")
    print(f"  C-PDIS (Perfect): {cpdis_perfect_runs.shape}")
    print(f"  CANDOR (Perfect): {candor_perfect_runs.shape}")


if __name__ == "__main__":
    main()