import numpy as np
import torch
from trajectory_classes import Trajectory
from stable_baselines3 import DQN


# ==========================================
#  Core Helper / Generation Functions
# ==========================================

def calculate_true_seq_policy_value(
        policy: np.ndarray,
        initial_state_dist: np.ndarray,
        transition_probs: np.ndarray,
        reward_means: np.ndarray,
        gamma: float,
        horizon: int
) -> float:
    """
    Calculate the true value of a policy in a finite-horizon MDP using Backward Induction.
    """
    num_states, num_actions = policy.shape
    V_next = np.zeros(num_states)

    for t in reversed(range(horizon)):
        expected_future_val = np.sum(transition_probs * V_next[None, None, :], axis=2)
        Q_curr = reward_means + gamma * expected_future_val
        V_curr = np.sum(policy * Q_curr, axis=1)
        V_next = V_curr

    return float(np.sum(initial_state_dist * V_next))


def calculate_q_table(
        policy: np.ndarray,
        transition_probs: np.ndarray,
        reward_means: np.ndarray,
        gamma: float,
        horizon: int
) -> np.ndarray:
    """
    Calculate the Exact Q-table (Horizon, S, A) for a given policy.
    Replaces the DQN to provide q_value.
    """
    num_states, num_actions = policy.shape

    # Store Q-values for each timestep: (H, S, A)
    q_table = np.zeros((horizon, num_states, num_actions))
    V_next = np.zeros(num_states)  # V_H = 0

    for t in reversed(range(horizon)):
        # Q_t(s,a) = R(s,a) + gamma * E[V_{t+1}(s')]
        expected_future_val = np.sum(transition_probs * V_next[None, None, :], axis=2)
        Q_curr = reward_means + gamma * expected_future_val

        # Store
        q_table[t] = Q_curr

        # Update V for next step
        V_next = np.sum(policy * Q_curr, axis=1)

    return q_table


def generate_sequential_dataset_of_trajectories(
        num_states: int,
        num_actions: int,
        horizon: int,
        initial_state_dist: np.ndarray,
        transition_probs: np.ndarray,
        reward_means: np.ndarray,
        reward_stds: np.ndarray,
        policy: np.ndarray,
        num_trajectories: int,
        rng: np.random.Generator,
) -> list[Trajectory]:
    """
    Generate a dataset of trajectories using explicit MDP dynamics.
    """
    trajectories = []

    for _ in range(num_trajectories):
        states_list = []
        actions_list = []
        rewards_list = []

        curr_state = rng.choice(num_states, p=initial_state_dist)

        for t in range(horizon):
            # Select Action
            action = rng.choice(num_actions, p=policy[curr_state])

            # Observe Reward
            mean = reward_means[curr_state, action]
            std = reward_stds[curr_state, action]
            reward = rng.normal(mean, std)

            # Record
            states_list.append(curr_state)
            actions_list.append(action)
            rewards_list.append(reward)

            # Transition
            next_state = rng.choice(num_states, p=transition_probs[curr_state, action])
            curr_state = next_state

        trajectories.append(Trajectory(
            np.array(states_list),
            np.array(actions_list),
            np.array(rewards_list),
            num_possible_states=num_states,
            num_possible_actions=num_actions
        ))

    return trajectories


def generate_q_annotations(
        dataset: list[Trajectory],
        # q_net: DQN,
        q_table: np.ndarray,
        num_annotations: int,
        rng: np.random.Generator
) -> np.ndarray:
    """
    Generate Counterfactual Annotations (Q-values) using a trained Q-Network.
    """
    num_trajectories = len(dataset)
    num_timesteps = len(dataset[0])
    num_actions = dataset[0].num_possible_actions

    all_annotations = np.full((num_trajectories, num_timesteps, num_actions), np.nan)

    # 1. Construct pool of all possible counterfactual queries
    cf_pool = []

    for i, traj in enumerate(dataset):
        states, actions, _ = traj.unpack()
        for t in range(num_timesteps):
            factual_a = actions[t]
            curr_s = states[t]
            for a in range(num_actions):
                if a != factual_a:
                    cf_pool.append((i, t, a, curr_s))

    # 2. Sample from pool based on budget
    max_possible = len(cf_pool)
    actual_budget = np.minimum(num_annotations, max_possible)
    chosen_indices = rng.choice(max_possible, size=actual_budget, replace=False)

    # # 3. Query Q-Net
    # with torch.no_grad():
    #     for idx in chosen_indices:
    #         traj_idx, t, a, s = cf_pool[idx]
    #
    #         # Manual Tensor Conversion
    #         obs_np = np.array([s])
    #         obs_tensor = torch.as_tensor(obs_np, device=q_net.device).float()
    #
    #         # Forward pass
    #         q_values, _ = q_net.q_net(obs_tensor)
    #
    #         # Extract scalar
    #         q_val = float(q_values[0, a].item())
    #
    #         all_annotations[traj_idx, t, a] = q_val

    # 3. Lookup Q table
    for idx in chosen_indices:
        traj_idx, t, a, s = cf_pool[idx]
        # Direct lookup from the Q-table (Ground Truth or Biased Truth)
        q_val = q_table[t, s, a]
        all_annotations[traj_idx, t, a] = q_val

    return all_annotations


# ==========================================
#  OPE Algorithms (Sequential)
# ==========================================

def run_pdis(
        pi_e: np.ndarray,
        pi_b: np.ndarray,
        dataset: list[Trajectory],
        gamma: float = 1.0,
) -> float:
    """
    Per-Decision Importance Sampling (PDIS) for sequential MDPs.
    Estimates V(pi_e).

    Formula (Recursive):
    v_T = 0
    v_{T-t} = rho_t * (r_t + gamma * v_{T-t+1})
    """
    if len(dataset) == 0:
        return 0.0

    # 1. Unpack data
    # Assuming all trajectories have same length
    states = np.array([t.states for t in dataset])  # (N, T)
    actions = np.array([t.actions for t in dataset])  # (N, T)
    rewards = np.array([t.rewards for t in dataset])  # (N, T)

    n_samples, n_timesteps = states.shape

    # 2. Compute Importance Ratios rho_t
    # pi(a|s) -> shape (N, T)
    # Using advanced indexing
    probs_e = pi_e[states, actions]
    probs_b = pi_b[states, actions]

    # Handle division by zero
    probs_b = np.clip(probs_b, 1e-12, None)
    rhos = probs_e / probs_b  # (N, T)

    # 3. Recursive Calculation (Backward)
    # v_vals represents the estimated value from step t onwards
    v_vals = np.zeros(n_samples)  # Initialize v_{T} = 0 (value after end of episode)

    for t in reversed(range(n_timesteps)):
        r_t = rewards[:, t]
        rho_t = rhos[:, t]

        # v_{t} = rho_t * (r_t + gamma * v_{t+1})
        v_vals = rho_t * (r_t + gamma * v_vals)

    return float(np.mean(v_vals))


def run_sequential_cis(
        pi_e: np.ndarray,
        pi_b: np.ndarray,
        dataset: list[Trajectory],
        annotations: np.ndarray,
        alpha: float = 0.5,
        gamma: float = 1.0
) -> float:
    """
    Based on Definition 4 in Tang & Wiens (2023).

    Args:
        pi_e: Evaluation policy (S, A)
        pi_b: Behavior policy (S, A)
        dataset: List of trajectories
        annotations: (N, T, A) array of Q-values (NaN for missing)
        alpha: Weight mixing parameter (0.0 to 1.0)
        gamma: Discount factor
    """
    if len(dataset) == 0:
        return 0.0

    # ----------------------------
    # 1. Pack Data
    # ----------------------------
    states = np.array([t.states for t in dataset])  # (N, T)
    actions_factual = np.array([t.actions for t in dataset])  # (N, T)
    rewards = np.array([t.rewards for t in dataset])  # (N, T)

    n_samples, n_timesteps = states.shape
    n_states, n_actions = pi_b.shape

    # ----------------------------
    # 2. Compute Weights w_t^a
    # ----------------------------
    # Count number of CF annotations (K) per sample-step
    # annotations shape: (N, T, A)
    is_annotated = ~np.isnan(annotations)  # (N, T, A) bool
    k_counts = np.sum(is_annotated, axis=2)  # (N, T)

    weights = np.zeros((n_samples, n_timesteps, n_actions), dtype=float)

    # Factual weights:
    # If K > 0 (has annotations): w = 1 - alpha
    # If K == 0 (no annotations): w = 1.0
    w_factual_vals = np.where(k_counts > 0, 1.0 - alpha, 1.0)

    # Assign factual weights using advanced indexing
    batch_idx = np.arange(n_samples)[:, None]
    time_idx = np.arange(n_timesteps)[None, :]
    weights[batch_idx, time_idx, actions_factual] = w_factual_vals

    # Counterfactual weights:
    # w = alpha / K (distributed equally among annotations)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_cf_vals = np.where(k_counts > 0, alpha / k_counts, 0.0)

    # Broadcast w_cf_vals (N, T) to (N, T, A) where annotated
    weights = np.where(is_annotated, w_cf_vals[..., None], weights)

    # ----------------------------
    # 3. Estimate pi_b_plus
    # ----------------------------
    # pi_b_plus(a|s) = E(w^a) * pi_b(a|s) + sum_{a'} pi_b(a'|s) * E[w^a']

    flat_states = states.ravel()
    flat_af = actions_factual.ravel()
    flat_w = weights.reshape(-1, n_actions)

    # sums_w[s, a_target, a_fact]: sum of weights for a_target when factual was a_fact
    sums_w = np.zeros((n_states, n_actions, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.float64)

    for a_target in range(n_actions):
        np.add.at(sums_w, (flat_states, np.full_like(flat_states, a_target), flat_af), flat_w[:, a_target])

    np.add.at(counts, (flat_states, flat_af), 1.0)

    # bar_w[s, a_target, a_fact] = E[w^{a_target} | s, a_fact]
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_w = np.where(counts[:, None, :] > 0, sums_w / counts[:, None, :], 0.0)

    # Explicit Calculation of pi_b_plus(a_target | s)
    # Formula: pi_b(a|s) * bar_w(a|s,a) + sum_{a'!=a} pi_b(a'|s) * bar_w(a|s,a')

    # Term 1: Factual Contribution (Diagonal of bar_w where a_target == a_fact)
    # bar_w diagonal: (S, A) -> bar_w[s, a, a]
    diag_bar_w = np.diagonal(bar_w, axis1=1, axis2=2)  # (S, A)
    term_factual = pi_b * diag_bar_w

    # Term 2: Counterfactual Contribution (Sum over a_fact where a_fact != a_target)
    # Total sum over a_fact: sum_p pi_b(p|s) * bar_w(a|s, p)
    total_weighted_sum = np.einsum("sp,sap->sa", pi_b, bar_w)
    term_cf = total_weighted_sum - term_factual

    # Final pi_b_plus
    pi_b_plus = term_factual + term_cf
    pi_b_plus = np.clip(pi_b_plus, 1e-12, None)

    # ----------------------------
    # 4. Recursive Value Estimation (C-PDIS)
    # ----------------------------
    # v_{T-t+1} = w^at * rho^at * (r_t + gamma * v_{T-t}) + sum_{cf} w^cf * rho^cf * g^cf

    v_vals = np.zeros(n_samples)  # Initialize v_T = 0

    for t in reversed(range(n_timesteps)):
        s_t = states[:, t]  # (N,)
        a_fact = actions_factual[:, t]  # (N,)
        r_t = rewards[:, t]  # (N,)
        w_t = weights[:, t, :]  # (N, A)
        g_t = annotations[:, t, :]  # (N, A), contains Q-values

        # Calculate rho_t for ALL actions at this step: (N, A)
        # pi_e(a|s_t) / pi_b_plus(a|s_t)
        rho_all = pi_e[s_t] / pi_b_plus[s_t]  # (N, A)

        # Factual Term: w[a_fact] * rho[a_fact] * (r + gamma * v_next)
        # Use indexing to get specific values for factual actions
        w_fact_t = w_t[np.arange(n_samples), a_fact]
        rho_fact_t = rho_all[np.arange(n_samples), a_fact]

        term_factual = w_fact_t * rho_fact_t * (r_t + gamma * v_vals)

        # Counterfactual Term: sum_{a' != a_fact} w[a'] * rho[a'] * g[a']
        # g_t is already Q-value, so we use it directly (Tang Def 4)

        # Mask out factual action to ensure we sum only CFs (though w should handle this, safety first)
        # w_t should be 0 for non-annotated CFs and non-selected CFs.

        # Element-wise multiply and sum
        g_t_safe = np.nan_to_num(g_t, nan=0.0)

        term_cf_all = w_t * rho_all * g_t_safe
        # Zero out the factual action contribution in this sum (it's handled in term_factual)

        mask_fact = np.zeros_like(term_cf_all)
        mask_fact[np.arange(n_samples), a_fact] = 1.0

        term_cf_sum = np.sum(term_cf_all * (1.0 - mask_fact), axis=1)

        # Combine
        v_vals = term_factual + term_cf_sum

    return float(np.mean(v_vals))


def run_sequential_candor(
        pi_e: np.ndarray,
        pi_b: np.ndarray,
        dataset: list[Trajectory],
        annotations: np.ndarray,
        n_fold: int = 2,
        alpha: float = 0.5,
        gamma: float = 1.0
) -> float:
    """
    Sequential CANDOR.

    Uses K-Fold Cross-Fitting.
    On the training fold, estimates Q_hat(t, s, a) using a weighted combination of:
      1. Factual Data: r + gamma * V_hat_next
      2. Annotations: q_value

    Args:
        pi_e: Evaluation policy (S, A)
        pi_b: Behavior policy (S, A)
        dataset: List of trajectories
        annotations: (N, T, A) array of Q-values (NaN for missing)
        n_fold: Number of folds for cross-fitting
        alpha: Weight for annotations (0 to 1).
               Factuals get (1-alpha), Annotations get alpha.
        gamma: Discount factor
    """
    n_samples = len(dataset)
    if n_samples == 0:
        return 0.0

    # 1. Unpack Data
    states = np.array([t.states for t in dataset])  # (N, T)
    actions = np.array([t.actions for t in dataset])  # (N, T)
    rewards = np.array([t.rewards for t in dataset])  # (N, T)


    n_timesteps = states.shape[1]
    num_states, num_actions = pi_b.shape

    # 2. Prepare Folds
    indices = np.arange(n_samples)

    # rng_internal = np.random.default_rng(2025)
    # rng_internal.shuffle(indices)
    fold_indices = np.array_split(indices, n_fold)

    dr_estimates = []

    # 3. Cross-Fitting Loop
    for k in range(n_fold):
        eval_idx = fold_indices[k]
        train_idx = np.setdiff1d(indices, eval_idx)

        if len(train_idx) == 0:
            continue

        # --- Train Step: Estimate Q_hat on train_idx ---
        # We perform Backward Induction on the training data

        # Initialize Q-table: (T, S, A)
        q_hat = np.zeros((n_timesteps, num_states, num_actions))
        # Initialize V-table for next step: (S,) - starts at 0 for step T
        v_next = np.zeros(num_states)

        # Slice training data
        tr_states = states[train_idx]  # (N_tr, T)
        tr_actions = actions[train_idx]  # (N_tr, T)
        tr_rewards = rewards[train_idx]  # (N_tr, T)
        tr_anns = annotations[train_idx]  # (N_tr, T, A)

        # Iterate backwards
        for t in reversed(range(n_timesteps)):
            q_curr = np.zeros((num_states, num_actions))

            # Prepare next states for this step t
            if t < n_timesteps - 1:
                tr_next_states = tr_states[:, t + 1]
                # Value of next states: V_{t+1}(s')
                # shape (N_tr,)
                target_v_next = v_next[tr_next_states]
            else:
                # Terminal step
                target_v_next = np.zeros(len(train_idx))

            # Factual Targets: r + gamma * V'
            factual_targets = tr_rewards[:, t] + gamma * target_v_next

            # For each state-action pair, aggregate
            for s in range(num_states):
                for a in range(num_actions):
                    # 1. Factual Data Points
                    mask_fact = (tr_states[:, t] == s) & (tr_actions[:, t] == a)
                    fact_vals = factual_targets[mask_fact]
                    n_fact = len(fact_vals)

                    # 2. Annotation Data Points
                    # Annotation exists if it's not NaN
                    # We look at all samples in train set where state is s
                    mask_state = (tr_states[:, t] == s)
                    # Extract annotations for action a
                    ann_vals_all = tr_anns[mask_state, t, a]
                    ann_vals = ann_vals_all[~np.isnan(ann_vals_all)]
                    n_ann = len(ann_vals)

                    # 3. Weighted Combination
                    # Sum(val * weight) / Sum(weight)
                    # Weights: (1-alpha) for factuals, alpha for annotations

                    numerator = 0.0
                    denominator = 0.0

                    if n_fact > 0:
                        numerator += np.sum(fact_vals) * (1.0 - alpha)
                        denominator += n_fact * (1.0 - alpha)

                    if n_ann > 0:
                        numerator += np.sum(ann_vals) * alpha
                        denominator += n_ann * alpha

                    if denominator > 0:
                        q_curr[s, a] = numerator / denominator
                    else:
                        # Fallback for unseen (s,a): 0.0 or global mean
                        q_curr[s, a] = 0.0

                        # Store Q_t
            q_hat[t] = q_curr

            # Compute V_t for recursion: V(s) = sum pi_e(a|s) * Q(s,a)
            # shape (S,)
            v_curr = np.sum(pi_e * q_curr, axis=1)
            v_next = v_curr  # Update for next iter

        # --- Eval Step: Run Recursive DR on eval_idx using q_hat ---
        ev_states = states[eval_idx]  # (N_ev, T)
        ev_actions = actions[eval_idx]  # (N_ev, T)
        ev_rewards = rewards[eval_idx]  # (N_ev, T)

        N_ev = len(eval_idx)

        # Recursive DR:
        # V_DR_t = V_model_t + rho_t * (r_t + gamma * V_DR_{t+1} - Q_model_t)

        v_dr = np.zeros(N_ev)  # Represents V_{T} = 0

        for t in reversed(range(n_timesteps)):
            s_t = ev_states[:, t]
            a_t = ev_actions[:, t]
            r_t = ev_rewards[:, t]

            # Model values
            q_model_table = q_hat[t]  # (S, A)

            # Q_hat(s_t, a_t)
            q_hat_taken = q_model_table[s_t, a_t]

            # V_model(s_t) = sum pi_e(a|s_t) * Q_hat(s_t, a)
            v_model_val = np.sum(pi_e[s_t] * q_model_table[s_t], axis=1)

            # Importance Ratios
            rho_t = pi_e[s_t, a_t] / np.clip(pi_b[s_t, a_t], 1e-12, None)

            # DR Recursion
            # Term: r_t + gamma * v_dr_next - Q_hat_taken
            advantage = r_t + gamma * v_dr - q_hat_taken

            v_dr = v_model_val + rho_t * advantage

        # Average over eval fold
        dr_estimates.extend(v_dr.tolist())

    return float(np.mean(dr_estimates))