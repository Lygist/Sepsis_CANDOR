import numpy as np
from scipy import stats
from trajectory_classes import Trajectory
from typing import Optional, Union

'''
DESCRIPTION: Implements core functions for running importance sampling
methods. Specifically, generates datasets and runs importance sampling.
'''

RNG = np.random.default_rng(123)


def calculate_policy_value_rmse_and_ci(
    estimated_policy_values: np.ndarray,
    true_policy_value: float,
    alpha: float = 0.05
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Given the estimated_policy_values generated for one particular policy,
    and the true_policy_value of that policy,
    calculate the RMSE and the half-width of the confidence interval.

    One policy value should be estimated per dataset.
    Each dataset should have some number of trajectories.
    The estimated_policy_values for a dataset should be the average over the trajectories.
    Dimension 0 of the estimated_policy_values is over the dataset.
    """
    squared_errors = (estimated_policy_values - true_policy_value) ** 2
    mse = np.mean(squared_errors, axis=0)
    rmse = np.sqrt(mse)
    # SE of MSE
    se_mse = stats.sem(squared_errors, axis=0)
    # Delta-method: SE of RMSE
    se_rmse = se_mse / (2 * rmse)
    # Half-width of CI
    z = stats.norm.ppf(1 - alpha/2)
    half_width = z * se_rmse

    return rmse, half_width


def calculate_true_policy_value(policy, state_distribution, reward_means):
    return np.sum(state_distribution @ (policy * reward_means))


def generate_dataset_of_trajectories(
    state_distribution: np.ndarray,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    policy: np.ndarray,
    rng: np.random.Generator,
    # NOTE: In theory, bandits should always have only one timestep. In the
    # original code, they provided some flexibility to having multiple
    # timesteps, so we do the same in our implementation.
    trajectory_len: int = 1,
    num_trajectories: int = 1000,
):
    """
    Args:
    state_distribution:  Shape (num_states,).
        In the original code, this is d0.
        Since we're in a bandit setting, we can use this distribution for all timesteps.
    reward_means:  Shape (num_states, num_actions)
    reward_stds:  Shape (num_states, num_actions)
    policy:  Shape (num_states, num_actions)
    rng:  A numpy random number generator
    trajectory_len:  Trajectory length
    num_trajectories:  Number of trajectories to generate

    CREDIT: This function is based on a subsection of single_exp_setting() from
    https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
    """
    # Not hard-coding these values to 2, in case we want to use this code in a
    # more general library
    num_states, num_actions = policy.shape

    trajectories = []

    for _ in range(num_trajectories):
        states = rng.choice(
            num_states,
            size=trajectory_len,
            p=state_distribution)
        actions = np.array(
            [rng.choice(num_actions, p=policy[state]) for state in states])
        rewards = np.array([
            rng.normal(reward_means[state, action], reward_stds[state, action])
                for state, action in zip(states, actions)])

        trajectories.append(Trajectory(
            states,
            actions,
            rewards,
            num_possible_states=num_states,
            num_possible_actions=num_actions
        ))

    return trajectories


# STATUS: Needs testing
def generate_annotations(
    factual_dataset: list[Trajectory],
    num_annotations: int,
    annotated_reward_means: np.ndarray,
    annotated_reward_stds: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Args:
    factual_dataset: A list of batch_size equal-length Trajectories.
    num_annotations: The total number of annotations to generate.
        Replaces 'Pc' in the original code.
    annotated_reward_means: Shape (num_states, num_actions)
    annotated_reward_stds: Shape (num_states, num_actions)
    rng: A numpy random number generator

    Return Value:
    all_annotations: An array of counterfactual annotations. Effective shape
        (batch_size, trajectory_len, num_actions)

    If bias and/or variance are to be added to the annotations, it should be done
    *before* the reward_[means|stds] are passed to this function.

    This function should be called several times to generate annotations
    of varying fidelities.

    CREDIT: This function is built upon a subsection of single_exp_setting() from
    https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
    """
    # NOTE: In the original code, they determine the probability of getting a
    # counterfactual annotation based on the observed actual action (i.e.,
    # Pc[x_i, a_i]). This is fine for the 2-state scenario, but we will want to
    # modify this behavior if we need to generalize this code to multi-action
    # scenarios.
    num_trajectories = len(factual_dataset)
    num_timesteps = len(factual_dataset[0])
    num_actions = factual_dataset[0].num_possible_actions
    all_annotations = np.full((num_trajectories, num_timesteps, num_actions), np.nan)

    # Construct a Pool of all possible counterfactual actions
    cf_pool = []

    for i, traj in enumerate(factual_dataset):
        states, actions, _ = traj.unpack()
        for t in range(num_timesteps):
            factual_a = actions[t]
            for a in range(num_actions):
                if a != factual_a:
                    cf_pool.append((i, t, a, states[t])) #(traj_idx, timestep, cf_action, state)

    # Sample CFs from the pool based on budget
    max_possible = len(cf_pool)
    actual_budget = np.minimum(num_annotations, max_possible)
    chosen_indices = rng.choice(max_possible, size=actual_budget, replace=False)

    # Generate values for selected CFs
    for idx in chosen_indices:
        traj_idx, t, a, s = cf_pool[idx]
        # Generate reward based on the specific (state, action) mean/std
        mean = annotated_reward_means[s, a]
        std = annotated_reward_stds[s, a]
        all_annotations[traj_idx, t, a] = rng.normal(mean, std)

    return all_annotations


def collapse_annotations(
    annotations: Optional[np.ndarray],
    *,
    source_weights: Optional[np.ndarray] = None,  # shape (M,), optional
    dtype: type = float,
) -> Optional[np.ndarray]:
    """
    Collapse multiple annotation sources along axis=0 via (weighted) nan-mean.

    Parameters
    ----------
    annotations : (M, batch, T, A) or None
        M sources of annotations, NaN for missing
    source_weights : (M,), optional
        Non-negative weights per source. If None, uses equal weights.
        Only weights corresponding to non-NaN entries at a slot are applied
    dtype : output dtype

    Returns
    -------
    ann_mean : (batch, T, A) or None
        NaN where all sources are missing. None, if `annotations` is None
    """
    if annotations is None:
        return None

    if annotations.ndim != 4:
        raise ValueError(f"`annotations` must be (M, batch, T, A); got shape {annotations.shape}")

    n_ann = annotations.shape[0]
    anns = np.asarray(annotations, dtype=float)  # safe cast

    mask = ~np.isnan(anns)                       # (M, B, T, A)
    if source_weights is None:
        # Unweighted nan-mean: sum / count only where mask is True
        counts = np.sum(mask, axis=0)                             # (B, T, A)
        sums = np.nansum(anns, axis=0)                            # (B, T, A)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=(counts > 0))
    else:
        w = np.asarray(source_weights, dtype=float).reshape(n_ann, 1, 1, 1)   # (M, 1, 1, 1)
        if np.any(w < 0):
            raise ValueError("source_weights must be non-negative")
        # Zero out weights where value is NaN
        w_eff = np.where(mask, w, 0.0)                                 # (M, B, T, A)
        wsum = np.sum(w_eff, axis=0)                                   # (B, T, A)
        wval = np.nansum(w_eff * anns, axis=0)                         # (B, T, A)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.divide(wval, wsum, out=np.full_like(wsum, np.nan), where=(wsum > 0))

    return mean.astype(dtype, copy=False)


def combine_dataset_of_trajectories_with_annotations(
    factual_dataset: list[Trajectory],
    annotations: np.ndarray,
) -> np.ndarray:
    """
    This function exists separately from
    combine_single_trajectory_with_annotations for efficiency reasons.

    Args:
    factual_dataset: A list of batch_size Trajectories.
    annotations: Effective shape: (num_annotation_sets, batch_size,
        trajectory_len, num_actions)

    Output:
    combined_factual_rewards_and_annotations: np.ndarray of shape
        (1 + num_annotation_sets, batch_size, trajectory_len, num_actions).
        np.nan will be found wherever counterfactual observations are not
        observed.
    """
    stacked_nan_expanded_factual_rewards = np.array(
        [traj.create_nan_expanded_rewards() for traj in factual_dataset])
    stacked_nan_expanded_factual_rewards = np.expand_dims(
        stacked_nan_expanded_factual_rewards, 0)

    # Combine the rewards and annotations into one np.ndarray.
    return np.concatenate(
        (stacked_nan_expanded_factual_rewards, np.array(annotations)),
        axis=0)


def combine_single_trajectory_with_annotations(
    trajectory: Trajectory,
    annotations: np.ndarray,
) -> np.ndarray:
    """
    Args:
    trajectory: A single trajectory.
    annotations: An np.ndarray of shape:
        (num_annotation_sets, trajectory_len, num_actions)

    Output:
    combined_factual_rewards_and_annotations: np.ndarray of shape
        (1 + num_annotation_sets, trajectory_len, num_actions). np.nan will be
        found wherever counterfactual observations are not observed.
    """
    return np.concatenate(
      (np.expand_dims(trajectory.create_nan_expanded_rewards(), 0), annotations),
      axis=0)


def create_combined_states(
    factual_dataset: list[Trajectory]
) -> np.ndarray:
    """
    Output: Shape (batch_size, trajectory_len)
    """
    return np.stack([trajectory.states for trajectory in factual_dataset])


def run_vanilla_is(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    dataset: list[Trajectory],
    bx: int = 0,
    gamma: float = 1.0,
    mode: str = "per_decision",  # "per_decision" or "trajectory"
    dtype=np.float32
) -> Union[float, np.ndarray]:
    """
    Run vanilla Importance Sampling to generate an estimate of policy_e's value for the given dataset.

    Args:
    policy_e: The evaluation policy
    policy_b: The behavior policy
    dataset: A list of trajectories
    bx: 0 if you want the mean estimate over all trajectories, 1 if you want per-trajectory estimates
    gamma: discount factor for MDP.

    (The policy's value is not to be confused with the policy's value *function*)

    CREDIT: This function is based on a subsection of single_exp_setting() from
    https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
    """
    estimates = []

    for traj in dataset:
        states, actions, rewards = traj.unpack()
        beh = np.clip(policy_b[states, actions], 1e-12, None)
        evl = policy_e[states, actions]
        rho_t = evl / beh                                 # (T,)

        if len(traj) == 1:  # bandit fast-path
            estimates.append(float(rho_t[0] * rewards[0]))
            continue

        # MDP paths
        if mode == "trajectory":
            w = float(np.prod(rho_t))
            g = float(np.sum((gamma ** np.arange(len(traj))) * rewards))
            estimates.append(w * g)
        elif mode == "per_decision":
            cum_rho = np.cumprod(rho_t)                  # (T,)
            disc = gamma ** np.arange(len(traj))
            estimates.append(float(np.sum(cum_rho * disc * rewards)))
        else:
            raise ValueError('mode must be "per_decision" or "trajectory"')

    est = np.asarray(estimates, dtype=dtype)
    return est if bx != 0 else est.mean(dtype=dtype)


# This is the algorithm proposed by Tang & Wiens.
# Denoted as C-IS in their original paper, and as IS+ in Aishwarya's paper.
def run_is_plus(
    pi_e: np.ndarray,
    pi_b: np.ndarray,
    dataset: list[Trajectory],
    annotations: np.ndarray,
    alpha: float = 0.5,
    bx: int = 0,
    dtype=np.float32,
    source_weights: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    C-IS (IS+) with per-sample weights in a 2-action bandit.

    Weight rule:
      - If the counterfactual action is annotated: w_factual = 0.5, w_cf = 0.5
      - Else:                                      w_factual = 1.0, w_cf = 0.0
      (sum of weights = 1)

    Augmented behavior & ratio:
      pi_b_plus(a|s) = sum_{a_prime} pi_b(a_prime|s) * E[w^a | s, a_prime]
      rho_plus(a|s)  = pi_e(a|s) / pi_b_plus(a|s)
    """
    # ----------------------------
    # Pack dataset into arrays
    # ----------------------------
    n_samples = len(dataset)
    n_timesteps = dataset[0].states.shape[0] if n_samples > 0 else 0
    n_states, n_actions = pi_b.shape


    states = np.stack([tr.states for tr in dataset], axis=0)            # (n_samples, n_timesteps)
    actions_factual = np.stack([tr.actions for tr in dataset], axis=0)  # (n_samples, n_timesteps)
    rewards_factual = np.stack([tr.rewards for tr in dataset], axis=0)  # (n_samples, n_timesteps)

    # collapse multiple sets of annotations, if any. Shape (M, B, T, A)
    ann_mean = collapse_annotations(annotations, source_weights=source_weights)

    # ----------------------------
    # Calculate Weights
    # ----------------------------
    # Count number of CF annotations (K)
    is_annotated = ~np.isnan(ann_mean)
    k_counts = np.sum(is_annotated, axis=2)
    weights = np.zeros((n_samples, n_timesteps, n_actions), dtype=float)

    # For factual actions: If K > 0, weight is 1-alpha. If K == 0, weight is 1.0.
    w_factual_vals = np.where(k_counts > 0, 1.0 - alpha, 1.0)
    batch_idx = np.arange(n_samples)[:, None]  # (N, 1)
    time_idx = np.arange(n_timesteps)[None, :]  # (1, T)
    weights[batch_idx, time_idx, actions_factual] = w_factual_vals

    # For CF actions, weight = alpha / K.
    with np.errstate(divide='ignore', invalid='ignore'):
        w_cf_vals = np.where(k_counts > 0, alpha / k_counts, 0.0)

    weights = np.where(is_annotated, w_cf_vals[..., None], weights)

    # ----------------------------
    # Estimate bar_W(a|s,a')
    # ----------------------------
    flat_states = states.ravel()  # (n_samples * n_timesteps,)
    flat_af = actions_factual.ravel()  # (n_samples * n_timesteps,)
    flat_w = weights.reshape(-1, n_actions)  # (n_samples * n_timesteps, n_actions)

    # ---------- Allocate accumulators ----------
    # sums_w[s, a, a_prime] will hold the sum of weights for the target action "a"
    # gathered over all (i,t) with the factual pair (state=s, action=a_prime)
    sums_w = np.zeros((n_states, n_actions, n_actions), dtype=np.float64)  # (s, a, a_prime)
    counts = np.zeros((n_states, n_actions), dtype=np.float64)  # (s, a_prime) i.e., N(s, a_prime)

    # ---------- Scatter-add without looping over samples/timesteps ----------
    for a_idx in range(n_actions):
        # add the a_idx column of weights into sums_w[:, a_idx, :]
        np.add.at(sums_w, (flat_states, np.full_like(flat_states, a_idx), flat_af), flat_w[:, a_idx])

    # counts of (s, a_prime) occurrences (independent of the target action a)
    np.add.at(counts, (flat_states, flat_af), 1.0)

    # ---------- Normalize to get bar_w(s, a, a_prime) = E[w^a | s, a_prime] ----------
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_w = np.where(counts[:, None, :] > 0, sums_w / counts[:, None, :], 0.0)  # (n_states, n_actions, n_actions)

    # Sanity checks (for debugging):
    # 1) Each (s,a_prime) column should sum to 1 across the target action "a".
    # assert np.allclose(bar_w.sum(axis=1), 1.0, atol=1e-8)
    # 2) All entries between 0 and 1
    # assert np.all((bar_w >= -1e-12) & (bar_w <= 1 + 1e-12))

    # ----------------------------
    # Augmented behavior policy and importance sampling ratio
    # ----------------------------
    pi_b_plus = np.einsum("sp,sap->sa", pi_b, bar_w)   # (n_states, n_actions)
    pi_b_plus = np.clip(pi_b_plus, 1e-12, None)
    rho_plus = pi_e / pi_b_plus                                           # (n_states, n_actions)

    # ----------------------------
    # Rewards c_i^a
    # ----------------------------
    c = np.full((n_samples, n_timesteps, n_actions), np.nan, dtype=float)
    c[np.arange(n_samples)[:, None], np.arange(n_timesteps)[None, :], actions_factual] = rewards_factual
    # inner: fill the non-factuals by the ann_mean (ann_mean can be NAN)
    # outer: use the combined rewards if annotations are available, else use the factual rewards
    c = np.where(is_annotated, np.where(np.isnan(c), ann_mean, c), c)

    # ----------------------------
    # Calculate the estimate per-trajectory
    # ----------------------------
    per_traj = np.empty((n_samples,), dtype=float)
    for i in range(n_samples):
        wi = weights[i]
        rhoi = rho_plus[states[i]]  # rho_plus for "all" actions at this state
        ci = np.nan_to_num(c[i], nan=0.0)
        # The inner sum is over actions. The outer mean is over timesteps.
        per_traj[i] = np.mean(np.sum(wi * rhoi * ci, axis=1))

    est = np.asarray(per_traj, dtype=dtype)
    return est if bx != 0 else est.mean(dtype=dtype)


def run_dr(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    dataset: list[Trajectory],
    annotations: Optional[np.ndarray] = None,  # (M, batch, T, A) or None
    n_fold: int = 2,
    alpha: float = 0.5,
    bx: int = 0,
    dtype=np.float32,
    source_weights: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Doubly Robust (DR) with K-fold cross-fitting.
    - If `annotations` are None, this is standard DR with cross-fitting.
    - If `annotations` are provided (shape: (M, batch, T, A)), this becomes CANDOR-style DM^+–IS with cross-fitting.

    Args:
        policy_e: (S, A) evaluation policy
        policy_b: (S, A) behavior policy
        dataset: list of Trajectory objects (each has .unpack() and .create_nan_expanded_rewards())
        annotations: optional counterfactual annotations of shape (M, batch, T, A)
        n_fold: number of folds (>=2)
        alpha: Weight for counterfactual annotations in R_hat estimation. Factuals get weight (1-alpha). Range [0, 1].
        bx: if 0, return a scalar array [mean]; else return per-trajectory estimates
        dtype: output dtype

    Returns:
        np.ndarray of shape (n_trajectories,) if bx != 0, else (1,).
    """
    n = len(dataset)
    assert 2 <= n_fold <= n, "K must be between 2 and number of trajectories"

    # Shapes from the first trajectory
    num_states = dataset[0].num_possible_states
    num_actions = dataset[0].num_possible_actions

    # Pre-extract factual rewards and (s,a,r) tuples per trajectory
    factual_rewards = [traj.create_nan_expanded_rewards() for traj in dataset]     # list of (T, A)
    unpacked = [traj.unpack() for traj in dataset]                                 # list of (states_T, actions_T, rewards_T)
    states = np.stack([u[0] for u in unpacked], axis=0)                            # (batch, T)

    # collapse annotations if multiple
    ann_mean = collapse_annotations(annotations, source_weights=source_weights)

    # Split indices into K folds
    fold_indices = np.array_split(np.arange(n), n_fold)

    # Container for per-trajectory DR values (preserve order)
    dr_per_traj = np.empty(n, dtype=dtype)

    # ---- Helper: estimate r_hat(s,a) on a training subset (count-weighted) ----
    def _estimate_r_hat_on_train(idx: np.ndarray) -> np.ndarray:
        """
        Estimate r_hat(s,a) using *only* trajectories in train_idx.
        Count-weighted aggregation across factual + (optional) annotations.
        Returns: (S, A) array.
        """
        # Collect training subset
        train_rewards = np.stack([factual_rewards[i] for i in idx], axis=0)     # (b_tr, T, A)
        if ann_mean is None:
            combined_rewards = np.expand_dims(train_rewards, 0)   # (1, b_tr, T, A)
            combined_weights = np.ones_like(combined_rewards)
        else:
            ann_tr = ann_mean[idx, :, :]  # (b_tr, T, A)
            combined_rewards = np.stack([train_rewards, ann_tr], axis=0)    # (2, b_tr, T, A)

            w_vec = np.array([1.0 - alpha, alpha], dtype=float).reshape(2, 1, 1, 1)
            combined_weights = np.broadcast_to(w_vec, combined_rewards.shape)

        train_states  = states[idx]                                             # (b_tr, T)
        # Broadcast states to shape (1, b_tr, T, 1) for masking by state
        states_broadcast = np.expand_dims(np.expand_dims(train_states, 0), 3)  # (1, b_tr, T, 1)

        # Accumulate SUM and COUNT for each (s,a), then take the ratio (safe divide).
        r_sum = np.zeros((num_states, num_actions), dtype=np.float64)
        r_weight_sum = np.zeros((num_states, num_actions), dtype=np.float64)

        # Per-state accumulation
        for s in range(num_states):
            mask_s = (states_broadcast == s)                              # (1(+M), b_tr, T, 1)
            vals_s = np.where(mask_s, combined_rewards, np.nan)           # (1(+M), b_tr, T, A)

            valid_mask = ~np.isnan(vals_s)
            weights_s = np.where(valid_mask, combined_weights, 0.0)

            sum_sa = np.nansum(vals_s * weights_s, axis=(0, 1, 2))        # \sum(r*w)
            w_sum_sa = np.sum(weights_s, axis=(0, 1, 2))                  # \sum(w)

            r_sum[s, :] = sum_sa
            r_weight_sum[s, :] = w_sum_sa

        # Compute r_hat with the safe divide
        with np.errstate(divide='ignore', invalid='ignore'):
            r_hat = r_sum / r_weight_sum

        # ---- Vectorized fallback for unseen (s,a) ----
        unseen = (r_weight_sum == 0)   # Mask of unseen (s,a)
        if np.any(unseen):
            # Global action means across all states (where seen)
            with np.errstate(divide='ignore', invalid='ignore'):
                global_sum_a = np.nansum(r_sum, axis=0)   # (A,)
                global_w_sum_a = np.sum(r_weight_sum, axis=0)      # (A,)
                global_mean_a = global_sum_a / global_w_sum_a

            # Build fallback per action; if an action is never seen anywhere, fall back to 0.0
            fallback = np.nan_to_num(global_mean_a, nan=0.0)     # (A,)
            # Assign only where unseen
            r_hat[unseen] = fallback[np.nonzero(unseen)[1]]

        return r_hat.astype(dtype, copy=False)

    # ---- Helper: evaluate DR on a set of indices, given r_hat ----
    def _eval_dr_on_fold(idx: np.ndarray, r_hat: np.ndarray) -> None:
        for i in idx:
            states_i, actions_i, rewards_i = unpacked[i]  # each (T,)

            # Importance ratio on taken actions
            denom = np.clip(policy_b[states_i, actions_i], 1e-12, None)
            rho = policy_e[states_i, actions_i] / denom  # (T,)

            # DM baseline: sum_a pi_e(a|s) * r_hat(s,a)
            v_dm = np.sum(r_hat[states_i] * policy_e[states_i], axis=1)  # (T,)
            r_hat_taken = r_hat[states_i, actions_i]                     # (T,)

            # DR for this trajectory (average over T, bandit => T=1)
            dr_val = np.mean(v_dm + rho * (rewards_i - r_hat_taken))
            dr_per_traj[i] = dr_val

    # ---- Cross-fitting loop ----
    for k in range(n_fold):
        eval_idx = np.array(fold_indices[k], dtype=int)
        train_idx = np.setdiff1d(np.arange(n), eval_idx)
        r_hat_k = _estimate_r_hat_on_train(train_idx)
        _eval_dr_on_fold(eval_idx, r_hat_k)

    est = np.asarray(dr_per_traj, dtype=dtype)
    return est if bx != 0 else est.mean(dtype=dtype)