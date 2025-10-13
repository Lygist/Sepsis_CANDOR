import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from two_state import *
from trajectory_classes import *
import shutil


MASTER_SEED = 20250928
RNG_MASTER = np.random.default_rng(MASTER_SEED)

NUM_DATASETS = 100
TRAJECTORIES_PER_DATASET = 100
BASE_OUTPUT_DIR = "figures"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def run_all_experiments():
    """
    Main function to run the two sets of analyses:
    1. Fixed Cost Ratio: Iterate through different ratios, plotting RMSE vs. Budget.
    2. Fixed Budget: Iterate through different budgets, plotting RMSE vs. Cost Ratio.
    """

    # --- Experiment Parameters ---
    state_distribution = np.array([0.5, 0.5])
    true_reward_means = np.array([
        [1., 1.5],
        [0., 0.],
    ])
    true_reward_stds = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    behavior_policy = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    evaluation_policy = np.array([
        [0.4, 0.6],
        [0.5, 0.5],
    ])
    doctor_bias = np.array([
        [0.15, 0.15],
        [0.15, 0.15],
    ])
    doctor_std = np.array([
        [0.9, 0.9],
        [0.9, 0.9],
    ])
    llm_bias = np.array([
        [0.3, 0.3],
        [0.3, 0.3],
    ])
    llm_std = np.array([
        [0.7, 0.7],
        [0.7, 0.7],
    ])

    budgets_to_test = [100, 200, 300, 400, 500]
    cost_ratios_to_test = [5, 10, 15, 20, 25]
    expert_allocations = [0, 20, 40, 60, 80, 100]  # percentages

    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, true_reward_means)

    print("Pre-generating factual datasets...")
    factual_datasets = []
    for _ in tqdm(range(NUM_DATASETS)):
        rng = np.random.default_rng(RNG_MASTER.integers(0, 2 ** 31 - 1))
        factual_datasets.append(generate_dataset_of_trajectories(
            state_distribution, true_reward_means, true_reward_stds,
            behavior_policy, rng=rng, num_trajectories=TRAJECTORIES_PER_DATASET))

    common_params = {
        "expert_allocations": expert_allocations,
        "state_distribution": state_distribution,
        "true_reward_means": true_reward_means,
        "true_reward_stds": true_reward_stds,
        "behavior_policy": behavior_policy,
        "evaluation_policy": evaluation_policy,
        "doctor_bias": doctor_bias, "doctor_std": doctor_std,
        "llm_bias": llm_bias, "llm_std": llm_std,
        "true_evaluation_policy_value": true_evaluation_policy_value,
        "factual_datasets": factual_datasets
    }

    print("\nGenerating baseline IS estimates...")
    is_estimates = []
    for factual_dataset in tqdm(factual_datasets):
        is_estimates.append(run_vanilla_is(
            evaluation_policy, behavior_policy, factual_dataset))

    is_estimates = np.array(is_estimates)
    is_rmse, is_error = calculate_policy_value_rmse_and_ci(is_estimates, true_evaluation_policy_value)

    # --- Analysis 1: Fixed Cost Ratio ---
    print("\n--- Starting Analysis 1: Fixed Cost Ratio ---")
    for ratio in cost_ratios_to_test:
        output_dir = os.path.join(BASE_OUTPUT_DIR, "fix_ratio", f"ratio={ratio}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating graphs for Cost Ratio = {ratio}...")

        rmse_data, error_data = rmse_vs_budget(
            budgets=budgets_to_test,
            expert_cost=ratio,
            llm_cost=1,
            **common_params)

        plot_results_with_errors(
            rmse_data, error_data,
            budgets_to_test, expert_allocations,
            is_rmse, is_error,
            f"RMSE vs. Budget (Cost Ratio = {ratio})",
            "Budget", output_dir, "budget_analysis")

    # --- Analysis 2: Fixed Budget ---
    print("\n--- Starting Analysis 2: Fixed Budget ---")
    for budget in budgets_to_test:
        output_dir = os.path.join(BASE_OUTPUT_DIR, "fix_budget", f"budget_{budget}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating graphs for Budget = {budget}...")

        rmse_data, error_data = rmse_vs_cost_ratio(
            cost_ratios=cost_ratios_to_test,
            fixed_budget=budget,
            **common_params)

        plot_results_with_errors(
            rmse_data, error_data,
            cost_ratios_to_test, expert_allocations,
            is_rmse, is_error,
            f"RMSE vs. Cost Ratio (Budget = {budget})",
            "Expert:Predictive Model Cost Ratio", output_dir, "ratio_analysis")

    print("\nAll experiments complete!")


def rmse_vs_budget(budgets, expert_cost, llm_cost, expert_allocations,
                   state_distribution, true_reward_means, true_reward_stds,
                   behavior_policy, evaluation_policy,
                   doctor_bias, doctor_std, llm_bias, llm_std,
                   true_evaluation_policy_value, factual_datasets):
    """
    Generate RMSE data for different budgets with a fixed cost ratio.
    Includes the new annotation capping logic.
    """
    max_annotations = 2 * TRAJECTORIES_PER_DATASET  # Max annotations is 2x factual data points

    value_estimates = {
        'C-IS': {
            allocation: {budget: [] for budget in budgets}
            for allocation in expert_allocations},
        'CANDOR': {
            allocation: {budget: [] for budget in budgets}
            for allocation in expert_allocations},
    }

    for factual_dataset in tqdm(factual_datasets, desc=f"Budget Analysis (Ratio={expert_cost})"):
        for budget in budgets:
            for expert_percent in expert_allocations:
                expert_spend = (expert_percent / 100) * budget
                llm_spend = budget - expert_spend

                # Calculate the number of annotations with the new cap
                num_expert = int(expert_spend / expert_cost)
                num_llm = int(llm_spend / llm_cost)

                # Prioritize expert annotations
                num_expert_annotations = min(num_expert, max_annotations)
                remaining_capacity = max_annotations - num_expert_annotations
                num_llm_annotations = min(num_llm, remaining_capacity)

                doctor_annotations = generate_annotations(
                    factual_dataset, num_expert_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)

                value_estimates["C-IS"][expert_percent][budget].append(
                    run_is_plus(evaluation_policy, behavior_policy, factual_dataset,
                               [doctor_annotations, llm_annotations]))
                value_estimates["CANDOR"][expert_percent][budget].append(
                    run_dr(evaluation_policy, behavior_policy, factual_dataset,
                                  [doctor_annotations, llm_annotations]))

    results = {
        'C-IS': {allocation: [] for allocation in expert_allocations},
        'CANDOR': {allocation: [] for allocation in expert_allocations},
    }
    errors = {
        'C-IS': {allocation: [] for allocation in expert_allocations},
        'CANDOR': {allocation: [] for allocation in expert_allocations},
    }

    for allocation in expert_allocations:
        for budget in budgets:
            for method in ['C-IS', 'CANDOR']:
                estimates = value_estimates[method][allocation][budget]
                results[method][allocation].append(
                    calculate_policy_value_rmse(estimates, true_evaluation_policy_value))
                errors[method][allocation].append(
                    calculate_error_bounds(estimates, true_evaluation_policy_value))
    return results, errors


def rmse_vs_cost_ratio(cost_ratios, expert_allocations, fixed_budget,
                       state_distribution, true_reward_means, true_reward_stds,
                       behavior_policy, evaluation_policy,
                       doctor_bias, doctor_std, llm_bias, llm_std,
                       true_evaluation_policy_value, factual_datasets):
    """
    Generate RMSE data for different cost ratios with a fixed budget.
    Includes the new annotation capping logic.
    """
    max_annotations = 2 * TRAJECTORIES_PER_DATASET  # Max annotations is 2x factual data points

    value_estimates = {
        'C-IS': {
            allocation: {ratio: [] for ratio in cost_ratios}
            for allocation in expert_allocations},
        'CANDOR': {
            allocation: {ratio: [] for ratio in cost_ratios}
            for allocation in expert_allocations},
    }

    for factual_dataset in tqdm(factual_datasets, desc=f"Cost Ratio Analysis (Budget={fixed_budget})"):
        for ratio in cost_ratios:
            llm_cost = 1
            expert_cost = llm_cost * ratio

            for expert_percent in expert_allocations:
                expert_spend = (expert_percent / 100) * fixed_budget
                llm_spend = fixed_budget - expert_spend

                # Calculate the number of annotations with the new cap
                num_expert = int(expert_spend / expert_cost)
                num_llm = int(llm_spend / llm_cost)

                # Prioritize expert annotations
                num_expert_annotations = min(num_expert, max_annotations)
                remaining_capacity = max_annotations - num_expert_annotations
                num_llm_annotations = min(num_llm, remaining_capacity)

                doctor_annotations = generate_annotations(
                    factual_dataset, num_expert_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)

                value_estimates["C-IS"][expert_percent][ratio].append(
                    run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                               [doctor_annotations, llm_annotations]))
                value_estimates["CANDOR"][expert_percent][ratio].append(
                    run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                                  [doctor_annotations, llm_annotations]))

    results = {
        'C-IS': {allocation: [] for allocation in expert_allocations},
        'CANDOR': {allocation: [] for allocation in expert_allocations},
    }
    errors = {
        'C-IS': {allocation: [] for allocation in expert_allocations},
        'CANDOR': {allocation: [] for allocation in expert_allocations},
    }

    for allocation in expert_allocations:
        for ratio in cost_ratios:
            for method in ['C-IS', 'CANDOR']:
                estimates = value_estimates[method][allocation][ratio]
                results[method][allocation].append(
                    calculate_policy_value_rmse(estimates, true_evaluation_policy_value))
                errors[method][allocation].append(
                    calculate_error_bounds(estimates, true_evaluation_policy_value))
    return results, errors


def plot_results_with_errors(results, errors, x_values, expert_allocations,
                             baseline_IS_rmse, baseline_IS_error,
                             title_base, x_label, output_dir, file_prefix):
    """
    Create the three types of plots for each figure with error bars and save them
    to the specified directory.
    """
    all_rmses = [baseline_IS_rmse]
    for method in ['C-IS', 'CANDOR']:
        for allocation in expert_allocations:
            all_rmses.extend(results[method][allocation])

    base_min_rmse = min(all_rmses) * 0.95
    base_max_rmse = max(all_rmses) * 1.05

    dmplus_rmses = []
    for allocation in expert_allocations:
        dmplus_rmses.extend(results['CANDOR'][allocation])

    dmplus_min = min(dmplus_rmses) if dmplus_rmses else 0
    dmplus_max = max(dmplus_rmses) if dmplus_rmses else 1
    dmplus_min_with_buffer = dmplus_min - (dmplus_max - dmplus_min) * 0.05
    dmplus_max_with_buffer = dmplus_max + (dmplus_max - dmplus_min) * 0.05

    plt.rcParams.update({
        'font.size': 24, 'axes.labelsize': 28, 'axes.titlesize': 30,
        'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 22,
        'figure.titlesize': 32
    })
    colors = plt.cm.viridis(np.linspace(0, 1, len(expert_allocations)))

    # Plot 1: Combined
    plt.figure(figsize=(16, 12))
    plt.ylim(base_min_rmse, base_max_rmse)
    for i, allocation in enumerate(expert_allocations):
        plt.errorbar(x_values, results['C-IS'][allocation],
                     fmt='o-', color=colors[i], linewidth=3, markersize=10,
                     label=f'C-IS @ {allocation}% Expert')
        plt.errorbar(x_values, results['CANDOR'][allocation],
                     fmt='s--', color=colors[i], linewidth=3, markersize=10,
                     label=f'CANDOR @ {allocation}% Expert')

    plt.axhline(y=baseline_IS_rmse, color='black', linestyle='-.', linewidth=3, label='Ordinary IS')
    mid_x = x_values[len(x_values) // 2]

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.title(title_base)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_combined.png"), dpi=300)
    plt.close()

    # Plot 2: IS+
    plt.figure(figsize=(14, 10))
    plt.ylim(base_min_rmse, base_max_rmse)
    for i, allocation in enumerate(expert_allocations):
        plt.errorbar(x_values, results['C-IS'][allocation],
                     fmt='o-', color=colors[i], linewidth=3, markersize=10,
                     label=f'{allocation}% Expert')

    plt.axhline(y=baseline_IS_rmse, color='gray', linestyle='--', linewidth=3, label='Ordinary IS')

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.title(f"C-IS| {title_base}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_C-IS.png"), dpi=300)
    plt.close()

    # Plot 3: DM+-IS
    plt.figure(figsize=(14, 10))
    plt.ylim(dmplus_min_with_buffer, dmplus_max_with_buffer)
    for i, allocation in enumerate(expert_allocations):
        plt.errorbar(x_values, results['CANDOR'][allocation],
                     fmt='s-', color=colors[i], linewidth=3, markersize=10,
                     label=f'{allocation}% Expert')

    # DM-IS baseline (0% allocation at budget 0 or ratio 10)
    if 0 in expert_allocations and results['CANDOR'][0]:
        dmis_baseline = results['CANDOR'][0][0]
        plt.axhline(y=dmis_baseline, color='gray', linestyle='--', linewidth=3, label='DR')

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.title(f"CANDOR | {title_base}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_candor.png"), dpi=300)
    plt.close()

    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == "__main__":
    run_all_experiments()
