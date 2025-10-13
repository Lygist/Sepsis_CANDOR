import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def _ylim_with_padding(
    y: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]]
) -> Tuple[float, float]:
    y_min = np.nanmin(y) if y.size else np.inf
    y_max = np.nanmax(y) if y.size else -np.inf
    bval, _ = baseline
    if bval is not None:
        y_min = min(y_min, bval)
        y_max = max(y_max, bval)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0
    pad = 0.05 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _plot_one(
    *,
    x_values: list[float],
    allocations: list[float],
    rmse: np.ndarray,              # (n_alloc, n_sweep)
    baseline_label: str,           # "IS" or "DR"
    baseline_value: Optional[float],
    title: str,
    x_label: str,
    out_png: str,
) -> None:

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(allocations)))
    marker_map = {0: "o", 20: "s", 40: "D", 60: "^", 80: "v", 100: "P"}
    y_low, y_high = _ylim_with_padding(rmse, (baseline_value, None))

    plt.figure(figsize=(14, 10))
    for i, alloc in enumerate(allocations):
        marker = marker_map.get(int(alloc), "o")
        plt.plot(
            x_values, rmse[i, :],
            linestyle="-", marker=marker,
            linewidth=3, markersize=8,
            color=colors[i],
            label=f"{alloc}%",
        )

    if baseline_value is not None:
        plt.axhline(y=baseline_value, color="gray", linestyle="--", linewidth=2.5, label=f"{baseline_label}")

    plt.ylim(y_low, y_high)
    plt.xlabel(x_label, fontweight="bold", fontsize=16)
    plt.ylabel("RMSE", fontweight="bold", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(x_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_budget_figures(json_path: str, outdir: str = "figures") -> None:
    with open(json_path, "r") as f:
        d = json.load(f)

    # baselines (flat across x)
    is_base_val = float(d["IS"]["rmse"]) if "IS" in d else None
    is_base_half = float(d["IS"]["half_width"]) if "IS" in d else None
    dr_base_val = float(d["DR"]["rmse"]) if "DR" in d else None
    dr_base_half = float(d["DR"]["half_width"]) if "DR" in d else None

    # a generic renderer over (summary_key, axis_label, title suffix, filename stem, baseline mapping)
    jobs = [
        # budget sweep
        ("SUMMARY_BUDGET", "Budget",
         lambda dd: f' @ cost ratio = {dd.get("FIXED_RATIO")}' if dd.get("FIXED_RATIO") is not None else "",
         "rmse_vs_budget",
         {"C-IS": ("IS", is_base_val, is_base_half), "CANDOR": ("DR", dr_base_val, dr_base_half)}),
        # ratio sweep
        ("SUMMARY_RATIO", "Cost Ratio",
         lambda dd: f' @ budget = {dd.get("FIXED_BUDGET")}' if dd.get("FIXED_BUDGET") is not None else "",
         "rmse_vs_ratio",
         {"C-IS": ("IS", is_base_val, is_base_half), "CANDOR": ("DR", dr_base_val, dr_base_half)}),
    ]

    for summary_key, x_label, title_suffix_fn, stem, baseline_map in jobs:
        s = d[summary_key]
        x_values = s["sweep_values"]
        allocations = s["allocations"]
        # arrays: (n_alloc, n_sweep)
        rmse_isplus = np.array(s["rmse"]["C-IS"], dtype=float)
        half_isplus = np.array(s["half_width"]["C-IS"], dtype=float)
        rmse_candor = np.array(s["rmse"]["CANDOR"], dtype=float)
        half_candor = np.array(s["half_width"]["CANDOR"], dtype=float)

        # loop over methods with their baselines
        for method_label, rmse_arr, half_arr in [
            ("C-IS", rmse_isplus, half_isplus),
            ("CANDOR", rmse_candor, half_candor),
        ]:
            base_label, base_val, base_half = baseline_map[method_label]
            title = f"RMSE vs {x_label} ({method_label})" + title_suffix_fn(d)
            out_png = os.path.join(outdir, f"{stem}_{'isplus' if method_label=='C-IS' else 'candor'}.png")

            _plot_one(
                x_values=x_values,
                allocations=allocations,
                rmse=rmse_arr,
                baseline_label=base_label,
                baseline_value=base_val,
                title=title,
                x_label=x_label,
                out_png=out_png,
            )


if __name__ == "__main__":
    plot_budget_figures("results/budget.json", "results")
