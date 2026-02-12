from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_RENAMES = {
    "medium_corr_0_2": "n_200_p_240_corr_2",
    "medium_corr_0_0": "n_200_p_240_corr_0_0",
    "medium_corr_0_5": "n_200_p_240_corr_0_5",
    "large_corr_0_0": "n_320_p_384_corr_0_0",
    "large_corr_0_2": "n_320_p_384_corr_0_2",
    "large_corr_0_5": "n_320_p_384_corr_0_5",
}


def load_summary(folder: Path) -> pd.DataFrame:
    path = folder / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found in {folder}")
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].replace(SCENARIO_RENAMES)
    return df


def plot_metric(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    scenario_labels = df["scenario"].astype(str)
    width = 0.35
    x = range(len(scenario_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df[f"{metric}_lasso"], width, label="Double LASSO")
    ax.bar([i + width / 2 for i in x], df[f"{metric}_ols"], width, label="OLS")

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario_labels, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    outpath = outdir / f"{metric}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def load_runs(folder: Path, label: str) -> pd.DataFrame:
    rows = []
    for path in folder.glob("*.csv"):
        if path.name == "summary.csv":
            continue
        df = pd.read_csv(path)
        if "treatment_effect_hat" in df:
            vals = df["treatment_effect_hat"]
        elif "beta1_hat" in df:
            vals = df["beta1_hat"]
        else:
            continue
        scenario = SCENARIO_RENAMES.get(path.stem, path.stem)
        rows.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "label": label,
                    "treatment_effect_hat": vals,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["scenario", "label", "treatment_effect_hat"])
    return pd.concat(rows, ignore_index=True)


def plot_treatment_effect_distributions(
    df_runs: pd.DataFrame,
    outdir: Path,
    scenario_order: list[str],
    scenario_labels: dict[str, str],
) -> None:
    if df_runs.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in scenario_order:
        subset = df_runs[df_runs["scenario"] == scenario]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        lasso_vals = subset[subset["label"] == "Double LASSO"]["treatment_effect_hat"].dropna()
        ols_vals = subset[subset["label"] == "OLS"]["treatment_effect_hat"].dropna()
        if lasso_vals.empty or ols_vals.empty:
            plt.close(fig)
            continue
        all_vals = pd.concat([lasso_vals, ols_vals], ignore_index=True)
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
        if lasso_vals.size >= 2:
            lasso_kde = gaussian_kde(lasso_vals)
            ax.plot(x_grid, lasso_kde(x_grid), linewidth=2, label="Double LASSO")
        if ols_vals.size >= 2:
            ols_kde = gaussian_kde(ols_vals)
            ax.plot(x_grid, ols_kde(x_grid), linewidth=2, label="OLS")
        ax.set_title(
            f"Treatment effect hat distribution: {scenario_labels.get(scenario, scenario)}"
        )
        ax.set_xlabel("treatment_effect_hat")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / f"treatment_effect_hat_distribution_{scenario}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare results between Double LASSO and OLS summaries.")
    parser.add_argument("--lasso_dir", type=Path, default=Path("results"))
    parser.add_argument("--ols_dir", type=Path, default=Path("results_ols"))
    parser.add_argument("--outdir", type=Path, default=Path("cplots"))
    args = parser.parse_args()

    lasso_df = load_summary(args.lasso_dir)
    ols_df = load_summary(args.ols_dir)

    merged = pd.merge(
        lasso_df,
        ols_df,
        on="scenario",
        suffixes=("_lasso", "_ols"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping scenarios between the two summary files.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "coverage",
        "bias",
        "rmse",
        "ci_length_mean",
        "treatment_effect_hat_mean",
        "standard_error_HC3_mean",
    ]
    scenario_order = [
        "classical_low_dim",
        "near_p_equals_n",
        "p_equals_n",
        "n_200_p_240_corr_0_0",
        "n_200_p_240_corr_2",
        "n_200_p_240_corr_0_5",
        "n_320_p_384_corr_0_0",
        "n_320_p_384_corr_0_2",
        "n_320_p_384_corr_0_5",
    ]
    scenario_labels = {
        "classical_low_dim": "Classical low dim",
        "near_p_equals_n": "near p equals n",
        "p_equals_n": "p equals n",
        "n_200_p_240_corr_0_0": "n=200, p=240, corr=0.0",
        "n_200_p_240_corr_2": "n=200, p=240, corr=0.2",
        "n_200_p_240_corr_0_5": "n=200, p=240, corr=0.5",
        "n_320_p_384_corr_0_0": "n=320, p=384, corr=0.0",
        "n_320_p_384_corr_0_2": "n=320, p=384, corr=0.2",
        "n_320_p_384_corr_0_5": "n=320, p=384, corr=0.5",
    }
    merged["scenario"] = pd.Categorical(
        merged["scenario"],
        categories=scenario_order,
        ordered=True,
    )
    merged = merged.sort_values("scenario")
    merged["scenario"] = merged["scenario"].map(scenario_labels)
    for metric in metrics:
        if f"{metric}_lasso" in merged and f"{metric}_ols" in merged:
            plot_metric(merged[["scenario", f"{metric}_lasso", f"{metric}_ols"]], metric, args.outdir)

    lasso_runs = load_runs(args.lasso_dir, "Double LASSO")
    ols_runs = load_runs(args.ols_dir, "OLS")
    runs = pd.concat([lasso_runs, ols_runs], ignore_index=True)
    plot_treatment_effect_distributions(runs, args.outdir, scenario_order, scenario_labels)

    print(f"Saved comparison plots to {args.outdir}")


if __name__ == "__main__":
    main()
