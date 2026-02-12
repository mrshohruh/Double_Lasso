from __future__ import annotations

"""
Coverage probability plots for Double LASSO simulation summaries.

Expected inputs: summary.csv under
  - results
  - results_cv
  - results_easierdgp
  - results_ecv (optional)
  - results_heavy (optional)

Outputs are written to plots/:
  - coverage_by_scenario.png
  - coverage_vs_rho.png
  - coverage_vs_n.png
  - coverage_vs_p.png
  - ci_length_vs_rho.png
  - ci_length_vs_n.png
  - ci_length_vs_p.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Global styling for consistent, cleaner plots
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PLOTS_DIR = Path("plots")

RESULT_FOLDERS = {
    "results": Path("results"),
    "results_cv": Path("results_cv"),
    "results_easierdgp": Path("results_easierdgp"),
    "results_ecv": Path("results_ecv"),
    "results_heavy": Path("results_heavy"),
    "results_heavycv": Path("results_heavycv"),
}

LABELS = {
    "results": "Plugin alpha (static dgp)",
    "results_cv": "Cross-validated alpha (static dgp)",
    "results_easierdgp": "Plugin alpha (easier dgp)",
    "results_ecv": "Cross-validated alpha (easier dgp)",
    "results_heavy": "Plugin alpha (heavy-tailed dgp)",
    "results_heavycv": "Cross-validated alpha (heavy-tailed dgp)",
}

# Preserve a consistent legend order across all plots
LABEL_ORDER = [
    LABELS["results"],
    LABELS["results_easierdgp"],
    LABELS["results_heavy"],
    LABELS["results_cv"],
    LABELS["results_ecv"],
    LABELS["results_heavycv"],
]

COLORS = {
    # Okabe-Ito colorblind-safe palette
    "Plugin alpha (static dgp)": "#0072B2",       # blue
    "Cross-validated alpha (static dgp)": "#D55E00",  # vermillion
    "Plugin alpha (easier dgp)": "#009E73",       # green
    "Cross-validated alpha (easier dgp)": "#CC79A7",  # reddish purple
    "Plugin alpha (heavy-tailed dgp)": "#E69F00", # orange
    "Cross-validated alpha (heavy-tailed dgp)": "#000000",  # black
}

HLINE_STYLE = dict(color="#222222", linestyle="--", linewidth=1, alpha=0.8)
LEGEND_FIG_Y = 0.02
LEGEND_BOTTOM = 0.2


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)

# Scenario metadata based on scenarios.py
SCENARIO_RENAMES = {
    "medium_corr_0_2": "n_200_p_240_corr_2",
    "medium_corr_0_0": "n_200_p_240_corr_0_0",
    "medium_corr_0_5": "n_200_p_240_corr_0_5",
    "large_corr_0_0": "n_320_p_384_corr_0_0",
    "large_corr_0_2": "n_320_p_384_corr_0_2",
    "large_corr_0_5": "n_320_p_384_corr_0_5",
}

SCENARIO_PARAMS = {
    "n_200_p_240_corr_0_0": dict(n_samples=200, n_covariates=240, covariate_correlation=0.0),
    "n_200_p_240_corr_2": dict(n_samples=200, n_covariates=240, covariate_correlation=0.2),
    "n_200_p_240_corr_0_5": dict(n_samples=200, n_covariates=240, covariate_correlation=0.5),
    "n_320_p_384_corr_0_0": dict(n_samples=320, n_covariates=384, covariate_correlation=0.0),
    "n_320_p_384_corr_0_2": dict(n_samples=320, n_covariates=384, covariate_correlation=0.2),
    "n_320_p_384_corr_0_5": dict(n_samples=320, n_covariates=384, covariate_correlation=0.5),
}

SCENARIO_ORDER = [
    "n_320_p_384_corr_0_0",
    "n_320_p_384_corr_0_2",
    "n_320_p_384_corr_0_5",
    "n_200_p_240_corr_0_0",
    "n_200_p_240_corr_2",
    "n_200_p_240_corr_0_5",
]

EXCLUDED_SCENARIOS = {
    "classical_low_dim",
    "near_p_equals_n",
    "p_equals_n",
}


def load_run_data() -> pd.DataFrame:
    """
    Load per-replication CSVs (excluding summary.csv) for CI length distributions.
    """
    rows = []
    column_aliases = {
        "treatment_effect_hat": ["treatment_effect_hat", "beta1_hat"],
        "ci_length": ["ci_length"],
        "ci_low": ["ci_low", "ci_lower", "ci_l"],
        "ci_high": ["ci_high", "ci_upper", "ci_u"],
        "n_selected_outcome_controls": ["n_selected_outcome_controls", "k_y"],
        "n_selected_treatment_controls": ["n_selected_treatment_controls", "k_d"],
        "standard_error": ["standard_error", "se", "se_HC3", "standard_error_HC3"],
        "beta_true": ["beta1_true", "treatment_effect_true", "beta_true"],
    }
    for key, folder in RESULT_FOLDERS.items():
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            if path.name == "summary.csv":
                continue
            df = pd.read_csv(path)
            scenario = SCENARIO_RENAMES.get(path.stem, path.stem)
            if scenario in EXCLUDED_SCENARIOS:
                continue
            df = df.assign(source=key, label=LABELS.get(key, key), scenario=scenario)
            selected_cols: list[str] = []
            rename_map: dict[str, str] = {}
            for canonical_name, aliases in column_aliases.items():
                for alias in aliases:
                    if alias in df.columns:
                        selected_cols.append(alias)
                        rename_map[alias] = canonical_name
                        break
            if selected_cols:
                standardized = df[selected_cols + ["label", "scenario"]].rename(columns=rename_map)
                rows.append(standardized)
    if not rows:
        return pd.DataFrame(columns=[
            "treatment_effect_hat",
            "ci_length",
            "n_selected_outcome_controls",
            "n_selected_treatment_controls",
            "label",
            "scenario",
        ])
    return pd.concat(rows, ignore_index=True)


def _ordered_labels(labels: list[str]) -> list[str]:
    """Return labels in the global LABEL_ORDER, dropping missing ones."""
    present = set(labels)
    return [lbl for lbl in LABEL_ORDER if lbl in present]


def _ordered_scenarios(scenarios: list[str]) -> list[str]:
    """Return scenarios in the global SCENARIO_ORDER, dropping missing ones."""
    present = set(scenarios)
    ordered = [sc for sc in SCENARIO_ORDER if sc in present]
    extras = sorted(sc for sc in present if sc not in ordered)
    return ordered + extras


def _ci_length_series(df: pd.DataFrame) -> pd.Series | None:
    if "ci_length" in df.columns:
        return df["ci_length"]
    if "ci_length_mean" in df.columns:
        return df["ci_length_mean"]
    if "ci_high" in df.columns and "ci_low" in df.columns:
        return df["ci_high"] - df["ci_low"]
    if "ci_high_mean" in df.columns and "ci_low_mean" in df.columns:
        return df["ci_high_mean"] - df["ci_low_mean"]
    print("Warning: CI length columns not found; skipping CI-length plot.")
    return None


def _legend_below(ax: plt.Axes, ncol: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig = ax.get_figure()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_FIG_Y),
        ncol=ncol,
    )


def _legend_below_with_title(ax: plt.Axes, title: str, ncol: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig = ax.get_figure()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        title=title,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_FIG_Y),
        ncol=ncol,
    )


def _select_shortlist(scenarios: list[str], max_items: int = 5) -> list[str]:
    present = sorted(set(scenarios) - EXCLUDED_SCENARIOS)
    shortlist = []
    lower_map = {sc: sc.lower() for sc in present}

    explicit = {"baseline", "rho_05", "heavy_tail_baseline"}
    for sc in present:
        if sc in explicit:
            shortlist.append(sc)
    if shortlist:
        return shortlist

    baseline = next((sc for sc in present if "baseline" in lower_map[sc]), None)
    if baseline:
        shortlist.append(baseline)

    rho_candidates = []
    for sc in present:
        meta = SCENARIO_PARAMS.get(sc)
        if meta and isinstance(meta.get("covariate_correlation"), (float, int)):
            rho_candidates.append((meta["covariate_correlation"], sc))
        elif "rho" in lower_map[sc] or "corr_" in lower_map[sc]:
            rho_candidates.append((0.0, sc))
    if rho_candidates:
        rho_candidates.sort()
        shortlist.append(rho_candidates[-1][1])

    heavy = next((sc for sc in present if "heavy" in lower_map[sc]), None)
    if heavy:
        shortlist.append(heavy)

    for sc in present:
        if sc not in shortlist:
            shortlist.append(sc)
        if len(shortlist) >= max_items:
            break

    return shortlist


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_")


def load_summaries() -> pd.DataFrame:
    frames = []
    for key, folder in RESULT_FOLDERS.items():
        summary_path = folder / "summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        rename_map = {
            "beta1_hat_mean": "treatment_effect_hat_mean",
            "k_y_mean": "n_selected_outcome_controls_mean",
            "k_d_mean": "n_selected_treatment_controls_mean",
            "alpha_y_mean": "outcome_lasso_penalty_mean",
            "alpha_d_mean": "treatment_lasso_penalty_mean",
            "se_HC3_mean": "standard_error_HC3_mean",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["scenario"] = df["scenario"].replace(SCENARIO_RENAMES)
        df = df[~df["scenario"].isin(EXCLUDED_SCENARIOS)]
        df["source"] = key
        df["label"] = LABELS.get(key, key)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No summary.csv files found in configured result folders.")
    out = pd.concat(frames, ignore_index=True)
    # Backfill missing scenario metadata with NaNs to avoid shape errors
    scenario_meta = out["scenario"].map(SCENARIO_PARAMS).apply(
        lambda meta: meta if isinstance(meta, dict) else {"n_samples": np.nan, "n_covariates": np.nan, "covariate_correlation": np.nan}
    )
    out[["n_samples", "n_covariates", "covariate_correlation"]] = scenario_meta.apply(pd.Series)
    return out


def plot_coverage_by_scenario(df: pd.DataFrame, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values="coverage")
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(15, 7), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.axhline(0.95, **HLINE_STYLE)
    ax.set_xlabel("Scenario", labelpad=2)
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage across scenarios")
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    _legend_below_with_title(ax, title="Configuration", ncol=2)
    _style_ax(ax)
    ax.get_figure().tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def plot_metric_by_scenario(df: pd.DataFrame, metric: str, ylabel: str, title: str, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values=metric)
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(15, 7), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.set_xlabel("Scenario", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    _legend_below_with_title(ax, title="Configuration", ncol=2)
    _style_ax(ax)
    ax.get_figure().tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def line_plot(df: pd.DataFrame, x: str, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for label in _ordered_labels(df["label"].unique().tolist()):
        group = df[df["label"] == label]
        sorted_group = group.sort_values(x)
        ax.plot(sorted_group[x], sorted_group["coverage"], marker="o",
                label=label, color=COLORS.get(label, None))
    ax.axhline(0.95, **HLINE_STYLE)
    ax.set_xlabel(x)
    ax.set_ylabel("Coverage")
    ax.set_title(title)
    _legend_below(ax, ncol=2)
    _style_ax(ax)
    fig.tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def line_plot_metric(df: pd.DataFrame, x: str, y: str, ylabel: str, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for label in _ordered_labels(df["label"].unique().tolist()):
        group = df[df["label"] == label]
        sorted_group = group.sort_values(x)
        ax.plot(sorted_group[x], sorted_group[y], marker="o", label=label, color=COLORS.get(label, None))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, ncol=2)
    _style_ax(ax)
    fig.tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_coverage_vs_rho(df: pd.DataFrame, outfile: Path) -> None:
    # Use all available rho values; average over n/p within each label-rho
    agg = df.groupby(["label", "covariate_correlation"], as_index=False)["coverage"].mean()
    line_plot(agg, x="covariate_correlation", title="Coverage vs correlation (rho)", outfile=outfile)


def plot_coverage_vs_n(df: pd.DataFrame, outfile: Path) -> None:
    # Focus on rho=0.2 to align across sizes
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_samples"], as_index=False)["coverage"].mean()
    line_plot(agg, x="n_samples", title="Coverage vs sample size (rho=0.2)", outfile=outfile)


def plot_coverage_vs_p(df: pd.DataFrame, outfile: Path) -> None:
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_covariates"], as_index=False)["coverage"].mean()
    line_plot(agg, x="n_covariates", title="Coverage vs dimensionality (rho=0.2)", outfile=outfile)


def plot_ci_length_vs_rho(df: pd.DataFrame, outfile: Path) -> None:
    ci_series = _ci_length_series(df)
    if ci_series is None:
        return
    df = df.assign(ci_length_value=ci_series)
    agg = df.groupby(["label", "covariate_correlation"], as_index=False)["ci_length_value"].mean()
    if agg.empty:
        print("Warning: No data available for CI length vs rho plot.")
        return
    line_plot_metric(
        agg,
        x="covariate_correlation",
        y="ci_length_value",
        ylabel="Average CI length",
        title="CI length vs correlation (rho)",
        outfile=outfile,
    )


def plot_ci_length_vs_n(df: pd.DataFrame, outfile: Path) -> None:
    ci_series = _ci_length_series(df)
    if ci_series is None:
        return
    df = df.assign(ci_length_value=ci_series)
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_samples"], as_index=False)["ci_length_value"].mean()
    if agg.empty:
        print("Warning: No data available for CI length vs n plot.")
        return
    line_plot_metric(
        agg,
        x="n_samples",
        y="ci_length_value",
        ylabel="Average CI length",
        title="CI length vs sample size (rho=0.2)",
        outfile=outfile,
    )


def plot_ci_length_vs_p(df: pd.DataFrame, outfile: Path) -> None:
    ci_series = _ci_length_series(df)
    if ci_series is None:
        return
    df = df.assign(ci_length_value=ci_series)
    subset = df[df["covariate_correlation"] == 0.2]
    agg = subset.groupby(["label", "n_covariates"], as_index=False)["ci_length_value"].mean()
    if agg.empty:
        print("Warning: No data available for CI length vs p plot.")
        return
    line_plot_metric(
        agg,
        x="n_covariates",
        y="ci_length_value",
        ylabel="Average CI length",
        title="CI length vs dimensionality (rho=0.2)",
        outfile=outfile,
    )


def plot_ci_length_by_scenario(df: pd.DataFrame, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    pivot = df.pivot(index="scenario", columns="label", values="ci_length_mean")
    pivot = pivot.loc[order]
    pivot = pivot[_ordered_labels(pivot.columns.tolist())]
    ax = pivot.plot(kind="bar", figsize=(15, 7), color=[COLORS.get(c, None) for c in pivot.columns])
    ax.set_xlabel("Scenario", labelpad=2)
    ax.set_ylabel("Avg CI length")
    ax.set_title("Average CI length across scenarios")
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    _legend_below_with_title(ax, title="Configuration", ncol=2)
    _style_ax(ax)
    ax.get_figure().tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    ax.get_figure().savefig(outfile, dpi=300)
    plt.close(ax.get_figure())


def plot_beta_hat_distribution(
    df_runs: pd.DataFrame,
    outdir: Path,
    beta_true: float = 2.0,
) -> None:
    """
    Line-density plot of treatment_effect_hat with Gaussian overlay and true beta line, one file per scenario.
    """
    if df_runs.empty or "treatment_effect_hat" not in df_runs.columns:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in _ordered_scenarios(df_runs["scenario"].unique().tolist()):
        subset = df_runs[df_runs["scenario"] == scenario]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(16, 9))
        all_vals = subset["treatment_effect_hat"].dropna()
        if all_vals.empty:
            plt.close(fig)
            continue
        x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
        for label in _ordered_labels(subset["label"].unique().tolist()):
            group = subset[subset["label"] == label]
            vals = group["treatment_effect_hat"].dropna()
            if vals.size < 2:
                continue
            color = COLORS.get(label, None)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=f"{label} KDE")
        ax.axvline(beta_true, **HLINE_STYLE, label="True treatment effect")
        ax.set_title(f"Sampling distribution of treatment_effect_hat: {scenario}")
        ax.set_xlabel("treatment_effect_hat")
        ax.set_ylabel("Density")
        _legend_below(ax, ncol=2)
        _style_ax(ax)
        fig.tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
        fig.savefig(outdir / f"treatment_effect_hat_distribution_{scenario}.png", dpi=300)
        plt.close(fig)


def plot_zscore_calibration(df_runs: pd.DataFrame, outdir: Path) -> None:
    if df_runs.empty:
        return
    required_cols = {"treatment_effect_hat", "standard_error"}
    if not required_cols.issubset(df_runs.columns):
        print("Warning: Missing treatment effect or standard error columns; skipping z-score plots.")
        return

    beta_true_series = None
    if "beta_true" in df_runs.columns and df_runs["beta_true"].notna().any():
        beta_true_series = df_runs["beta_true"]
    else:
        print("Warning: beta_true missing; using 2.0 fallback for z-score plots.")
        beta_true_series = pd.Series(2.0, index=df_runs.index)

    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in _ordered_scenarios(df_runs["scenario"].unique().tolist()):
        subset = df_runs[df_runs["scenario"] == scenario].copy()
        if subset.empty:
            continue
        subset = subset.assign(beta_true=beta_true_series.loc[subset.index])
        subset = subset[subset["standard_error"] > 0]
        if subset.empty:
            print(f"Warning: No valid standard errors for z-score plot in {scenario}.")
            continue

        fig, ax = plt.subplots(figsize=(15, 7))
        for label in _ordered_labels(subset["label"].unique().tolist()):
            group = subset[subset["label"] == label]
            if group.empty:
                continue
            zscores = (group["treatment_effect_hat"] - group["beta_true"]) / group["standard_error"]
            if zscores.dropna().empty:
                continue
            ax.hist(zscores.dropna(), bins=30, alpha=0.35, label=label)

        ax.set_xlabel("Standardized error (beta_hat - beta_true) / SE")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Z-score calibration: {scenario}")
        _legend_below(ax, ncol=2)
        _style_ax(ax)
        fig.tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
        fig.savefig(outdir / f"zscore_distribution_{_sanitize_filename(scenario)}.png", dpi=300)
        plt.close(fig)


def plot_k_line(df: pd.DataFrame, metric: str, ylabel: str, title: str, outfile: Path) -> None:
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(15, 7))
    for label in _ordered_labels(df["label"].unique().tolist()):
        group = df[df["label"] == label]
        ordered_vals = [group[group["scenario"] == sc][metric].values[0] for sc in order if sc in set(group["scenario"])]
        ax.plot(x[:len(ordered_vals)], ordered_vals, marker="o", label=label, color=COLORS.get(label, None))
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=30)
    ax.set_xlabel("Scenario", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_below(ax, ncol=2)
    _style_ax(ax)
    fig.tight_layout(rect=[0, LEGEND_BOTTOM, 1, 1])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_summary_dashboard(df: pd.DataFrame, outfile: Path) -> None:
    """
    Combined dashboard: coverage, bias, CI length, and selected-controls counts across scenarios.
    """
    order = _ordered_scenarios(df["scenario"].unique().tolist())
    metrics = [
        ("coverage", "Coverage"),
        ("bias", "Bias"),
        ("ci_length_mean", "Avg CI length"),
        ("n_selected_outcome_controls_mean", "Avg outcome controls"),
        ("n_selected_treatment_controls_mean", "Avg treatment controls"),
    ]
    n_axes = len(metrics)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 5))
    axes_flat = axes.flatten()

    for ax, (metric, title) in zip(axes_flat, metrics):
        pivot = df.pivot(index="scenario", columns="label", values=metric)
        if pivot.empty:
            ax.axis("off")
            continue
        pivot = pivot.loc[order]
        pivot = pivot[_ordered_labels(pivot.columns.tolist())]
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, None) for c in pivot.columns])
        if metric == "coverage":
            ax.axhline(0.95, **HLINE_STYLE)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(metric)
        if ax is axes_flat[0]:
            ax.legend(title="Configuration", frameon=False, fontsize=8)
        else:
            ax.legend().remove()
        _style_ax(ax)

    # Hide any unused axes
    for ax in axes_flat[n_axes:]:
        ax.axis("off")

    fig.suptitle("Double LASSO performance summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_overall_summary(df: pd.DataFrame, outfile: Path) -> None:
    """
    Aggregate performance across all scenarios (per configuration).
    """
    metrics = [
        ("coverage", "Coverage"),
        ("bias", "Bias"),
        ("rmse", "RMSE"),
        ("ci_length_mean", "Avg CI length"),
        ("n_selected_outcome_controls_mean", "Avg outcome controls"),
        ("n_selected_treatment_controls_mean", "Avg treatment controls"),
    ]
    grouped = df.groupby("label").agg({m[0]: "mean" for m in metrics}).reset_index()
    grouped["label"] = pd.Categorical(grouped["label"], categories=LABEL_ORDER, ordered=True)
    grouped = grouped.sort_values("label")
    n_axes = len(metrics)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 5))
    axes_flat = axes.flatten()

    for ax, (metric, title) in zip(axes_flat, metrics):
        subset = grouped[["label", metric]].set_index("label")
        subset.plot(kind="bar", ax=ax, legend=False, color=[COLORS.get(lbl, "#888888") for lbl in subset.index])
        if metric == "coverage":
            ax.axhline(0.95, **HLINE_STYLE)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.set_ylabel(metric)
        _style_ax(ax)

    for ax in axes_flat[n_axes:]:
        ax.axis("off")

    fig.suptitle("Overall performance summary (averaged across scenarios)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries()
    run_data = load_run_data()

    plot_ci_length_vs_rho(summaries, PLOTS_DIR / "ci_length_vs_rho.png")
    plot_ci_length_vs_n(summaries, PLOTS_DIR / "ci_length_vs_n.png")
    plot_ci_length_vs_p(summaries, PLOTS_DIR / "ci_length_vs_p.png")
    plot_zscore_calibration(run_data, PLOTS_DIR)
    plot_coverage_by_scenario(summaries, PLOTS_DIR / "coverage_by_scenario.png")
    plot_metric_by_scenario(
        summaries,
        metric="bias",
        ylabel="Bias",
        title="Bias across scenarios",
        outfile=PLOTS_DIR / "bias_by_scenario.png",
    )
    plot_metric_by_scenario(
        summaries,
        metric="rmse",
        ylabel="RMSE",
        title="RMSE across scenarios",
        outfile=PLOTS_DIR / "rmse_by_scenario.png",
    )
    plot_coverage_vs_rho(summaries, PLOTS_DIR / "coverage_vs_rho.png")
    plot_coverage_vs_n(summaries, PLOTS_DIR / "coverage_vs_n.png")
    plot_coverage_vs_p(summaries, PLOTS_DIR / "coverage_vs_p.png")
    plot_ci_length_by_scenario(summaries, PLOTS_DIR / "ci_length_by_scenario.png")
    plot_beta_hat_distribution(run_data, PLOTS_DIR)
    plot_k_line(
        summaries,
        metric="n_selected_outcome_controls_mean",
        ylabel="Average selected outcome controls",
        title="Selected controls (outcome model) across scenarios",
        outfile=PLOTS_DIR / "n_selected_outcome_controls_line.png",
    )
    plot_k_line(
        summaries,
        metric="n_selected_treatment_controls_mean",
        ylabel="Average selected treatment controls",
        title="Selected controls (treatment model) across scenarios",
        outfile=PLOTS_DIR / "n_selected_treatment_controls_line.png",
    )
    plot_summary_dashboard(summaries, PLOTS_DIR / "summary_dashboard.png")
    plot_overall_summary(summaries, PLOTS_DIR / "summary_overall.png")


if __name__ == "__main__":
    main()
