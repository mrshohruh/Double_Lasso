from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys
import pandas as pd
import numpy as np

from dgps.static import simulate_dgp as static_dgp
from dgps.static_easier import simulate_dgp as static_easier_dgp
from dgps.heavy_tail_dgp import simulate_dgp as heavy_tail_dgp
from estimators.lasso import double_lasso_ci
from estimators.ols import ols_ci
from runner import run_simulation
from scenarios import get_scenarios, SimulationScenario

DGP_MAP = {
    "static": static_dgp,
    "static_easier": static_easier_dgp,
    "heavy_tail": heavy_tail_dgp,
}

ESTIMATOR_MAP = {
    "double_lasso": double_lasso_ci,
    "ols": ols_ci,
}


def resolve_dgp(name: str):
    try:
        return DGP_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dgp '{name}'. Options: {', '.join(DGP_MAP)}") from exc


def resolve_estimator(name: str):
    try:
        return ESTIMATOR_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown estimator '{name}'. Options: {', '.join(ESTIMATOR_MAP)}") from exc


def write_summary(folder: str, beta1_true: float = 2.0, scenario_names: list[str] | None = None) -> pd.DataFrame:
    """Aggregate per-scenario CSVs in a folder into summary.csv.

    If scenario_names is provided, only include CSVs whose stem is in that list.
    """
    rows = []
    def pick_column(df: pd.DataFrame, new: str, old: str, fill_value=np.nan) -> pd.Series:
        if new in df:
            return df[new]
        if old in df:
            return df[old]
        return pd.Series([fill_value] * len(df))

    for path in glob.glob(os.path.join(folder, "*.csv")):
        if os.path.basename(path) == "summary.csv":
            continue
        df = pd.read_csv(path)
        scenario = os.path.splitext(os.path.basename(path))[0]
        if scenario_names is not None and scenario not in scenario_names:
            continue
        treatment_hat_series = pick_column(df, "treatment_effect_hat", "beta1_hat")
        bias_series = treatment_hat_series - beta1_true
        rows.append({
            "scenario": scenario,
            "rows": len(df),
            "coverage": df["covered"].mean(),
            "treatment_effect_hat_mean": treatment_hat_series.mean(),
            "bias": bias_series.mean(),
            "rmse": np.sqrt((bias_series ** 2).mean()),
            "ci_length_mean": df["ci_length"].mean(),
            "n_selected_outcome_controls_mean": pick_column(df, "n_selected_outcome_controls", "k_y").mean(),
            "n_selected_treatment_controls_mean": pick_column(df, "n_selected_treatment_controls", "k_d").mean(),
            "outcome_lasso_penalty_mean": pick_column(df, "outcome_lasso_penalty", "alpha_y").mean(),
            "treatment_lasso_penalty_mean": pick_column(df, "treatment_lasso_penalty", "alpha_d").mean(),
            "standard_error_HC3_mean": pick_column(df, "standard_error_HC3", "se_HC3").mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(folder, "summary.csv"), index=False)
    return summary

def run_scenarios(
    *,
    dgp_name: str,
    outdir: str,
    use_cv: bool = False,
    estimator_name: str = "double_lasso",
    scenario_filter: str = "",
) -> None:
    dgp = resolve_dgp(dgp_name)
    estimator = resolve_estimator(estimator_name)
    os.makedirs(outdir, exist_ok=True)
    scenarios = get_scenarios()
    selected = [s.strip() for s in scenario_filter.split(",") if s.strip()]
    if selected:
        scenarios = [sc for sc in scenarios if sc.name in selected]
        if not scenarios:
            raise ValueError(f"No matching scenarios for names: {selected}")
    for sc in scenarios:
        df = run_simulation(
            **sc.to_run_kwargs(),
            use_cv=use_cv,
            dgp=dgp,
            estimator=estimator,
        )
        out = f"{outdir}/{sc.name}.csv"
        df.to_csv(out, index=False)
        print(f"Saved {sc.name} -> {out}")
    scenario_names = [sc.name for sc in scenarios]
    summary = write_summary(outdir, beta1_true=scenarios[0].treatment_effect, scenario_names=scenario_names)
    print("Updated summary.csv:")
    print(summary)

def run_all() -> None:
    # Baseline Double LASSO (plug-in)
    run_scenarios(dgp_name="static", outdir="results")
    run_scenarios(dgp_name="static_easier", outdir="results_easierdgp")
    run_scenarios(dgp_name="heavy_tail", outdir="results_heavy")

    # Cross-validated penalties
    run_scenarios(dgp_name="static", outdir="results_cv", use_cv=True)
    run_scenarios(dgp_name="static_easier", outdir="results_ecv", use_cv=True)
    run_scenarios(dgp_name="heavy_tail", outdir="results_heavycv", use_cv=True)

    # OLS baseline
    run_scenarios(dgp_name="static", outdir="results_ols", estimator_name="ols")

    # Plots
    subprocess.run([sys.executable, "plots.py"], check=True)
    subprocess.run([sys.executable, "plots_compare.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Double LASSO simulations (modular layout).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # simple run
    runp = sub.add_parser("run", help="Run a single design R times.")
    runp.add_argument("--R", type=int, default=500, help="Number of Monte Carlo replications")
    runp.add_argument("--n", type=int, default=200)
    runp.add_argument("--p", type=int, default=240)
    runp.add_argument("--s", type=int, default=5)
    runp.add_argument("--beta1", type=float, default=2.0)
    runp.add_argument("--rho", type=float, default=0.2)
    runp.add_argument("--c", type=float, default=0.6)
    runp.add_argument("--ci", type=float, default=0.95)
    runp.add_argument("--seed", type=int, default=123)
    runp.add_argument("--out", type=str, default="results.csv")
    runp.add_argument("--dgp", type=str, default="static", choices=list(DGP_MAP))
    runp.add_argument("--estimator", type=str, default="double_lasso", choices=list(ESTIMATOR_MAP))
    runp.add_argument("--use_cv", action="store_true",
                      help="Use cross-validated alpha instead of plugin alpha.")

    # scenarios
    scp = sub.add_parser("scenarios", help="Run predefined scenarios.")
    scp.add_argument("--outdir", type=str, default="results")
    scp.add_argument("--dgp", type=str, default="static", choices=list(DGP_MAP))
    scp.add_argument("--estimator", type=str, default="double_lasso", choices=list(ESTIMATOR_MAP))
    scp.add_argument("--scenarios", type=str, default="",
                     help="Comma-separated scenario names to run; defaults to all.")
    scp.add_argument("--use_cv", action="store_true",
                     help="Use cross-validated alpha instead of plugin alpha.")

    if len(sys.argv) == 1:
        run_all()
        return

    args = parser.parse_args()
    dgp = resolve_dgp(getattr(args, "dgp", "static"))
    estimator = resolve_estimator(getattr(args, "estimator", "double_lasso"))

    if args.cmd == "run":
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        scenario = SimulationScenario(
            name="cli_single_run",
            n_samples=args.n,
            n_covariates=args.p,
            n_relevant_covariates=args.s,
            treatment_effect=args.beta1,
            covariate_correlation=args.rho,
            R=args.R,
            plugin_c=args.c,
            ci_level=args.ci,
            seed=args.seed,
        )
        df = run_simulation(
            **scenario.to_run_kwargs(),
            use_cv=args.use_cv,
            dgp=dgp,
            estimator=estimator,
        )
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

    elif args.cmd == "scenarios":
        run_scenarios(
            dgp_name=args.dgp,
            outdir=args.outdir,
            use_cv=args.use_cv,
            estimator_name=args.estimator,
            scenario_filter=args.scenarios,
        )

if __name__ == "__main__":
    main()
