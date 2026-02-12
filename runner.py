
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

from protocols import DGPProtocol, EstimatorProtocol

def run_simulation(*,
                   R: int = 500,
                   n_samples: int = 200,
                   n_covariates: int = 240,  # default to p > n for high-dimensional design
                   n_relevant_covariates: int = 5,
                   treatment_effect: float = 2.0,
                   covariate_correlation: float = 0.2,
                   ci_level: float = 0.95,
                   plugin_c: float = 0.6,
                   plugin_alpha_level: float = 0.1,
                   seed: int = 123,
                   use_cv: bool = False,
                   dgp: DGPProtocol | None = None,
                   estimator: EstimatorProtocol | None = None) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(R):
        outcome, treatment, covariates = dgp(
            n_samples=n_samples,
            n_covariates=n_covariates,
            n_relevant_covariates=n_relevant_covariates,
            treatment_effect=treatment_effect,
            covariate_correlation=covariate_correlation,
            seed=rng.integers(0, 1_000_000),
        )
        estimator_results = estimator(
            outcome,
            treatment,
            covariates,
            ci_level=ci_level,
            use_cv=use_cv,
            plugin_c=plugin_c,
            plugin_alpha_level=plugin_alpha_level,
        )
        covered = int((estimator_results["ci_lower"] <= treatment_effect) and (treatment_effect <= estimator_results["ci_upper"]))
        rows.append({
            "treatment_effect_hat": estimator_results["treatment_effect_hat"],
            "standard_error_HC3": estimator_results["standard_error_HC3"],
            "ci_lower": estimator_results["ci_lower"],
            "ci_upper": estimator_results["ci_upper"],
            "ci_length": estimator_results["ci_upper"] - estimator_results["ci_lower"],
            "covered": covered,
            "n_selected_outcome_controls": estimator_results["n_selected_outcome_controls"],
            "n_selected_treatment_controls": estimator_results["n_selected_treatment_controls"],
            "lasso_penalty": estimator_results.get("outcome_lasso_penalty", np.nan),
            "outcome_lasso_penalty": estimator_results.get("outcome_lasso_penalty", np.nan),
            "treatment_lasso_penalty": estimator_results.get("treatment_lasso_penalty", np.nan),
        })
    return pd.DataFrame(rows)
