
from __future__ import annotations
import numpy as np

def simulate_dgp(
    n_samples: int = 200,
    n_covariates: int = 240,
    n_relevant_covariates: int = 5,
    treatment_effect: float = 2.0,
    covariate_correlation: float = 0.0,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    if covariate_correlation == 0.0:
        covariates = rng.normal(size=(n_samples, n_covariates))
    else:
        covariance_matrix = (
            (1 - covariate_correlation) * np.eye(n_covariates)
            + covariate_correlation * np.ones((n_covariates, n_covariates))
        )
        cholesky_factor = np.linalg.cholesky(covariance_matrix)
        covariates = rng.normal(size=(n_samples, n_covariates)) @ cholesky_factor.T

    confounding_signal = covariates[:, :n_relevant_covariates].sum(axis=1)
    treatment_noise = rng.normal(size=n_samples)
    outcome_noise = rng.normal(size=n_samples)

    treatment = confounding_signal + treatment_noise
    outcome = treatment_effect * treatment + confounding_signal + outcome_noise
    return outcome, treatment, covariates
