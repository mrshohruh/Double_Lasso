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
    """
    Coverage-friendly DGP for Double LASSO experiments.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_covariates : int
        Number of covariates.
    n_relevant_covariates : int
        Number of relevant covariates (first columns of X).
    treatment_effect : float
        True treatment effect.
    covariate_correlation : float
        Equicorrelation parameter between covariates.
    seed : int or None
        Random seed for reproducibility.

    DGP:
        X ~ N(0, covariance_matrix) where covariance_matrix has off-diagonal covariate_correlation
        Let b_j ∝ 1 / j^2 for j = 1,...,n_relevant_covariates (approximate sparsity)

        signal = X_{1:s} b

        D = gamma_D * signal + v
        Y = treatment_effect * D + gamma_Y * signal + u

    Returns
    -------
    Y : (n_samples,) ndarray
        Outcome variable.
    D : (n_samples,) ndarray
        Treatment / target regressor.
    X : (n_samples, n_covariates) ndarray
        Covariate matrix.
    """

    rng = np.random.default_rng(seed)

    if covariate_correlation == 0.0:
        covariates = rng.normal(size=(n_samples, n_covariates))
    else:
        covariance_matrix = (
            (1.0 - covariate_correlation) * np.eye(n_covariates)
            + covariate_correlation * np.ones((n_covariates, n_covariates))
        )
        cholesky_factor = np.linalg.cholesky(covariance_matrix)
        covariates = rng.normal(size=(n_samples, n_covariates)) @ cholesky_factor.T

    idx = np.arange(1, n_relevant_covariates + 1)
    coef_decay = 1.0 / (idx**2)

    coef_decay = coef_decay / np.linalg.norm(coef_decay) * np.sqrt(n_relevant_covariates)

    signal = covariates[:, :n_relevant_covariates] @ coef_decay

    treatment_noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    outcome_noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    gamma_treatment = 1.0
    gamma_outcome = 0.5

    treatment = gamma_treatment * signal + treatment_noise
    outcome = treatment_effect * treatment + gamma_outcome * signal + outcome_noise

    return outcome, treatment, covariates
