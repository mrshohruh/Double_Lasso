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
    Heavy-tailed DGP using multivariate Student-t covariates and t noise terms.
    """
    rng = np.random.default_rng(seed)
    degrees_of_freedom = 3.0  # low degrees of freedom to induce heavy tails

    if covariate_correlation == 0.0:
        gaussian_draws = rng.standard_normal(size=(n_samples, n_covariates))
    else:
        covariance_matrix = (
            (1.0 - covariate_correlation) * np.eye(n_covariates)
            + covariate_correlation * np.ones((n_covariates, n_covariates))
        )
        cholesky_factor = np.linalg.cholesky(covariance_matrix)
        gaussian_draws = rng.standard_normal(size=(n_samples, n_covariates)) @ cholesky_factor.T

    chi_square_draws = rng.chisquare(df=degrees_of_freedom, size=n_samples)
    scale_factors = np.sqrt(degrees_of_freedom / chi_square_draws)
    covariates = gaussian_draws * scale_factors[:, None]

    idx = np.arange(1, n_relevant_covariates + 1)
    sparse_coefficients = 1.0 / np.sqrt(idx)
    sparse_coefficients = sparse_coefficients / np.linalg.norm(sparse_coefficients) * np.sqrt(n_relevant_covariates)
    confounding_signal = covariates[:, :n_relevant_covariates] @ sparse_coefficients

    noise_scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    treatment_noise = rng.standard_t(degrees_of_freedom, size=n_samples) * noise_scale
    outcome_noise = rng.standard_t(degrees_of_freedom, size=n_samples) * noise_scale

    gamma_treatment = 1.0
    gamma_outcome = 0.5

    treatment = gamma_treatment * confounding_signal + treatment_noise
    outcome = treatment_effect * treatment + gamma_outcome * confounding_signal + outcome_noise
    return outcome, treatment, covariates
