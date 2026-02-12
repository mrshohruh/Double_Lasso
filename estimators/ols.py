import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


def ols_ci(outcome, treatment, covariates, ci_level=0.95, **kwargs):
    """
    Plain OLS with HC3 standard errors:
    outcome = alpha + beta1 * treatment + covariates * gamma + eps
    Returns beta1 inference in the same format as the Double Lasso estimator.
    """
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    covariates = np.asarray(covariates)
    if covariates.ndim == 1:
        covariates = covariates[:, None]

    design = np.column_stack([treatment, covariates])
    design = sm.add_constant(design)

    ols_fit = sm.OLS(outcome, design).fit(cov_type="HC3")

    beta1_hat = float(ols_fit.params[1])
    se = float(ols_fit.bse[1])
    z = norm.ppf(0.5 + ci_level / 2.0)
    ci_lower = beta1_hat - z * se
    ci_upper = beta1_hat + z * se

    n_controls = int(covariates.shape[1])
    return {
        "treatment_effect_hat": beta1_hat,
        "standard_error_HC3": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_selected_outcome_controls": n_controls,
        "n_selected_treatment_controls": n_controls,
        "lasso_penalty": float("nan"),
        "outcome_lasso_penalty": float("nan"),
        "treatment_lasso_penalty": float("nan"),
    }
