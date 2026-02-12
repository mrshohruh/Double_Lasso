import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.api as sm
from scipy.stats import norm


def _plugin_alpha_from_sigma(
    n_samples,
    n_covariates,
    sigma_hat,
    c=1.1,
    alpha_level=0.1,
):
    """
    Compute sklearn's alpha from a given sigma_hat using the plug-in rule
    from Chernozhukov et al. (Chapter 3).

    Book-style Lasso problem:
        sum_i (Y_i - X_i' b)^2 + lambda * sum_j |b_j|

    Plug-in rule (upper bound form):
        lambda ~ 2 * c * sigma_hat * sqrt(2 * n * log(2p / alpha_level))

    sklearn solves:
        (1/(2n)) ||Y - Xb||^2_2 + alpha * ||b||_1

    Matching coefficients:
        lambda = 2n * alpha  =>  alpha = lambda / (2n).

    So:
        alpha_sklearn = c * sigma_hat * sqrt(2 * log(2p / alpha_level) / n)
    """
    if n_covariates <= 0:
        raise ValueError("Number of predictors n_covariates must be positive.")
    if not (0 < alpha_level < 1):
        raise ValueError("alpha_level must be in (0, 1).")

    # alpha for sklearn's Lasso
    alpha_sklearn = (
        c * sigma_hat * np.sqrt(2.0 * np.log(2.0 * n_covariates / alpha_level) / n_samples)
    )
    return alpha_sklearn


def plugin_alpha(covariates, outcome, c=1.1, alpha_level=0.1, max_iter=1, tol=1e-4):
    """
    Plug-in style L1 penalty for sklearn's Lasso:
        (1/(2n)) ||y - Xb||^2 + alpha ||b||_1

    This implements the plug-in rule from Chernozhukov et al. (Ch. 3),
    with a simple iterative update of sigma_hat as in Section 3.A.

    Parameters
    ----------
    covariates : array-like, shape (n_samples, n_covariates)
        Design matrix.
    outcome : array-like, shape (n_samples,)
        Response vector.
    c : float
        Multiplicative constant in the penalty rule (typically slightly > 1).
    alpha_level : float
        Tail probability parameter 'a' in log(2p/a). Default ~ 0.1.
    max_iter : int
        Maximum number of iterations for refining sigma_hat. In practice,
        1-2 iterations are usually enough (book suggests K = 1 works well).
    tol : float
        Tolerance for convergence of sigma_hat.

    Returns
    -------
    alpha : float
        Penalty parameter in sklearn scale to be passed to Lasso(alpha=...).
    """
    covariates = np.asarray(covariates)
    outcome = np.asarray(outcome)
    n_samples, n_covariates = covariates.shape

    # Initial sigma_hat: residual std from intercept-only model
    outcome_centered = outcome - outcome.mean()
    sigma_hat = float(np.sqrt(np.mean(outcome_centered ** 2)))

    if sigma_hat <= 0:
        # Degenerate case; fall back to 1.0 to avoid crashes
        sigma_hat = 1.0

    for _ in range(max_iter):
        alpha_current = _plugin_alpha_from_sigma(
            n_samples=n_samples,
            n_covariates=n_covariates,
            sigma_hat=sigma_hat,
            c=c,
            alpha_level=alpha_level,
        )

        # Fit Lasso with current alpha to update sigma_hat
        model = Lasso(alpha=alpha_current, fit_intercept=True, max_iter=5000)
        model.fit(covariates, outcome)
        residuals = outcome - model.predict(covariates)
        sigma_new = float(np.sqrt(np.mean(residuals ** 2)))

        if abs(sigma_new - sigma_hat) <= tol:
            sigma_hat = sigma_new
            break
        sigma_hat = sigma_new

    # Final alpha based on final sigma_hat
    alpha_final = _plugin_alpha_from_sigma(
        n_samples=n_samples,
        n_covariates=n_covariates,
        sigma_hat=sigma_hat,
        c=c,
        alpha_level=alpha_level,
    )
    return alpha_final


def cv_alpha(covariates, outcome):
    """
    Compute alpha via 10-fold cross-validation (sklearn's LassoCV).
    Returns the selected alpha_.
    """
    covariates = np.asarray(covariates)
    outcome = np.asarray(outcome)
    model = LassoCV(cv=10, fit_intercept=True, n_jobs=-1)
    model.fit(covariates, outcome)
    return float(model.alpha_)


def lasso_residuals(covariates, outcome, alpha):
    """
    Fit Lasso(outcome ~ covariates) and return residuals outcome - (intercept + covariates b_hat),
    along with the fitted model.
    """
    covariates = np.asarray(covariates)
    outcome = np.asarray(outcome)

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    model.fit(covariates, outcome)
    y_hat = model.predict(covariates)
    residuals = outcome - y_hat
    return residuals, model


def double_lasso_ci(outcome, treatment, covariates, alpha=None, ci_level=0.95, use_cv=False,
                    plugin_c=1.1, plugin_alpha_level=0.1):
    """
    Double LASSO confidence interval for the coefficient of treatment in:
        outcome = treatment_effect * treatment + covariates * gamma + eps

    Steps:
      1) Lasso outcome ~ covariates -> residuals.
      2) Lasso treatment ~ covariates -> residuals.
      3) OLS residual(outcome) on residual(treatment), HC3 robust SE -> CI for treatment_effect.

    Parameters
    ----------
    outcome : array-like, shape (n_samples,)
        Outcome.
    treatment : array-like, shape (n_samples,)
        Treatment / target regressor.
    covariates : array-like, shape (n_samples, n_covariates)
        Controls.
    alpha : float or None
        If not None and use_cv=False, this value is used as the Lasso alpha
        for both outcome and treatment equations (sklearn scale).
    ci_level : float
        Confidence level (e.g., 0.95).
    use_cv : bool
        If True, use LassoCV to select alpha for outcome and treatment separately.
        If False and alpha is None, use the plug-in rule (Chernozhukov et al.).
    plugin_c : float
        Constant c used in the plug-in penalty rule (default 0.6).
    plugin_alpha_level : float
        Tail probability 'a' in log(2p/a) in the plug-in rule.

    Returns
    -------
    results : dict with keys
        - treatment_effect_hat : float
        - standard_error_HC3   : float
        - ci_lower             : float
        - ci_upper             : float
        - n_selected_outcome_controls    : int   (number of selected controls in outcome regression)
        - n_selected_treatment_controls  : int   (number of selected controls in treatment regression)
        - outcome_lasso_penalty          : float (alpha used for outcome regression)
        - treatment_lasso_penalty        : float (alpha used for treatment regression)
    """
    outcome = np.asarray(outcome)
    treatment = np.asarray(treatment)
    covariates = np.asarray(covariates)
    n_samples, n_covariates = covariates.shape

    # Choose penalty levels
    if use_cv:
        outcome_lasso_penalty = cv_alpha(covariates, outcome)
        treatment_lasso_penalty = cv_alpha(covariates, treatment)
    else:
        if alpha is not None:
            # User-specified alpha (same for both equations)
            outcome_lasso_penalty = treatment_lasso_penalty = float(alpha)
        else:
            # True plug-in rule, potentially different for Y and D equations
            outcome_lasso_penalty = plugin_alpha(
                covariates, outcome, c=plugin_c, alpha_level=plugin_alpha_level
            )
            treatment_lasso_penalty = plugin_alpha(
                covariates, treatment, c=plugin_c, alpha_level=plugin_alpha_level
            )

    outcome_residual, outcome_model = lasso_residuals(covariates, outcome, outcome_lasso_penalty)

    treatment_residual, treatment_model = lasso_residuals(covariates, treatment, treatment_lasso_penalty)

    covariates_ols = sm.add_constant(treatment_residual)
    ols_fit = sm.OLS(outcome_residual, covariates_ols).fit(cov_type="HC3")

    treatment_effect_hat = float(ols_fit.params[1])
    standard_error = float(ols_fit.bse[1])

    # z-quantile for desired CI level
    z = norm.ppf(0.5 + ci_level / 2.0)
    ci_lower = treatment_effect_hat - z * standard_error
    ci_upper = treatment_effect_hat + z * standard_error

    return {
        "treatment_effect_hat": treatment_effect_hat,
        "standard_error_HC3": standard_error,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_selected_outcome_controls": int(np.sum(outcome_model.coef_ != 0)),
        "n_selected_treatment_controls": int(np.sum(treatment_model.coef_ != 0)),
        "outcome_lasso_penalty": float(outcome_lasso_penalty),
        "treatment_lasso_penalty": float(treatment_lasso_penalty),
    }
