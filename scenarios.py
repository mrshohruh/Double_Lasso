from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SimulationScenario:
    name: str
    n_samples: int = 200
    n_covariates: int = 240
    n_relevant_covariates: int = 5
    treatment_effect: float = 2.0
    covariate_correlation: float = 0.2
    R: int = 500
    plugin_c: float = 0.6
    ci_level: float = 0.95
    seed: int = 123

    def to_run_kwargs(self) -> Dict[str, Any]:
        """Convert the scenario into run_simulation keyword arguments."""
        return dict(
            R=self.R,
            n_samples=self.n_samples,
            n_covariates=self.n_covariates,
            n_relevant_covariates=self.n_relevant_covariates,
            treatment_effect=self.treatment_effect,
            covariate_correlation=self.covariate_correlation,
            ci_level=self.ci_level,
            plugin_c=self.plugin_c,
            seed=self.seed,
        )


def get_scenarios() -> list[SimulationScenario]:

    return [
        SimulationScenario(
            name="classical_low_dim",
            n_samples=200,
            n_covariates=20,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.2,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="near_p_equals_n",
            n_samples=200,
            n_covariates=180,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.2,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="p_equals_n",
            n_samples=200,
            n_covariates=200,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.2,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="medium_corr_0_0",
            n_samples=200,
            n_covariates=240,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.0,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="medium_corr_0_2",
            n_samples=200,
            n_covariates=240,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.2,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="medium_corr_0_5",
            n_samples=200,
            n_covariates=240,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.5,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="large_corr_0_0",
            n_samples=320,
            n_covariates=384,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.0,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="large_corr_0_2",
            n_samples=320,
            n_covariates=384,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.2,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="large_corr_0_5",
            n_samples=320,
            n_covariates=384,
            n_relevant_covariates=5,
            treatment_effect=2.0,
            covariate_correlation=0.5,
            R=500,
            plugin_c=0.6,
            ci_level=0.95,
            seed=123,
        ),

    ]
