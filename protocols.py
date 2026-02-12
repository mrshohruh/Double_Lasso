from __future__ import annotations
from typing import Protocol, Tuple
import numpy as np

class DGPProtocol(Protocol):
    def __call__(
        self,
        n_samples: int,
        n_covariates: int,
        n_relevant_covariates: int,
        treatment_effect: float,
        covariate_correlation: float,
        seed: int | None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class EstimatorProtocol(Protocol):
    def __call__(
        self,
        outcome: np.ndarray,
        treatment: np.ndarray,
        covariates: np.ndarray,
        **kwargs,
    ) -> dict: ...
