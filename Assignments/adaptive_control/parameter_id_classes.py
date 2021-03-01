import numpy as np
from copy import deepcopy

F_FACTOR_0 = 0.75
GAMMA_MAX_EIG = 50.0

estimator_init_default = {
    "Gamma": np.identity(4),
    "params": [[0.0], [0.0], [0.0], [0.0]],
    "f_factor": 0.975
}


class Estimator_RLS:
    """
    Rec. Least Squares (for recursive_lse.py)
    """
    def __init__(self, P=None, params=None, f_factor=None):
        self.P = np.array(P) if P is not None else np.identity(3)
        self.params = np.array(params) if params else [[0.0], [0.0], [0.0]]
        self.f_factor = f_factor if f_factor else 1.0

    def update_P(self, regressors):
        Gamma = (1/self.f_factor) * self.P
        self.P = Gamma @(np.identity(3) - regressors @ np.linalg.inv(
            1 + regressors.T @ Gamma @ regressors
        )@ regressors.T @ Gamma)

    def update_params(self, regressors, measurement):
        prior_error = measurement - self.regress(regressors)
        self.params = self.params + self.P @ regressors * prior_error

    def regress(self, regressors):
        return float(regressors.T @ self.params)


class Estimator_RLS_EE:
    """
    Equation Error Model with RLS, using variable forgetting factor (for armax.py)
    """
    def __init__(self, Gamma=None, params=None, f_factor=None):
        self.Gamma = np.array(Gamma) if Gamma is not None else estimator_init_default["Gamma"]
        self.params = np.array(params) if params else estimator_init_default["params"]
        self.f_factor = f_factor if f_factor else estimator_init_default["f_factor"]
        self.regressors = np.array([[0.0], [0.0], [0.0], [0.0]])

    def update(self, measurement, regressors):
        self.update_Gamma(regressors)
        self.update_params(measurement)

    def update_Gamma(self, regressors):
        # Note: Regressors are updated here!!
        self.regressors = deepcopy(regressors)

        Gamma = (1/self.f_factor) * (self.Gamma - (1/(1 + regressors.T @ self.Gamma @ regressors)) * self.Gamma @ regressors @ regressors.T @ self.Gamma)
        self.Gamma = Gamma if np.max(np.linalg.eigvals(Gamma)) < GAMMA_MAX_EIG else self.Gamma  # Covariance Limiting
        self.f_factor = F_FACTOR_0*self.f_factor + 1 - F_FACTOR_0  # Variable forgetting factor, Lambda_1 = Lambda_2

    def update_params(self, measurement):
        prior_error = measurement - self.regress()
        self.params = self.params + (1/(1 + self.regressors.T @ self.Gamma @ self.regressors)) * self.Gamma @ self.regressors * prior_error

    def regress(self):
        return float(self.regressors.T @ self.params)


class Estimator_ARMAX:
    """
    ARMAX Model with Extended Least Squares (for armax.py)
    y(t) = f(y(t-1), .., e(t), e(t-1))
    """
    def __init__(self, Gamma=None, params=None, f_factor=None):
        self.Gamma = np.array(Gamma) if Gamma is not None else 1.0*np.identity(6)
        self.params = np.array(params) if params else np.array([[0.0], [0.0], [0.0], [0.0], [2.5], [1.5]])
        self.f_factor = f_factor if f_factor else estimator_init_default["f_factor"]
        self.regressors = np.array([[0.0], [0.0], [0.0], [0.0], [0.5], [0.5]])  # Noise signal estimated using prior est. error

    def update(self, measurement, regressors):
        # We need to estimate e(t) using epsilon(t)
        self.regressors[0:4] = deepcopy(regressors)
        self.regressors[5] = deepcopy(self.regressors[4])
        self.regressors[4] = [0.0]
        self.regressors[4] = (1 / (1 + self.params[4])) * (measurement - self.regress())

        self.update_Gamma(regressors)
        self.update_params(measurement)

    def update_Gamma(self, regressors):
        Gamma = (1/self.f_factor) * (self.Gamma - (1/(1 + self.regressors.T @ self.Gamma @ self.regressors)) * self.Gamma @ self.regressors @ self.regressors.T @ self.Gamma)
        self.Gamma = Gamma if np.max(np.linalg.eigvals(Gamma)) < GAMMA_MAX_EIG else self.Gamma  # Covariance Limiting
        self.f_factor = F_FACTOR_0*self.f_factor + 1 - F_FACTOR_0  # Variable forgetting factor, Lambda_1 = Lambda_2

    def update_params(self, measurement):
        prior_error = measurement - self.regress()
        self.params = self.params + (1/(1 + self.regressors.T @ self.Gamma @ self.regressors)) * self.Gamma @ self.regressors * prior_error

    def regress(self):
        return float(self.regressors.T @ self.params)


class Estimator_ARMAX5:
    """
    ARMAX Model with Extended Least Squares (for armax.py)
    Uses one less regressor than Estimator_ARMAX, since e(t) is difficult to approximate at same time-step
    y(t) = f(y(t-1), .., u(t-3), e(t-1))
    """

    def __init__(self, Gamma=None, params=None, f_factor=None):
        self.Gamma = np.array(Gamma) if Gamma is not None else 5.0 * np.identity(5)
        self.params = np.array(params) if params else np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        self.f_factor = f_factor if f_factor else estimator_init_default["f_factor"]
        self.regressors = np.array(
            [[0.0], [0.0], [0.0], [0.0], [0.0]])  # Noise signal estimated using prior est. error

    def update(self, measurement, regressors):
        # We need to estimate e(t) using epsilon(t)
        self.regressors[0:4] = deepcopy(regressors)
        self.update_Gamma()
        self.update_params(measurement)
        self.regressors[4] = measurement - self.regress()

    def update_Gamma(self, ):
        Gamma = (1 / self.f_factor) * (self.Gamma - (1 / (
                    1 + self.regressors.T @ self.Gamma @ self.regressors)) * self.Gamma @ self.regressors @ self.regressors.T @ self.Gamma)
        self.Gamma = Gamma if np.max(
            np.linalg.eigvals(Gamma)) < GAMMA_MAX_EIG else self.Gamma  # Covariance Limiting
        self.f_factor = F_FACTOR_0 * self.f_factor + 1 - F_FACTOR_0  # Variable forgetting factor, Lambda_1 = Lambda_2

    def update_params(self, measurement):
        prior_error = measurement - self.regress()
        self.params = self.params + (1 / (
                    1 + self.regressors.T @ self.Gamma @ self.regressors)) * self.Gamma @ self.regressors * prior_error

    def regress(self):
        return float(self.regressors.T @ self.params)

