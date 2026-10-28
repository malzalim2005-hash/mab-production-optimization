# =======================================================================
# CLASSES
# Models used to predict mAb production from the initial conditions.
# =======================================================================

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import IDX_MAB


class PolynomialSurrogate:
    """
    Direct response-surface model of the mAb titre.

    Inputs:  initial glucose [mM], initial glutamine [mM], initial VCD [cells/mL] and time [d]
    Output:  mAb concentration [mg/L] at that time

    The model never feeds its own predictions back into itself, so a whole
    production curve is obtained in one step from the initial conditions.
    Time can be used as-is ("linear") or as log(1 + t) ("log"), which
    linearizes the fast-rise / plateau shape of the titre curve.
    """

    def __init__(self, degree, alpha, time_transform):
        self.degree = degree
        self.alpha = alpha
        self.time_transform = time_transform
        self.pipeline = None

    def build_features(self, x_ic, t_d):
        """
        Combine an (n, 3) array of initial conditions with an (n,) array of times [d].
        VCD is expressed in millions of cells/mL so that the polynomial terms stay well conditioned.
        """
        x_ic = np.asarray(x_ic, dtype=float).copy()
        x_ic[:, 2] = x_ic[:, 2] / 1.0e6

        if self.time_transform == "log":
            t_feature = np.log1p(t_d)
        else:
            t_feature = np.asarray(t_d, dtype=float)

        return np.column_stack([x_ic, t_feature])

    def fit(self, x_ic, t_d, y):
        self.pipeline = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=self.degree, include_bias=False),
            Ridge(alpha=self.alpha)
        )
        self.pipeline.fit(self.build_features(x_ic, t_d), y)
        return self

    def predict(self, x_ic, t_d):
        y_pred = self.pipeline.predict(self.build_features(x_ic, t_d))

        # A titre can never be negative
        return np.clip(y_pred, a_min=0.0, a_max=None)

    def predict_curves(self, x_ic, t_d):
        """
        Predict a full titre curve for every row of x_ic.

        Returns an array of shape (n_conditions, n_times).
        """
        x_ic = np.asarray(x_ic, dtype=float)
        n_ic = x_ic.shape[0]
        n_t = t_d.shape[0]

        x_rep = np.repeat(x_ic, n_t, axis=0)
        t_rep = np.tile(t_d, n_ic)

        return self.predict(x_rep, t_rep).reshape(n_ic, n_t)

    def label(self):
        return f"Surrogate (degree={self.degree}, alpha={self.alpha:g}, time={self.time_transform})"


class NVARModel:
    """
    Non-linear vector autoregression (NVAR) of the full process state.

    State: glucose, glutamine, lactate, ammonia, VCD, TCD and mAb.
    The model learns the change in the state over one sampling interval
    from the previous p states, and a production curve is obtained by
    feeding predictions back into the model recursively. Residuals of
    the training fit are re-sampled as white noise to produce a
    distribution of forecasts.
    """

    def __init__(self, model_type, hyperparameter, p):
        self.model_type = model_type
        self.hyperparameter = hyperparameter
        self.p = p
        self.model = None
        self.residuals = None
        self.x_max = None

    def fit(self, x_lag, x_cur):
        if self.model_type == "poly":
            self.model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=self.hyperparameter, include_bias=False),
                Ridge(alpha=1.0e-3)
            )
        else:
            self.model = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=self.hyperparameter)
            )

        # Learn the increment x(t) - x(t-1) rather than the level x(t)
        n_var = x_cur.shape[1]
        delta = x_cur - x_lag[:, :n_var]
        self.model.fit(x_lag, delta)

        self.residuals = delta - self.model.predict(x_lag)

        # Forecasts are not allowed to leave the range seen during training
        self.x_max = 1.2 * np.max(x_cur, axis=0)

        return self

    def forecast(self, x_initial, n_steps, n_simulations, rng):
        """
        Simulate n_simulations stochastic trajectories from one initial state.

        Returns an array of shape (n_simulations, n_steps, n_variables).
        """
        n_var = x_initial.shape[0]
        x_now = np.repeat(np.expand_dims(x_initial, axis=0), n_simulations, axis=0)
        history = [x_now]

        while len(history) < n_steps:
            # Most recent state first; repeat the initial state when fewer than p states exist
            lags = [history[-k] if len(history) >= k else history[0] for k in range(1, self.p + 1)]
            x_lag = np.concatenate(lags, axis=1)

            delta = self.model.predict(x_lag)

            # Bootstrap one residual per variable for every simulation (white noise)
            idx = rng.integers(0, self.residuals.shape[0], size=(n_simulations, n_var))
            noise = self.residuals[idx, np.arange(n_var)]

            x_next = np.clip(history[-1] + delta + noise, a_min=0.0, a_max=self.x_max)
            history.append(x_next)

        return np.stack(history, axis=1)

    def forecast_mab(self, x_initial, n_steps, n_simulations, rng):
        return self.forecast(x_initial, n_steps, n_simulations, rng)[:, :, IDX_MAB]

    def label(self):
        name = "degree" if self.model_type == "poly" else "k"
        return f"NVAR-{self.model_type} ({name}={self.hyperparameter}, p={self.p})"