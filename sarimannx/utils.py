"""Utilities for the sarimannx modules
"""

import logging
import warnings

import numpy as np
from scipy.special import expit as logistic_sigmoid


def identity(x):
    """Simply leave the input array unchanged.

    Parameters
    ----------
    x : ndarray
        ANN activations.
    """
    return x


def logistic(x):
    """Compute the logistic function.

    Parameters
    ----------
    x : ndarray
        ANN activations.
    """
    return logistic_sigmoid(x)


def tanh(x):
    """Compute the hyperbolic tan function.

    Parameters
    ----------
    x : ndarray
        ANN activations.
    """
    return np.tanh(x)


def relu(x):
    """Compute the rectified linear unit function.

    Parameters
    ----------
    x : ndarray
            ANN activations.
        """
    return np.maximum(x, 0)


ACTIVATIONS = {"identity": identity,
               "tanh": tanh,
               "logistic": logistic,
               "relu": relu}


def identity_derivative(z):
    """Apply the derivative of the identity function.

    Parameters
    ----------
    z : ndarray
        The data which was output from the identity activation function during
        the ANN forward pass.
    """
    return np.ones_like(z)


def logistic_derivative(z):
    """Apply the derivative of the logistic function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    z : ndarray
        The data which was output from the identity activation function during
        the ANN forward pass.
    """
    return z * (1 - z)


def tanh_derivative(z):
    """Apply the derivative of the hyperbolic tan function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    z : ndarray
        The data which was output from the identity activation function during
        the ANN forward pass.
    """
    return 1 - z ** 2


def relu_derivative(z):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    z : ndarray
        The data which was output from the identity activation function during
        the ANN forward pass.
    """
    z_copy = z.copy()
    z_copy[z_copy > 0] = 1
    return z_copy


DERIVATIVES = {"identity": identity_derivative,
               "tanh": tanh_derivative,
               "logistic": logistic_derivative,
               "relu": relu_derivative}


# SARIMANNX hyperparameters and inputs validation
class Validation:
    """User inputs validation class.

    The Validation contains methods for validating user inputs
    in the SARIMANNX class methods.

    Parameters
    ----------
    model : SARIMANNX class object
        SARIMANNX model whose parameters will be validated.
    """
    def __init__(self, model):
        self.model = model

    def _order(self, order):
        """Validate input `order` in __init__ SARIMANNX method.
        """
        try:
            if len(order) != 5:
                raise ValueError

            valid_order = [0] * len(order)
            for i in range(len(order)):
                elem = int(order[i])
                if elem < 0:
                    raise ValueError
                valid_order[i] = elem
            valid_order = tuple(valid_order)
        except (TypeError, ValueError):
            raise ValueError("The `order` must be an iterable, which contains a"
                             " 5 non-negative integers.")
        else:
            return valid_order

    def _seasonal_order(self, seasonal_order):
        """Validate input `seasonal_order` in __init__ SARIMANNX method."""
        try:
            if len(seasonal_order) != 6:
                raise ValueError

            valid_seasonal_order = [0] * len(seasonal_order)
            for i in range(len(seasonal_order)):
                elem = seasonal_order[i]
                if i == 2 or i == 5:
                    elem = int(elem)
                    if elem < 0:
                        raise ValueError
                    valid_seasonal_order[i] = elem
                elif elem is None:
                    valid_seasonal_order[i] = tuple()
                elif isinstance(elem, (int, float)):
                    elem = int(elem)
                    if elem < 0:
                        raise ValueError
                    elif elem == 0:
                        valid_seasonal_order[i] = tuple()
                    else:
                        valid_seasonal_order[i] = (elem,)
                else:
                    lags = [0] * len(elem)
                    for j in range(len(elem)):
                        lag = int(elem[j])
                        if lag < 0:
                            raise ValueError
                        lags[j] = lag
                    valid_seasonal_order[i] = tuple(lags)
            valid_seasonal_order = tuple(valid_seasonal_order)
        except (TypeError, ValueError):
            raise ValueError("The `seasonal_order` must be an iterable of size 6"
                             " with the following structure: Each element by"
                             " index 0, 1, 3, 4 is either a `None`, a non-negative"
                             " integer or an iterable of non-negative integers;"
                             " Each element by index 2, 5 is a non-negative integer.")
        else:
            return valid_seasonal_order

    def _ann_hidden_layer_sizes(self, ann_hidden_layer_sizes):
        """Validate input `ann_hidden_layer_sizes` in __init__ SARIMANNX method."""
        try:
            if isinstance(ann_hidden_layer_sizes, (int, float)):
                if ann_hidden_layer_sizes < 0:
                    raise ValueError
                valid_ann_hidden_layer_sizes = int(ann_hidden_layer_sizes),
            elif ann_hidden_layer_sizes is None:
                valid_ann_hidden_layer_sizes = tuple()
            else:
                valid_ann_hidden_layer_sizes = [0] * len(ann_hidden_layer_sizes)
                for i in range(len(ann_hidden_layer_sizes)):
                    elem = int(ann_hidden_layer_sizes[i])
                    if elem < 0:
                        raise ValueError
                    valid_ann_hidden_layer_sizes[i] = elem
                valid_ann_hidden_layer_sizes = tuple(valid_ann_hidden_layer_sizes)
        except (TypeError, ValueError):
            raise ValueError("The `ann_hidden_layer_sizes` must be a `None`,"
                             " non-negative integer or an iterable of non-negative"
                             " integers.")
        else:
            return valid_ann_hidden_layer_sizes

    def _ann_activation(self, ann_activation):
        """Validate input `ann_activation` in __init__ SARIMANNX method."""
        if ann_activation not in ACTIVATIONS:
            raise ValueError(f"The activation {ann_activation} is not supported."
                             " Supported activations is"
                             f" {', '.join(map(str, ACTIVATIONS.keys()))}")
        return ann_activation

    def _trend(self, trend):
        """Validate input `trend` in __init__ SARIMANNX method."""
        try:
            if isinstance(trend, str):
                if trend == "n":
                    valid_trend = np.array([])
                elif trend == "c":
                    valid_trend = np.array([0])
                elif trend == "t":
                    valid_trend = np.array([1])
                elif trend == "ct":
                    valid_trend = np.array([0, 1])
                else:
                    raise ValueError
            else:
                valid_trend = np.array([float(power) for power in trend])
        except (TypeError, ValueError):
            raise ValueError("The `trend` must be a char 'n', 'c', 't' or 'ct' or"
                             " an iterable of float numbers.")
        else:
            if valid_trend.shape[0] > 0 and self.model.d + self.model.seasonal_d > 0:
                warnings.warn(f"Be careful about adding a {valid_trend.max()}th degree polinomial"
                              " trend when using time series differencing, because this"
                              " means that the original time series has"
                              f" {valid_trend.max() + self.model.d + self.model.seasonal_d}th"
                              " degree polinomial trend.")
            return valid_trend

    def _optimize_init_shocks(self, optimize_init_shocks):
        """Validate input `optimize_init_shocks` in __init__ SARIMANNX method."""
        try:
            valid_optimize_init_shocks = bool(optimize_init_shocks)
        except TypeError:
            raise ValueError("The `optimize_init_shocks` must be boolean.")
        else:
            return valid_optimize_init_shocks

    def _y(self, y, it_input_shocks=False):
        """Validate input time series `y`."""
        try:
            if y is None:
                y = list()
            y = np.array(y, dtype=float).squeeze()
            if y.ndim != 1:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"The {'`y`' if not it_input_shocks else '`input_shocks`'}"
                             " must be a 1 dimensional iterable of float numbers.")
        else:
            return y

    def _X(self, X):
        """Validate input exogenous regressors `X`."""
        try:
            if X is None or len(X) == 0:
                X = np.array([], dtype=float)
            else:
                X = np.array(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                elif X.ndim > 2:
                    raise ValueError
        except (TypeError, ValueError):
            raise ValueError("The `X` must be a 2 dimensional iterable of float numbers.")
        else:
            return X

    def _sarima_exogs(self, sarima_exogs):
        """Validate input `sarima_exogs` in fit SARIMANNX method."""
        try:
            if sarima_exogs == slice(None):
                valid_sarima_exogs = sarima_exogs
            elif sarima_exogs is None or len(sarima_exogs) == 0:
                valid_sarima_exogs = tuple()
            else:
                valid_sarima_exogs = [0] * len(sarima_exogs)
                for i in range(len(sarima_exogs)):
                    exog = int(sarima_exogs[i])
                    if exog < 0:
                        raise ValueError
                    valid_sarima_exogs[i] = exog
                valid_sarima_exogs = tuple(valid_sarima_exogs)
        except (TypeError, ValueError):
            raise ValueError("The `sarima_exogs` must be a None, slice(None)"
                             " or an iterable of non-negative integers"
                             " less than number of columns in `X`.")
        else:
            return valid_sarima_exogs

    def _ann_exogs(self, ann_exogs):
        """Validate input `ann_exogs` in fit SARIMANNX method."""
        try:
            if ann_exogs == slice(None):
                valid_ann_exogs = ann_exogs
            elif ann_exogs is None or len(ann_exogs) == 0:
                valid_ann_exogs = tuple()
            else:
                valid_ann_exogs = [0] * len(ann_exogs)
                for i in range(len(ann_exogs)):
                    exog = int(ann_exogs[i])
                    if exog < 0:
                        raise ValueError
                    valid_ann_exogs[i] = exog
                valid_ann_exogs = tuple(valid_ann_exogs)
        except (TypeError, ValueError):
            raise ValueError("The`ann_exogs` must be a None, slice(None)"
                             " or an iterable of non-negative integers"
                             " less than number of columns in `X`.")
        else:
            return valid_ann_exogs

    def _dtype(self, dtype):
        """Validate input data type."""
        try:
            valid_dtype = np.dtype(dtype)
            if valid_dtype.kind not in ("i", "u", "f"):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("The `dtype` must be one of the Python or Numpy"
                             " floats or ints or Numpy dtype.")
        else:
            return valid_dtype

    def _fit_input(self, y, X, sarima_exogs, ann_exogs, dtype, init_weights_shocks,
                   return_preds_resids):
        """Validate inputs in SARIMANNX method fit."""
        dtype = self._dtype(dtype)
        y = (self._y(y)).astype(dtype, copy=False)
        X = (self._X(X)).astype(dtype, copy=False)
        sarima_exogs = self._sarima_exogs(sarima_exogs)
        ann_exogs = self._ann_exogs(ann_exogs)

        try:
            init_weights_shocks = bool(init_weights_shocks)
            return_preds_resids = bool(return_preds_resids)
        except TypeError:
            raise ValueError("The `init_weights_shocks` and `return_preds_resids`"
                             " must be a booleans.")

        if y.shape[0] < (self.model.d + self.model.seasonal_d * self.model.s +
                         self.model.max_ar_lag + 1):
            raise ValueError(f"Number of observations {y.shape[0]} in `y` is not"
                             " enough for training. At least"
                             f""" {self.model.d + self.model.seasonal_d * self.model.s +
                                  self.model.max_ar_lag + 1}"""
                             " observations needed.")

        if X.shape[0] != 0 and y.shape[0] != X.shape[0]:
            raise ValueError(f"Number of observations in `y` {y.shape[0]} and"
                             f" `X`(number of rows) {X.shape[0]} must be equal.")

        if (sarima_exogs != slice(None) and len(sarima_exogs) != 0 and X.ndim == 2 and
                max(sarima_exogs) >= X.shape[1]):
            raise ValueError(f"The `sarima_exogs` has out-of-range `X` index {max(sarima_exogs)}."
                             f" Maximum index is {X.shape[1]-1}.")

        if (ann_exogs != slice(None) and len(ann_exogs) != 0 and X.ndim == 2 and
                max(ann_exogs) >= X.shape[1]):
            raise ValueError(f"The `ann_exogs` has out-of-range `X` index {max(ann_exogs)}."
                             f" Maximum index is {X.shape[1] - 1}.")

        return (y, X, sarima_exogs, ann_exogs, dtype, init_weights_shocks,
                return_preds_resids)

    def _check_optimize_result(self, opt_res):
        """Check optimization result."""
        if not opt_res.success:
            warnings.warn("The optimizer was not exited successfully.\n"
                          f"Status: {opt_res.status};\n"
                          f"Message: {opt_res.message};\n"
                          f"NumIters: {opt_res.nit};")
        return opt_res.nit

    def _predict_input(self, y, X, input_shocks, t, horizon, intervals,
                       return_last_input_shocks):
        """Validate inputs in SARIMANNX method predict."""
        y = self._y(y)
        X = self._X(X)
        input_shocks = self._y(input_shocks, it_input_shocks=True)

        try:
            if t is not None:
                t = int(t)
            horizon = int(horizon)
            if horizon <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("The forecasting origin `t` must be integer or None."
                             " The forecasting horizon `horizon` must be positive integer.")

        try:
            intervals = bool(intervals)
            return_last_input_shocks = bool(return_last_input_shocks)
        except TypeError:
            raise ValueError("The `intervals` and `return_last_input_shocks` must be boolean.")

        return y, X, input_shocks, t, horizon, intervals, return_last_input_shocks


def squared_loss(sarima_coefs, ann_coefs, ann_intercepts, ann_activation, trend,
                 trend_coefs, init_shocks, p, q, d, r, g, seasonal_p, seasonal_q,
                 seasonal_d, seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag,
                 sarima_exogs, ann_exogs, y, X, sarima_in, ann_in, trend_in):
    """Calculates mean squared error loss (MSE)."""
    shocks = init_shocks.copy()
    loss = 0
    for t in range(max_ar_lag, y.shape[0]):
        # SARIMA input
        sarima_in[:p] = y[t - 1:t - p - 1 if t != p else None:-1]
        sarima_in[p:p + len(seasonal_p)] = y[[t - i for i in seasonal_p]]
        sarima_in[p + len(seasonal_p):p + len(seasonal_p) + q] = shocks[:q]
        sarima_in[p + len(seasonal_p) + q:p + len(seasonal_p) + q + len(seasonal_q)] = \
            shocks[[i - 1 for i in seasonal_q]]
        if hasattr(X, "ndim") and X.ndim == 2:
            # Add eXogs
            sarima_in[p + len(seasonal_p) + q + len(seasonal_q):] = \
                X[t + d + seasonal_d * s, sarima_exogs]
        # Trend input
        trend_in = np.power(t + d + seasonal_d * s, trend, dtype=y.dtype)
        # ANN input
        ann_in[:r] = y[t - 1:t - r - 1 if t != r else None:-1]
        ann_in[r:r + len(seasonal_r)] = y[[t - i for i in seasonal_r]]
        ann_in[r + len(seasonal_r):r + len(seasonal_r) + g] = shocks[:g]
        ann_in[r + len(seasonal_r) + g:r + len(seasonal_r) + g + len(seasonal_g)] = \
            shocks[[i - 1 for i in seasonal_g]]
        if hasattr(X, "ndim") and X.ndim == 2:
            # Add eXogs
            ann_in[r + len(seasonal_r) + g + len(seasonal_g):] = \
                X[t + d + seasonal_d * s, ann_exogs]

        # ANN fast forward pass
        ann_hid_unit = ann_in.copy()
        ann_hid_activation = ACTIVATIONS[ann_activation]
        for i in range(len(ann_coefs) - 1):
            ann_hid_unit = ann_hid_activation(
                np.dot(ann_coefs[i], ann_hid_unit) + ann_intercepts[i]
            )

        # Calculate loss
        pred = np.dot(sarima_coefs, sarima_in) + np.dot(trend_coefs, trend_in)
        if len(ann_coefs):
            pred += np.dot(ann_coefs[-1], ann_hid_unit)
        shock = y[t] - pred
        shocks[1:] = shocks[:-1]
        if len(shocks):
            shocks[0] = shock
        loss += 0.5 * shock ** 2 / y.shape[0]

    return loss


def squared_loss_grad_numapprox(sarima_coefs, ann_coefs, ann_intercepts, ann_activation,
                                trend, trend_coefs, init_shocks, optimize_init_shocks,
                                p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d,
                                seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag,
                                sarima_exogs, ann_exogs, y, X, grad_loss_total,
                                sarima_in, ann_in, trend_in, epsilon=1e-5):
    """Computes numeric approximation of squared loss gradient with respect to
    all model's optimized parameters.

    To calculate the gradient numeric approximation, the finite central differences
    method is used, residual corrections of which are O(e^2).
    """
    # Calculate loss
    loss = squared_loss(sarima_coefs, ann_coefs, ann_intercepts, ann_activation,
                        trend, trend_coefs, init_shocks, p, q, d, r, g, seasonal_p,
                        seasonal_q, seasonal_d, seasonal_r, seasonal_g, s, max_ar_lag,
                        max_ma_lag, sarima_exogs, ann_exogs, y, X, sarima_in, ann_in,
                        trend_in)

    # Calculate grad
    # SARIMA weights grad
    delta = np.zeros(sarima_coefs.shape)

    for i in range(sarima_coefs.shape[0]):
        delta[i] = epsilon
        grad_loss_total["sarima_coefs"][i] = squared_loss(
            sarima_coefs + delta, ann_coefs, ann_intercepts, ann_activation, trend,
            trend_coefs, init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
            seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs, y, X,
            sarima_in, ann_in, trend_in
        )
        grad_loss_total["sarima_coefs"][i] -= squared_loss(
            sarima_coefs - delta, ann_coefs, ann_intercepts, ann_activation, trend,
            trend_coefs, init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
            seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs, y, X,
            sarima_in, ann_in, trend_in
        )
        grad_loss_total["sarima_coefs"][i] /= 2. * epsilon
        delta[i] = 0.

    # Trend weights grad
    delta = np.zeros(trend_coefs.shape)

    for i in range(trend_coefs.shape[0]):
        delta[i] = epsilon
        grad_loss_total["trend_coefs"][i] = squared_loss(
            sarima_coefs, ann_coefs, ann_intercepts, ann_activation, trend,
            trend_coefs + delta, init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d,
            seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
            y, X, sarima_in, ann_in, trend_in
        )
        grad_loss_total["trend_coefs"][i] -= squared_loss(
            sarima_coefs, ann_coefs, ann_intercepts, ann_activation, trend,
            trend_coefs - delta, init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d,
            seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
            y, X, sarima_in, ann_in, trend_in
        )
        grad_loss_total["trend_coefs"][i] /= 2. * epsilon
        delta[i] = 0.

    # ANN weights grad
    if len(ann_coefs):
        delta = np.zeros(ann_coefs[-1].shape)

        for i in range(ann_coefs[-1].shape[0]):
            delta[i] = epsilon
            grad_loss_total["ann_coefs"][-1][i] = squared_loss(
                sarima_coefs, ann_coefs[:-1] + [ann_coefs[-1] + delta],
                ann_intercepts, ann_activation, trend, trend_coefs, init_shocks,
                p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r, seasonal_g,
                s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs, y, X, sarima_in,
                ann_in, trend_in
            )
            grad_loss_total["ann_coefs"][-1][i] -= squared_loss(
                sarima_coefs, ann_coefs[:-1] + [ann_coefs[-1] - delta],
                ann_intercepts, ann_activation, trend, trend_coefs, init_shocks,
                p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r, seasonal_g,
                s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs, y, X, sarima_in,
                ann_in, trend_in
            )
            grad_loss_total["ann_coefs"][-1][i] /= 2. * epsilon
            delta[i] = 0.

        for l in range(len(ann_coefs) - 1):
            delta_coef = np.zeros(ann_coefs[l].shape)
            delta_intercept = np.zeros(ann_intercepts[l].shape)

            for i in range(ann_coefs[l].shape[0]):
                for j in range(ann_coefs[l].shape[1]):
                    delta_coef[i, j] = epsilon
                    grad_loss_total["ann_coefs"][l][i, j] = squared_loss(
                        sarima_coefs,
                        ann_coefs[:l] + [ann_coefs[l] + delta_coef] + ann_coefs[l + 1:],
                        ann_intercepts, ann_activation, trend, trend_coefs,
                        init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
                        seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
                        y, X, sarima_in, ann_in, trend_in
                    )
                    grad_loss_total["ann_coefs"][l][i, j] -= squared_loss(
                        sarima_coefs,
                        ann_coefs[:l] + [ann_coefs[l] - delta_coef] + ann_coefs[l + 1:],
                        ann_intercepts, ann_activation, trend, trend_coefs,
                        init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
                        seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
                        y, X, sarima_in, ann_in, trend_in
                    )
                    grad_loss_total["ann_coefs"][l][i, j] /= 2. * epsilon
                    delta_coef[i, j] = 0.

                # Intercepts grad
                delta_intercept[i] = epsilon
                grad_loss_total["ann_intercepts"][l][i] = squared_loss(
                    sarima_coefs, ann_coefs,
                    ann_intercepts[:l] + [ann_intercepts[l] + delta_intercept] +
                    ann_intercepts[l + 1:], ann_activation, trend, trend_coefs,
                    init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
                    seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
                    y, X, sarima_in, ann_in, trend_in
                )
                grad_loss_total["ann_intercepts"][l][i] -= squared_loss(
                    sarima_coefs, ann_coefs,
                    ann_intercepts[:l] + [ann_intercepts[l] - delta_intercept] +
                    ann_intercepts[l + 1:], ann_activation, trend, trend_coefs,
                    init_shocks, p, q, d, r, g, seasonal_p, seasonal_q, seasonal_d, seasonal_r,
                    seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs, ann_exogs,
                    y, X, sarima_in, ann_in, trend_in
                )
                grad_loss_total["ann_intercepts"][l][i] /= 2. * epsilon
                delta_intercept[i] = 0.

    # Initial Shocks grad
    if optimize_init_shocks:
        delta = np.zeros(init_shocks.shape)

        for i in range(init_shocks.shape[0]):
            delta[i] = epsilon
            grad_loss_total["init_shocks"][i] = squared_loss(
                sarima_coefs, ann_coefs, ann_intercepts, ann_activation, trend,
                trend_coefs, init_shocks + delta, p, q, d, r, g, seasonal_p, seasonal_q,
                seasonal_d, seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs,
                ann_exogs, y, X, sarima_in, ann_in, trend_in
            )
            grad_loss_total["init_shocks"][i] -= squared_loss(
                sarima_coefs, ann_coefs, ann_intercepts, ann_activation, trend,
                trend_coefs, init_shocks - delta, p, q, d, r, g, seasonal_p, seasonal_q,
                seasonal_d, seasonal_r, seasonal_g, s, max_ar_lag, max_ma_lag, sarima_exogs,
                ann_exogs, y, X, sarima_in, ann_in, trend_in
            )
            grad_loss_total["init_shocks"][i] /= 2. * epsilon
            delta[i] = 0.

    return loss, grad_loss_total


# def _loss_grad(self, packed_params, args):
#     """(DEBUG VERSION)Compute the mean squared loss function (MSE) and its corresponding
#     derivatives with respect to the different parameters given in the initialization.
#
#     Returned gradients are packed in a single vector.
#
#     Parameters
#     ----------
#     packed_params : ndarray
#         A vector comprising the flattened coefficients, intercepts and initial shocks.
#
#     args : tuple
#         Arguments for the loss function.
#
#     Returns
#     -------
#     loss : float
#         Loss.
#     grad : ndarray
#         Packed in a single vector gradient.
#     """
#     global ITER_NUM
#     ITER_NUM += 1
#     logger.info(f"Start {ITER_NUM} iteration")
#
#     # Calculate analytic loss and gradient
#     self._unpack(packed_params)
#     loss, grad_loss_total = self._squared_loss_grad(*args)
#     logger.info(f"Loss: {loss}")
#     grad = self._pack(grad_loss_total)
#
#     # Calculate gradient norm
#     grad_norm = np.linalg.norm(grad)
#     logger.info(f"Gradient norm: {grad_norm}")
#     if grad_norm > 1e+10 or not np.isfinite(grad_norm):
#         warn_msg = (f"The gradient norm is {grad_norm}, so it looks like the"
#                     " gradient is exploding. To fix this try reinitializing the model"
#                     " or changing the architecture using other hyperparameters.")
#         logger.warning(warn_msg)
#         warnings.warn(warn_msg, RuntimeWarning)
#
#         # Take random step if gradient values is Inf or NaN
#         if not np.isfinite(grad_norm):
#             logger.info("The gradient norm is Inf or NaN, so a random step will be taken.")
#             grad = np.random.uniform(size=grad.shape)
#             grad *= self.max_grad_norm / np.linalg.norm(grad)
#             grad_norm = self.max_grad_norm
#
#     # Calculate numeric approximation of gradient
#     grad_analytic = grad_loss_total.copy()
#     y, X, sarima_in, ann_in, trend_in = args[0], args[1], args[-5].copy(), args[-4].copy(), args[-3]
#     loss_approx, grad_approx = \
#         squared_loss_grad_numapprox(self.sarima_coefs, self.ann_coefs, self.ann_intercepts,
#                                     self.ann_activation, self.trend, self.trend_coefs,
#                                     self.init_shocks, self.optimize_init_shocks,
#                                     self.p, self.q, self.d, self.r, self.g,
#                                     self.seasonal_p, self.seasonal_q, self.seasonal_d,
#                                     self.seasonal_r, self.seasonal_g, self.s, self.max_ar_lag,
#                                     self.max_ma_lag, self.sarima_exogs, self.ann_exogs,
#                                     y, X, grad_loss_total, sarima_in, ann_in, trend_in,
#                                     epsilon=1e-5)
#     logger.debug(f"Loss approx: {loss_approx}")
#
#     # Checking analytic and approximate gradients closeness
#     for key in grad_analytic:
#         if key == "ann_coefs" or key == "ann_intercepts":
#             for i in range(len(grad_analytic[key])):
#                 logger.debug("Maximum difference between analytic and approximate"
#                              f" gradients for {key} at {i}th level:"
#                              f" {(grad_analytic[key][i] - grad_approx[key][i]).max(initial=0)}")
#         else:
#             logger.debug("Maximum difference between analytic and approximate"
#                          f" gradients for {key}:"
#                          f" {(grad_analytic[key] - grad_approx[key]).max(initial=0)}")
#
#     # Normalize gradient
#     if np.floor(grad_norm) > self.max_grad_norm:
#         grad *= self.max_grad_norm / grad_norm
#
#     return loss, grad
