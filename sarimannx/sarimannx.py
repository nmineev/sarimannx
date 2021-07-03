"""Seasonal AutoRegressive Integrated MovingAverage Neural Network with eXogenous regressors
"""

import logging
import warnings
import time

import numpy as np
import scipy.optimize

from .utils import ACTIVATIONS, DERIVATIVES, Validation

validate = None
ITER_NUM = None

logger = logging.getLogger(__name__)


class SARIMANNX:
    """Seasonal AutoRegressive Integrated MovingAverage Neural Network with eXogenous regressors.

    This model optimizes the squared-loss (MSE) using LBFGS or other optimizers
    available in scipy.optimize.minimize(
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    ).

    Parameters
    ----------
    order : iterable, optional
        The (p, q, d, r, g) order of the model. All values must be an integers.
        Default is (1, 0, 0, 1, 0).

    seasonal_order : iterable, optional
        The (P, Q, D, R, G, s) order of the seasonal component of the model.
        D and s must be an integers, while P, Q, R and G may either be an integers
        or iterables of integers. s needed only for differencing, so all necessary
        seasonal lags must be specified explicitly.
        Default is no seasonal effect.

    ann_hidden_layer_sizes : iterable, optional
        The ith element represents the number of neurons in the ith hidden layer
        in ANN part of the model. All values must be an integers.
        Default is (10,).

    ann_activation : {"identity", "logistic", "tanh", "relu"}
        Activation function for the hidden layer in ANN part of the model.

        - "identity", no-op activation,
          returns f(x) = x

        - "logistic", the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - "tanh", the hyperbolic tan function,
          returns f(x) = tanh(x).

        - "relu", the rectified linear unit function,
          returns f(x) = max(0, x)

        Default is "tanh".

    trend : str{"n","c","t","ct"} or iterable, optional
        Parameter controlling the deterministic trend polynomial Trend(t).
        Can be specified as a string where "c" indicates a constant
        (i.e. a degree zero component of the trend polynomial), "t" indicates
        a linear trend with time, and "ct" is both. Can also be specified as
        an iterable defining the powers of t included in polynomial. For example,
        [1, 2, 0, 8] denotes a*t + b*t^2 + c + d*t^8.
        Default is to not include a trend component.

    optimize_init_shocks : bool, optional
        Whether to optimize first MAX_MA_LAG shocks as additional model
        parameters or assume them as zeros. If the sample size is relatively
        small, initial shocks optimization is more preferable.
        Default is True.

    grad_clip_value : int, optional
        Maximum allowed value of the gradients. The gradients are clipped in
        the range [-grad_clip_value, grad_clip_value]. Gradient clipping by
        value used for intermediate gradients, where gradient clipping by
        norm is not applicable. Clipping needed for fixing gradint explosion.
        Default is 1e+140.

    max_grad_norm : int, optional
        Maximum allowed norm of the final gradient. If the final gradient
        norm is greater, final gradient will be normalized and multiplied by
        max_grad_norm. Gradient clipping by norm used for final gradient to
        fix its explosion.
        Default is 10.

    logging_level : int, optional
        If logging is needed, firstly necessary to initialize logging config
        and then choose appropriate logging level for logging training progress.
        Without config no messages will be displayed at either logging level
        (Do not confuse with warning messages from warnings library which
        simply printing in stdout. For disable it use
        warnings.filterwarnings("ignore") for example). For more details
        see logging HOWTO(https://docs.python.org/3/howto/logging.html).
        Default is 30.

    solver : str, optional
        The solver for weights optimization. For a full list of available
        solvers, see scipy.optimize.minimize(
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        ).
        Default is "L-BFGS-B".

    **solver_kwargs
        Additional keyword agruments for the solver(For example, maximum
        number of iterations or optimization tolerance). For more details,
        see scipy.optimize.minimize(
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        ).
    Attributes
    ----------
    loss_ : float
        The final loss computed with the loss function.

    n_iter_ : int
        The number of iterations the solver has run.

    trend_coefs : numpy ndarray
        Trend polynomial coefficients.

    sarima_coefs : numpy ndarray
        Weights vector of SARIMA part of the model. First p coefficients
        corresponds to AR part, next len(P) corresponds to seasonal AR part,
        next q corresponds to MA part and last len(Q) coefficients corresponds
        to seasonal MA part.

    ann_coefs : list of numpy ndarrays
        Weights matrices of ANN part of the model. The ith element in the list
        represents the weight matrix corresponding to layer i.

    ann_intercepts : list of numpy ndarrays
        Bias vectors of ANN part of the model. The ith element in the list
        represents the bias vector corresponding to layer i + 1. Output layer has no bias.

    init_shocks : numpy ndarray
        The first MAX_MA_LAG shocks. If optimize_init_shocks is False, then after
        training they will be zeros, else initial shocks will be optimized as
        other model weights.

    max_ma_lag : int
        Highest moving average lag in the model.

    max_ar_lag : int
        Highest autoregressive lag in the model

    num_exogs : int
        Number of exogenous regressors used by model. Equals to X.shape[1] or
        len(sarima_exogs)+len(ann_exogs).

    train_score : float
        r2_score on training data after training.

    train_std : float
        Standard deviation of model residuals after training.

    Examples
    --------
    >>> import sys
    >>> import os
    >>> import numpy as np
    >>> import warnings
    >>> warnings.filterwarnings("ignore")
    >>> module_path = os.path.join(os.path.abspath("."), "sarimannx")
    >>> if module_path not in sys.path:
    ...     sys.path.append(module_path)
    ...
    >>> from sarimannx import SARIMANNX
    >>> np.random.seed(888)
    >>> y = np.random.normal(1., 1., size=(200,))
    >>> model = SARIMANNX(options={"maxiter": 500}).fit(y)
    >>> model.predict()
    1.093410687884555

    References
    ----------
    R. Hyndman, G. Athanasopoulos, Forecasting: principles and practice,
        3rd ed. Otexts, 2021, p. 442. Available: https://otexts.com/fpp3/

    I. Goodfellow, Y. Bengio, A. Courville, Deep Learning.
        The MIT Press, 2016, p. 800. Available: https://www.deeplearningbook.org/
    """
    def __init__(self, order=(1, 0, 0, 1, 0), seasonal_order=(0, 0, 0, 0, 0, 0),
                 ann_hidden_layer_sizes=10, ann_activation="tanh", trend="n",
                 optimize_init_shocks=True, grad_clip_value=1e+140, max_grad_norm=10,
                 logging_level=logging.WARNING, solver="L-BFGS-B", **solver_kwargs):
        # logger setting (NullHandler for no printing messages without config)
        logger.setLevel(logging_level)
        logger.addHandler(logging.NullHandler())

        # Define Validate object
        global validate
        validate = Validation(self)

        # Validate and assign orders and seasonal orders
        self.p, self.q, self.d, self.r, self.g = validate._order(order)
        self.seasonal_p, self.seasonal_q, self.seasonal_d, self.seasonal_r, \
        self.seasonal_g, self.s = validate._seasonal_order(seasonal_order)

        # Compute maximum AutoRegressive and MovingAverage lags
        self.max_ar_lag = max(self.p, self.r,
                              max(self.seasonal_p) if len(self.seasonal_p) else 0,
                              max(self.seasonal_r) if len(self.seasonal_r) else 0)
        self.max_ma_lag = max(self.q, self.g,
                              max(self.seasonal_q) if len(self.seasonal_q) else 0,
                              max(self.seasonal_g) if len(self.seasonal_g) else 0)

        # Validate and assign other hyperparameters
        self.ann_hidden_layer_sizes = validate._ann_hidden_layer_sizes(ann_hidden_layer_sizes)
        self.ann_activation = validate._ann_activation(ann_activation)
        self.trend = validate._trend(trend)
        self.optimize_init_shocks = validate._optimize_init_shocks(optimize_init_shocks)
        self.grad_clip_value = grad_clip_value
        self.max_grad_norm = max_grad_norm
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def _time_series_differencing(self, y):
        """Differentiates a time series accordingly to specified at initialization
        differencing first `d` and seasonal `D` orders and time series periodicity `s`.

        Parameters
        ----------
        y : ndarray of shape (nobs,)
            The input time series.

        Returns
        -------
        y : ndarray of shape (nobs - d - D * s,)
            The differentiated time series.
        first_ys : ndarray of shape (d,)
            Values of input y (or its combinations) needed for y first order undifferenicng.
        seasonal_ys : ndarray of shape (D, s)
            Values of input y (or its combinations) needed for y seasonal undifferenicng.
        """
        # First order differencing
        first_ys = np.zeros(self.d, dtype=y.dtype)
        for i in range(self.d):
            first_ys[i] = y[0]
            y = y[1:] - y[:-1]

        # Seasonal differencing
        seasonal_ys = np.zeros((self.seasonal_d, self.s), dtype=y.dtype)
        for i in range(self.seasonal_d):
            seasonal_ys[i] = y[:self.s]
            y = y[self.s:] - y[:-self.s]

        return y, first_ys, seasonal_ys

    def _time_series_undifferencing(self, y, first_ys, seasonal_ys):
        """Undifferentiates a time series using saved values of input original time series.

        Parameters
        ----------
        y : ndarray of shape (nobs - d - D * s,)
            The differentiated time series.

        first_ys : ndarray of shape (d,)
            Values of input y (or its combinations) needed for y first order undifferenicng.

        seasonal_ys : ndarray of shape (D, s)
            Values of input y (or its combinations) needed for y seasonal undifferenicng.

        Returns
        -------
        y : ndarray of shape (nobs,)
            The input original time series.
        """
        # Seasonal undifferencing
        for y_vals in seasonal_ys[::-1]:
            y = np.hstack((y_vals, y))
            for i in range(self.s, len(y)):
                y[i] += y[i - self.s]

        # First order undifferencing
        for y_val in first_ys[::-1]:
            y = np.hstack((y_val, y))
            for i in range(1, len(y)):
                    y[i] += y[i - 1]

        return y

    def _undifferencing_preds(self, preds, y, first_ys, seasonal_ys):
        """Undifferentiates predictions using input original differentiated time
        series and its saved for undifferencing original values.

        Parameters
        ----------
        preds : ndarray of shape (nobs - d - D * s,)
            One step y predictions.

        y : ndarray of shape (nobs - d - D * s,)
            The differenced time series.

        first_ys : ndarray of shape (d,)
            Values of input y (or its combinations) needed for y first order undifferenicng.

        seasonal_ys : ndarray of shape (D, s)
            Values of input y (or its combinations) needed for y seasonal undifferenicng.

        Returns
        -------
        preds : ndarray of shape (nobs,)
            One step undifferentiated y predictions.
        """
        for t in range(self.max_ar_lag, y.shape[0]):
            y_t = y[t]
            y[t] = preds[t]
            preds[t] = self._time_series_undifferencing(y[:t+1],
                                                        first_ys,
                                                        seasonal_ys)[-1]
            y[t] = y_t

        return np.hstack([
            self._time_series_undifferencing(y[:self.max_ar_lag],
                                             first_ys,
                                             seasonal_ys),
            preds[self.max_ar_lag:]
        ])

    def _ann_init_coef(self, fan_in, fan_out, factor, dtype):
        # Use the initialization method recommended by
        # Glorot et al.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = np.random.uniform(-init_bound, init_bound, (fan_out, fan_in))
        intercept_init = np.random.uniform(-init_bound, init_bound, fan_out)

        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)

        return coef_init, intercept_init

    def _init_weights_shocks(self, dtype):
        """Declares and initializes all model weights and initial shocks by
        Glorot et al. initialization method.
        """
        factor = 6.
        if self.ann_activation == 'logistic':
            factor = 2.

        # SARIMA coefs initialization
        fan_in = self.p + self.q + len(self.seasonal_p) + len(self.seasonal_q) + \
                 (self.num_exogs if self.sarima_exogs == slice(None) else len(self.sarima_exogs))
        fan_out = 1
        init_bound = np.sqrt(factor / (fan_in + fan_out))
        self.sarima_coefs = np.random.uniform(-init_bound, init_bound, size=fan_in)
        self.sarima_coefs = self.sarima_coefs.astype(dtype, copy=False)

        # Trend coefs initialization
        self.trend_coefs = np.zeros(self.trend.shape, dtype=dtype)

        # ANN coefs and intercepts initialization
        num_ann_exogs = self.num_exogs if self.ann_exogs == slice(None) else len(self.ann_exogs)
        if self.r + self.g + len(self.seasonal_r) + len(self.seasonal_g) + num_ann_exogs:
            layer_units = [self.r + self.g + len(self.seasonal_r) +
                           len(self.seasonal_g) + num_ann_exogs] + \
                          list(self.ann_hidden_layer_sizes) + [1]
        else:
            layer_units = []

        self.ann_coefs = [None] * (len(layer_units) - 1)
        self.ann_intercepts = [None] * (len(layer_units) - 2)

        if len(self.ann_coefs):
            fan_in = layer_units[-2]
            fan_out = layer_units[-1]
            init_bound = np.sqrt(factor / (fan_in + fan_out))
            self.ann_coefs[-1] = np.random.uniform(-init_bound, init_bound, size=fan_in)
            self.ann_coefs[-1] = self.ann_coefs[-1].astype(dtype, copy=False)

            for i in range(len(layer_units) - 2):
                fan_in = layer_units[i]
                fan_out = layer_units[i + 1]
                self.ann_coefs[i], self.ann_intercepts[i] = \
                    self._ann_init_coef(fan_in, fan_out, factor, dtype)

        # Initial Shocks initialization
        self.init_shocks = np.zeros(self.max_ma_lag, dtype=dtype)

    def _save_indices(self):
        """Save start-end indices (for `packed_params` vector) and shapes of
        coefficients for faster unpacking.
        """
        # SARIMA indices and size
        start = 0
        end = self.sarima_coefs.shape[0]
        self.sarima_coefs_indptr = start, end
        start = end

        # ANN indices and size
        if len(self.ann_coefs):
            self.ann_coefs_indptr = [None] * len(self.ann_coefs)
            self.ann_intercepts_indptr = [None] * len(self.ann_intercepts)

            for i in range(len(self.ann_coefs) - 1):
                fan_out, fan_in = self.ann_coefs[i].shape
                end = start + (fan_in * fan_out)
                self.ann_coefs_indptr[i] = start, end, (fan_out, fan_in)
                start = end

            end = start + self.ann_coefs[-1].shape[0]
            self.ann_coefs_indptr[-1] = start, end
            start = end

            for i in range(len(self.ann_intercepts)):
                end = start + self.ann_intercepts[i].shape[0]
                self.ann_intercepts_indptr[i] = start, end
                start = end

        # Trend indptr
        end = start + self.trend_coefs.shape[0]
        self.trend_coefs_indptr = start, end
        start = end

        # Initial Shocks indptr
        if self.optimize_init_shocks:
            end = start + self.init_shocks.shape[0]
            self.init_shocks_indptr = start, end

    def _initialize(self, y, X, init_weights_shocks=True):
        """Prepares for model fitting."""
        # Declare and initialize all model weights and initial shocks
        if init_weights_shocks:
            self._init_weights_shocks(y.dtype)

        # Save start-end indices (for `packed_params` vector) and shapes of
        # coefficients for faster unpacking
        if not init_weights_shocks and not hasattr(self, "sarima_coefs"):
            raise ValueError("Model optimization parameters are not initialized."
                             " Change `init_weights_shocks` to `True` or define"
                             " specific parameters via `set_params` method.")
        else:
            self._save_indices()

        # Initialization of loss-function args
        # Total gradient of loss
        grad_loss_total = {"sarima_coefs":
                               np.zeros(self.sarima_coefs.shape, dtype=y.dtype),
                           "ann_coefs":
                               [np.zeros(l.shape, dtype=y.dtype) for l in self.ann_coefs],
                           "ann_intercepts":
                               [np.zeros(l.shape, dtype=y.dtype) for l in self.ann_intercepts],
                           "trend_coefs":
                               np.zeros(self.trend_coefs.shape, dtype=y.dtype)}

        # Gradient of loss on current `t` iteration
        grad_loss_curr = {"sarima_coefs": None,
                          "ann_coefs": [None] * len(self.ann_coefs),
                          "ann_intercepts": [None] * len(self.ann_intercepts),
                          "trend_coefs": None}

        # History of past (on `t-1`, ..., `t-max_ma_lag` iterations) gradients
        # of pred with respect to weights
        if self.max_ma_lag:
            grad_pred_weights_past = {
                "sarima_coefs":
                    np.zeros((self.max_ma_lag,) + self.sarima_coefs.shape, dtype=y.dtype)
                    if len(self.sarima_coefs) else np.array([], dtype=y.dtype),
                "ann_coefs":
                    [np.zeros((self.max_ma_lag,) + l.shape, dtype=y.dtype)
                     for l in self.ann_coefs],
                "ann_intercepts":
                    [np.zeros((self.max_ma_lag,) + l.shape, dtype=y.dtype)
                     for l in self.ann_intercepts],
                "trend_coefs":
                    np.zeros((self.max_ma_lag,) + self.trend_coefs.shape, dtype=y.dtype)
                    if len(self.trend_coefs) else np.array([], dtype=y.dtype)
            }
        else:
            grad_pred_weights_past = None

        # Gradient of pred with respect to input shocks
        grad0_pred_shocks = np.zeros(self.max_ma_lag, dtype=y.dtype)

        if self.optimize_init_shocks:
            grad_loss_total["init_shocks"] = np.zeros(self.init_shocks.shape, dtype=y.dtype)
            # History of past (on `t-1`, ..., `t-max_ma_lag` iters) gradients of
            # pred with respect to init shocks
            grad_pred_init_shocks_past = np.identity(self.max_ma_lag, dtype=y.dtype)
        else:
            grad_pred_init_shocks_past = None

        # Input vectors
        sarima_in = np.empty(
            self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q) +
            (self.num_exogs if self.sarima_exogs == slice(None) else len(self.sarima_exogs)),
            dtype=y.dtype)
        ann_in = np.empty(
            self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g) +
            (self.num_exogs if self.ann_exogs == slice(None) else len(self.ann_exogs)),
            dtype=y.dtype)
        trend_in = None

        # ANN hidden units and activations lists
        ann_hid_units = [None] * len(self.ann_coefs)
        ann_activations = [None] * (len(self.ann_coefs) - 1)

        return (y, X, grad_loss_total, grad_loss_curr, grad_pred_weights_past,
                grad0_pred_shocks, grad_pred_init_shocks_past, sarima_in, ann_in,
                trend_in, ann_hid_units, ann_activations)

    def _ann_forward_pass(self, hid_units, activations):
        """Perform a forward pass on the ANN part of SARIMANNX by computing the
        values of the neurons in the hidden and output layers.

        Parameters
        ----------
        hid_units : list, length = len(ann_coefs)
            The ith element of the list holds the values of the ith layer.

        activations : list, length = len(ann_coefs) - 1
            The ith element of the list holds the values of the (i+1)th layer activations
            (after applying to it activaion function we will get (i+1)th hid_units).
        """
        hid_activation = ACTIVATIONS[self.ann_activation]
        for i in range(len(self.ann_coefs) - 1):
            activations[i] = np.dot(self.ann_coefs[i], hid_units[i])
            activations[i] += self.ann_intercepts[i]
            hid_units[i + 1] = hid_activation(activations[i])
        return hid_units, activations

    def _ann_backprop(self, delta, hid_units, activations, coef_grads, intercept_grads):
        """Perform a backward pass on the ANN part of SARIMANNX by computing the
        loss function derivatives with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        delta : float(ndarray)
            At ith iteration it is the gradient of loss with respect to (i+1)th layer
            activations.

        hid_units : list, length = len(ann_coefs)
            The ith element of the list holds the values of the ith layer.

        activations : list, length = len(ann_coefs) - 1
            The ith element of the list holds the values of the (i+1)th layer activations
            (after applying to it activaion function we will get (i+1)th hid_units).

        coef_grads : list, length = len(ann_coefs)
            The ith element contains derivatives of loss with respect to ith
            layer weights.

        intercept_grads : list, length = len(ann_coefs) - 1
            The ith element contains derivatives of loss with respect to ith
            layer biases(last layer has no bias).
        """
        derivative = DERIVATIVES[self.ann_activation]
        for i in range(len(self.ann_coefs) - 2, -1, -1):
            delta = derivative(hid_units[i + 1]) * np.dot(delta, self.ann_coefs[i + 1])
            coef_grads[i] = np.tensordot(delta, hid_units[i], axes=0)
            intercept_grads[i] = delta
        return coef_grads, intercept_grads, delta

    def _compute_total_grad(self, shock, grad_pred_shocks, grad_loss_curr,
                            grad_loss_total, grad_pred_weights_past):
        """Complete the current gradient to total current gradient and add it
        to total loss gradient.

        Completes the current gradient to total current by subtracting
        multiplications of derivatives of past preds with respect to weights
        and derivatives of current pred with respect to input shocks and then
        adding it to total loss gradient.

        Parameters
        ----------
        shock : float
            true_t - pred_t (divided by nobs)

        grad_pred_shocks : ndarray of shape (max_ma_lag,)
            The derivatives of pred_t with respect to input shocks.

        grad_loss_curr : dict of ndarrays
            Dict which contains derivatives of loss with respect to weights(of
            SARIMANNX parts corresponding to keys) for current `t` iteration.

        grad_loss_total : dict of ndarrays
            Dict which contains sum of derivatives of loss with respect to weights(of
            SARIMANNX parts corresponding to keys) calculated from `0` to `t-1` iterations.

        grad_pred_weights_past : dict of ndarrays
            Dict which contains total derivatives of pred_{t-1}, ..., pred_{t-max_ma_lag}
            with respect to weights(of SARIMANNX parts corresponding to keys).
        """
        # SARIMA
        if len(self.sarima_coefs):
            if self.max_ma_lag:
                # Complete derivative to total
                grad_loss_curr["sarima_coefs"] += \
                    shock * np.tensordot(grad_pred_weights_past["sarima_coefs"],
                                         grad_pred_shocks,
                                         axes=([0], [0]))

                # Clip gradient values if necessary to fix gradient explosion
                np.clip(grad_loss_curr["sarima_coefs"], -self.grad_clip_value,
                        self.grad_clip_value, out=grad_loss_curr["sarima_coefs"])

                # Add current total derivative to history
                # TODO: Need to redesign because this way is too long
                grad_pred_weights_past["sarima_coefs"][1:] = \
                    grad_pred_weights_past["sarima_coefs"][:-1]
                grad_pred_weights_past["sarima_coefs"][0] = \
                    -1 * grad_loss_curr["sarima_coefs"] / shock

            # Sum gradient
            grad_loss_total["sarima_coefs"] += grad_loss_curr["sarima_coefs"]

        # Trend
        if len(self.trend_coefs):
            if self.max_ma_lag:
                # Complete derivative to total
                grad_loss_curr["trend_coefs"] += \
                    shock * np.tensordot(grad_pred_weights_past["trend_coefs"],
                                         grad_pred_shocks,
                                         axes=([0], [0]))

                # Clip gradient values if necessary to fix gradient explosion
                np.clip(grad_loss_curr["trend_coefs"], -self.grad_clip_value,
                        self.grad_clip_value, out=grad_loss_curr["trend_coefs"])

                # Add current total derivative to history(Need to redesign it)
                grad_pred_weights_past["trend_coefs"][1:] = \
                    grad_pred_weights_past["trend_coefs"][:-1]
                grad_pred_weights_past["trend_coefs"][0] = \
                    -1 * grad_loss_curr["trend_coefs"] / shock

            # Sum gradient
            grad_loss_total["trend_coefs"] += grad_loss_curr["trend_coefs"]

        # ANN
        if len(self.ann_coefs):
            # For the last layer
            if self.max_ma_lag:
                # Complete derivative to total
                grad_loss_curr["ann_coefs"][-1] += \
                    shock * np.tensordot(grad_pred_weights_past["ann_coefs"][-1],
                                         grad_pred_shocks,
                                         axes=([0], [0]))

                # Clip gradient values if necessary to fix gradient explosion
                np.clip(grad_loss_curr["ann_coefs"][-1], -self.grad_clip_value,
                        self.grad_clip_value, out=grad_loss_curr["ann_coefs"][-1])

                # Add current total derivative to history(Need to redesign it)
                grad_pred_weights_past["ann_coefs"][-1][1:] = \
                    grad_pred_weights_past["ann_coefs"][-1][:-1]
                grad_pred_weights_past["ann_coefs"][-1][0] = \
                    -1 * grad_loss_curr["ann_coefs"][-1] / shock

            # Sum gradient
            grad_loss_total["ann_coefs"][-1] += grad_loss_curr["ann_coefs"][-1]

            # For other layers
            for i in range(len(self.ann_coefs) - 1):
                if self.max_ma_lag:
                    # Complete derivative to total
                    grad_loss_curr["ann_coefs"][i] += \
                        shock * np.tensordot(grad_pred_weights_past["ann_coefs"][i],
                                             grad_pred_shocks,
                                             axes=([0], [0]))

                    # Clip gradient values if necessary to fix gradient explosion
                    np.clip(grad_loss_curr["ann_coefs"][i], -self.grad_clip_value,
                            self.grad_clip_value, out=grad_loss_curr["ann_coefs"][i])

                    # Add current total derivative to history(Need to redesign it)
                    grad_pred_weights_past["ann_coefs"][i][1:] = \
                        grad_pred_weights_past["ann_coefs"][i][:-1]
                    grad_pred_weights_past["ann_coefs"][i][0] = \
                        -1 * grad_loss_curr["ann_coefs"][i] / shock

                    # Complete derivative to total
                    grad_loss_curr["ann_intercepts"][i] += \
                        shock * np.tensordot(grad_pred_weights_past["ann_intercepts"][i],
                                             grad_pred_shocks,
                                             axes=([0], [0]))

                    # Clip gradient values if necessary to fix gradient explosion
                    np.clip(grad_loss_curr["ann_intercepts"][i], -self.grad_clip_value,
                            self.grad_clip_value, out=grad_loss_curr["ann_intercepts"][i])

                    # Add current total derivative to history(Need to redesign it)
                    grad_pred_weights_past["ann_intercepts"][i][1:] = \
                        grad_pred_weights_past["ann_intercepts"][i][:-1]
                    grad_pred_weights_past["ann_intercepts"][i][0] = \
                        -1 * grad_loss_curr["ann_intercepts"][i] / shock

                # Sum gradients
                grad_loss_total["ann_coefs"][i] += grad_loss_curr["ann_coefs"][i]
                grad_loss_total["ann_intercepts"][i] += grad_loss_curr["ann_intercepts"][i]

        return grad_loss_curr, grad_loss_total, grad_pred_weights_past

    def _squared_loss_grad(self, y, X, grad_loss_total, grad_loss_curr, grad_pred_weights_past,
                           grad0_pred_shocks, grad_pred_init_shocks_past, sarima_in,
                           ann_in, trend_in, ann_hid_units, ann_activations):
        """Compute the mean squared loss function (MSE) and its corresponding
        derivatives with respect to each parameter: weights, bias and
        (optionally) init shocks vectors.

        Parameters
        ----------
        y : ndarray of shape (nobs - d - D * s,)
            The (differenced) time series.

        X : ndarray of shape (nobs, number of exogenous regressors)
            The exogenous regressors. Must have number of rows equals to number
            of observations in original undiffereced y.

        grad_loss_total : dict of ndarrays
            Total gradient of loss.

        grad_loss_curr : dict of ndarrays
            Gradient of loss on current `t` iteration.

        grad_pred_weights_past : dict of ndarrays
            History of past (on `t-1`, ..., `t-max_ma_lag` iterations) gradients
            of pred with respect to weights

        grad0_pred_shocks : ndarray of shape (max_ma_lag,)
            Gradient of pred with respect to input shocks

        grad_pred_init_shocks_past : ndarray of shape (max_ma_lag, max_ma_lag)
            History of past (on `t-1`, ..., `t-max_ma_lag` iters) gradients of
            pred with respect to init shocks

        sarima_in : ndarray of shape (p + q + len(P) + len(Q) + len(sarima_exogs),)
            SARIMA input.

        ann_in : ndarray of shape (r + g + len(R) + len(G) + len(ann_exogs),)
            ANN input.

        trend_in : None
            Trend input.

        ann_hid_units : list, length = len(ann_coefs)
            The ith element of the list holds the values of the ith layer.

        ann_activations : list, length = len(ann_coefs) - 1
            The ith element of the list holds the values of the (i+1)th layer activations
            (after applying to it activaion function we get (i+1)th hid_units).

        Returns
        -------
        loss : float
            Loss.
        total_loss_grad : dict
            Total gradient of loss.
        """
        # Zeroing gradients from previous optimizer iteration
        for key in ["sarima_coefs", "ann_coefs", "ann_intercepts", "trend_coefs"]:
            if key == "ann_coefs" or key == "ann_intercepts":
                for i in range(len(grad_loss_total[key])):
                    grad_loss_total[key][i] *= 0
                    if self.max_ma_lag:
                        grad_pred_weights_past[key][i] *= 0
            else:
                grad_loss_total[key] *= 0
                if self.max_ma_lag:
                    grad_pred_weights_past[key] *= 0

        if self.optimize_init_shocks:
            grad_loss_total["init_shocks"] *= 0
            grad_pred_init_shocks_past *= 0
            np.fill_diagonal(grad_pred_init_shocks_past, 1)

        grad0_pred_shocks *= 0
        grad_pred_shocks = grad0_pred_shocks.copy()

        # Precompute gradient of pred with respect to SARIMA input shocks
        for i in range(self.q):
            grad0_pred_shocks[i] = \
                self.sarima_coefs[self.p + len(self.seasonal_p) + i]
        for i in range(len(self.seasonal_q)):
            grad0_pred_shocks[self.seasonal_q[i] - 1] = \
                self.sarima_coefs[self.p + len(self.seasonal_p) + self.q + i]

        shocks = self.init_shocks.copy() # store shocks in reverse time order
        loss = 0                         # (so by 0 index holds last t-1 shock, by -1 index holds t - max_ma_lag shock)
        for t in range(self.max_ar_lag, y.shape[0]):
            # SARIMA input
            sarima_in[:self.p] = y[t - 1:t - self.p - 1 if t != self.p else None:-1]
            sarima_in[self.p:
                      self.p + len(self.seasonal_p)] = y[[t - i for i in self.seasonal_p]]
            sarima_in[self.p + len(self.seasonal_p):
                      self.p + len(self.seasonal_p) + self.q] = shocks[:self.q]
            sarima_in[self.p + len(self.seasonal_p) + self.q:
                      self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q)] = \
                shocks[[i - 1 for i in self.seasonal_q]]
            if self.num_exogs:
                sarima_in[self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q):] = \
                    X[t + self.d + self.seasonal_d * self.s, self.sarima_exogs]  # Add eXogs
            # Trend input
            trend_in = np.power(t + self.s * self.seasonal_d + self.d, self.trend, dtype=y.dtype)
            # ANN input
            ann_in[:self.r] = y[t - 1:t - self.r - 1 if t != self.r else None:-1]
            ann_in[self.r:
                   self.r + len(self.seasonal_r)] = y[[t - i for i in self.seasonal_r]]
            ann_in[self.r + len(self.seasonal_r):
                   self.r + len(self.seasonal_r) + self.g] = shocks[:self.g]
            ann_in[self.r + len(self.seasonal_r) + self.g:
                   self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g)] = \
                shocks[[i - 1 for i in self.seasonal_g]]
            if self.num_exogs:
                ann_in[self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g):] = \
                    X[t + self.d + self.seasonal_d * self.s, self.ann_exogs]  # Add eXogs

            if len(ann_hid_units):
                ann_hid_units[0] = ann_in

            # ANN Forward propagate
            ann_hid_units, ann_activations = self._ann_forward_pass(ann_hid_units,
                                                                    ann_activations)

            # Calculate loss
            pred = np.dot(self.sarima_coefs, sarima_in) + np.dot(self.trend_coefs, trend_in)
            if len(self.ann_coefs): pred += np.dot(self.ann_coefs[-1], ann_hid_units[-1])
            shock = y[t] - pred
            shocks[1:] = shocks[:-1]
            if len(shocks): shocks[0] = shock
            loss += shock ** 2 / 2. / y.shape[0]

            # Calculate grad
            delta = -1 * shock / y.shape[0]
            # SARIMA grad (not total yet if max_ma_lag not equal 0)
            grad_loss_curr["sarima_coefs"] = delta * sarima_in
            # Trend grad (not total yet if max_ma_lag not equal 0)
            grad_loss_curr["trend_coefs"] = delta * trend_in
            # ANN grad (not total yet if max_ma_lag not equal 0)
            if len(self.ann_coefs):
                grad_loss_curr["ann_coefs"][-1] = delta * ann_hid_units[-1]
                grad_loss_curr["ann_coefs"], grad_loss_curr["ann_intercepts"], delta = \
                    self._ann_backprop(delta, ann_hid_units, ann_activations,
                                       grad_loss_curr["ann_coefs"],
                                       grad_loss_curr["ann_intercepts"])

            # gradient of pred with respect to input shocks
            grad_pred_shocks[:] = grad0_pred_shocks[:]
            if len(self.ann_coefs):
                # Getting row if ann_coefs[0]==ann_coefs[-1], i.e.
                # is the last layer weights vector, else getting column
                slice_ = tuple() if self.ann_coefs[0].ndim == 1 else ...,
                for i in range(self.g):
                    grad_pred_shocks[i] -= \
                        (np.dot(delta, self.ann_coefs[0][slice_ + (self.r + len(self.seasonal_r) + i,)]) *
                         y.shape[0] / shock)
                for i in range(len(self.seasonal_g)):
                    grad_pred_shocks[self.seasonal_g[i] - 1] -= \
                        (np.dot(delta, self.ann_coefs[0][slice_ + (self.r + len(self.seasonal_r) + self.g + i,)]) *
                         y.shape[0] / shock)

            # Complete gradient to total gradient
            grad_loss_curr, grad_loss_total, grad_pred_weights_past = \
                self._compute_total_grad(shock / y.shape[0], grad_pred_shocks, grad_loss_curr,
                                         grad_loss_total, grad_pred_weights_past)

            # Initial Shocks gradient
            if self.optimize_init_shocks and self.max_ma_lag:
                # gradient of pred with respect to init shocks
                grad_pred_init_shocks = -1 * np.dot(grad_pred_init_shocks_past,
                                                    grad_pred_shocks)

                # Clip gradient values if necessary to fix gradient explosion
                np.clip(grad_pred_init_shocks, -self.grad_clip_value,
                        self.grad_clip_value, out=grad_pred_init_shocks)

                # Add gradient to history (Need to redesign it)
                grad_pred_init_shocks_past[:, 1:] = grad_pred_init_shocks_past[:, :-1]
                grad_pred_init_shocks_past[:, 0] = grad_pred_init_shocks
                # Sum gradients
                grad_loss_total["init_shocks"] += shock * grad_pred_init_shocks / y.shape[0]

        return loss, grad_loss_total

    def _pack(self, grad_loss_total=None):
        """Pack the parameters into a single vector.

        If grad_loss_total is not provided, pack model parameters,
        else pack gradient.
        """
        if grad_loss_total is None:
            pack_list = [self.sarima_coefs] + [l.ravel() for l in self.ann_coefs] + \
                        self.ann_intercepts + [self.trend_coefs]
            if self.optimize_init_shocks:
                pack_list += [self.init_shocks]
        else:
            pack_list = [grad_loss_total["sarima_coefs"]] + \
                        [l.ravel() for l in grad_loss_total["ann_coefs"]] + \
                        grad_loss_total["ann_intercepts"] + \
                        [grad_loss_total["trend_coefs"]]
            if self.optimize_init_shocks:
                pack_list += [grad_loss_total["init_shocks"]]

        return np.hstack(pack_list)

    def _unpack(self, packed_params):
        """Extract the coefficients and intercepts from packed parameters."""
        # SARIMA unpack
        start, end = self.sarima_coefs_indptr
        self.sarima_coefs = packed_params[start:end]

        # Trend unpack
        start, end = self.trend_coefs_indptr
        self.trend_coefs = packed_params[start:end]

        # ANN unpack
        if len(self.ann_coefs):
            start, end = self.ann_coefs_indptr[-1]
            self.ann_coefs[-1] = packed_params[start:end]

            for i in range(len(self.ann_coefs) - 1):
                start, end, shape = self.ann_coefs_indptr[i]
                self.ann_coefs[i] = np.reshape(packed_params[start:end], shape)

                start, end = self.ann_intercepts_indptr[i]
                self.ann_intercepts[i] = packed_params[start:end]

        # Initial Shocks unpack
        if self.optimize_init_shocks:
            start, end = self.init_shocks_indptr
            self.init_shocks = packed_params[start:end]

    def _loss_grad(self, packed_params, args):
        """Compute the mean squared loss function (MSE) and its corresponding
        derivatives with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector.

        Parameters
        ----------
        packed_params : ndarray
            A vector comprising the flattened coefficients, intercepts and initial shocks.

        args : tuple
            Arguments for the loss function.

        Returns
        -------
        loss : float
            Loss.
        grad : ndarray
            Packed in a single vector gradient.
        """
        global ITER_NUM
        ITER_NUM += 1
        logger.info(f"Start {ITER_NUM} iteration")

        # Calculate analytic loss and gradient
        self._unpack(packed_params)
        loss, grad_loss_total = self._squared_loss_grad(*args)
        logger.info(f"Loss: {loss}")
        grad = self._pack(grad_loss_total)

        # Calculate gradient norm
        grad_norm = np.linalg.norm(grad)
        logger.info(f"Gradient norm: {grad_norm}")
        if not np.isfinite(grad_norm) or grad_norm > 1e+10:
            warn_msg = (f"The gradient norm is {grad_norm}, so it looks like the"
                        " gradient is exploding. To fix this try reinitializing the model"
                        " or changing the architecture using other hyperparameters.")
            logger.warning(warn_msg)
            warnings.warn(warn_msg, RuntimeWarning)

            # Take random step if gradient values is Inf or NaN
            if not np.isfinite(grad_norm):
                logger.info("The gradient norm is Inf or NaN, so a random step will be taken.")
                grad = np.random.uniform(size=grad.shape)
                grad *= self.max_grad_norm / np.linalg.norm(grad)
                grad_norm = self.max_grad_norm

        # Normalize gradient
        if np.floor(grad_norm) > self.max_grad_norm:
            grad *= self.max_grad_norm / grad_norm

        return loss, grad

    def fit(self, y, X=None, sarima_exogs=slice(None), ann_exogs=slice(None),
            dtype=float, init_weights_shocks=True, return_preds_resids=False):
        """Fits the model to time series data y.

        Parameters
        ----------
        y : ndarray of shape (nobs,)
            Training time series data.

        X : ndarray, optional
            Matrix of exogenous regressors. If provided, it must be shaped to
            (nobs, k), where k is number of regressors.
            Default is no exogenous regressors in the model.

        sarima_exogs : iterable, optional
            Specify regressors which will be included in SARIMA input by specifying
            columns indices of the X matrix.
            Default is all provided regressors will be included.

        ann_exogs : iterable, optional
            Specify regressors which will be included in ANN input by specifying
            columns indices of the X matrix.
            Default is all provided regressors will be included.

        dtype : dtype, optional
            Data type in which input data y and X will be converted before training.
            Default is numpy float64

        init_weights_shocks : bool, optional
            Wheter or not to initialize all model trained parameters. If this is
            the first time calling fit method and parameters did not specified via
            set_params method then init_weights_shocks must be set to True.
            Default is True.

        return_preds_resids : bool, optional
            Wheter or not to return all one step predictions along y and
            corresponding residuals together with trained model. If true,
            returns self and python dictionary.
            Default is False.

        Returns
        -------
        self : return a trained SARIMANNX model.

        python dictionary
            Returns python dict with keys "predictions" and "residuals" and
            corresponding numpy ndarrays, if return_preds_resids was set to True.
        """
        # Validate input
        y, X, sarima_exogs, ann_exogs, dtype, init_weights_shocks, return_preds_resids = \
            validate._fit_input(y, X, sarima_exogs, ann_exogs, dtype,
                                init_weights_shocks, return_preds_resids)

        self.sarima_exogs = sarima_exogs
        self.ann_exogs = ann_exogs

        # Define number of used exogenous regressors
        if X.shape[0] == 0 or X.ndim != 2:
            self.num_exogs = 0
        elif self.sarima_exogs == slice(None) or self.ann_exogs == slice(None):
            self.num_exogs = X.shape[1]
        else:
            self.num_exogs = max(len(self.sarima_exogs), len(self.ann_exogs))

        # Save last train info for predicting
        self.last_train_ys = y[-(self.max_ar_lag + self.s * self.seasonal_d + self.d):]
        self.last_train_t = y.shape[0]-1

        # Differencing a time series
        y, first_ys, seasonal_ys = self._time_series_differencing(y)

        # Initialize args for _loss_grad func for optimizing
        args = self._initialize(y, X, init_weights_shocks)

        # Pack all model's optimization initial params
        packed_params = self._pack()

        # Run the solver
        global ITER_NUM
        ITER_NUM = 0
        logger.info("Start Fitting")
        ts = time.time()

        opt_res = scipy.optimize.minimize(
            self._loss_grad, packed_params, args=(args,), method=self.solver, jac=True,
            **self.solver_kwargs
        )

        ITER_NUM = 0
        logger.info(f"Fitting complete in {time.time() - ts:.3f}s")

        self.n_iter_ = validate._check_optimize_result(opt_res)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

        # Get one step predictions and shocks with optimized parameters
        preds, shocks = self._full_forward_pass_fast(y, X[self.d+self.s*self.seasonal_d+self.max_ar_lag:],
                                                     None, self.last_train_t)

        # Save last train info for predicting
        self.last_train_shocks = shocks[-1:-self.max_ma_lag-1:-1]

        # Undifferentiate y and predictions
        preds = self._undifferencing_preds(preds, y, first_ys, seasonal_ys)
        y = self._time_series_undifferencing(y, first_ys, seasonal_ys)

        # Calculate residuals, its standard deviation and R^2 score
        residuals = y - preds

        ss_res = (residuals**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        if ss_tot == 0:
            if ss_res == 0:
                self.train_score = 0.
            else:
                self.train_score = -np.inf
        else:
            self.train_score = 1. - ss_res / ss_tot

        self.train_std = residuals.std()

        if return_preds_resids:
            return self, {"predictions": preds, "residuals": residuals}
        return self

    def _forward_pass_fast(self, sarima_in, ann_in, trend_in):
        """Predict using the trained SARIMANNX model.

        Function just substitute inputs to model, get prediction and return it.

        Parameters
        ----------
        sarima_in : ndarray of shape (p + q + len(P) + len(Q) + len(sarima_exogs),)
            The SARIMA input.

        ann_in : ndarray of shape (r + g + len(R) + len(G) + len(ann_exogs),)
            The ANN input.

        trend_in : ndarray of shape (len(trend),)
            The Trend input.

        Returns
        -------
        pred : float
            Prediction.
        """
        # ANN fast forward pass
        ann_hid_unit = self._ann_forward_pass_fast(ann_in)
        # Calculate prediction
        pred = np.dot(self.sarima_coefs, sarima_in) + np.dot(self.trend_coefs, trend_in)
        if len(self.ann_coefs):
            pred += np.dot(self.ann_coefs[-1], ann_hid_unit)
        return pred

    def _ann_forward_pass_fast(self, ann_in):
        """Perform a fast forward pass on the ANN part of SARIMANNX.

        This is the same as _ann_forward_pass but does not record the activations
        and hid_units of all layers and only returns
        the last hidden units(after dot product it on last layer ANN weights we will get ANN prediction)

        Parameters
        ----------
        ann_in : ndarray of shape (r + g + len(R) + len(G) + len(ann_exogs),)
            The ANN input.

        Returns
        -------
        ann_hid_unit : ndarray of shape ann_coefs[-1].shape
            The last hidden units(after dot product it on last layer ANN weights we will get ANN prediction).
        """
        # ANN fast forward pass
        ann_hid_unit = ann_in.copy()
        ann_hid_activation = ACTIVATIONS[self.ann_activation]
        for i in range(len(self.ann_coefs) - 1):
            ann_hid_unit = ann_hid_activation(
                np.dot(self.ann_coefs[i], ann_hid_unit) + self.ann_intercepts[i]
            )
        return ann_hid_unit

    def _full_forward_pass_fast(self, y, X=None, input_shocks=None, last_t=None):
        """Collect all one step predictions along provided data using the trained model.

        Compute one step prediction for t = max_ar_lag..y.shape[0], store it and
        store corresponding shocks.

        Parameters
        ----------
        y : ndarray of shape (nobs,)
            The time series. First max_ar_lag - 1 values will be saved in preds_hist,
            then preds_hist will be appended with predictions.

        X : None or ndarray of shape (nobs - max_ar_lag, number of exog regressors)
            The exogenous regressors. Time moments of X[i-max_ar_lag] row values and y[i]
            are equal for i=max_ar_lag..y.shape[0]-1.

        input_shocks : None or ndarray of shape (max_ma_lag,)
            The input shocks in reversed time order, needed to find first,
            corresponding to y[max_ar_lag], prediction. input_shocks[0] holds last
            t-1 input shock, input_shocks[-1] holds t-max_ma_lag shock. If not provided,
            init_shocks will be used as input shocks.

        last_t : None or int
            The time moment corresponding to last y value. If not provided, defines
            as y.shape[0] - 1.

        Returns
        -------
        preds_hist : ndarray of shape (y.shape[0],)
            The all one step predictions alog y.
        shocks_hist : ndarray of shape (y.shape[0] - max_ar_lag + max_ma_lag)
            The input_shocks and shocks corresponding to each prediction in preds_hist.
        """
        if input_shocks is None:
            shocks = self.init_shocks.copy()
        else:
            shocks = input_shocks.copy()

        if last_t is None:
            last_t = y.shape[0] - 1

        # Input vectors
        sarima_in = np.empty(
            self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q) +
            (self.num_exogs if self.sarima_exogs == slice(None) else len(self.sarima_exogs)),
            dtype=y.dtype)
        ann_in = np.empty(
            self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g) +
            (self.num_exogs if self.ann_exogs == slice(None) else len(self.ann_exogs)),
            dtype=y.dtype)
        trend_in = None

        # Arrays of shocks and predictions history
        shocks_hist = np.empty(y.shape[0] - self.max_ar_lag + self.max_ma_lag, dtype=y.dtype)
        shocks_hist[:self.max_ma_lag] = shocks[self.max_ma_lag-1::-1]
        preds_hist = np.empty(y.shape[0], dtype=y.dtype)
        preds_hist[:self.max_ar_lag] = y[:self.max_ar_lag]

        for t in range(self.max_ar_lag, y.shape[0]):
            real_t = last_t - (y.shape[0] - 1) + t
            # SARIMA input
            sarima_in[:self.p] = y[t - 1:t - self.p - 1 if t != self.p else None:-1]
            sarima_in[self.p:
                      self.p + len(self.seasonal_p)] = y[[t - i for i in self.seasonal_p]]
            sarima_in[self.p + len(self.seasonal_p):
                      self.p + len(self.seasonal_p) + self.q] = shocks[:self.q]
            sarima_in[self.p + len(self.seasonal_p) + self.q:
                      self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q)] = \
                shocks[[i - 1 for i in self.seasonal_q]]
            if self.num_exogs:
                sarima_in[self.p + len(self.seasonal_p) + self.q + len(self.seasonal_q):] = \
                    X[t-self.max_ar_lag, self.sarima_exogs]  # Add eXogs
            # Trend input
            trend_in = np.power(real_t, self.trend, dtype=y.dtype)
            # ANN input
            ann_in[:self.r] = y[t - 1:t - self.r - 1 if t != self.r else None:-1]
            ann_in[self.r:
                   self.r + len(self.seasonal_r)] = y[[t - i for i in self.seasonal_r]]
            ann_in[self.r + len(self.seasonal_r):
                   self.r + len(self.seasonal_r) + self.g] = shocks[:self.g]
            ann_in[self.r + len(self.seasonal_r) + self.g:
                   self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g)] = \
                shocks[[i - 1 for i in self.seasonal_g]]
            if self.num_exogs:
                ann_in[self.r + len(self.seasonal_r) + self.g + len(self.seasonal_g):] = \
                    X[t-self.max_ar_lag, self.ann_exogs]  # Add eXogs

            # Calculate prediction
            preds_hist[t] = self._forward_pass_fast(sarima_in, ann_in, trend_in)
            if np.isnan(y[t]):
                y[t] = preds_hist[t] # if y_t is nan, it is replaced with prediction
            shocks_hist[t-self.max_ar_lag+self.max_ma_lag] = y[t] - preds_hist[t]
            shocks[1:] = shocks[:-1]
            if len(shocks):
                shocks[0] = shocks_hist[t-self.max_ar_lag+self.max_ma_lag]

        return preds_hist, shocks_hist

    def _prepare_predict_input(self, y, X, input_shocks, t, horizon):
        """Prepares predict function input for finding prediction using the trained model.

        Parameters
        ----------
        y : ndarray of shape (nobs,)
            The time series. If y has observations less than d + D * s + max_ar_lag + num_shocks_to_find
            it will be extended by last_train_ys values.

        X : ndarray of shape (num_shocks_to_find + horizon, number of exog regressors)
            The exogenous regressors.

        input_shocks : ndarray
            The input shocks in reversed time order. If input_shocks has shocks less than max_ma_lag
            it will be extended by last_train_shocks and num_shocks_to_find will be
            t - (last_train_t + input_shocks.shape[0]), else num_shocks_to_find is 0.

        t : None or int
            The forecasting origin. If not provided, defines as last_train_t.

        horizon : int
            The forecasting horizon.
        """
        # Prepare forecasting origin t
        if t is None:
            t = self.last_train_t + y.shape[0]
            warnings.warn("Forecasting origin `t` is not specified,"
                          " so it will be assumed equals to"
                          " sum of the last train time moment + size of"
                          f" input observations `y`: {t}.")

        # Calculate number of shocks which needed to find before calculating final prediction
        if (input_shocks.shape[0] < self.max_ma_lag and
            t - self.last_train_t - input_shocks.shape[0] > 0):
            num_shocks_to_find = t - self.last_train_t - input_shocks.shape[0]
        else:
            num_shocks_to_find = 0

        # Check if it is possible to find `num_shocks_to_find` shocks
        if (self.d + self.seasonal_d * self.s + self.max_ar_lag +
                num_shocks_to_find > y.shape[0] + self.last_train_ys.shape[0]):
            raise ValueError(f"For getting predictions {num_shocks_to_find} shocks"
                             " needed, but entered `y` with `last_train_ys`"
                             " observations can provide finding only"
                             f""" {y.shape[0] + self.last_train_ys.shape[0] -
                                   self.d - self.seasonal_d * self.s -
                                   self.max_ar_lag}"""
                             " shocks. Reduce forecasting origin `t` or"
                             " add more observations `y`.")

        # Prepare input shocks
        input_shocks = input_shocks[:self.max_ma_lag]
        if input_shocks.shape[0] < self.max_ma_lag:
            warnings.warn("Number of shocks in `input_shocks`:"
                          f" {input_shocks.shape[0]} is not enough for"
                          f" getting predictions (necessary {self.max_ma_lag}),"
                          " so last"
                          f" {self.max_ma_lag - input_shocks.shape[0]}"
                          " shocks from train will be added.")
            input_shocks = np.hstack(
                [input_shocks, self.last_train_shocks[:self.max_ma_lag - input_shocks.shape[0]]]
            )

        # Prepare y
        y = y[-(self.d + self.seasonal_d * self.s +
                self.max_ar_lag + num_shocks_to_find):]
        if (self.d + self.seasonal_d * self.s +
            self.max_ar_lag + num_shocks_to_find) > y.shape[0]:
            warnings.warn(f"Number of observesions in `y`: {y.shape[0]} "
                          f" is not enough to find all {num_shocks_to_find}"
                          " missing shocks, taking into account that for"
                          f" getting prediction necessary {self.max_ar_lag}"
                          " observations and after differencing first"
                          f" {self.d + self.seasonal_d * self.s} observations"
                          f" will be lost, so last"
                          f""" {self.d + self.seasonal_d * self.s + self.max_ar_lag +
                                num_shocks_to_find - y.shape[0]}"""
                          " observations from training data will be added.")
            y = np.hstack([
                self.last_train_ys[-(self.d + self.seasonal_d * self.s + self.max_ar_lag +
                                     num_shocks_to_find - y.shape[0]):],
                y
            ])

        # Prepare exogenous regressors
        X = X[-(horizon + num_shocks_to_find):]
        if X.shape[0] < horizon + num_shocks_to_find and self.num_exogs:
            raise ValueError("Model was fitted with"
                             " exogenous regressors and also"
                             f" necessary to find {num_shocks_to_find}"
                             " shocks, so for getting"
                             " predictions you must give on input number"
                             " of exogenous regressors observations"
                             " greater or equals to :"
                             f" {horizon + num_shocks_to_find},"
                             " but number of given observations is"
                             f" {X.shape[0]}; or you can refit the model"
                             " without exogenous regressors, or give"
                             f" to input {num_shocks_to_find} more shocks.")

        return y, X, input_shocks, t, horizon

    def predict(self, y=None, X=None, input_shocks=None, t=None, horizon=1,
                intervals=False, return_last_input_shocks=False):
        """Makes forecasts from fitted model.

        Suppose that the last time moment in train data was T.

        By default, predict returns prediction of value at T+horizon time moment,
        but if a model has exogenous regressors, for getting prediction you need
        to provide exogenous regressors matrix X of shape (horizon, k) where k is
        number of exogenous regressors.

        If new data y was provided with N number of observations, then predict
        returns prediction of value at T+N+horizon time moment. All shocks
        corresponds to new data will be calculated. And again, if a model has
        exogenous regressors, you need to provide exogenous regressors matrix X
        but now of shape (N+horizon, k).

        Also, you can just substitute new data y, shocks, time moment t and
        exogenous regressors X into model and get prediction of value at t+horizon
        time moment, but make sure that new data y has at least d+D*s+max_ar_lag
        observations, shocks has at least max_ma_lag values and exogenous regressors
        matrix has shape (horizon, k).

        Parameters
        ----------
        y : ndarray, optional
            New time series data.

        X : ndarray, optional
            Matrix of exogenous regressors.

        input_shocks : ndarray, optional
            Input shocks. Must be in reverse time order, i.e. t-1 shock by 0 index
            t-2 shock by 1 index etc.

        t : int, optional
            Forecasing origin.

        horizon : float, optional
            Forecasting horizon.

        intervals : bool, optional
            Wheter or not to return prediction intervals. Intervals correctly
            calculates only for one step prediction(i.e when horizon is 1) yet.

        return_last_input_shocks : bool, optional
        Wheter or not to return last calculated input shocks. Useful when in next
        time you want to just substitute inputs in model for getting prediction.

        Returns
        -------
        pred : float
            Model prediction of value at t+horizon time moment.
        pred_intervals : tuple
            Returns prediction intervals if intervals was True.
        last_input_shocks : ndarray
            Returns last input shocks in reverse time order, if return_input_shocks was True.
        """
        # Validate input
        y, X, input_shocks, t, horizon, intervals, return_last_input_shocks = \
            validate._predict_input(y, X, input_shocks, t, horizon, intervals,
                                    return_last_input_shocks)
        # Prepare input
        y, X, input_shocks, t, horizon =\
            self._prepare_predict_input(y, X, input_shocks, t, horizon)

        # Differencing y
        y, first_ys, seasonal_ys = self._time_series_differencing(y)

        # Add nans which will be replaced with predictions
        y = np.hstack([y, [np.nan] * horizon])

        # Calculate predictions
        preds, shocks = self._full_forward_pass_fast(y, X, input_shocks, t + horizon)

        # Taking last t,...,t-max_ma_lag+1 shocks, where t is forecasting origin
        last_input_shocks = shocks[-horizon-1:-horizon-self.max_ma_lag-1:-1]

        # Undifference predictions and get last value
        y[-horizon:] = preds[-horizon:]
        pred = self._time_series_undifferencing(y, first_ys, seasonal_ys)[-1]

        # Return prediction function result
        res = pred
        if intervals:
            res = pred, (pred - 1.64 * self.train_std, pred + 1.64 * self.train_std),
        if return_last_input_shocks:
            if isinstance(res, tuple):
                res += last_input_shocks
            else:
                res = res, last_input_shocks,

        return res

    def score(self):
        pass

    def get_params(self):
        """Returns all trained model parameters and initial shocks.

        Returns
        -------
        python dictionary
            Returns all model trained parameters and initial shocks in python
            dictionary by keys "sarima_coefs", "trend_coefs", "ann_coefs",
            "ann_intercepts" and "init_shocks".
        """
        return {"sarima_coefs": self.sarima_coefs,
                "ann_coefs": self.ann_coefs,
                "ann_intercepts": self.ann_intercepts,
                "trend_coefs": self.trend_coefs,
                "init_shocks": self.init_shocks}

    def set_params(self, sarima_coefs, ann_coefs, ann_intercepts, trend_coefs,
                   init_shocks):
        """Sets all trainable model parameters and initial shocks.

        Parameters
        ----------
        sarima_coefs : ndarray
            Weights vector of SARIMA part of the model.

        ann_coefs : list of ndarrays
            Weights matrices of ANN part of the model. The ith element in the
            list represents the weight matrix corresponding to layer i.

        ann_intercepts : list of ndarrays
            Bias vectors of ANN part of the model. The ith element in the list
            represents the bias vector corresponding to layer i + 1. Output layer
            has no bias.

        trend_coefs : ndarray
            Coefficients of trend polynomial.

        init_shocks : ndarray
            The first MAX_MA_LAG shocks.
        """
        # TODO: Add validation for the input data
        self.sarima_coefs = sarima_coefs
        self.ann_coefs = ann_coefs
        self.ann_intercepts = ann_intercepts
        self.trend_coefs = trend_coefs
        self.init_shocks = init_shocks
