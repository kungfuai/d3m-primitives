import logging

from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Arima:
    def __init__(
        self,
        seasonal: bool = False,
        seasonal_differencing: int = 1,
        max_order: int = 5,
        dynamic: bool = True,
        log_transform: bool = False,
    ):
        """initialize ARIMA class

        Keyword Arguments:
            seasonal {bool} -- whether time series has seasonal component (default: {False})
            seasonal_differencing {int} -- period for seasonal differencing (default: {1})
            max_order {int} -- maximum order of p and q terms on which to fit model (default: {5})
            dynamic {bool} -- whether in-sample lagged values should be used for in-sample prediction
            log_transform {bool} -- whether to apply log_transform before fitting arima model
        """

        self.seasonal = seasonal
        self.seasonal_differencing = seasonal_differencing
        self.max_order = max_order
        self.dynamic = dynamic
        self.log_transform = log_transform

    def fit(self, train):
        """fit ARIMA model on training data, automatically selecting p (AR), q (MA),
        P (AR - seasonal), Q (MA - seasonal), d, and D (differencing) amongst other parameters
        based on AIC

        Arguments:
            np array -- endogenous time series on which model should select parameters and fit
        """

        self.min_train = min(train)

        if self.log_transform:
            train = self._transform(train)

        self.arima_model = auto_arima(
            train,
            start_p=1,
            start_q=1,
            max_p=self.max_order,
            max_q=self.max_order,
            m=self.seasonal_differencing,
            seasonal=self.seasonal,
            stepwise=True,
            suppress_warnings=True,
        )

        self.arima_model.fit(train)
        logger.info(
            f"Fit ARIMA model with {self.arima_model.df_model()} degrees of freedom"
        )

    def predict(self, n_periods=1, return_conf_int=False, alpha=0.05):
        """forecasts the time series n_periods into the future

        Keyword Arguments:
            n_periods {int} -- number of periods to forecast into the future (default: {1})
            return_conf_int {bool} -- whether to return confidence intervals instead of
                forecasts (default: {False})
            alpha {float} -- significance level for confidence interval, i.e. alpha = 0.05
                returns a 95% confdience interval from alpha / 2 to 1 - (alpha / 2)
                (default: {0.05})

        Returns:
            np array -- (n_periods, 1) time series forecast n_periods into the future
                OR (n_periods, 3) if returning confidence interval forecast
        """
        logger.info(
            f"Making predictions for ARIMA model with {self.arima_model.df_model()} degrees of freedom"
        )
        if return_conf_int:
            forecast, interval = self.arima_model.predict(
                n_periods=n_periods, return_conf_int=True, alpha=alpha
            )
            if self.log_transform:
                return (
                    self._inverse_transform(forecast),
                    self._inverse_transform(interval[:, 0]),
                    self._inverse_transform(interval[:, 1]),
                )
            else:
                return (
                    forecast,
                    interval[:, 0],
                    interval[:, 1],
                )
        else:
            if self.log_transform:
                return self._inverse_transform(
                    self.arima_model.predict(n_periods=n_periods)
                )
            else:
                return self.arima_model.predict(n_periods=n_periods)

    def predict_in_sample(self, return_conf_int=False, alpha=0.05):
        """thin wrapper for ARIMA predict_in_sample f(). always predicts all in-sample
        points (except for first point). dynamic parameter controlled by instance variable

        Keyword Arguments:
            return_conf_int {bool} -- whether to return confidence intervals instead of
                forecasts (default: {False})
            alpha {float} -- significance level for confidence interval, i.e. alpha = 0.05
                returns a 95% confdience interval from alpha / 2 to 1 - (alpha / 2)
                (default: {0.05})

        Returns:
            np array -- (n_in_sample, 1) time series forecast for n in-sample timesteps
                OR (n_in_sample, 3) if returning confidence interval forecast
        """

        forecast = self.arima_model.predict_in_sample(dynamic=self.dynamic)
        if return_conf_int:
            interval = self.arima_model.conf_int(alpha=alpha)
            if self.log_transform:
                return (
                    self._inverse_transform(forecast),
                    self._inverse_transform(interval[:, 0]),
                    self._inverse_transform(interval[:, 1]),
                )
            else:
                return (
                    forecast,
                    interval[:, 0],
                    interval[:, 1],
                )
        else:
            if self.log_transform:
                return self._inverse_transform(forecast)
            else:
                return forecast

    def get_absolute_value_params(self):
        """get absolute value of trend, AR, and MA parameters of
            fit ARIMA model (no exogenous variables)

        Returns:
            pandas df -- df with column for each parameter
        """

        try:
            ar_count = self.arima_model.arparams().shape[0]
        except AttributeError:
            logger.debug("There are no ar parameters in this model")
            ar_count = 0
        try:
            ma_count = self.arima_model.maparams().shape[0]
        except AttributeError:
            logger.debug("There are no ma parameters in this model")
            ma_count = 0
        trend_count = self.arima_model.df_model() - ar_count - ma_count

        ar_cols = ["ar_" + str(i + 1) for i in range(ar_count)]
        ma_cols = ["ma_" + str(i + 1) for i in range(ma_count)]
        trend_cols = ["trend_" + str(i) for i in range(trend_count)]

        return pd.DataFrame(
            np.absolute(self.arima_model.params().reshape(1, -1)),
            columns=trend_cols + ar_cols + ma_cols,
        )

    def _transform(self, input):
        """transforms data according to defined transformation

        Arguments:
            input {np array} -- data pre-transform

        Returns:
            np array -- data post-transform
        """
        return np.log(input - self.min_train + 1)

    def _inverse_transform(self, input):
        """inverse transform of data according to defined transformation

        Arguments:
            input {np array} -- data pre-inverse-transform

        Returns:
            np array -- data post-inverse-transform
        """
        return np.exp(input) + self.min_train - 1
