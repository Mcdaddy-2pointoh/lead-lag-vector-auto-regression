## Config Universe Errors
class UniverseParamsNotFound(Exception):
    """Raised when `universe_params` not found in the `config_params.yml` file"""

class MarketParamsNotFound(Exception):
    """Raised when `time_params` not found in the `config_params.yml` file"""

class UniverseNameKeyNotFound(Exception):
    """Raised when `universe_name` not found in key `universe_params` from the `config_params.yml` file"""

class UniverseNameNotFound(Exception):
    """Raised when specified `universe_name` not found in key `universe_params` from the `config_params.yml` file"""


## Config Time Errors
class TimeParamsNotFound(Exception):
    """Raised when `time_params` not found in the `config_params.yml` file"""

class TimeParamIntervalNotFound(Exception):
    """Raised when `interval` not found in key `time_params` from the `config_params.yml` file"""

class TimeParamPeriodNotFound(Exception):
    """Raised when `period` not found in key `time_params` from the `config_params.yml` file"""


## Config Market Errors
class MarketParamsNotFound(Exception):
    """Raised when `market_params` not found in the `config_params.yml` file"""

class MarketParamTimezoneNotFound(Exception):
    """Raised when `time_zone` not found in key `market_params` from the `config_params.yml` file"""

class MarketParamStartTimeNotFound(Exception):
    """Raised when `start_time` not found in key `market_params` from the `config_params.yml` file"""

class MarketParamEndTimeNotFound(Exception):
    """Raised when `end_time` not found in key `market_params` from the `config_params.yml` file"""


## Config ADF Errors
class ADFParamsNotFound(Exception):
    """Raised when `adf_testing_params` not found in the `config_params.yml` file"""

class ADFParamAlphaNotFound(Exception):
    """Raised when `alpha` not found in key `adf_testing_params` from the `config_params.yml` file"""

## Config VAR Errors
class VARParamsNotFound(Exception):
    """Raised when `var_params` not found in the `config_params.yml` file"""

class VARParamMaxLagsNotFound(Exception):
    """Raised when `max_lags` not found in key `var_params` from the `config_params.yml` file"""

class VARParamICCriterionNotFound(Exception):
    """Raised when `ic_criterion` not found in key `var_params` from the `config_params.yml` file"""

class VARParamAlphaNotFound(Exception):
    """Raised when `alpha` not found in key `var_params` from the `config_params.yml` file"""

class VARParamEstimationMethodNotFound(Exception):
    """Raised when `estimation_method` not found in key `var_params` from the `config_params.yml` file"""

class InvalidICCriterion(Exception):
    """Raised when `ic_criterion` is not one of ['aic', 'bic', 'hqic', 'fpe']"""

class InvalidMaxLagsError(Exception):
    """Raised when `max_lags` is not a positive integer (>= 1)"""

class InvalidAlphaError(Exception):
    """Raised when `alpha` significance level is not between 0 and 1"""

class InvalidEstimationMethod(Exception):
    """Raised when `estimation_method` is not one of ['ols', 'mle']"""


### Data Key Not founnd error
class DataKeyNotFound(Exception):
    "Raised when specific datakey not found in the `data` section of `config.yaml`"

    def __init__(self, message=None):
        if message is None:
            message = "Raised when specific datakey not found in the `data` section of `config.yaml`"
        super().__init__(message)


### Log Key Not founnd error
class DataKeyNotFound(Exception):
    "Raised when specific datakey not found in the `data` section of `config.yaml`"

    def __init__(self, message=None):
        if message is None:
            message = "Raised when specific datakey not found in the `data` section of `config.yaml`"
        super().__init__(message)