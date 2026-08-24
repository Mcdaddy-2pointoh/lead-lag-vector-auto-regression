## Config Universe Errors
class UniverseParamsNotFound(Exception):
    """Raised when `universe_params` not found in the `config_params.yml` file"""

class MarketParamsNotFound(Exception):
    """Raised when `time_params` not found in the `config_params.yml` file"""

class UniverseNameKeyNotFound(Exception):
    """Raised when `universe_name` not found in key `universe_params` from the `config_params.yml` file"""

class UniverseNameNotFound(Exception):
    """Raised when `universe_name` not found in key `universe_params` from the `config_params.yml` file"""


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


## Config Market Errors
class ADFParamsNotFound(Exception):
    """Raised when `adf_testing_params` not found in the `config_params.yml` file"""

class ADFParamAlphaNotFound(Exception):
    """Raised when `alpha` not found in key `adf_testing_params` from the `config_params.yml` file"""