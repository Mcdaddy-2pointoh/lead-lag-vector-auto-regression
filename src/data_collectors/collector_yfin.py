import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import traceback
import yaml

# Internal Imports
from src.utils.file_operators import *
from src.utils.text_operators import *
from src.utils.error_classes import * 

class YFinanceNSEPipeline:
    """
    Class: To fetch and process raw data for lead-lag-vector-autoregression on NSE Tickers
    """
    def __init__(
            self, 
            config_params: dict, 
            config_catalog: dict
        ) -> None:
        """
        Function: Initialisation function
        Args:
            tickers: List of Indian stock symbols without suffix (e.g. ['TCS', 'INFY'])
            interval: Bar frequency ('1m' for 1-minute, '5m' for 5-minute)
            period: Historical lookback ('7d' max for 1m data on Yahoo Finance)
        Returns: None
        """

        # Keep a local copy of config catalog
        self.config_catalog = config_catalog
        self.data_catalog = self.config_catalog.get("data", {})
        self.logs_catalog = self.config_catalog.get("logs", {})

        # Keep a local copy of config params
        self.config_params = config_params

        # set all rest vars to none

        # Universe params
        self.universe_params = None
        self.universe_name = None
        self.universe_list = None

        # Time Params
        self.time_params = None
        self.interval = None
        self.period = None

        # Market Params
        self.market_params = None
        self.time_zone = None
        self.market_start_time = None
        self.market_end_time = None

        # ADF Params
        self.adf_testing_params = None
        self.alpha = None

        # Log and Data Params
        self.error_log_dir = None
        self.raw_prices = None
        self.aligned_prices = None
        self.log_returns = None
        self.yf_tickers = None

        ## Universe Params
        # Get the universe_params
        if config_params.get('universe_params', None) is None:
            raise UniverseParamsNotFound()
        else: 
            self.universe_params = config_params.get('universe_params', None)

        # Get universe_name
        if self.universe_params.get('universe_name', None) is None:
            raise UniverseNameKeyNotFound()
        else: 
            self.universe_name = self.universe_params.get('universe_name', None)

        # Get universe_list based on the name
        if self.universe_params.get(self.universe_name, None) is None:
            raise UniverseNameKeyNotFound(
                f"Universe name {self.universe_name} specified in the config however now key with the same name available in config containing ticker list"
            )
        else: 
            self.universe_list = self.universe_params.get(self.universe_name, None)

        # Warn if error log config not setup
        if self.logs_catalog.get('collector_yfin_error_logs', None) is None:
            print(
                format_warn_text(f"No Error Logger Setup in the catalog config for `collector_yfin_error_logs`")
            )
        else: 
            self.error_log_dir = self.logs_catalog.get('collector_yfin_error_logs', None)


        ## Time Params
        # Get the time_params
        if config_params.get('time_params', None) is None:
            raise TimeParamsNotFound()
        else: 
            self.time_params = config_params.get('time_params', None)


        # Get the period from time_params
        if self.time_params.get('period', None) is None:
            raise TimeParamPeriodNotFound()
        else: 
            self.period = self.time_params.get('period', None)


        # Get the interval from time_params
        if self.time_params.get('interval', None) is None:
            raise TimeParamsNotFound()
        else: 
            self.interval = self.time_params.get('interval', None)


        ## Market Params
        # Get the market_params
        if config_params.get('market_params', None) is None:
            raise MarketParamsNotFound()
        else: 
            self.market_params = config_params.get('market_params', None)


        # Get the timezone from time_params
        if self.market_params.get('time_zone', None) is None:
            raise MarketParamTimezoneNotFound()
        else: 
            self.time_zone = self.market_params.get('time_zone', None)


        # Get the timezone from time_params
        if self.market_params.get('start_time', None) is None:
            raise MarketParamStartTimeNotFound()
        else: 
            self.market_start_time = self.market_params.get('start_time', None)


        # Get the timezone from time_params
        if self.market_params.get('end_time', None) is None:
            raise MarketParamEndTimeNotFound()
        else: 
            self.market_end_time = self.market_params.get('end_time', None)


        ## ADF Params
        # Get the market_params
        if config_params.get('adf_testing_params', None) is None:
            raise ADFParamsNotFound()
        else: 
            self.adf_testing_params = config_params.get('adf_testing_params', None)


        # Get the timezone from time_params
        if self.adf_testing_params.get('alpha', None) is None:
            raise ADFParamAlphaNotFound()
        else: 
            self.alpha = self.adf_testing_params.get('alpha', None)


        # Process all ticker list
        try:
            # Load the tickers
            self.yf_tickers = [
                f"{t}.NS" if not t.endswith(".NS") else t for t in self.universe_list
            ]

        except Exception as e:
            print(
                format_error_text(f"Failed to process ticker data for universe: {self.universe_name}")
            )

            error_message = (
                f"Failed to process ticker data for universe: "
                f"{self.universe_name}\n\n"
                f"Error: {str(e)}\n\n"
                f"Traceback:\n"
                f"{traceback.format_exc()}"
            )

            if self.error_log_dir is not None:
                save_text(
                    text=error_message,
                    directory=self.error_log_dir['directory'],
                    file_name=self.error_log_dir['file_name'],
                    versioned=self.error_log_dir['versioned'],
                    suffix="ticker_processing"
                )

            raise

    # def fetch_raw_data(self) -> pd.DataFrame:
    #     """
    #     Function: Ingests raw intraday bars data via yfinance aggregator
    #     Args:
    #         self
    #     Returns:
    #         raw_data (pd.DataFrame): Raw Intraday Prices for all specified tickers
    #     """

    #     # Log the fetch raw_data
    #     print(
    #         format_info_text(f"[INGESTION]: Fetching {self.period} of {self.interval} data for {len(self.universe_list)} assets from Yahoo Finance")
    #     )

    #     # Downloads batch data
    #     y_fin_data = None
    #     try:
    #         y_fin_data = yf.download(
    #             tickers=self.yf_tickers,
    #             period=self.period,
    #             interval=self.interval,
    #             group_by="column",
    #             progress=False
    #         )
    #     except Exception as e:
    #         print(
    #             format_error_text(f"Failed to download ticker data for universe: {self.universe_name} from Yahoo Finance")
    #         )

    #         error_message = (
    #             f"Failed to downloadload ticker data for universe: "
    #             f"{self.universe_name}\n\n"
    #             f"Error: {str(e)}\n\n"
    #             f"Traceback:\n"
    #             f"{traceback.format_exc()}"
    #         )

    #         if self.error_log_dir is not None:
    #             save_text(
    #                 text=error_message,
    #                 directory=self.error_log_dir['directory'],
    #                 file_name=self.error_log_dir['file_name'],
    #                 versioned=self.error_log_dir['versioned'],
    #                 suffix="ticker_data_download"
    #             )

    #         raise

    #     # Extract Close Values only
    #     if len(self.yf_tickers) > 1:
    #         prices = y_fin_data["Close"]
    #     else:
    #         prices = y_fin_data[["Close"]]

    #     # Rename Columns back to original names
    #     prices.columns = [col.replace(".NS", "") for col in prices.columns]
    #     self.raw_prices = prices
    #     print(format_success_text(f"[INGESTION]: Downloaded {len(prices)} raw rows, {len(self.universe_list)} assets from Yahoo Finance"))

    #     # Drop any column if it is all null
    #     self.raw_prices.dropna(axis=1, how='all')

    #     # Save only if not empty
    #     if not(self.raw_prices.empty or self.raw_prices.shape[1] == 0):

    #         # Warn if data catalog config not setup
    #         if self.data_catalog.get('collector_yfin_raw_data', None) is None:
    #             print(
    #                 format_warn_text(f"No Data Catalog Setup in the catalog config for `collector_yfin_raw_data`. Will run on In Memory form")
    #             )
    #         else: 
    #             raw_data_save_confs = self.data_catalog.get('collector_yfin_raw_data', None)
    #             save_dataframe(
    #                 df=self.raw_prices,
    #                 directory=raw_data_save_confs['directory'],
    #                 file_name=raw_data_save_confs['file_name'],
    #                 file_format=raw_data_save_confs['file_format'],
    #                 versioned=raw_data_save_confs['versioned'],
    #                 suffix=self.universe_name,
    #                 save_index=True
    #             )

    #     else:
    #         self.raw_prices = None

    #     return self.raw_prices

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Function: Ingests raw intraday bars data via yfinance aggregator 
                  using sequential requests with custom HTTP headers to avoid
                  anti-scraping connection drops (curl 56).
        Args:
            self
        Returns:
            raw_data (pd.DataFrame): Raw Intraday Prices for all specified tickers
        """
        import time
        import requests

        # Log the fetch raw_data initiation
        print(
            format_info_text(
                f"[INGESTION]: Fetching {self.period} of {self.interval} data for "
                f"{len(self.universe_list)} assets from Yahoo Finance (Sequential Mode)"
            )
        )

        # Build custom HTTP session with a modern browser User-Agent
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        })

        fetched_series = {}

        try:
            for ticker_symbol, clean_name in zip(self.yf_tickers, self.universe_list):
                print(format_info_text(f"[INGESTION]: Downloading {clean_name} ({ticker_symbol})..."))
                
                try:
                    ticker = yf.Ticker(ticker_symbol, session=session)
                    df_ticker = ticker.history(period=self.period, interval=self.interval)

                    if not df_ticker.empty and "Close" in df_ticker.columns:
                        fetched_series[clean_name] = df_ticker["Close"]
                        print(format_success_text(f"  └─ Received {len(df_ticker)} bars for {clean_name}"))
                    else:
                        print(format_warn_text(f"  └─ Warning: Empty or invalid payload for {clean_name}"))

                except Exception as sym_err:
                    print(format_warn_text(f"  └─ Failed fetching {clean_name}: {str(sym_err)}"))

                # 1.5 second delay to prevent IP rate-limiting / socket termination
                time.sleep(1.5)

            if not fetched_series:
                raise ValueError("All asset downloads failed. Check network connection or IP block status.")

            # Combine individual series into a single multi-asset DataFrame
            prices = pd.DataFrame(fetched_series)

        except Exception as e:
            print(
                format_error_text(
                    f"Failed to download ticker data for universe: {self.universe_name} from Yahoo Finance"
                )
            )

            error_message = (
                f"Failed to download ticker data for universe: "
                f"{self.universe_name}\n\n"
                f"Error: {str(e)}\n\n"
                f"Traceback:\n"
                f"{traceback.format_exc()}"
            )

            if self.error_log_dir is not None:
                save_text(
                    text=error_message,
                    directory=self.error_log_dir['directory'],
                    file_name=self.error_log_dir['file_name'],
                    versioned=self.error_log_dir['versioned'],
                    suffix="ticker_data_download"
                )

            raise

        self.raw_prices = prices
        print(
            format_success_text(
                f"[INGESTION]: Successfully downloaded {len(prices)} raw rows across "
                f"{len(prices.columns)} valid assets from Yahoo Finance"
            )
        )

        # Drop any column if it is all null
        self.raw_prices = self.raw_prices.dropna(axis=1, how='all')

        # Save only if not empty
        if not (self.raw_prices.empty or self.raw_prices.shape[1] == 0):

            # Warn if data catalog config not setup
            if self.data_catalog.get('collector_yfin_raw_data', None) is None:
                print(
                    format_warn_text(
                        f"No Data Catalog Setup in the catalog config for `collector_yfin_raw_data`. "
                        f"Will run on In Memory form"
                    )
                )
            else: 
                raw_data_save_confs = self.data_catalog.get('collector_yfin_raw_data', None)
                save_dataframe(
                    df=self.raw_prices,
                    directory=raw_data_save_confs['directory'],
                    file_name=raw_data_save_confs['file_name'],
                    file_format=raw_data_save_confs['file_format'],
                    versioned=raw_data_save_confs['versioned'],
                    suffix=self.universe_name,
                    save_index=True
                )

        else:
            self.raw_prices = None

        return self.raw_prices

    def align_and_clean_data(
        self,
        use_latest: bool = True,
        use_specific: str = "",
    ) -> pd.DataFrame:


        # Log the align_and_clean_data
        print(
            format_info_text(f"[PROCESSING]: Processing {self.period} of {self.interval} data for {len(self.universe_list)} assets")
        )

        # Use latest price
        if use_latest:
            if self.raw_prices is None:
                try:
                    raw_data_save_confs = self.data_catalog.get('collector_yfin_raw_data', None)
                    self.raw_prices = load_latest_dataframe(
                        directory=raw_data_save_confs['directory'],
                        file_name=raw_data_save_confs['file_name'],
                        file_format=raw_data_save_confs['file_format'],
                        suffix=self.universe_name
                    )

                except Exception as e:
                    self.fetch_raw_data()

        # Use a specific raw file
        else:
            self.raw_prices = pd.read_csv(use_specific)

        # Copy df
        df = self.raw_prices.copy(deep=True)

        # Convert the index to IST timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(self.time_zone)
        else:
            df.index = df.index.tz_convert(self.time_zone)

        # Filter strictly for market hours only
        df = df.between_time(self.market_start_time, self.market_end_time)

        # Drop empty timestamps and apply Forward-Fill then Back-Fill
        df = df.dropna(how="all")
        df = df.ffill().bfill()

        self.aligned_prices = df
        print(format_success_text(f"[PROCESSING] Aligned grid size: {len(df)} rows across {len(self.universe_list)} stocks."))

        # Save the aligned prices
        if not(self.aligned_prices.empty or self.aligned_prices.shape[1] == 0):

            # Warn if data catalog config not setup
            if self.data_catalog.get('collector_yfin_cleaned_data', None) is None:
                print(
                    format_warn_text(f"No Data Catalog Setup in the catalog config for `collector_yfin_cleaned_data`. Will run on In Memory form")
                )
            else: 
                aligned_data_save_confs = self.data_catalog.get('collector_yfin_cleaned_data', None)
                save_dataframe(
                    df=self.aligned_prices,
                    directory=aligned_data_save_confs['directory'],
                    file_name=aligned_data_save_confs['file_name'],
                    file_format=aligned_data_save_confs['file_format'],
                    versioned=aligned_data_save_confs['versioned'],
                    suffix=self.universe_name,
                    save_index=True
                )

        else:
            self.aligned_prices = None

        return self.aligned_prices

    def compute_log_returns(
        self,
        use_latest: bool = True,
        use_specific: str = "",
    ) -> pd.DataFrame:
        
        # Log the compute_log_returns
        print(
            format_info_text(f"[PROCESSING]: Computing Log Returns for {len(self.universe_list)} assets")
        )

        # Use latest aligned price
        if use_latest:
            if self.aligned_prices is None:
                try:
                    aligned_data_save_confs = self.data_catalog.get('collector_yfin_cleaned_data', None)
                    self.aligned_prices = load_latest_dataframe(
                        directory=aligned_data_save_confs['directory'],
                        file_name=aligned_data_save_confs['file_name'],
                        file_format=aligned_data_save_confs['file_format'],
                        suffix=self.universe_name
                    )

                except Exception as e:
                    self.align_and_clean_data()

        # Use a specific raw file
        else:
            self.aligned_prices = pd.read_csv(use_specific)

        # Copy df
        df = self.aligned_prices.copy(deep=True)

        # Compute Log Returns
        self.log_returns = np.log(self.aligned_prices / self.aligned_prices.shift(1)).dropna()

        # Save the aligned prices
        if not(self.log_returns.empty or self.log_returns.shape[1] == 0):

            # Warn if data catalog config not setup
            if self.data_catalog.get('collector_yfin_log_returns', None) is None:
                print(
                    format_warn_text(f"No Data Catalog Setup in the catalog config for `collector_yfin_log_returns`. Will run on In Memory form")
                )
            else: 
                log_returns_data_save_confs = self.data_catalog.get('collector_yfin_log_returns', None)
                save_dataframe(
                    df=self.log_returns,
                    directory=log_returns_data_save_confs['directory'],
                    file_name=log_returns_data_save_confs['file_name'],
                    file_format=log_returns_data_save_confs['file_format'],
                    versioned=log_returns_data_save_confs['versioned'],
                    suffix=self.universe_name,
                    save_index=True
                )

        else:
            self.log_returns = None

        return self.log_returns

    def validate_stationarity(
        self,
        use_latest: bool = True,
        use_specific: str = "",
    ) -> pd.DataFrame:
        
        # Log the validate_stationarity
        print(
            format_info_text(f"[ANALYSING]: Performing ADF for {len(self.universe_list)} assets")
        )

        # Use latest log returns
        if use_latest:
            if self.log_returns is None:
                try:
                    log_returns_data_save_confs = self.data_catalog.get('collector_yfin_log_returns', None)
                    self.log_returns = load_latest_dataframe(
                        directory=log_returns_data_save_confs['directory'],
                        file_name=log_returns_data_save_confs['file_name'],
                        file_format=log_returns_data_save_confs['file_format'],
                        suffix=self.universe_name
                    )

                except Exception as e:
                    self.compute_log_returns()

        # Use a specific raw file
        else:
            self.log_returns = pd.read_csv(use_specific)

        # Compute ADF
        report = {}
        for col in self.log_returns.columns:
            series = self.log_returns[col].replace([np.inf, -np.inf], np.nan).dropna()
            adf_stat, p_value, _, _, _, _ = adfuller(series)
            report[col] = {
                "ADF Statistic": round(adf_stat, 4),
                "p-value": round(p_value, 6),
                "Stationary (I(0))": p_value < self.alpha
            }

        adf_res = pd.DataFrame(report).T

        # Save the aligned prices
        if not(adf_res.empty or adf_res.shape[1] == 0):

            # Warn if data catalog config not setup
            if self.data_catalog.get('collector_yfin_stationarity_check', None) is None:
                print(
                    format_warn_text(f"No Data Catalog Setup in the catalog config for `collector_yfin_stationarity_check`. Will run on In Memory form")
                )
            else: 
                adf_results_path = self.data_catalog.get('collector_yfin_stationarity_check', None)
                save_dataframe(
                    df=adf_res,
                    directory=adf_results_path['directory'],
                    file_name=adf_results_path['file_name'],
                    file_format=adf_results_path['file_format'],
                    versioned=adf_results_path['versioned'],
                    suffix=self.universe_name,
                    save_index=True
                )

        else:
            adf_res = None

        return adf_res




