import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path

# Local File Operator & Formatting Imports
from src.utils.file_operators import save_dataframe, load_latest_dataframe
from src.utils.text_operators import format_error_text, format_info_text, format_success_text
from src.utils.error_classes import *


class GrangerCausalityMaskedSignal:
    """
    Execution & Signal Generation Engine for Lead-Lag Quantitative Strategies.
    
    Driven by config_params and config_catalog. Loads data directly from catalog keys:
      - data.collector_yfin_log_returns (Historical log returns for volatility scaling)
      - data.var_engine_forecast_signals (Raw 1-step return predictions)
      - data.var_engine_granger_matrix (Pairwise Granger Causality p-values / mask)
    """

    def __init__(self, config_params: Dict, config_catalog: Dict):
        """
        Args:
            config_params (Dict): Runtime hyperparameters dictionary.
            config_catalog (Dict): Catalog directory mapping for dataset and artifact I/O.
        """
        self.config_params = config_params
        self.config_catalog = config_catalog

        # Extract Signal Generation Hyperparameters
        sig_params = self.config_params.get("granger_causal_signal_gen_params", {})
        self.min_log_returns_threshold = sig_params.get("min_log_returns_threshold", 0.0005)
        self.max_position_size = sig_params.get("max_position_size", 1.0)
        self.volatility_scaling = sig_params.get("volatility_scaling", True)
        self.target_volatility = sig_params.get("target_volatility", 0.15)

        # Extract Bar Interval (Determines annualization factor for volatility scaling)
        time_params = self.config_params.get("time_params", {})
        self.interval = time_params.get("interval", "1m")
        self.bars_per_day = 72 if self.interval == "5m" else 360  # 360 bars/day for 1m interval (09:30-15:30)

        # Catalog Input Resolution
        self.catalog_data = self.config_catalog.get("data", {})

        # Universe Params
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

        # Get the output file_metadata
        self.output_signal_weight_metadata = self.catalog_data.get('output_signal_weights', False)
        self.output_signal_metadata_metadata = self.catalog_data.get('output_signal_metadata', False)
        

    def apply_granger_mask(
        self, 
        raw_forecasts: pd.DataFrame, 
        granger_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Filters raw VAR log-return forecasts using the Granger Causality binary signal matrix.
        """
        # Extract forecast Series
        if isinstance(raw_forecasts, pd.DataFrame):
            if "forecast" in raw_forecasts.columns:
                forecast_series = raw_forecasts["forecast"]
            else:
                forecast_series = raw_forecasts.iloc[-1]
        else:
            forecast_series = raw_forecasts

        tickers = list(forecast_series.index)
        alpha = self.config_params.get("var_params", {}).get("alpha", 0.05)

        mask_df = granger_matrix.copy()
        
        # Standardize matrix into binary flags (1 = Causal, 0 = Non-causal) if p-values are passed
        if (mask_df.values > 1.0).any() or ((mask_df.values > 0.0) & (mask_df.values < 1.0)).any():
            binary_mask = (mask_df < alpha).astype(float)
        else:
            binary_mask = mask_df.astype(float)

        # Active causal leader mask: Target asset j gets 1.0 if any leading asset i Granger-causes it
        active_causal_leaders = (binary_mask.sum(axis=0) > 0).astype(float)
        active_mask_series = pd.Series(active_causal_leaders, index=mask_df.columns).reindex(tickers).fillna(0.0)

        # Apply mask element-wise to raw return forecasts
        masked_forecasts = forecast_series * active_mask_series
        return pd.Series(masked_forecasts, index=tickers, name="masked_forecasts")

    def generate_positions(
        self,
        raw_forecasts: Optional[pd.DataFrame] = None,
        granger_matrix: Optional[pd.DataFrame] = None,
        historical_returns: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Master Execution Pipeline. Automates loading catalog artifacts when explicit inputs are omitted.
        
        Args:
            raw_forecasts (pd.DataFrame, optional): Manual forecast input; loaded from catalog if None.
            granger_matrix (pd.DataFrame, optional): Manual Granger mask input; loaded from catalog if None.
            historical_returns (pd.DataFrame, optional): Manual returns input; loaded from catalog if None.
            output_catalog_entry (Dict, optional): Catalog dict specifying directory/file_name for saving.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (target_weights_df, signal_metadata)
        """
        # Step 0: Catalog Resolution (Automated Loading via load_latest_dataframe)
        if raw_forecasts is None:
            entry = self.catalog_data["var_engine_forecast_signals"]
            print(format_info_text(f"[SignalGenerator]: Loading forecasts from '{entry['file_name']}'..."))
            print(entry)
            raw_forecasts = load_latest_dataframe(
                directory=entry["directory"],
                file_name=entry["file_name"],
                file_format=entry.get("file_format", "csv"),
                suffix=self.universe_name
            )

            raw_forecasts.rename(
                columns={
                    "Unnamed: 0" : "Index"
                },
                inplace=True
            )
            raw_forecasts.set_index(keys=['Index'], inplace=True)
            
        if granger_matrix is None:
            entry = self.catalog_data["var_engine_granger_matrix"]
            print(format_info_text(f"[SignalGenerator]: Loading Granger matrix from '{entry['file_name']}'..."))
            granger_matrix = load_latest_dataframe(
                directory=entry["directory"],
                file_name=entry["file_name"],
                file_format=entry.get("file_format", "csv"),
                suffix=self.universe_name
            )

            granger_matrix.rename(
                columns={
                    "Unnamed: 0" : "Index"
                },
                inplace=True
            )
            granger_matrix.set_index(keys=['Index'], inplace=True)

        if self.volatility_scaling and historical_returns is None:
            entry = self.catalog_data["collector_yfin_log_returns"]
            print(format_info_text(f"[SignalGenerator]: Loading historical log returns from '{entry['file_name']}'..."))
            historical_returns = load_latest_dataframe(
                directory=entry["directory"],
                file_name=entry["file_name"],
                file_format=entry.get("file_format", "csv"),
                suffix=self.universe_name
            )

        # Step 1: Apply Granger Binary Mask Filtering
        masked_forecasts = self.apply_granger_mask(raw_forecasts, granger_matrix)
        tickers = list(masked_forecasts.index)

        # Step 2: Directional Thresholding (Deadband Filter)
        directional_signals = np.zeros(len(tickers))
        
        long_condition = masked_forecasts.values > self.min_log_returns_threshold
        short_condition = masked_forecasts.values < -self.min_log_returns_threshold

        directional_signals[long_condition] = 1.0   # LONG Signal
        directional_signals[short_condition] = -1.0  # SHORT Signal

        # Step 3: Volatility Scaling (Risk Parity Adjustment)
        position_weights = directional_signals.copy()
        
        if self.volatility_scaling and historical_returns is not None:
            valid_returns = historical_returns.reindex(columns=tickers).dropna()
            
            annual_factor = np.sqrt(self.bars_per_day * 252)
            asset_vols = valid_returns.std() * annual_factor
            
            vol_scalars = (self.target_volatility / (asset_vols + 1e-6)).values
            vol_scalars = np.clip(vol_scalars, 0.1, 2.0)
            
            position_weights = position_weights * vol_scalars

        # Step 4: Cap Position Weights & Format DataFrame
        target_weights = np.clip(position_weights, -self.max_position_size, self.max_position_size)
        
        target_weights_df = pd.DataFrame(
            target_weights, 
            index=tickers, 
            columns=["target_position_weight"]
        )

        # Step 5: Construct Audit Trail Metadata DataFrame
        raw_vals = raw_forecasts.values.flatten() if hasattr(raw_forecasts, "values") else raw_forecasts
        if len(raw_vals) != len(tickers):
            raw_vals = raw_forecasts.reindex(tickers).values.flatten()

        signal_metadata = pd.DataFrame({
            "raw_forecast": raw_vals,
            "masked_forecast": masked_forecasts.values,
            "directional_signal": directional_signals,
            "final_target_weight": target_weights_df["target_position_weight"].values
        }, index=tickers)

        # Step 6: Save DataFrame to Catalog via save_dataframe
        if self.output_signal_weight_metadata:
            target_weights_save_df = target_weights_df.reset_index()
            saved_path = save_dataframe(
                df=target_weights_save_df,
                directory=self.output_signal_weight_metadata["directory"],
                file_name=self.output_signal_weight_metadata["file_name"],
                file_format=self.output_signal_weight_metadata.get("file_format", "csv"),
                versioned=self.output_signal_weight_metadata.get("versioned", True),
                suffix=self.universe_name,
                save_index=False
            )
            print(format_success_text(f"[SignalGenerator]: Exported target weights to: {saved_path}"))

        if self.output_signal_metadata_metadata:
            signal_metadata_save_df = signal_metadata.reset_index()
            saved_path = save_dataframe(
                df=signal_metadata_save_df,
                directory=self.output_signal_metadata_metadata["directory"],
                file_name=self.output_signal_metadata_metadata["file_name"],
                file_format=self.output_signal_metadata_metadata.get("file_format", "csv"),
                versioned=self.output_signal_metadata_metadata.get("versioned", True),
                suffix=self.universe_name,
                save_index=False
            )
            print(format_success_text(f"[SignalGenerator]: Exported target metadata to: {saved_path}"))

        return target_weights_df, signal_metadata

