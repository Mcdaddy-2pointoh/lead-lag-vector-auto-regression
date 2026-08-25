import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import ValueWarning
import traceback

# Internal Imports
from src.utils.text_operators import format_info_text, format_success_text, format_warn_text, format_error_text
from src.utils.file_operators import *
from src.utils.error_classes import *


class LeadLagVAREngine:
    """
    Class: Fits a Vector Autoregression (VAR) model on stationary log returns,
           conducts Granger Causality testing to detect information diffusion,
           and generates 1-step-ahead price return forecasts.
    """

    def __init__(
        self, 
        config_params: dict, 
        config_catalog: dict
    ) -> None:
        """
        Function: Initialization function for LeadLagVAREngine
        Args:
            config_params (dict): Parameters for VAR estimation (max_lags, ic_criterion, alpha, etc.)
            config_catalog (dict): Catalog mappings for logging, models, and data outputs
        Returns:
            None
        """

        # Keep a local copy of config catalog
        self.config_catalog = config_catalog
        self.data_catalog = self.config_catalog.get("data", {})
        self.logs_catalog = self.config_catalog.get("logs", {})
        self.models_catalog = self.config_catalog.get("models", {})

        # Keep a local copy of config params
        self.config_params = config_params

        # Initialize attributes
        self.var_params = None
        self.max_lags = None
        self.ic_criterion = None
        self.alpha = None
        self.estimation_method = None
        self.error_log_dir = None
        self.universe_name = None
        self.universe_params = None

        # Model and Metadata catalog configs
        self.var_model_catalog = None
        self.var_meta_catalog = None

        self.returns_df = None
        self.model = None
        self.fitted_model = None
        self.optimal_lag = None
        self.phi_matrices = None
        self.granger_matrix = None

        ## Error Logger Setup
        if self.logs_catalog.get('var_engine_error_logs', None) is None:
            print(
                format_warn_text(f"[VAR ENGINE]: No Error Logger Setup in the catalog config for `var_engine_error_logs`")
            )
        else: 
            self.error_log_dir = self.logs_catalog.get('var_engine_error_logs', None)

        ## Model & Metadata Catalog Validation
        if self.models_catalog.get('var_engine_models', None) is None:
            raise VARParamsNotFound(
                format_warn_text(f"[VAR ENGINE]: No Model Catalog Setup in the catalog config for `var_engine_models`")
            )
        else:
            self.var_model_catalog = self.models_catalog.get('var_engine_models', None)

        if self.models_catalog.get('var_engine_metadata', None) is None:
            raise VARParamsNotFound(
                format_warn_text(f"[VAR ENGINE]: No Metadata Catalog Setup in the catalog config for `var_engine_metadata`")
            )
        else:
            self.var_meta_catalog = self.models_catalog.get('var_engine_metadata', None)

        ## VAR Config Validation
        if config_params.get('var_params', None) is None:
            raise VARParamsNotFound()
        else:
            self.var_params = config_params.get('var_params', None)

        # Get max_lags
        if self.var_params.get('max_lags', None) is None:
            raise VARParamMaxLagsNotFound()
        else:
            self.max_lags = self.var_params.get('max_lags', None)

        # Get ic_criterion
        if self.var_params.get('ic_criterion', None) is None:
            raise VARParamICCriterionNotFound()
        else:
            self.ic_criterion = self.var_params.get('ic_criterion', None)

        # Get alpha
        if self.var_params.get('alpha', None) is None:
            raise VARParamAlphaNotFound()
        else:
            self.alpha = self.var_params.get('alpha', None)

        # Get estimation_method
        if self.var_params.get('estimation_method', None) is None:
            raise VARParamEstimationMethodNotFound()
        else:
            self.estimation_method = self.var_params.get('estimation_method', None)

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

        # Process & Validate Config Attributes
        try:
            # Normalize criterion string
            self.ic_criterion = str(self.ic_criterion).lower()
            if self.ic_criterion not in ["aic", "bic", "hqic", "fpe"]:
                raise InvalidICCriterion(format_error_text(f"[VAR ENGINE]: Unsupported information criterion: {self.ic_criterion}"))

            # Ensure positive bounds
            self.max_lags = int(self.max_lags)
            if self.max_lags < 1:
                raise InvalidMaxLagsError(format_error_text(f"[VAR ENGINE]: max_lags must be >= 1, received: {self.max_lags}"))

            self.alpha = float(self.alpha)
            if not (0.0 < self.alpha < 1.0):
                raise InvalidAlphaError(format_error_text(f"[VAR ENGINE]: alpha must be between 0 and 1, received: {self.alpha}"))

            # Normalize estimator string
            self.estimation_method = str(self.estimation_method).lower()
            if self.estimation_method not in ["ols", "mle"]:
                raise InvalidEstimationMethod(format_error_text(f"[VAR ENGINE]: Unsupported estimation method: {self.estimation_method}"))

        except Exception as e:
            print(
                format_error_text(f"[VAR ENGINE]: Failed to process VAR configuration parameters")
            )

            error_message = (
                f"Failed to process VAR configuration parameters\n\n"
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
                    suffix="var_config_validation"
                )

            raise

    def _prepare_returns_dataframe(self) -> None:
        """
        Helper method to clean DATETIME index, retain numeric-only float returns, 
        and assign index frequency to avoid statsmodels parsing errors.
        """
        if 'Datetime' in self.returns_df.columns:
            self.returns_df['Datetime'] = pd.to_datetime(self.returns_df['Datetime'])
            self.returns_df = self.returns_df.set_index('Datetime')

        if not isinstance(self.returns_df.index, pd.DatetimeIndex):
            self.returns_df.index = pd.to_datetime(self.returns_df.index)

        # Filter numeric return columns strictly
        self.returns_df = self.returns_df.select_dtypes(include=[np.number])

        # Infer index frequency if possible
        inferred_freq = pd.infer_freq(self.returns_df.index)
        if inferred_freq is not None:
            self.returns_df.index.freq = inferred_freq

    def fit_var_model(
        self,
        log_returns_df: pd.DataFrame = None
    ) -> dict:
        """
        Function: Selects optimal lag order (p) via Information Criteria and estimates 
                  the VAR coefficient matrices (Phi_1 ... Phi_p) via OLS/MLE.
        Args:
            log_returns_df (pd.DataFrame | optional): Stationary log-returns matrix (T x N).
        Returns:
            summary_dict: Fitted parameters and diagnostic metrics.
        """
        try:
            # Load the latest returns file if log_returns_df is None
            if (log_returns_df is None) or not isinstance(log_returns_df, pd.DataFrame):
                log_returns_catalog_metadata = self.data_catalog.get('collector_yfin_log_returns', {})

                if log_returns_catalog_metadata == {}:
                    raise DataKeyNotFound(
                        format_error_text(f"[VAR ENGINE]: Could not find key `collector_yfin_log_returns` in `data` from `config_catalog.yaml`. To load the log returns data")
                    )
                else:
                    self.returns_df = load_latest_dataframe(
                        directory=log_returns_catalog_metadata['directory'],
                        file_name=log_returns_catalog_metadata['file_name'],
                        file_format=log_returns_catalog_metadata['file_format'],
                        suffix=self.universe_name
                    )
            else:
                self.returns_df = log_returns_df.copy()

            # Clean DATETIME index and filter strictly numeric data matrix
            self._prepare_returns_dataframe()

            # Initialise a VAR model with warning suppression for intraday date gaps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ValueWarning)
                self.model = VAR(self.returns_df)

                # Select Optimal Lags (P) based on the information criterion
                lag_selection = self.model.select_order(maxlags=self.max_lags)
                selected_lags = getattr(lag_selection, self.ic_criterion)

                # Fallback to 1st lag if model selects 0 and warn
                self.optimal_lag = max(1, selected_lags) if selected_lags is not None else 1

                if (selected_lags is None) or (selected_lags == 0):
                    print(format_warn_text(f"[VAR ENGINE]: No optimal lag selected, defaulted to 1"))
                else:
                    print(format_success_text(
                        f"[VAR ENGINE]: Selected optimal lag p = {self.optimal_lag} minute(s) via '{self.ic_criterion.upper()}' criterion."
                    ))

                # Fit the model using OLS or MLE
                if self.estimation_method == "ols":
                    self.fitted_model = self.model.fit(self.optimal_lag)
                    self.phi_matrices = self.fitted_model.params
                    print(format_info_text(f"[VAR Engine]: Fitted VAR({self.optimal_lag}) using OLS."))       

                else:
                    self.fitted_model = self.model.fit(self.optimal_lag)
                    residuals = self.fitted_model.resid
                    n_obs = len(residuals)
                    
                    # MLE Sigma = (E^T * E) / T
                    sigma_mle = (residuals.T @ residuals) / n_obs
                    self.phi_matrices = self.fitted_model.params
                    print(format_info_text(f"[VAR Engine]: Fitted VAR({self.optimal_lag}) using MLE (Uncorrected Sigma Divisor T={n_obs})."))

            phi_file_path = None

            # Save only if not empty
            if not (self.phi_matrices.empty or self.phi_matrices.shape[1] == 0):

                # Warn if data catalog config not setup
                if self.data_catalog.get('var_engine_phi_matrice', None) is None:
                    print(
                        format_warn_text(
                            f"No Data Catalog Setup in the catalog config for `var_engine_phi_matrice`. Will run on In Memory form"
                        )
                    )
                else: 
                    phi_matix_metadata = self.data_catalog.get('var_engine_phi_matrice', None)
                    phi_file_path = save_dataframe(
                        df=self.phi_matrices,
                        directory=phi_matix_metadata['directory'],
                        file_name=phi_matix_metadata['file_name'],
                        file_format=phi_matix_metadata['file_format'],
                        versioned=phi_matix_metadata['versioned'],
                        suffix=self.universe_name,
                        save_index=True
                    )

            else:
                self.phi_matrices = None

            # Save Model and Metadata
            model_path = save_model(
                model_object=self.fitted_model,
                directory=self.var_model_catalog['directory'],
                file_name=self.var_model_catalog['file_name'],
                suffix=self.universe_name,
                versioned=self.var_model_catalog['versioned']
            )

            model_metadata = {
                "model_name": self.var_meta_catalog['file_name'],
                "optimal_lag": int(self.optimal_lag) if self.optimal_lag else None,
                "ic_criterion": str(self.ic_criterion),
                "alpha": float(self.alpha) if self.alpha else None,
                "asset_universe": list(self.returns_df.columns) if self.returns_df is not None else [],
                "n_assets": len(self.returns_df.columns) if self.returns_df is not None else 0,
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
            }

            model_metadata_path = save_dictionary(
                data=model_metadata,
                directory=self.var_meta_catalog['directory'],
                file_name=self.var_meta_catalog['file_name'],
                suffix=self.universe_name,
                versioned=self.var_meta_catalog['versioned'],
            )

            return {
                "optimal_lag" : self.optimal_lag,
                "method" : self.estimation_method,
                "phi_matrice_path" : phi_file_path,
                "phi_matrie": self.phi_matrices,
                "fitted_model" : self.fitted_model,
                "fitted_model_path" : model_path,
                "model_metdata" : model_metadata,
                "model_metdata_path" : model_metadata_path
            }

        except Exception as e:
            print(format_error_text(f"[VAR ENGINE]: Failed to fit VAR model for universe: {self.universe_name}"))
            error_message = f"Failed to fit VAR model: {self.universe_name}\n\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            if self.error_log_dir is not None:
                save_text(
                    text=error_message,
                    directory=self.error_log_dir['directory'],
                    file_name=self.error_log_dir['file_name'],
                    versioned=self.error_log_dir['versioned'],
                    suffix="var_model_fitting"
                )
            raise

    def compute_grangers_causality(
        self,
        log_returns_df: pd.DataFrame = None,
        phi_matrices: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Function: Runs pairwise Granger Causality tests (F-tests) across all asset pairs (A -> B),
                  interprets directional precedence (Strict Lead-Lag vs. Feedback Loops),
                  prints a formatted quantitative analysis, and saves the binary signal mask.
        Args:
            log_returns_df (pd.DataFrame | optional): Stationary log-returns matrix (T x N).
            phi_matrices (pd.DataFrame | optional): Coefficient matrix extracted from VAR model.
        Returns:
            granger_pvalue_matrix (pd.DataFrame): Matrix containing p-values where rows are leaders and columns are laggers.
        """
        try:
            # 1. Load the latest returns file if log_returns_df is None
            if (log_returns_df is None) or not isinstance(log_returns_df, pd.DataFrame):
                log_returns_catalog_metadata = self.data_catalog.get('collector_yfin_log_returns', {})

                if log_returns_catalog_metadata == {}:
                    raise DataKeyNotFound(
                        format_error_text(
                            f"[VAR ENGINE]: Could not find key `collector_yfin_log_returns` in `data` from `config_catalog.yaml`."
                        )
                    )
                else:
                    self.returns_df = load_latest_dataframe(
                        directory=log_returns_catalog_metadata['directory'],
                        file_name=log_returns_catalog_metadata['file_name'],
                        file_format=log_returns_catalog_metadata['file_format'],
                        suffix=self.universe_name
                    )
            else:
                self.returns_df = log_returns_df

            # Clean DATETIME index and filter strictly numeric data matrix
            self._prepare_returns_dataframe()

            # 2. Smart Check: Try Loading Latest Model from Disk if fitted_model is None
            if self.fitted_model is None:
                try:
                    if self.var_model_catalog and self.var_meta_catalog:
                        print(format_info_text(f"[VAR ENGINE]: Checking catalog for pre-trained VAR model..."))
                        
                        self.fitted_model = load_latest_model(
                            directory=self.var_model_catalog['directory'],
                            file_name=self.var_model_catalog['file_name'],
                            suffix=self.universe_name
                        )

                        model_metadata = load_latest_dictionary(
                            directory=self.var_meta_catalog['directory'],
                            file_name=self.var_meta_catalog['file_name'],
                            suffix=self.universe_name
                        )

                        self.optimal_lag = model_metadata.get("optimal_lag", self.optimal_lag)
                        self.phi_matrices = self.fitted_model.params
                        print(format_success_text(f"[VAR ENGINE]: Successfully loaded existing VAR({self.optimal_lag}) model from catalog."))

                except Exception as load_err:
                    print(format_warn_text(f"[VAR ENGINE]: Could not load existing model from storage ({str(load_err)}). Backtracking to fit a new model..."))
                    self.fitted_model = None

            # 3. Fallback: Re-fit VAR Model if no model could be loaded
            if self.fitted_model is None:
                print(format_info_text(f"[VAR ENGINE]: Fitting new VAR model on input dataset..."))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ValueWarning)
                    self.model = VAR(self.returns_df)
                    lag_selection = self.model.select_order(maxlags=self.max_lags)
                    selected_lags = getattr(lag_selection, self.ic_criterion)
                    self.optimal_lag = max(1, selected_lags) if selected_lags is not None else 1
                    self.fitted_model = self.model.fit(self.optimal_lag)
                    self.phi_matrices = self.fitted_model.params

            # 4. Ensure phi_matrices is loaded or assigned
            if (phi_matrices is not None) and isinstance(phi_matrices, pd.DataFrame):
                self.phi_matrices = phi_matrices

            symbols = list(self.returns_df.columns)
            n_assets = len(symbols)

            print(format_info_text(f"[VAR ENGINE]: Computing pairwise Granger Causality tests across {n_assets} assets..."))

            p_val_matrix = pd.DataFrame(
                np.ones((n_assets, n_assets)),
                index=symbols,
                columns=symbols
            )

            for leader in symbols:
                for lagger in symbols:
                    if leader == lagger:
                        continue
                    try:
                        test_res = self.fitted_model.test_causality(
                            caused=lagger,
                            causing=leader,
                            kind='f'
                        )
                        p_val_matrix.loc[leader, lagger] = round(float(test_res.pvalue), 6)

                    except Exception as pair_err:
                        print(
                            format_warn_text(
                                f"[VAR ENGINE]: Could not compute Granger test for pair {leader} -> {lagger}: {str(pair_err)}"
                            )
                        )

            self.granger_matrix = p_val_matrix
            print(format_success_text(f"[VAR ENGINE]: Successfully computed Granger Causality p-value matrix."))

            # 5. DIRECTIONAL INTERPRETATION & SIGNAL MASK GENERATION
            signal_mask = pd.DataFrame(
                np.zeros((n_assets, n_assets), dtype=int),
                index=symbols,
                columns=symbols
            )

            pair_records = []
            processed_pairs = set()

            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    stock_a = symbols[i]
                    stock_b = symbols[j]

                    pair_key = tuple(sorted([stock_a, stock_b]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    p_a_to_b = self.granger_matrix.loc[stock_a, stock_b]
                    p_b_to_a = self.granger_matrix.loc[stock_b, stock_a]

                    sig_a_to_b = p_a_to_b < self.alpha
                    sig_b_to_a = p_b_to_a < self.alpha

                    if sig_a_to_b and not sig_b_to_a:
                        rel_type = "STRICT LEAD-LAG"
                        leader_name, lagger_name = stock_a, stock_b
                        action = f"Trade {lagger_name} using {leader_name} forecast"
                        signal_mask.loc[stock_a, stock_b] = 1

                    elif sig_b_to_a and not sig_a_to_b:
                        rel_type = "STRICT LEAD-LAG (REVERSE)"
                        leader_name, lagger_name = stock_b, stock_a
                        action = f"Trade {lagger_name} using {leader_name} forecast"
                        signal_mask.loc[stock_b, stock_a] = 1

                    elif sig_a_to_b and sig_b_to_a:
                        rel_type = "FEEDBACK LOOP"
                        leader_name, lagger_name = "BOTH", "BOTH"
                        action = "BLOCK / REDUCE WEIGHT (Systemic Co-movement)"

                    else:
                        rel_type = "INDEPENDENT"
                        leader_name, lagger_name = "NONE", "NONE"
                        action = "IGNORE (No Statistical Causality)"

                    pair_records.append({
                        "Asset A": stock_a,
                        "Asset B": stock_b,
                        "p(A->B)": p_a_to_b,
                        "p(B->A)": p_b_to_a,
                        "Relationship": rel_type,
                        "Leader": leader_name,
                        "Lagger": lagger_name,
                        "Action": action
                    })

            self.granger_pairs_df = pd.DataFrame(pair_records)
            self.signal_mask = signal_mask

            # Print Terminal Breakdown
            print("\n" + "=" * 80)
            print(f"       GRANGER CAUSALITY QUANTITATIVE INTERPRETATION (Alpha = {self.alpha})")
            print("=" * 80)
            print("\n[1] PAIRWISE DIRECTIONAL BREAKDOWN:\n")
            for _, row in self.granger_pairs_df.iterrows():
                if row["Relationship"].startswith("STRICT"):
                    tag = "[VALID LINK]   "
                elif row["Relationship"] == "FEEDBACK LOOP":
                    tag = "[FEEDBACK LOOP]"
                else:
                    tag = "[NO LINK]      "

                print(f"{tag} {row['Asset A']} vs {row['Asset B']}:")
                print(f"    - p({row['Asset A']} -> {row['Asset B']}): {row['p(A->B)']}")
                print(f"    - p({row['Asset B']} -> {row['Asset A']}): {row['p(B->A)']}")
                print(f"    - Classification : {row['Relationship']}")
                print(f"    - Action         : {row['Action']}\n")

            print("-" * 80)
            print("[2] TRADING RECOMMENDATION SUMMARY:")
            strict_pairs = self.granger_pairs_df[self.granger_pairs_df["Relationship"].str.startswith("STRICT")]
            if not strict_pairs.empty:
                for _, row in strict_pairs.iterrows():
                    print(f"    >>> SIGNAL ALLOWED: Predict {row['Lagger']} based on {row['Leader']} moves.")
            else:
                print("    >>> NO STRICT UNIDIRECTIONAL PAIRS FOUND.")
            print("=" * 80 + "\n")

            # 6. SAVE OUTPUTS TO CATALOG
            granger_catalog_metadata = self.data_catalog.get('var_engine_granger_matrix', {})
            if granger_catalog_metadata != {}:
                save_dataframe(
                    df=self.granger_matrix,
                    directory=granger_catalog_metadata['directory'],
                    file_name=granger_catalog_metadata['file_name'],
                    file_format=granger_catalog_metadata.get('file_format', 'csv'),
                    versioned=granger_catalog_metadata.get('versioned', True),
                    suffix=self.universe_name,
                    save_index=True
                )

            mask_catalog_metadata = self.data_catalog.get('var_engine_signal_mask', {})
            if mask_catalog_metadata != {}:
                save_dataframe(
                    df=self.signal_mask,
                    directory=mask_catalog_metadata['directory'],
                    file_name=mask_catalog_metadata['file_name'],
                    file_format=mask_catalog_metadata.get('file_format', 'csv'),
                    versioned=mask_catalog_metadata.get('versioned', True),
                    suffix=self.universe_name,
                    save_index=True
                )

            return self.granger_matrix

        except Exception as e:
            print(format_error_text(f"[VAR ENGINE]: Failed to compute Granger Causality matrix for universe: {self.universe_name}"))
            error_message = f"Failed to compute Granger Causality matrix: {self.universe_name}\n\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            if self.error_log_dir is not None:
                save_text(
                    text=error_message,
                    directory=self.error_log_dir['directory'],
                    file_name=self.error_log_dir['file_name'],
                    versioned=self.error_log_dir['versioned'],
                    suffix="var_granger_causality"
                )
            raise

    def generate_next_bar_signal(
        self,
        log_returns_df: pd.DataFrame = None,
        phi_matrices: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Function: Generates 1-step-ahead (t+1) price log-return forecasts (hat{Y}_{t+1}) 
                  for all assets in the universe using the estimated VAR model and recent lag history.
        Args:
            log_returns_df (pd.DataFrame | optional): Stationary log-returns matrix (T x N).
            phi_matrices (pd.DataFrame | optional): Coefficient matrix extracted from VAR model.
        Returns:
            forecast_df (pd.DataFrame): Predicted 1-step-ahead returns for each stock (1 x N).
        """
        try:
            # 1. Load the latest returns file if log_returns_df is None
            if (log_returns_df is None) or not isinstance(log_returns_df, pd.DataFrame):
                log_returns_catalog_metadata = self.data_catalog.get('collector_yfin_log_returns', {})

                if log_returns_catalog_metadata == {}:
                    raise DataKeyNotFound(
                        format_error_text(
                            f"[VAR ENGINE]: Could not find key `collector_yfin_log_returns` in `data` from `config_catalog.yaml`."
                        )
                    )
                else:
                    self.returns_df = load_latest_dataframe(
                        directory=log_returns_catalog_metadata['directory'],
                        file_name=log_returns_catalog_metadata['file_name'],
                        file_format=log_returns_catalog_metadata['file_format'],
                        suffix=self.universe_name
                    )
            else:
                self.returns_df = log_returns_df

            # Clean DATETIME index and filter strictly numeric data matrix
            self._prepare_returns_dataframe()

            # 2. Smart Check: Try Loading Latest Model from Disk if fitted_model is None
            if self.fitted_model is None:
                try:
                    if self.var_model_catalog and self.var_meta_catalog:
                        print(format_info_text(f"[VAR ENGINE]: Checking catalog for pre-trained VAR model..."))

                        self.fitted_model = load_latest_model(
                            directory=self.var_model_catalog['directory'],
                            file_name=self.var_model_catalog['file_name'],
                            suffix=self.universe_name
                        )

                        model_metadata = load_latest_dictionary(
                            directory=self.var_meta_catalog['directory'],
                            file_name=self.var_meta_catalog['file_name'],
                            suffix=self.universe_name
                        )

                        self.optimal_lag = model_metadata.get("optimal_lag", self.optimal_lag)
                        self.phi_matrices = self.fitted_model.params
                        print(format_success_text(f"[VAR ENGINE]: Successfully loaded existing VAR({self.optimal_lag}) model from catalog."))

                except Exception as load_err:
                    print(format_warn_text(f"[VAR ENGINE]: Could not load existing model from storage ({str(load_err)}). Backtracking to fit a new model..."))
                    self.fitted_model = None

            # 3. Fallback: Re-fit VAR Model if no model could be loaded
            if self.fitted_model is None:
                print(format_info_text(f"[VAR ENGINE]: Fitting new VAR model on input dataset..."))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ValueWarning)
                    self.model = VAR(self.returns_df)
                    lag_selection = self.model.select_order(maxlags=self.max_lags)
                    selected_lags = getattr(lag_selection, self.ic_criterion)
                    self.optimal_lag = max(1, selected_lags) if selected_lags is not None else 1
                    self.fitted_model = self.model.fit(self.optimal_lag)
                    self.phi_matrices = self.fitted_model.params

            # 4. Extract recent p-lag observations and produce 1-step forecast
            k_ar = self.fitted_model.k_ar
            last_observations = self.returns_df.values[-k_ar:]

            forecast_values = self.fitted_model.forecast(y=last_observations, steps=1)

            forecast_df = pd.DataFrame(
                forecast_values,
                columns=self.returns_df.columns,
                index=["forecast_return_t_plus_1"]
            )

            print(format_success_text(f"[VAR ENGINE]: Successfully generated 1-step-ahead return forecasts for '{self.universe_name}'."))

            # Save Forecast signal
            forecast_catalog_metadata = self.data_catalog.get('var_engine_forecast_signals', {})
            if forecast_catalog_metadata != {}:
                save_dataframe(
                    df=forecast_df,
                    directory=forecast_catalog_metadata['directory'],
                    file_name=forecast_catalog_metadata['file_name'],
                    file_format=forecast_catalog_metadata.get('file_format', 'csv'),
                    versioned=forecast_catalog_metadata.get('versioned', True),
                    suffix=self.universe_name,
                    save_index=True
                )

            return forecast_df

        except Exception as e:
            print(format_error_text(f"[VAR ENGINE]: Failed to generate forecasts for universe: {self.universe_name}"))
            error_message = f"Failed to generate return forecasts: {self.universe_name}\n\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            if self.error_log_dir is not None:
                save_text(
                    text=error_message,
                    directory=self.error_log_dir['directory'],
                    file_name=self.error_log_dir['file_name'],
                    versioned=self.error_log_dir['versioned'],
                    suffix="var_forecast_generation"
                )
            raise