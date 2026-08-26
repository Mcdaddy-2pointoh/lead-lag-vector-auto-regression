# Lead-Lag Vector Autoregression (VAR) Trading Engine

An end-to-end quantitative trading pipeline designed for high-frequency intraday trading on the Indian Equity Market (NSE). 

This engine identifies **directional lead-lag relationships** between equities using Vector Autoregression (VAR) and Granger Causality. It automatically downloads market data, fits econometric models, filters out statistical noise, and generates risk-adjusted portfolio position weights for live execution.

---

## System Architecture

The pipeline is fully modular, configuration-driven, and divided into three core phases:

### 1. Data Ingestion & QC (`YFinanceNSEPipeline`)
* **Intraday Data Extraction:** Fetches 1-minute or 5-minute bar data via `yfinance`.
* **Market Session Normalization:** Cleans data and restricts bars to NSE trading hours (09:30 - 15:30 IST).
* **Stationarity Assurance:** Runs Augmented Dickey-Fuller (ADF) tests and computes log-returns to ensure stationary inputs for the econometric engine.

### 2. Econometric Engine (`LeadLagVAREngine`)
* **VAR(p) Modeling:** Fits multi-asset Vector Autoregression models on the log-return matrices.
* **Lag Optimization:** Automatically selects the optimal lag length using Information Criteria (AIC/BIC).
* **Granger Causality Matrix:** Computes directional pairwise causality to map which stocks lead and which lag.
* **Forecasting:** Generates 1-step-ahead log-return predictions.

### 3. Execution & Risk Layer (`SignalGenerator`)
* **Binary Masking:** Multiplies raw forecasts by the Granger Causality matrix to block signals from non-causal assets and feedback loops.
* **Deadband Thresholding:** Drops weak signals below a configurable threshold (e.g., 5 bps) to protect against exchange fees and slippage.
* **Risk-Parity Volatility Scaling:** Scales high-beta assets down to meet an annualized target portfolio volatility (e.g., 15%).
* **Output:** Generates a structured DataFrame of target `LONG`/`SHORT`/`FLAT` portfolio weights.

---

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/lead-lag-vector-auto-regression.git](https://github.com/your-username/lead-lag-vector-auto-regression.git)
   cd lead-lag-vector-auto-regression
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv ll-var-venv
   source ll-var-venv/bin/activate  # On Windows: ll-var-venv\Scripts\activate
   pip install -r requirements.txt
   ```

*Core Dependencies: `pandas`, `numpy`, `statsmodels`, `yfinance`, `joblib`, `pyyaml`*

---

## Configuration

The entire pipeline is driven by two YAML files located in the `config/` directory.

### 1. `config_params.yaml` (Hyperparameters)
Defines the trading universe, model constraints, and risk limits.

```yaml
time_params:
  interval: "1m"
  period: "7d"

universe_params:
  universe_name: "tech_eqs"
  tech_eqs: ["TCS", "INFY", "MPHASIS", "LTIM", "COFORGE"]

var_params:
  max_lags: 10
  ic_criterion: "aic"
  alpha: 0.05

granger_causal_signal_gen_params:
  min_log_returns_threshold: 0.0005  # 5 bps minimum edge
  max_position_size: 1.0             # Cap leverage at 100% per asset
  volatility_scaling: true
  target_volatility: 0.15            # 15% annualized target volatility
```

### 2. `config_catalog.yaml` (Data I/O)
An enterprise-grade data catalog mapping all intermediate artifacts (raw data, cleaned data, log-returns, models, matrices, and final signals) to specific directory paths. The pipeline relies on custom `file_operators.py` to seamlessly read and write versioned files using this catalog.

---

## Usage

To run the complete end-to-end pipeline (Data Ingestion -> Econometric Modeling -> Signal Generation):

```bash
notebooks/main.ipynb
```

**Expected Output:**
The script will output a target position weight matrix to your designated catalog directory (e.g., `data/signals/target_position_weights_YYYYMMDD_HHMMSS.csv`), looking like this:

| target_position_weight |
| :--- |
| **TCS** | 0.0000 |
| **INFY** | 0.0000 |
| **MPHASIS** | 0.0000 |
| **COFORGE** | -0.5475 |

---

## Roadmap

**Next Release (v1.1.0): Bayesian VAR (BVAR) Engine**
To prevent parameter explosion and overfitting when expanding the asset universe ($N > 30$), the next major update will introduce **Bayesian VAR with Minnesota Priors**. This will apply algorithmic shrinkage toward a Random Walk baseline, allowing for robust, sub-50ms high-frequency forecasting on large asset baskets.

---

## Disclaimer
**For Educational and Research Purposes Only.** 
This software is not financial advice. Trading equities, especially at high frequencies, carries significant financial risk. The authors and contributors are not responsible for any financial losses incurred from using this codebase in live markets. Always backtest and paper-trade your strategies before deploying capital.
