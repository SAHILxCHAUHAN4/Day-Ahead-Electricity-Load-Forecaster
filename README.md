# Day-Ahead Electricity Load Forecaster

An end-to-end machine learning pipeline that builds a day-ahead hourly electricity load forecasting model for a single smart meter. It utilizes the UCI Individual Household Electric Power Consumption dataset to forecast power consumption 24 hours into the future, comparing an XGBoost model against a naive persistence baseline.

## Project Structure
```text
electricity-load-forecaster/
│
├── data/
│   └── household_power_consumption.txt    # Raw UCI dataset (Downloaded automatically)
│
├── main.py                                # Core end-to-end pipeline script
├── forecast_results.png                   # Output chart of model performance
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

## Methodology

### Step 1: Data Loading & Cleaning
- **Raw Data Loading**: The semicolon-separated text file is loaded using `pandas`. Missing values represented as `?` are treated as `NaN` and dropped.
- **Datetime Indexing**: The separate `Date` and `Time` columns are combined into a single, continuous `Datetime` index.
- **Type Conversion**: All columns are converted to floating-point numbers to ensure suitability for mathematical modeling.

### Step 2: Resampling to Hourly Data
To smooth out high-frequency noise, the minute-level readings are resampled to hourly blocks (`1h`):
- **Energy consumption metrics** (`Global_active_power` and sub-metering columns) are **summed** to represent total active energy.
- **Instantaneous state metrics** (`Voltage`, `Global_intensity`, and `Global_reactive_power`) are **averaged** to represent the steady state.

### Step 3: Feature Engineering
To enable day-ahead forecasting without future data leakage, all features for a target hour $T$ are constructed using information from $T-24$ or earlier:
- **Cyclical Time Encoding**: Sine and cosine transformations are applied to the hour of the day (24h period), day of the week (7d period), and month (12m period) to preserve the temporal continuity of calendar time.
- **Weekend Indicator**: A binary feature distinguishing weekends from weekdays.
- **Historical Target Encoding**: Average load by hour of the day, day of the week, and hour-day combinations (computed strictly on the training set to prevent leakage).
- **Lag Features**: Lags from 24h to 168h (e.g. 24, 25, 26, 48, 72, 96, 120, 144, 168) are added to capture multi-day trends.
- **Rolling Statistics**: Rolling means and standard deviations over 6h, 12h, and 24h are calculated on the 24-hour shifted target.

### Step 4: Chronological Train/Test Split
A strict chronological split is used to prevent data leakage (predicting the past using future information):
- **Train Set**: All records before `2010-01-01` (26,503 samples).
- **Test Set**: All records on or after `2010-01-01` (7,918 samples).

### Step 5: Model Training & Baseline Comparison
- **Naive Baseline**: A 24-hour persistence baseline is established (i.e. forecasting that tomorrow's load equals today's load, $y_T = y_{T-24}$).
- **Predictive Model**: An XGBoost Regressor is trained on the engineered features to predict the target load.
- **Metrics**: Forecast error is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Results & Performance

- **Naive Baseline (Tomorrow = Today)**:
  - **MAE**: 33.39
  - **RMSE**: 49.61
- **XGBoost Regressor Model**:
  - **MAE**: 27.22
  - **RMSE**: 37.63
- **Forecasting Accuracy Improvement**:
  - **MAE** reduced by **18.49%**
  - **RMSE** reduced by **24.16%**

The visual comparison plot is saved to `forecast_results.png`.

---

## Real-World Business Context

### 1. Scaling to Hundreds of Thousands of Meters
To scale this forecasting system from a single smart meter to hundreds of thousands of meters:
- **Data Infrastructure**: Replace in-memory Pandas with a distributed framework like **Apache Spark (PySpark)** to handle out-of-core memory requirements. Ingestion pipelines should write to a scalable cloud data warehouse (e.g., Snowflake) or a dedicated time-series database (e.g., TimescaleDB, ClickHouse).
- **Modeling Approach**: Training 100,000+ local models is operationally complex and computationally expensive. Instead, train a **Global Forecasting Model** (like LightGBM or a Temporal Fusion Transformer) across all meters simultaneously. Introduce static metadata features (e.g., `meter_id`, household size, building type, geographic location) so the model learns shared cross-meter patterns while customizing predictions for individual households.
- **Pipeline Automation**: Containerize the pipeline using Docker and orchestrate it with tools like **Apache Airflow** or **Prefect** to automate data ingestion, daily feature engineering, batch inference, and model monitoring.

### 2. Industry Practice: Complex Models vs Simpler Heuristics
The choice between a complex ML model and a simple heuristic depends on the aggregation level and cost-benefit analysis:
- **Grid/Substation Level (Complex Models Win)**: For aggregated forecasting at the substation or transmission grid level, complex models (like LightGBM, XGBoost, or Deep Learning) are standard. Because load errors on a large scale lead to millions of dollars in peak generation costs or grid instability, a 1% reduction in error yields massive returns, easily justifying the infrastructure cost of running complex models.
- **Individual Meter/Edge Level (Simpler Models Win)**: For individual smart meters, especially if running on the edge or within home energy management devices, computation and memory are heavily constrained. For millions of endpoints, simple heuristics (e.g., dynamic rolling averages, seasonal persistence, or lightweight SARIMA models) are preferred. They offer "good enough" accuracy at a fraction of the computational and maintenance cost, yielding a better overall business ROI.
