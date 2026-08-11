# Time Series Project Report

## 1. Project Overview

This report explains the complete logic implemented in `app.py` for the Time Series forecasting application. It covers:

- how the data is read and prepared,
- how datetime and target columns are detected,
- how the dataset is cleaned,
- how series characteristics are analyzed,
- which forecasting models are trained,
- how each model computes forecasts,
- how models are ranked,
- and how the final recommended model is selected.

The purpose of this application is to provide a robust time series forecasting pipeline with automatic data detection, cleaning, model training, scoring, and selection.

---

## 2. Key Imports and Their Purpose

`app.py` imports libraries for the following reasons:

- `streamlit as st`: build the interactive web UI, show progress spinners, display messages, and manage caching.
- `warnings`: suppress non-critical warnings while preserving important numerical warnings.
- `numpy as np` and `pandas as pd`: core numeric and table operations for time series manipulation.
- `plotly.express`, `plotly.graph_objects`, `make_subplots`: plot interactive charts.
- `scipy.stats.linregress`: compute linear regression for trend detection.
- `sklearn.ensemble.IsolationForest`: optional anomaly detection for outlier handling.
- `sklearn.metrics.mean_absolute_error`, `mean_squared_error`, `r2_score`: compute forecast performance metrics.
- `statsmodels.tsa.arima.model.ARIMA`: fit ARIMA models.
- `statsmodels.tsa.holtwinters.ExponentialSmoothing`: fit Holt-Winters exponential smoothing models.
- `statsmodels.tsa.seasonal.seasonal_decompose`: separate trend, seasonal, and residual components.
- `statsmodels.tsa.statespace.sarimax.SARIMAX`: fit seasonal ARIMA models.
- `statsmodels.tsa.stattools.acf`, `adfuller`, `pacf`: compute autocorrelation and check stationarity.

Utility modules imported from `utils/` provide file upload handling, memory management, and session state management.

---

## 3. Data Detection and Format Helpers

### 3.1 `safe_to_datetime(series)`

This function converts a pandas Series to datetime safely and robustly. It supports:

- already-datetime data,
- numeric timestamps,
- Excel serial dates,
- Unix timestamps in seconds and milliseconds,
- ISO8601 strings,
- mixed string formats,
- missing values.

How it works:

1. If already datetime, it returns the original series after stripping time zone information.
2. If numeric, it tests the values against common timestamp ranges:
   - Excel dates when median value is between 20,000 and 90,000.
   - Unix milliseconds when median > 1e11.
   - Unix seconds when median is between 1e8 and 1e11.
3. For string-like values, it trims whitespace and normalizes empty/null markers before parsing.
4. It uses pandas' datetime parser with different strategies:
   - exact parsing,
   - day-first parsing,
   - inferred format parsing.

The function returns the best datetime conversion when at least 60% of values parse successfully.

### 3.2 `detect_datetime_column(df)`

Automatically finds the most likely datetime column in the dataset.

Process:

1. Search column names for common datetime keywords such as `date`, `timestamp`, `time`, `created_at`, and `log_date`.
2. For each candidate column, attempt to parse it with `safe_to_datetime()` and measure parse success.
3. If a candidate column parses successfully for at least 60% of values, it is selected.
4. If no name-based candidate is found, every column is scored based on parsing success and uniqueness.
5. If still not found, numeric columns are tested for Excel or Unix timestamp behavior with a 80% success threshold.

### 3.3 `detect_target_column(df)`

Finds the best numeric target variable for forecasting.

It scores numeric columns based on:

- whether the name matches priority forecast targets like `sales`, `revenue`, `temperature`, `demand`, `load`, `price`, `volume`, `close`, etc.
- the percentage of missing values.
- whether the column has positive variance.
- how continuous it is (unique ratio > 0.30).
- whether it has more than 20 distinct values.

The column with the highest score is returned as the forecasting target.

### 3.4 `detect_frequency(df, date_col)`

Detects the temporal frequency of the date index and returns:

- `code`: pandas frequency alias, e.g. `D`, `W`, `M`, `Q`, `Y`, `H`
- `name`: human-friendly description, e.g. Daily, Weekly
- `seasonal_period`: expected seasonal cycle length
- `delta`: observed interval

Detection uses:

- `pd.infer_freq()` for standard frequencies, and if that fails,
- manual analysis of the median difference between sorted dates.

It supports irregular series and returns `IRREGULAR` when spacing does not fit standard calendar units.

---

## 4. Dataset Analysis

### 4.1 `analyze_dataset(df)`

Builds a summary of dataset quality and the time series characteristics.

The output includes:

- row and column counts,
- missing value percentage,
- duplicate row count,
- detected date column and target column,
- list of numeric columns,
- inferred dataset type (weather, sales, stock, IoT, energy, traffic, or generic),
- frequency detection,
- trend and seasonality analysis,
- time span (`start_date`, `end_date`, `total_days`).

When both a date column and a target column are present, it calls `detect_trend_and_seasonality()` for richer intelligence.

### 4.2 Duplicate `detect_trend_and_seasonality()` Definitions

The file defines `detect_trend_and_seasonality()` twice. In Python, the second definition overrides the first one, so the actual runtime behavior is controlled by the second version.

The later version is a master intelligence engine that combines:

- `detect_trend()`
- `detect_seasonality()`
- `analyze_signal_quality()`

Then it selects a recommended model and computes a forecast readiness grade.

---

## 5. Time Series Cleaning and Preparation

### 5.1 `clean_time_series(df, date_col, value_col)`

This function cleans the time series for forecasting and returns a cleaned dataframe plus a report of changes.

Key steps:

1. Convert the selected date column to datetime using `safe_to_datetime()`.
2. Remove rows with invalid timestamps.
3. Remove duplicate timestamps.
4. Sort chronologically.
5. Set the date column as the index.
6. Detect frequency again.
7. If the frequency is regular, reindex the dataset to fill missing timestamps.
8. Interpolate missing target values using time-based interpolation, then forward-fill and backward-fill.
9. Remove duplicate rows and reset the index.

The cleaning report tracks:

- how many rows were removed,
- how many duplicates were removed,
- how many missing dates were filled,
- how many missing values were filled,
- final row count,
- detected frequency.

### 5.2 `prepare_forecast_data(df, date_col, value_col)`

This is the main preparation function for the forecasting pipeline.

It performs:

- cleaning with `clean_time_series()`,
- dataset analysis with `analyze_dataset()`,
- setting the date column as the DataFrame index,
- selecting only the target column,
- numeric coercion of the target,
- dropping rows with missing target values,
- dataset downsampling for extremely large series (over 50,000 observations),
- validation to ensure at least 20 values remain.

The returned metadata includes both cleaning and analysis results.

### 5.3 `split_train_test(df, target_col)`

Splits the cleaned dataset into training and test sets using an adaptive holdout size:

- 30% test for datasets smaller than 100 rows,
- 20% test for 100–999 rows,
- 15% test for 1,000–9,999 rows,
- 10% test for 10,000+ rows.

This preserves the temporal order by taking the first portion as training and the last portion as test.

---

## 6. Forecast Performance Metrics

### 6.1 `calculate_forecast_metrics(actual, predicted)`

Computes four standard forecast error metrics:

- `MAE` (Mean Absolute Error): average absolute error.
  - formula: `mean(|actual - predicted|)`
- `RMSE` (Root Mean Squared Error): square root of average squared error.
  - formula: `sqrt(mean((actual - predicted)^2))`
- `MAPE` (Mean Absolute Percentage Error): average absolute percent error.
  - formula: `mean(|actual - predicted| / max(|actual|, 1e-9)) * 100`
  - uses `max(|actual|, 1e-9)` to avoid division by zero.
- `R2` (Coefficient of Determination): proportion of variance explained.

The function returns rounded values for readability.

---

## 7. Model Training and Forecast Algorithms

`app.py` trains several forecasting approaches and evaluates each against the test set.

### 7.1 `train_forecasting_models(train_df, test_df, target_col, models_to_train=None)`

By default, the following models are trained:

- Naive
- Moving Average
- ARIMA
- SARIMA
- Holt-Winters

Each model trains sequentially, and the results dictionary stores:

- the fitted model object,
- the forecast values for the test window,
- evaluation metrics,
- any additional metadata.

#### 7.1.1 Naive Forecast

- Forecast for all future points using the last observed training value.
- formula: `forecast = [last_value] * steps`
- serves as a simple baseline.

#### 7.1.2 Moving Average Forecast

- Uses the mean of the last `window` training values.
- `window` is chosen as `min(10, max(3, len(y_train)//10))`.
- forecast is constant: `forecast = [average_of_last_window] * steps`

### 7.2 `train_arima_model(train_df, test_df, target_col, max_p=3, max_d=2, max_q=3)`

Trains an ARIMA model with automatic order search.

ARIMA stands for Autoregressive Integrated Moving Average:

- `p` = autoregressive order,
- `d` = number of differences to achieve stationarity,
- `q` = moving average order.

How it works:

1. Calls `check_stationarity()` to determine the recommended differencing order `d`.
2. Adapts search ranges based on dataset size.
3. Iterates over candidate `(p, d, q)` combinations.
4. Fits each ARIMA model and forecasts the test window.
5. Computes RMSE and retains the best model by lowest RMSE.

The returned output includes:

- the best fitted model,
- the best forecast,
- evaluation metrics,
- chosen order,
- stationarity diagnostics.

### 7.3 `train_sarima_model(train_df, test_df, date_col, target_col, max_p=2, max_q=2, max_P=1, max_Q=1)`

Trains a Seasonal ARIMA model when seasonality is present.

SARIMA includes:

- nonseasonal order `(p, d, q)`,
- seasonal order `(P, D, Q, m)` where `m` is the seasonal period.

How it works:

1. Uses `check_stationarity()` to determine `d`.
2. Uses `detect_seasonality()` to determine the seasonal period `m`.
3. Throws a clear error if seasonality is not detected.
4. Searches over a restricted parameter grid based on dataset size.
5. Fits `SARIMAX` models with `enforce_stationarity=False` and `enforce_invertibility=False`.
6. Chooses the best model by RMSE.

### 7.4 `train_holt_winters_model(train_df, test_df, date_col, target_col)`

Trains a Holt-Winters exponential smoothing model.

Exponential smoothing models are suitable for series with level, trend, and seasonality.

How it works:

1. Detects seasonality using `detect_seasonality()`.
2. Detects long-term trend using `detect_trend()`.
3. Chooses additive trend if the series has a strong trend.
4. Chooses additive seasonality if moderate or strong seasonality exists.
5. Fits `ExponentialSmoothing` and forecasts the test horizon.

This model is especially useful when the series has consistent trend and seasonality patterns but may not require an ARIMA structure.

### 7.5 `generate_future_forecast(df, date_col, value_col, model_name, forecast_steps)`

Generates future forecast values for the selected model and returns prediction intervals when available.

It supports:

- ARIMA: selects the best `(p,d,q)` by AIC, uses `get_forecast()`, and returns confidence intervals.
- SARIMA: fits a seasonal model with default order `(1,d,1)` and seasonal order `(1,d,1,m)`.
- Holt-Winters: fits an exponential smoothing model and returns point forecasts.
- Moving Average: constant forecast equal to the recent rolling mean.
- Naive: constant forecast equal to the latest observed value.

This function also computes future dates using the detected frequency or daily frequency if detection fails.

---

## 8. Model Ranking and Selection

### 8.1 `select_best_forecasting_model(results)`

Ranks trained models using a leaderboard built from metrics.

For each model, the function records:

- `RMSE`
- `MAE`
- `MAPE`
- `R2`

Then it computes rankings:

- `RMSE Rank`: lower RMSE is better.
- `MAE Rank`: lower MAE is better.
- `MAPE Rank`: lower MAPE is better.
- `R2 Rank`: higher R2 is better.

The overall score is the sum of the four ranks:

- `Overall Score = RMSE Rank + MAE Rank + MAPE Rank + R2 Rank`

The model with the lowest overall score is selected as the best model.

This approach treats each metric equally and rewards models that perform consistently well across error and explained variance metrics.

### 8.2 `forecast_pipeline(df, date_col, value_col)`

This is the end-to-end forecasting pipeline used in the app.

It performs:

1. Preparation of the data via `prepare_forecast_data()`.
2. Train/test split via `split_train_test()`.
3. Training of all model candidates via `train_forecasting_models()`.
4. Model comparison via `select_best_forecasting_model()`.
5. Metadata enrichment with the best recommended model.

The pipeline result includes:

- the best model name,
- the leaderboard,
- detailed model results,
- cleaning and analysis metadata,
- train and test data subsets.

The pipeline is cached with `@st.cache_data(ttl=3600)` to avoid re-computing expensive training when the same inputs are reused.

---

## 9. Trend, Seasonality, and Signal Quality Intelligence

### 9.1 `detect_trend(df, date_col, value_col)`

Detects long-term trend behavior using linear regression.

Process:

1. Sort the time series.
2. Convert the series to numeric values.
3. Fit a linear regression to index positions `x = [0, 1, 2, ...]` vs values `y`.
4. Extract slope, intercept, and R-squared.
5. Determine direction:
   - `Increasing` if slope > tolerance,
   - `Decreasing` if slope < -tolerance,
   - `Flat` otherwise.
6. Determine trend strength from R-squared:
   - `Very Strong` if R2 >= 0.85,
   - `Strong` if R2 >= 0.65,
   - `Moderate` if R2 >= 0.40,
   - `Weak` if R2 >= 0.20,
   - `Very Weak` otherwise.
7. Compare linear vs exponential fit if all values are positive:
   - if exponential RMSE is more than 10% lower than linear RMSE, adopt `Exponential`.

Outputs include trend slope, intercept, R2, trend type, direction, strength, and confidence.

### 9.2 `detect_seasonality(df, date_col, value_col)`

Detects seasonal structure using decomposition and autocorrelation.

Process:

1. Determine the seasonal period using frequency detection.
2. If there is enough data for at least two seasons, perform additive seasonal decomposition.
3. Compute seasonal strength as:
   - `std(seasonal) / std(y)`.
4. Compute autocorrelation values using `acf()` and examine lags after the seasonal period.
5. Combine seasonal strength and autocorrelation peak into a score:
   - `score = strength * 0.6 + peak * 0.4`
6. Label seasonality as:
   - `Strong` if score >= 0.70,
   - `Moderate` if score >= 0.45,
   - `Weak` if score >= 0.25,
   - `None` otherwise.

This method captures both additive seasonality and repeated periodic behavior.

### 9.3 `check_stationarity(df, value_col)`

Uses the Augmented Dickey-Fuller (ADF) test to determine whether the series is stationary.

The ADF test returns a p-value. Interpretation rules:

- if `pvalue < 0.05`, the series is stationary and no differencing is required.
- otherwise, attempt first differencing and test again.
- if the first difference is stationary, recommend `d = 1`.
- if not, recommend `d = 2`.

This result is critical for ARIMA and SARIMA modeling because these models require stationary or differenced stationary inputs.

### 9.4 `analyze_signal_quality(df, date_col, value_col)`

Evaluates the dataset's forecasting suitability.

It computes:

- missing percentage,
- outlier percentage using the IQR rule,
- noise level as `std(y - smooth)`,
- signal strength as `std(smooth)`,
- signal-to-noise ratio `SNR = signal / max(noise, 1e-9)`.

Then it computes a data quality score out of 100 by penalizing missing values, outliers, and low SNR.

Forecast difficulty is classified as:

- `Easy` for high quality,
- `Moderate` for reasonable quality,
- `Hard` for weak quality,
- `Very Hard` for very low quality.

### 9.5 `detect_trend_and_seasonality(df, date_col, value_col)`

This master intelligence function combines trend, seasonality, and data quality.

It also selects a recommended model based on:

- strong trend + strong seasonality → `SARIMA`
- strong seasonality but weak trend → `Exponential Smoothing`
- strong trend without seasonality → `ARIMA`
- regular frequency without strong trend/seasonality → `Prophet`

It also computes a forecast readiness grade `A+` to `Needs Improvement` using missing values, outliers, and SNR.

> Note: This function is defined twice in `app.py`. The second definition overrides the first one in Python.

---

## 10. Preprocessing and Utility Functions

### 10.1 `optimize_dataframe(df, date_parse_success_threshold=0.9)`

Reduces memory usage without changing semantics.

It:

- converts object columns to datetime only when at least 90% of values parse successfully,
- downcasts integer columns to smaller integer types with `pd.to_numeric(..., downcast="integer")`.

Floating point columns are intentionally left at full precision to avoid small errors that can bias statistical tests.

### 10.2 `preprocess_time_series(df, value_col)`

Provides a Streamlit-driven preprocessing interface for users.

It supports:

- resampling to daily, weekly, monthly, quarterly, or yearly frequency,
- missing value handling via forward fill, backward fill, linear interpolation, mean, or median,
- outlier removal using Z-score, IQR, or Isolation Forest.

This function is designed for interactive user control only and is not part of the automatic forecast pipeline.

### 10.3 `seasonal_decomposition_full(series, model='additive', period=12)`

Runs seasonal decomposition on the full series without subsampling.

It warns when multiplicative decomposition is invalid for series with non-positive values.

This preserves the statistical validity of the decomposition.

### 10.4 `efficient_rolling_stats(df, value_col, window)`

Computes rolling mean, standard deviation, min, and max using pandas' native `rolling()` method.

This avoids manual chunking errors and ensures correct results at all points.

### 10.5 `_downsample_for_plot(df)`

Downsampling is used only for visualization speed. It keeps every `step`th row when the dataset exceeds 10,000 rows.

It is explicitly not used for modeling or statistical calculations because sampling can change temporal spacing and distort results.

---

## 11. Forecast Calculation Summary

### 11.1 Forecasting Models Summary

- **Naive**: repeats the last observed value. Good as a baseline and sanity check.
- **Moving Average**: repeats the mean of the last window of recent values. This is a simple smoothing baseline.
- **ARIMA**: uses autoregression, differencing, and moving average on the stationary series.
- **SARIMA**: extends ARIMA with explicit seasonal components.
- **Holt-Winters**: uses exponential smoothing for level, trend, and seasonal components.

### 11.2 Model Ranking Logic

The ranking logic uses four metrics and converts them into ranks.

- Lower RMSE, MAE, and MAPE are better.
- Higher R2 is better.
- The model with the best combined rank wins.

This ensemble-style ranking reduces the chance that a model is chosen solely by one metric.

### 11.3 Error and Quality Metrics

- `MAE` measures the average size of forecast errors.
- `RMSE` penalizes larger errors more strongly.
- `MAPE` measures error relative to the actual value, making it easier to compare datasets on different scales.
- `R2` measures how much of the variance in the actual series is explained by the forecast.

Use of these metrics together provides a balanced evaluation between absolute error and explained variance.

---

## 12. Why These Methods Were Used

### 12.1 Robust Data Preparation

The app puts a lot of effort into automatic detection because real-world time series often come with:

- inconsistent date columns,
- missing timestamps,
- duplicate records,
- irregular frequency,
- mixed numeric and string formats.

Robust parsing and cleaning ensure the models receive valid input.

### 12.2 Model Variety

Using several different forecasting strategies is important because no single model fits every dataset.

- `Naive` and `Moving Average` provide simple baselines.
- `ARIMA` is suitable for trend-driven and stationary series.
- `SARIMA` is better when strong periodic seasonality exists.
- `Holt-Winters` is useful for series with smooth trend and seasonality patterns.

This set of models covers a wide range of real-world time series behaviors.

### 12.3 Automatic Selection

The model selection applies data intelligence rather than hard-coding a single model.

- it measures trend,
- it measures seasonality,
- it measures signal quality,
- it runs multiple learned models,
- and it selects the best candidate using an objective leaderboard.

That means the application can adapt to different datasets and still make a defensible recommendation.

---

## 13. Implementation Notes and Observations

- The file uses `st.cache_data` to avoid repeating expensive computations across Streamlit reruns.
- `st.spinner()` shows progress while models are training.
- The code includes high-quality data parsing, and it avoids forcing every object column to datetime unless it appears to be mostly dates.
- The forecast pipeline is designed to protect against too-small datasets.
- Model training uses adaptive search spaces so large datasets do not require an excessively large hyperparameter search.
- The report acknowledges that the second definition of `detect_trend_and_seasonality()` overrides its earlier version.

---

## 14. Practical Forecasting Workflow

1. Upload a dataset.
2. Detect the date and target variables automatically.
3. Clean and regularize the series.
4. Analyze trend, seasonality, and data quality.
5. Train baseline and advanced models.
6. Score each model on test data.
7. Rank models and select the best.
8. Generate future predictions and confidence intervals.

This workflow supports presentation to a professor by showing both the automated intelligence and the statistical reasoning behind the forecast.

---

## 15. Conclusion

`app.py` implements a complete time series forecasting system with:

- intelligent data detection,
- strong cleaning and validation,
- multiple forecast approaches,
- comparative ranking,
- and readiness scoring.

It is designed for real datasets where the input may be messy, and it provides transparent metrics so that the recommended model can be justified objectively.

