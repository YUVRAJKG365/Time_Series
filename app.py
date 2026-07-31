"""
Time Series Analysis module for the EDA App.

Fixes applied vs. the original version (see chat for full rationale):
  1. Forecast "confidence intervals" are now real, model-derived intervals
     (ARIMA/SARIMA via get_forecast().summary_frame(), Prophet's native
     yhat_lower/yhat_upper, Holt-Winters via simulation) instead of a
     cosmetic +/-10% band that misrepresented uncertainty.
  2. Moving-Average "forecast" now projects the last rolling value forward
     instead of relabeling historical rolling-mean values as future data.
  3. Statistical routines that depend on regular time spacing (seasonal
     decomposition, ACF/PACF, ADF) always run on the FULL series. Only
     *plotting* of very large series is downsampled (for rendering speed),
     never the underlying statistics -- naive `iloc[::step]` striding
     changes the effective sampling interval and silently corrupts
     seasonality/autocorrelation results.
  4. Rolling statistics are computed directly with pandas' vectorized
     `.rolling()` (already O(n), no chunking needed) instead of chunking
     the series into independent 10k-row blocks, which produced wrong
     values at every chunk boundary.
  5. MAPE is guarded against division by zero.
  6. The sidebar "Refresh" / "Clear All Data" buttons no longer reference
     an undefined `session_manager` at module scope (NameError bug).
  7. Datetime coercion no longer blindly force-converts every object
     column; it only converts columns that parse with a high success
     rate, avoiding false-positive date columns.
  8. Isolation Forest inputs are NaN-dropped first.
  9. Only integer columns are downcast for memory savings; floats are left
     at full precision to avoid biasing statistical tests.
 10. Removed unused imports (dask, pmdarima, seaborn, matplotlib, sklearn
     StandardScaler / train_test_split, tqdm) to reduce memory footprint
     and startup time.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

from utils.data_loader import display_file_info, handle_file_upload
from utils.memory_manager import MemoryManager
from utils.session_state_manager import get_session_manager

# Only silence noisy, expected convergence/optimizer warnings -- not
# everything -- so genuine numerical problems are still visible if needed.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
MAX_ROWS_FOR_PLOTTING = 10_000  # only affects rendering, never statistics

PROFESSIONAL_PALETTE = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "accent": "#3498DB",
    "background": "#F9F9F9",
    "text": "#333333",
    "highlight": "#F1C40F",
    "success": "#27AE60",
    "diverging": ["#D62728", "#FF7F0E", "#2CA02C", "#1F77B4", "#9467BD"],
}


# --------------------------------------------------------------------------
# Memory / preprocessing helpers
# --------------------------------------------------------------------------

def optimize_dataframe(df: pd.DataFrame, date_parse_success_threshold: float = 0.9) -> pd.DataFrame:
    """Reduce memory footprint without changing the data's meaning.

    Only downcasts integers (lossless within range). Floats are left at
    full precision since downcasting can introduce small errors that bias
    downstream statistical tests (ADF, ARIMA, etc.).

    Object columns are only converted to datetime if a high fraction of
    values actually parse as dates -- avoids silently mis-treating
    ordinary text/numeric columns as dates.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        success_rate = parsed.notna().mean() if len(df) else 0
        if success_rate >= date_parse_success_threshold:
            df[col] = parsed

    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def preprocess_time_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Preprocess time series data: resampling, missing values, outliers."""
    try:
        df = optimize_dataframe(df.copy())

        if value_col not in df.columns:
            st.error(f"Value column '{value_col}' not found in dataframe")
            return df

        with st.expander("⚙️ Data Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Resampling")
                resample_freq = st.selectbox(
                    "Resampling Frequency",
                    ["Raw", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                    index=0,
                    key="resample_freq",
                )

                if resample_freq != "Raw":
                    freq_map = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "ME",
                        "Quarterly": "QE",
                        "Yearly": "YE",
                    }
                    df = df.resample(freq_map[resample_freq]).mean(numeric_only=True)

            with col2:
                st.markdown("#### Missing Values")
                missing_method = st.selectbox(
                    "Handle Missing Values",
                    ["None", "Forward Fill", "Backward Fill", "Linear Interpolation", "Mean", "Median"],
                    index=0,
                    key="missing_method",
                )

                if missing_method == "Forward Fill":
                    df[value_col] = df[value_col].ffill()
                elif missing_method == "Backward Fill":
                    df[value_col] = df[value_col].bfill()
                elif missing_method == "Linear Interpolation":
                    df[value_col] = df[value_col].interpolate(method="linear")
                elif missing_method == "Mean":
                    df[value_col] = df[value_col].fillna(df[value_col].mean())
                elif missing_method == "Median":
                    df[value_col] = df[value_col].fillna(df[value_col].median())

            st.markdown("#### Outlier Detection")
            outlier_col1, outlier_col2 = st.columns(2)

            with outlier_col1:
                remove_outliers = st.checkbox("Remove Outliers", False, key="remove_outliers")

            if remove_outliers:
                with outlier_col2:
                    method = st.selectbox(
                        "Outlier Detection Method",
                        ["Z-Score", "IQR", "Isolation Forest"],
                        index=0,
                        key="outlier_method",
                    )

                if method == "Z-Score":
                    threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key="z_threshold")
                    z_scores = (df[value_col] - df[value_col].mean()) / df[value_col].std()
                    df = df[np.abs(z_scores) < threshold]
                elif method == "IQR":
                    q1 = df[value_col].quantile(0.25)
                    q3 = df[value_col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    df = df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]
                elif method == "Isolation Forest":
                    contamination = st.slider("Contamination", 0.01, 0.5, 0.05, key="contamination")
                    clean = df[[value_col]].dropna()
                    if len(clean) > 0:
                        model = IsolationForest(
                            contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
                        )
                        preds = model.fit_predict(clean)
                        keep_index = clean.index[preds == 1]
                        df = df.loc[df.index.isin(keep_index) | df[value_col].isna()]

        return df

    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return df


def seasonal_decomposition_full(series: pd.Series, model: str = "additive", period: int = 12):
    """Seasonal decomposition on the FULL series (never subsampled) so the
    seasonal period argument stays meaningful."""
    try:
        series = series.dropna()

        if model == "multiplicative" and (series <= 0).any():
            st.warning(
                "Multiplicative decomposition isn't valid for series with values <= 0. "
                "Switching to additive."
            )
            model = "additive"

        if len(series) < 2 * period:
            st.warning(
                f"Series has {len(series)} points, which is too short for a seasonal "
                f"period of {period}. Decomposition may be unreliable or fail."
            )

        return seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

    except Exception as e:
        st.error(f"Error in decomposition: {e}")
        return None


def efficient_rolling_stats(df: pd.DataFrame, value_col: str, window: int):
    """Rolling statistics via pandas' native vectorized rolling -- correct
    at every point (no chunk-boundary artifacts) and already efficient."""
    try:
        rolling = df[value_col].rolling(window=window)
        return rolling.mean(), rolling.std(), rolling.min(), rolling.max()
    except Exception as e:
        st.error(f"Error calculating rolling stats: {e}")
        return None, None, None, None


def _downsample_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample ONLY for rendering speed. Never use this before running
    a statistical computation (decomposition, ADF, ACF/PACF, model
    fitting) -- striding breaks the assumption of regular time spacing."""
    if len(df) <= MAX_ROWS_FOR_PLOTTING:
        return df
    step = max(1, len(df) // MAX_ROWS_FOR_PLOTTING)
    return df.iloc[::step].copy()


def plot_large_data(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Line plot with rendering-only downsampling for very large series."""
    try:
        plot_df = _downsample_for_plot(df)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode="lines",
                line=dict(color=PROFESSIONAL_PALETTE["accent"]),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            plot_bgcolor=PROFESSIONAL_PALETTE["background"],
            paper_bgcolor=PROFESSIONAL_PALETTE["background"],
            font_color=PROFESSIONAL_PALETTE["text"],
        )
        return fig
    except Exception as e:
        st.error(f"Error plotting data: {e}")
        return None


def safe_mape(actual: pd.Series, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring points where actual == 0
    (otherwise divides by zero and produces inf/NaN)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def time_series_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Full time series analysis: overview, decomposition, forecasting,
    anomaly detection, feature engineering."""
    try:
        df = df.copy()

        with st.spinner("Initializing time series data..."):
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col])
                df = df.sort_values(date_col)
                df.set_index(date_col, inplace=True)

                if value_col in df.columns:
                    df = df[[value_col]]
                else:
                    st.error(f"Value column '{value_col}' not found in dataframe")
                    return df
            except Exception as e:
                st.error(f"Initial preprocessing error: {e}")
                return df

            st.markdown("### 📊 Data Overview")

            # Display all three metrics in columns with consistent sizing
            col1, col2, col3 = st.columns(3)
            col1.metric("Data Points", len(df))
            col2.metric("Missing Values", int(df.isnull().sum().sum()))

            # Date range as a metric with consistent formatting
            if len(df) > 0:
                date_range_text = f"{df.index.min().date()} → {df.index.max().date()}"
                col3.metric("Date Range", date_range_text)

        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(100), width="stretch")

        df = preprocess_time_series(df, value_col)

        if df.empty:
            st.warning("No data remains after preprocessing. Adjust your options above.")
            return df

        st.markdown("### 📈 Time Series Analysis")
        st.write(f"**Analyzing:** `{value_col}` over `{date_col}`")

        # Display data points and date range on separate lines with consistent formatting
        st.write(f"**Data Points:** {len(df):,}")
        if len(df) > 0:
            st.write(f"**Time Range:** {df.index.min().date()} to {df.index.max().date()}")

        analysis_section = st.radio(
            "Select Analysis Section:",
            ["📊 Overview", "🔍 Decomposition", "📈 Forecasting", "⚠️ Anomaly Detection", "🛠 Feature Engineering"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if analysis_section == "📊 Overview":
            _render_overview(df, date_col, value_col)
        elif analysis_section == "🔍 Decomposition":
            _render_decomposition(df, value_col)
        elif analysis_section == "📈 Forecasting":
            _render_forecasting(df, date_col, value_col)
        elif analysis_section == "⚠️ Anomaly Detection":
            _render_anomaly_detection(df, date_col, value_col)
        elif analysis_section == "🛠 Feature Engineering":
            df = _render_feature_engineering(df, value_col)

        return df

    except Exception as e:
        st.error(f"Unexpected error in time series analysis: {e}")
        return df


def _render_overview(df, date_col, value_col):
    st.markdown("#### Basic Statistics")
    st.write(df[value_col].describe().to_frame("Statistics"))

    st.markdown("#### Rolling Statistics")
    rolling_window = st.slider(
        "Select Rolling Window Size", min_value=1, max_value=min(365, len(df)), value=7,
        key="rolling_window_overview",
    )

    with st.spinner("Calculating rolling statistics..."):
        rolling_mean, rolling_std, _, _ = efficient_rolling_stats(df, value_col, rolling_window)
        if rolling_mean is not None:
            df[f"{value_col}_Rolling_Avg"] = rolling_mean
            df[f"{value_col}_Rolling_Std"] = rolling_std

    with st.spinner("Generating plot..."):
        fig1 = plot_large_data(df.reset_index(), date_col, value_col, "Time Series with Rolling Statistics")
        if fig1 is not None and rolling_mean is not None:
            plot_df = _downsample_for_plot(df)
            fig1.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[f"{value_col}_Rolling_Avg"],
                name=f"{rolling_window}-period Rolling Avg",
                line=dict(color=PROFESSIONAL_PALETTE["secondary"]),
            ))
            upper = plot_df[f"{value_col}_Rolling_Avg"] + plot_df[f"{value_col}_Rolling_Std"]
            lower = plot_df[f"{value_col}_Rolling_Avg"] - plot_df[f"{value_col}_Rolling_Std"]
            fig1.add_trace(go.Scatter(x=plot_df.index, y=upper, name="Rolling Avg + Std",
                                       line=dict(width=0), showlegend=False))
            fig1.add_trace(go.Scatter(x=plot_df.index, y=lower, name="Rolling Avg - Std",
                                       line=dict(width=0), fillcolor="rgba(231, 76, 60, 0.2)",
                                       fill="tonexty", showlegend=False))
            fig1.update_layout(showlegend=True)
            st.plotly_chart(fig1, width="stretch")

    st.markdown("#### Stationarity Test (Augmented Dickey-Fuller)")
    st.caption("Run on the full series (not downsampled) so the result is statistically valid.")
    with st.spinner("Performing ADF test..."):
        clean_series = df[value_col].dropna()
        if len(clean_series) < 10:
            st.warning("Not enough data points to run a reliable ADF test.")
        else:
            adf_result = adfuller(clean_series)
            st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
            st.write(f"**p-value:** {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                st.success("The series is likely stationary (p < 0.05).")
            else:
                st.warning("The series is likely non-stationary (p >= 0.05). Consider differencing or detrending.")

    if st.checkbox("Show Differenced Series", key="show_diff_series"):
        diff_order = st.slider("Differencing Order", 1, 3, 1, key="diff_order_overview")
        with st.spinner("Calculating differenced series..."):
            diff_series = df[value_col].diff(periods=diff_order).dropna()
            fig_diff = plot_large_data(
                diff_series.reset_index().rename(columns={0: value_col, "index": date_col}),
                date_col if date_col in diff_series.reset_index().columns else diff_series.reset_index().columns[0],
                value_col if value_col in diff_series.reset_index().columns else diff_series.reset_index().columns[1],
                f"{diff_order}-Order Differenced Series",
            )
            if fig_diff is not None:
                st.plotly_chart(fig_diff, width="stretch")
            if len(diff_series) >= 10:
                adf_diff = adfuller(diff_series)
                st.write(f"**ADF Statistic (Differenced):** {adf_diff[0]:.4f}")
                st.write(f"**p-value (Differenced):** {adf_diff[1]:.4f}")


def _render_decomposition(df, value_col):
    st.markdown("#### Seasonal Decomposition")
    st.caption("Computed on the full series so the seasonal period stays meaningful; only the plot below is thinned for rendering on very large datasets.")

    decomp_model = st.radio("Decomposition Model", ["Additive", "Multiplicative"], index=0, key="decomp_model")
    period = st.slider("Seasonal Period", 2, max(2, min(365, len(df) // 2 or 2)), min(12, max(2, len(df) // 2 or 2)), key="seasonal_period")

    with st.spinner("Performing decomposition..."):
        decomposition = seasonal_decomposition_full(
            df[value_col], model="additive" if decomp_model == "Additive" else "multiplicative", period=period
        )

        if decomposition is not None:
            fig2 = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
            for i, (name, series) in enumerate([
                ("Observed", decomposition.observed),
                ("Trend", decomposition.trend),
                ("Seasonal", decomposition.seasonal),
                ("Residual", decomposition.resid),
            ], start=1):
                color = [PROFESSIONAL_PALETTE["accent"], PROFESSIONAL_PALETTE["secondary"],
                         PROFESSIONAL_PALETTE["highlight"], PROFESSIONAL_PALETTE["primary"]][i - 1]
                fig2.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(color=color)), row=i, col=1)

            fig2.update_layout(
                height=800, title_text=f"Time Series Decomposition ({decomp_model} Model)",
                plot_bgcolor=PROFESSIONAL_PALETTE["background"], paper_bgcolor=PROFESSIONAL_PALETTE["background"],
                font_color=PROFESSIONAL_PALETTE["text"], showlegend=False,
            )
            st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Autocorrelation & Partial Autocorrelation")
    st.caption("Computed on the full series.")
    clean_series = df[value_col].dropna()
    max_lags = max(1, min(60, len(clean_series) // 2 - 1))
    lags = st.slider("Number of Lags", 1, max_lags, min(30, max_lags), key="acf_lags")

    with st.spinner("Calculating ACF/PACF..."):
        acf_values = acf(clean_series, nlags=lags, fft=True)
        pacf_values = pacf(clean_series, nlags=lags, method="ywm")

        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(x=list(range(1, lags + 1)), y=acf_values[1:], name="ACF",
                                  marker_color=PROFESSIONAL_PALETTE["accent"]))
        acf_fig.update_layout(title="Autocorrelation (ACF)", xaxis_title="Lag", yaxis_title="Correlation",
                               plot_bgcolor=PROFESSIONAL_PALETTE["background"],
                               paper_bgcolor=PROFESSIONAL_PALETTE["background"],
                               font_color=PROFESSIONAL_PALETTE["text"])
        st.plotly_chart(acf_fig, width="stretch")

        pacf_fig = go.Figure()
        pacf_fig.add_trace(go.Bar(x=list(range(1, lags + 1)), y=pacf_values[1:], name="PACF",
                                   marker_color=PROFESSIONAL_PALETTE["secondary"]))
        pacf_fig.update_layout(title="Partial Autocorrelation (PACF)", xaxis_title="Lag", yaxis_title="Correlation",
                                plot_bgcolor=PROFESSIONAL_PALETTE["background"],
                                paper_bgcolor=PROFESSIONAL_PALETTE["background"],
                                font_color=PROFESSIONAL_PALETTE["text"])
        st.plotly_chart(pacf_fig, width="stretch")


def _render_forecasting(df, date_col, value_col):
    st.markdown("#### Time Series Forecasting")
    forecast_df = df.copy()

    forecast_method = st.selectbox(
        "Select Forecasting Method",
        ["Naive", "Moving Average", "Exponential Smoothing", "ARIMA", "SARIMA", "Prophet"],
        index=0, key="forecast_method",
    )
    forecast_periods = st.slider("Forecast periods (future steps)", 1, 365, 30, key="forecast_periods")
    test_size = st.slider("Test Size (%) for Evaluation", 10, 50, 20, key="test_size")

    train_size = int(len(forecast_df) * (1 - test_size / 100))
    train, test = forecast_df.iloc[:train_size], forecast_df.iloc[train_size:]

    lower, upper = None, None  # real confidence interval, when available
    model_name = "Forecast"
    forecast = None

    with st.spinner(f"Training {forecast_method} model..."):
        try:
            if forecast_method == "Naive":
                last_value = train[value_col].iloc[-1]
                forecast = np.repeat(last_value, forecast_periods)
                model_name = "Naive Forecast"

            elif forecast_method == "Moving Average":
                window = st.slider("Moving Average Window", 1, 30, 3, key="ma_window")
                last_ma = train[value_col].rolling(window=window).mean().iloc[-1]
                # Project the last rolling value forward as a flat
                # continuation -- this IS what a moving-average forecast
                # is; it must not reuse historical rolling values as if
                # they were future predictions.
                forecast = np.repeat(last_ma, forecast_periods)
                model_name = f"{window}-period Moving Average"

            elif forecast_method == "Exponential Smoothing":
                seasonal_periods = st.slider("Seasonal Periods", 2, 52, 12, key="es_seasonal_periods")
                model = ExponentialSmoothing(
                    train[value_col], trend="add", seasonal="add", seasonal_periods=seasonal_periods
                ).fit()
                forecast = np.asarray(model.forecast(forecast_periods))
                model_name = "Holt-Winters Exponential Smoothing"
                # Approximate prediction interval via simulation.
                try:
                    sims = model.simulate(forecast_periods, repetitions=200, random_state=RANDOM_STATE)
                    lower = sims.quantile(0.05, axis=1).to_numpy()
                    upper = sims.quantile(0.95, axis=1).to_numpy()
                except Exception:
                    lower, upper = None, None

            elif forecast_method in ("ARIMA", "SARIMA"):
                order_p = st.slider("AR Order (p)", 0, 5, 1, key="order_p")
                order_d = st.slider("Difference Order (d)", 0, 2, 1, key="order_d")
                order_q = st.slider("MA Order (q)", 0, 5, 1, key="order_q")

                if forecast_method == "ARIMA":
                    fitted = ARIMA(train[value_col], order=(order_p, order_d, order_q)).fit()
                    model_name = f"ARIMA({order_p},{order_d},{order_q})"
                else:
                    seasonal_p = st.slider("Seasonal AR Order (P)", 0, 2, 0, key="seasonal_p")
                    seasonal_d = st.slider("Seasonal Difference (D)", 0, 1, 0, key="seasonal_d")
                    seasonal_q = st.slider("Seasonal MA Order (Q)", 0, 2, 0, key="seasonal_q")
                    seasonal_period = st.slider("Seasonal Period (s)", 4, 24, 12, key="seasonal_period_sarima")
                    fitted = SARIMAX(
                        train[value_col], order=(order_p, order_d, order_q),
                        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                    ).fit(disp=False)
                    model_name = (
                        f"SARIMA({order_p},{order_d},{order_q})"
                        f"({seasonal_p},{seasonal_d},{seasonal_q})[{seasonal_period}]"
                    )

                pred = fitted.get_forecast(steps=forecast_periods)
                forecast = np.asarray(pred.predicted_mean)
                ci = pred.conf_int(alpha=0.10)  # 90% interval
                lower = ci.iloc[:, 0].to_numpy()
                upper = ci.iloc[:, 1].to_numpy()

                converged = getattr(fitted, "mle_retvals", {}).get("converged", True)
                if not converged:
                    st.warning(
                        "The optimizer did not fully converge -- treat this forecast "
                        "with caution and consider different orders."
                    )

            elif forecast_method == "Prophet":
                prophet_df = train[value_col].reset_index()
                prophet_df.columns = ["ds", "y"]
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_periods)
                forecast_result = model.predict(future)
                tail = forecast_result.iloc[-forecast_periods:]
                forecast = tail["yhat"].to_numpy()
                lower = tail["yhat_lower"].to_numpy()
                upper = tail["yhat_upper"].to_numpy()
                model_name = "Facebook Prophet"

        except Exception as e:
            st.error(f"Error fitting {forecast_method} model: {e}")
            forecast = np.repeat(train[value_col].mean(), forecast_periods)
            model_name = "Fallback Mean Forecast"

    freq = pd.infer_freq(forecast_df.index)
    future_dates = pd.date_range(forecast_df.index[-1], periods=forecast_periods + 1, freq=freq)[1:]

    with st.spinner("Generating forecast plot..."):
        fig_forecast = go.Figure()
        plot_train = _downsample_for_plot(train)
        fig_forecast.add_trace(go.Scatter(x=plot_train.index, y=plot_train[value_col], name="Training Data",
                                           line=dict(color=PROFESSIONAL_PALETTE["accent"])))
        if len(test) > 0:
            plot_test = _downsample_for_plot(test)
            fig_forecast.add_trace(go.Scatter(x=plot_test.index, y=plot_test[value_col], name="Actual Test Data",
                                               line=dict(color=PROFESSIONAL_PALETTE["primary"])))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=forecast, name=f"{model_name} Forecast",
                                           line=dict(color=PROFESSIONAL_PALETTE["secondary"], dash="dot")))

        if lower is not None and upper is not None:
            fig_forecast.add_trace(go.Scatter(x=future_dates, y=upper, name="Upper Bound (90% CI)",
                                               line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=future_dates, y=lower, name="Lower Bound (90% CI)",
                                               line=dict(width=0), fillcolor="rgba(231, 76, 60, 0.2)",
                                               fill="tonexty", showlegend=True))
        else:
            st.caption("No statistically derived confidence interval is available for this method.")

        fig_forecast.update_layout(
            title=f"Time Series Forecast - {model_name}", xaxis_title="Date", yaxis_title=value_col,
            plot_bgcolor=PROFESSIONAL_PALETTE["background"], paper_bgcolor=PROFESSIONAL_PALETTE["background"],
            font_color=PROFESSIONAL_PALETTE["text"],
        )
        st.plotly_chart(fig_forecast, width="stretch")

        st.markdown("---")
        st.markdown("### Export Forecast Results")
        export_data = {"Date": future_dates, "Forecast": forecast}
        if lower is not None and upper is not None:
            export_data["Lower_CI"] = lower
            export_data["Upper_CI"] = upper
        forecast_export_df = pd.DataFrame(export_data)
        st.download_button(
            label="📥 Download Forecast as CSV", data=forecast_export_df.to_csv(index=False),
            file_name=f"{value_col}_forecast.csv", mime="text/csv", key="forecast_download",
        )

        if st.button("💾 Download Forecast Plot as PNG"):
            try:
                img_bytes = fig_forecast.to_image(format="png")
                st.download_button(label="Download", data=img_bytes, file_name=f"{value_col}_forecast.png",
                                    mime="image/png", key="instant_download")
                st.success("Plot ready to download.")
            except Exception as e:
                st.error(f"Error generating plot image: {e}")

    if len(test) > 0 and len(test) >= forecast_periods:
        actual = test[value_col].iloc[:forecast_periods]
        mae = mean_absolute_error(actual, forecast)
        rmse = float(np.sqrt(mean_squared_error(actual, forecast)))
        mape = safe_mape(actual, forecast)

        st.markdown("#### Forecast Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A (zero actuals)")


def _render_anomaly_detection(df, date_col, value_col):
    st.markdown("#### Anomaly Detection")
    st.caption("Computed on the full series; only the plot is thinned for very large datasets.")
    anomaly_df = df.copy()

    anomaly_method = st.selectbox(
        "Select Anomaly Detection Method",
        ["Z-Score", "IQR", "Isolation Forest", "Moving Average Deviation"],
        index=0, key="anomaly_method",
    )

    anomalies = anomaly_df.iloc[0:0]  # empty default

    with st.spinner("Detecting anomalies..."):
        if anomaly_method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, key="z_score_threshold")
            mean = anomaly_df[value_col].mean()
            std = anomaly_df[value_col].std()
            if std and not np.isnan(std) and std != 0:
                anomaly_df["Z-Score"] = (anomaly_df[value_col] - mean) / std
                anomalies = anomaly_df[np.abs(anomaly_df["Z-Score"]) > threshold]
            else:
                st.warning("Standard deviation is zero or undefined; cannot compute Z-scores.")

        elif anomaly_method == "IQR":
            q1 = anomaly_df[value_col].quantile(0.25)
            q3 = anomaly_df[value_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            anomalies = anomaly_df[(anomaly_df[value_col] < lower_bound) | (anomaly_df[value_col] > upper_bound)]

        elif anomaly_method == "Isolation Forest":
            contamination = st.slider("Expected Anomaly Fraction", 0.01, 0.5, 0.05, key="contamination_anomaly")
            clean = anomaly_df[[value_col]].dropna()
            if len(clean) > 0:
                model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
                preds = model.fit_predict(clean)
                anomalies = anomaly_df.loc[clean.index[preds == -1]]
            else:
                st.warning("No valid (non-missing) data available for Isolation Forest.")

        elif anomaly_method == "Moving Average Deviation":
            window = st.slider("Moving Average Window", 1, 30, 7, key="ma_window_anomaly")
            threshold = st.slider("Deviation Threshold (STD)", 1.0, 5.0, 2.0, key="deviation_threshold")

            rolling_mean, rolling_std, _, _ = efficient_rolling_stats(anomaly_df, value_col, window)
            if rolling_mean is not None:
                anomaly_df["Moving_Avg"] = rolling_mean
                anomaly_df["Moving_Std"] = rolling_std
                anomaly_df["Upper_Bound"] = anomaly_df["Moving_Avg"] + (threshold * anomaly_df["Moving_Std"])
                anomaly_df["Lower_Bound"] = anomaly_df["Moving_Avg"] - (threshold * anomaly_df["Moving_Std"])
                anomalies = anomaly_df[
                    (anomaly_df[value_col] > anomaly_df["Upper_Bound"]) |
                    (anomaly_df[value_col] < anomaly_df["Lower_Bound"])
                ]

    with st.spinner("Generating anomaly plot..."):
        fig_anomalies = plot_large_data(
            anomaly_df.reset_index(), date_col, value_col, f"Anomaly Detection - {anomaly_method}"
        )
        if fig_anomalies is not None:
            fig_anomalies.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies[value_col], mode="markers", name="Anomalies",
                marker=dict(color=PROFESSIONAL_PALETTE["secondary"], size=8, line=dict(width=1, color="DarkSlateGrey")),
            ))
            if anomaly_method == "Moving Average Deviation" and "Upper_Bound" in anomaly_df:
                plot_bounds = _downsample_for_plot(anomaly_df)
                fig_anomalies.add_trace(go.Scatter(x=plot_bounds.index, y=plot_bounds["Upper_Bound"],
                                                    name="Upper Bound", line=dict(color=PROFESSIONAL_PALETTE["highlight"], dash="dash")))
                fig_anomalies.add_trace(go.Scatter(x=plot_bounds.index, y=plot_bounds["Lower_Bound"],
                                                    name="Lower Bound", line=dict(color=PROFESSIONAL_PALETTE["highlight"], dash="dash")))
            st.plotly_chart(fig_anomalies, width="stretch")

    st.markdown("#### Detected Anomalies")
    st.write(f"**{len(anomalies)}** anomalies detected out of **{len(anomaly_df)}** points.")
    st.write(anomalies[[value_col]])


def _render_feature_engineering(df, value_col):
    st.markdown("#### Time Series Feature Engineering")
    new_features = []

    if st.checkbox("Extract Date Features", key="extract_date_features"):
        with st.spinner("Adding date features..."):
            date_features = ["Year", "Month", "Day", "DayOfWeek", "DayOfYear", "Quarter", "IsWeekend"]
            df["Year"] = df.index.year
            df["Month"] = df.index.month
            df["Day"] = df.index.day
            df["DayOfWeek"] = df.index.dayofweek
            df["DayOfYear"] = df.index.dayofyear
            df["Quarter"] = df.index.quarter
            df["IsWeekend"] = df.index.dayofweek >= 5
            new_features.extend(date_features)

            fig_date = px.histogram(df, x="Month", y=value_col, histfunc="avg", title="Monthly Averages")
            st.plotly_chart(fig_date, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Date Features (CSV)", df[date_features].to_csv(),
                                    "date_features.csv", "text/csv")
            with col2:
                try:
                    st.download_button("📥 Download Monthly Averages Plot (PNG)", fig_date.to_image(format="png"),
                                        "monthly_averages.png", "image/png")
                except Exception:
                    pass

    if st.checkbox("Create Lag Features", key="create_lag_features"):
        n_lags = st.slider("Number of Lag Features", 1, 10, 3, key="lag_features_slider")
        with st.spinner(f"Creating {n_lags} lag features..."):
            lag_features = [f"Lag_{i}" for i in range(1, n_lags + 1)]
            for i in range(1, n_lags + 1):
                df[f"Lag_{i}"] = df[value_col].shift(i)
            new_features.extend(lag_features)

            fig_lag = go.Figure()
            for i in range(1, min(4, n_lags + 1)):
                fig_lag.add_trace(go.Scatter(x=df.index, y=df[f"Lag_{i}"], name=f"Lag {i}"))
            fig_lag.update_layout(title="Lag Features Visualization")
            st.plotly_chart(fig_lag, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Lag Features (CSV)", df[lag_features].to_csv(),
                                    "lag_features.csv", "text/csv")
            with col2:
                try:
                    st.download_button("📥 Download Lag Plot (PNG)", fig_lag.to_image(format="png"),
                                        "lag_visualization.png", "image/png")
                except Exception:
                    pass

    if st.checkbox("Create Rolling Statistics", key="create_rolling_stats"):
        window = st.slider("Rolling Window Size", 2, 30, 7, key="rolling_window_slider")
        with st.spinner("Calculating rolling statistics..."):
            rolling_mean, rolling_std, rolling_min, rolling_max = efficient_rolling_stats(df, value_col, window)
            if rolling_mean is not None:
                roll_features = [f"Rolling_Mean_{window}", f"Rolling_Std_{window}",
                                  f"Rolling_Min_{window}", f"Rolling_Max_{window}"]
                df[f"Rolling_Mean_{window}"] = rolling_mean
                df[f"Rolling_Std_{window}"] = rolling_std
                df[f"Rolling_Min_{window}"] = rolling_min
                df[f"Rolling_Max_{window}"] = rolling_max
                new_features.extend(roll_features)

                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=df.index, y=df[value_col], name="Original", line=dict(color="blue")))
                fig_roll.add_trace(go.Scatter(x=df.index, y=df[f"Rolling_Mean_{window}"],
                                               name=f"{window}-period Rolling Mean", line=dict(color="red")))
                fig_roll.update_layout(title="Rolling Statistics Visualization")
                st.plotly_chart(fig_roll, width="stretch")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 Download Rolling Stats (CSV)", df[roll_features].to_csv(),
                                        "rolling_stats.csv", "text/csv")
                with col2:
                    try:
                        st.download_button("📥 Download Rolling Plot (PNG)", fig_roll.to_image(format="png"),
                                            "rolling_visualization.png", "image/png")
                    except Exception:
                        pass

    if st.checkbox("Create Differenced Series", key="create_diff_series"):
        diff_order = st.slider("Differencing Order", 1, 3, 1, key="diff_order_slider")
        with st.spinner("Creating differenced series..."):
            diff_feature = f"Diff_{diff_order}"
            df[diff_feature] = df[value_col].diff(periods=diff_order)
            new_features.append(diff_feature)

            fig_diff = px.line(df, y=diff_feature, title=f"{diff_order}-Order Differenced Series")
            st.plotly_chart(fig_diff, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Differenced Data (CSV)", df[[diff_feature]].to_csv(),
                                    "differenced_series.csv", "text/csv")
            with col2:
                try:
                    st.download_button("📥 Download Differenced Plot (PNG)", fig_diff.to_image(format="png"),
                                        "differenced_series.png", "image/png")
                except Exception:
                    pass

    if new_features:
        st.markdown("---")
        st.markdown("### Export All Engineered Features")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download All Features (CSV)", df[new_features].to_csv(),
                                "all_engineered_features.csv", "text/csv")
        with col2:
            st.download_button("📥 Download Complete Dataset (CSV)", df.to_csv(),
                                "complete_dataset.csv", "text/csv")

    return df


# --------------------------------------------------------------------------
# Section wrapper / entry point
# --------------------------------------------------------------------------

def render_time_series_section():
    session_manager = get_session_manager()
    section = "Time Series"

    st.markdown("## ⏳ Time Series Analysis")
    st.markdown("Analyze, forecast, and visualize trends, patterns, seasonality, and anomalies in time-based data.")

    with st.expander("ℹ️ Getting Started - Complete Guide to Time Series Analysis", expanded=True):
        st.markdown("""
      # 📚 Complete Guide to Time Series Analysis

      Welcome to the **Time Series Analysis Module** of the EDA Application.

      This module is designed not only to analyze your data but also to help you understand **why each analysis is performed, when it should be used, and how to correctly interpret the results.**

      Whether you are a beginner learning data analytics or a professional data scientist working with business datasets, this guide will walk you through every major concept involved in Time Series Analysis.

      ---

      # 🌍 What is Time Series Analysis?

      **Definition:** Time Series Analysis is the statistical process of analyzing data points collected or recorded at specific time intervals. The key characteristic is that observations are dependent on time and often on previous observations.

      **Simple Explanation:** Think of it like watching a movie frame by frame - each moment is connected to what came before and what comes after. Unlike a photograph (which is a single moment), a time series is like a video that shows how things change over time.

      **Key Concept:**
      > **Time Series = Data + Time**

      This means every piece of data has a timestamp telling us WHEN it happened, not just WHAT happened.

      ### Real-World Examples:
      - 📈 **Daily Stock Prices** - How much a company's stock changes each day
      - 💰 **Monthly Sales Revenue** - How much money a business makes each month
      - 🌡️ **Hourly Temperature Readings** - How temperature changes throughout the day
      - 🚗 **Daily Traffic Counts** - How many cars pass through a road each day
      - ⚡ **Electricity Consumption** - How much power a household uses each hour
      - 🏥 **Hospital Patient Admissions** - How many patients come to the hospital each day
      - 📱 **Website Visitors** - How many people visit a website each hour
      - 🌾 **Crop Production** - How much wheat is produced each season
      - 💳 **Bank Transactions** - How many transactions occur each day

      ### Main Questions Time Series Analysis Answers:
      ✔ **What happened?** - Understanding past behavior
      ✔ **Why did it happen?** - Finding causes and patterns
      ✔ **Is there a pattern?** - Identifying regular occurrences
      ✔ **Will it happen again?** - Predicting future based on patterns
      ✔ **What is likely to happen next?** - Forecasting future values

      ---

      # 🎯 Why Time Series Analysis is Important

      **Definition:** Time series analysis helps organizations move from reactive decision-making to proactive planning by understanding historical patterns and predicting future outcomes.

      **Simple Explanation:** Imagine driving a car - you don't just look at the road directly in front of you, you look ahead to anticipate turns, traffic, and obstacles. Time series analysis does the same thing for business decisions - it helps you see what's coming before it arrives.

      ### Key Benefits:
      • **Understand Historical Performance** - See how metrics have changed over months or years
      • **Detect Long-term Growth or Decline** - Know if you're moving in the right direction
      • **Discover Seasonal Behavior** - Identify patterns that repeat at specific times
      • **Forecast Future Demand** - Predict how much product you'll need
      • **Detect Unusual Events** - Find anomalies that need investigation
      • **Improve Planning and Budgeting** - Make informed financial decisions
      • **Optimize Inventory** - Keep the right amount of stock
      • **Predict Equipment Failures** - Fix machines before they break
      • **Improve Customer Experience** - Anticipate customer needs

      ### The Power of Being Proactive:
      Instead of reacting after something happens (like running out of stock), businesses can become proactive by making predictions before events occur (like ordering more inventory before demand spikes).

      ---

      # 🏢 Real-World Business Applications

      Time Series Analysis is used across almost every industry. Here's how different sectors use it:

      ### 🛒 Retail

      **What they analyze:** Sales data, customer footfall, inventory levels

      **How they use it:**
      - **Sales Forecasting** - Predict how much they'll sell next month
      - **Inventory Planning** - Know what products to stock and when
      - **Demand Prediction** - Anticipate customer needs before they arise
      - **Seasonal Product Analysis** - Understand which products sell during holidays

      **Example:** A clothing store analyzes sales from last summer to predict what sizes and styles they should stock for this summer.

      ---

      ### 💹 Finance

      **What they analyze:** Stock prices, trading volumes, economic indicators

      **How they use it:**
      - **Stock Market Analysis** - Understand price movements
      - **Portfolio Monitoring** - Track investment performance
      - **Fraud Detection** - Spot unusual transaction patterns
      - **Cryptocurrency Prediction** - Forecast digital currency trends
      - **Risk Assessment** - Evaluate financial risks

      **Example:** An investment firm predicts future stock movement based on historical price patterns to make better investment decisions.

      ---

      ### 🏭 Manufacturing

      **What they analyze:** Machine sensors, production rates, quality metrics

      **How they use it:**
      - **Machine Health Monitoring** - Track equipment performance
      - **Predictive Maintenance** - Fix machines before they break down
      - **Production Planning** - Schedule manufacturing efficiently
      - **Sensor Monitoring** - Detect abnormal machine behavior

      **Example:** A factory monitors vibration patterns on machines and detects unusual patterns before a machine fails, preventing costly downtime.

      ---

      ### 🏥 Healthcare

      **What they analyze:** Patient records, admissions, disease statistics

      **How they use it:**
      - **Disease Outbreak Monitoring** - Track spread of diseases
      - **Patient Admission Prediction** - Prepare for influx of patients
      - **Medicine Demand Forecasting** - Ensure drugs are in stock
      - **ICU Occupancy Prediction** - Plan intensive care capacity

      **Example:** A hospital predicts how many patients will need ICU beds during flu season and prepares accordingly.

      ---

      ### 🌦 Weather & Climate

      **What they analyze:** Temperature, rainfall, atmospheric pressure

      **How they use it:**
      - **Rainfall Forecasting** - Predict when it will rain
      - **Temperature Prediction** - Forecast weather conditions
      - **Climate Change Analysis** - Understand long-term climate trends
      - **Storm Monitoring** - Track and predict severe weather

      **Example:** Meteorologists predict hurricane paths using historical storm data and current conditions.

      ---

      ### 🚗 Transportation

      **What they analyze:** Traffic patterns, vehicle movements, passenger counts

      **How they use it:**
      - **Traffic Prediction** - Forecast congestion
      - **Vehicle Demand** - Plan fleet requirements
      - **Route Optimization** - Find most efficient paths
      - **Fuel Consumption Analysis** - Track and optimize fuel usage

      **Example:** A ride-sharing company predicts where demand will be high during different times and positions drivers accordingly.

      ---

      ### ⚡ Energy

      **What they analyze:** Power consumption, generation data, grid metrics

      **How they use it:**
      - **Electricity Demand Forecasting** - Predict power needs
      - **Solar Energy Prediction** - Forecast solar power generation
      - **Wind Power Forecasting** - Predict wind energy output
      - **Smart Grid Optimization** - Manage power distribution

      **Example:** A power company predicts how much electricity will be needed on a hot summer day and ensures sufficient generation capacity.

      ---

      ### 🌐 IoT & Smart Devices

      **What they analyze:** Sensor data, device usage patterns, system logs

      **How they use it:**
      - **Sensor Monitoring** - Track device performance
      - **Smart Home Automation** - Predict and optimize home systems
      - **Industrial IoT** - Monitor industrial equipment
      - **Predictive Alerts** - Warn about potential issues

      **Example:** A smart home system learns your heating patterns and adjusts temperature before you arrive home.

      ---

      # 📂 What Kind of Dataset Can Be Used?

      **Definition:** A time series dataset is structured data where each observation has both a timestamp and one or more measured values.

      **Simple Explanation:** Think of a diary - each entry has a date (when it happened) and something you wrote down (what happened). In time series data, each row is like a diary entry with a date and a measurement.

      ### Essential Components:

      ## 1️⃣ Date / Time Column (The "When")
      **Definition:** This column contains the temporal information that orders your observations chronologically.

      **Simple Explanation:** This tells us WHEN each measurement was taken. Without this, we don't know the sequence of events.

      **Examples:**
      - Date (like "2024-01-15")
      - Timestamp (like "2024-01-15 14:30:00")
      - Order Date (when an order was placed)
      - Invoice Date (when a bill was sent)
      - Year (like "2024")
      - Month (like "January")
      - Datetime (full date and time)

      ---

      ## 2️⃣ Numerical Value Column (The "What")
      **Definition:** This column contains the actual measurements or quantities being tracked over time.

      **Simple Explanation:** This tells us WHAT we're measuring. It's the actual data we want to analyze.

      **Examples:**
      - Sales (dollars earned)
      - Revenue (total income)
      - Temperature (degrees)
      - Visitors (number of people)
      - Price (cost of an item)
      - Profit (earnings after costs)
      - Demand (quantity needed)
      - Population (number of people)

      ---

      # ✅ Characteristics of a Good Time Series Dataset

      **Definition:** These are the qualities that make a dataset suitable for time series analysis.

      **Simple Explanation:** Just like you need good ingredients to cook a good meal, you need good data to get good analysis results.

      ### Key Characteristics:

      ✔ **Chronological Order** - Data arranged from earliest to latest (like reading a timeline)

      ✔ **Consistent Time Intervals** - Regular spacing between observations (daily, monthly, etc.)

      ✔ **Numeric Target Values** - Measurements expressed as numbers (prices, counts, etc.)

      ✔ **Minimal Missing Timestamps** - Few gaps in the timeline

      ✔ **Proper Datetime Format** - Dates recognized as dates by the computer

      ✔ **No Duplicate Timestamps** - Only one value per time point

      ✔ **Reliable Measurements** - Data accurately collected

      ---

      # ❌ Common Problems in Time Series Data

      **Definition:** Issues that can make analysis difficult or misleading if not addressed.

      **Simple Explanation:** These are like obstacles on a road - you need to know about them so you can remove them before proceeding.

      ### Common Issues:

      • **Missing Dates** - Gaps in the timeline (like missing diary entries)

      • **Duplicate Dates** - Multiple entries for the same time

      • **Randomly Ordered Records** - Data not sorted by time

      • **Mixed Time Frequencies** - Some data daily, some weekly

      • **Incorrect Date Formats** - Dates the computer doesn't understand

      • **Large Number of Missing Values** - Too many blank entries

      • **Non-Numeric Measurements** - Text instead of numbers

      • **Inconsistent Sampling** - Irregular time gaps between measurements

      **Important:** These issues should be corrected before performing forecasting or statistical analysis.

      ---

      # 🧩 Components of a Time Series

      **Definition:** Every time series is generally composed of four major components.

      **Simple Explanation:** Think of music - a song is made up of melody (trend), rhythm (seasonality), and background noise (random). Similarly, time series data has different layers that combine to form the complete picture.

      ```
                    Original Time Series
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
        Trend          Seasonality       Random Noise
                            │
                       Cyclical Effects
      ```

      Understanding these components helps determine which forecasting model is most appropriate.

      ---

      # 📈 Trend

      **Definition:** The long-term direction in which the data is moving over an extended period.

      **Simple Explanation:** Like watching a ball roll - is it going uphill (increasing), downhill (decreasing), or staying flat (stable)? Trend is the big picture direction.

      ### Examples:
      📈 **Increasing Revenue** - Sales going up year after year
      📉 **Declining Sales** - Sales gradually dropping
      📈 **Population Growth** - More people over time
      📉 **Falling Product Demand** - Fewer people wanting a product

      ### Types of Trend:
      • **Upward Trend** - Values generally increasing over time
      • **Downward Trend** - Values generally decreasing over time
      • **No Trend (Stationary)** - Values moving around a fixed average

      ### Causes of Trend:
      **Definition:** Factors that cause long-term changes in data.

      • **Economic Growth** - The economy getting bigger
      • **Inflation** - Prices generally rising
      • **Business Expansion** - Company growing bigger
      • **Customer Demand** - Changing preferences over time
      • **Technology Changes** - New innovations changing behavior

      ---

      # 🔁 Seasonality

      **Definition:** Regular, predictable patterns that repeat at fixed, known intervals.

      **Simple Explanation:** Like the seasons of the year - summer comes every year, then fall, then winter, then spring. Seasonality is any pattern that repeats at regular intervals.

      ### Key Characteristics:
      - Pattern repeats at the same time every cycle
      - The duration is fixed and known
      - It's predictable and expected

      ### Examples:
      • 🍦 Higher ice cream sales every summer
      • 🎄 Increased shopping during December holidays
      • 🍽️ Weekend restaurant rush
      • 💰 Monthly salary deposits
      • 🌙 Nighttime vs daytime activity
      • ☀️ Hotter temperatures in July
      • 🏫 School traffic in September

      ### Common Seasonal Intervals:
      - **Daily** - Pattern repeats every day
      - **Weekly** - Pattern repeats every week
      - **Monthly** - Pattern repeats every month
      - **Quarterly** - Pattern repeats every 3 months
      - **Yearly** - Pattern repeats every year

      ---

      # 🔄 Cyclical Patterns

      **Definition:** Long-term oscillations that don't have a fixed period and are typically driven by broader economic or business cycles.

      **Simple Explanation:** Unlike seasons which are predictable (summer always comes), cycles are like the economy - sometimes good (growth), sometimes bad (recession), but we can't predict exactly when.

      ### Key Characteristics:
      - Pattern repeats but not at regular intervals
      - Length of each cycle varies
      - Often linked to economic conditions

      ### Examples:
      • **Business Cycles** - Periods of economic growth and contraction
      • **Economic Recessions** - Periods of economic decline
      • **Housing Market Cycles** - Property price ups and downs
      • **Inflation Cycles** - Price changes over time
      • **Oil Price Fluctuations** - Rising and falling energy costs

      ### Difference from Seasonality:
      **Seasonality** = Fixed interval (every year at the same time)
      **Cyclical** = Variable interval (could be 5 years or 10 years)

      ---

      # 🎲 Random Noise (Residuals)

      **Definition:** The unpredictable, irregular fluctuations that cannot be explained by trend, seasonality, or cyclical patterns.

      **Simple Explanation:** Life is unpredictable - sometimes things just happen that we couldn't see coming. Random noise is the "surprise" element in data.

      ### Key Characteristics:
      - Completely unpredictable
      - No pattern to learn from
      - Random variation

      ### Examples:
      • 🌪️ **Natural Disasters** - Unexpected storms or earthquakes
      • 🏛️ **Political Instability** - Sudden government changes
      • 💻 **Unexpected System Failures** - Computers crashing
      • 🦠 **Pandemics** - Disease outbreaks like COVID-19
      • 🖥️ **Cyber Attacks** - Unexpected security breaches

      ### Important Note:
      Noise cannot be forecast accurately and is generally treated as random variation that we accept as normal fluctuation.

      ---

      # 🔄 Complete Time Series Analysis Workflow

      **Definition:** The step-by-step process followed to properly analyze time series data.

      **Simple Explanation:** Just like following a recipe step by step, this is the proven sequence to follow for reliable analysis.

      The following workflow summarizes how this module performs analysis.

      ```
      Upload Dataset
            │
            ▼
      Select Date Column
            │
            ▼
      Validate Date Format
            │
            ▼
      Sort Data Chronologically
            │
            ▼
      Handle Missing Values
            │
            ▼
      Resample (Optional)
            │
            ▼
      Remove Outliers
            │
            ▼
      Explore Statistics
            │
            ▼
      Rolling Statistics
            │
            ▼
      Stationarity Testing
            │
            ▼
      Differencing (If Required)
            │
            ▼
      Seasonal Decomposition
            │
            ▼
      ACF / PACF Analysis
            │
            ▼
      Forecast Model Selection
            │
            ▼
      Generate Forecast
            │
            ▼
      Evaluate Forecast Accuracy
            │
            ▼
      Detect Anomalies
            │
            ▼
      Feature Engineering
            │
            ▼
      Export Results
      ```

      ---

      ## 💡 Important Pro Tip

      **Do not immediately jump to forecasting.**

      Always follow this sequence:

      1. **Understand the dataset** - Know what you're working with
      2. **Clean the dataset** - Fix problems and fill gaps
      3. **Explore statistical properties** - Understand patterns and distributions
      4. **Check stationarity** - Ensure the data is stable
      5. **Understand trend and seasonality** - Identify patterns
      6. **Choose the correct forecasting model** - Pick the right tool
      7. **Evaluate the prediction** - Check if the forecast is good

      **Why this matters:** Following this workflow significantly improves forecasting accuracy and prevents common analytical mistakes. Think of it like building a house - you need a solid foundation before you can add the roof.

      ---

      ## 📊 Section Navigation Guide

      ### 📊 Overview Tab
      - Basic statistics of your data
      - Rolling averages and moving trends
      - Check if your data is stable

      ### 🔍 Decomposition Tab
      - Break down data into components
      - Understand trend, seasonality, and noise
      - See correlations with past values

      ### 📈 Forecasting Tab
      - Predict future values
      - Multiple forecasting methods
      - Evaluate prediction accuracy

      ### ⚠️ Anomaly Detection Tab
      - Find unusual data points
      - Detect unexpected patterns
      - Investigate problems

      ### 🛠 Feature Engineering Tab
      - Create new time-based columns
      - Extract useful patterns
      - Prepare for deeper analysis

      ---

      **Ready to begin?** Upload your data below and start exploring your time series! 🚀
              """)

    with st.expander("📁 Upload Time Series Data (CSV/Excel only)", expanded=True):
        uploaded_file = handle_file_upload(
            section=section, file_types=["csv", "xlsx", "xls"],
            title="Upload a CSV or Excel file",
            help_text="Only structured CSV or Excel files with rows and columns are supported.",
        )
        if session_manager.get_data(section, "file_processed", False):
            display_file_info(section)

    df = session_manager.get_dataframe(section)
    if df is None:
        st.info("Please upload and load a dataset first to use time series analysis features.")
        return

    datetime_cols = list(df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns)

    if not datetime_cols:
        # Try to find a column that reliably parses as a date rather than
        # blindly converting the first column that doesn't error out.
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.9:
                df[col] = parsed
                datetime_cols = [col]
                session_manager.set_data(section, "df", df)
                break

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)

    if datetime_cols and numeric_cols:
        with st.expander("📅 Time Series Setup", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select Date Column", datetime_cols, key="date_col")
            with col2:
                value_col = st.selectbox("Select Value Column", numeric_cols, key="value_col")

        if len(df) > 10_000 or len(df.columns) > 50:
            st.warning(
                f"Large dataset detected ({len(df):,} rows, {len(df.columns)} columns). "
                f"Statistical computations still run on the full data for accuracy; "
                f"only plots are thinned for rendering speed."
            )

        with st.spinner("Performing time series analysis..."):
            time_series_analysis(df, date_col, value_col)
    else:
        st.info("The dataset requires at least one datetime column and one numeric column for time series analysis.")


def render_sidebar_controls():
    """Sidebar utility controls. Requires an active session manager, fetched
    fresh here rather than relying on one defined elsewhere in the script."""
    session_manager = get_session_manager()

    with st.sidebar.expander("🚀 Performance Mode"):
        st.session_state.performance_mode = st.toggle(
            "Enable Performance Mode", value=st.session_state.get("performance_mode", False)
        )
        if st.session_state.performance_mode:
            st.info("Performance mode reduces animations for faster processing.")

    with st.sidebar.expander("🔄 Refresh"):
        if st.button("Refresh", help="Refresh the entire application and clear session state"):
            session_manager.clear_all_data()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.button("🧹 Clear All Data", help="Remove all uploaded files and reset the app"):
            session_manager.clear_all_data()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        MemoryManager.display_memory_usage()


def main():
    st.set_page_config(
        page_title="EDA App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_sidebar_controls()
    render_time_series_section()


if __name__ == "__main__":
    main()