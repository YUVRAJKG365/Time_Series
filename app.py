import streamlit as st
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
def safe_to_datetime(series):
    """
    Enterprise-grade datetime parser.

    Supports:
    ----------
    ✓ Mixed datetime formats
    ✓ Mixed timezones
    ✓ Excel serial dates
    ✓ Unix timestamps (seconds)
    ✓ Unix timestamps (milliseconds)
    ✓ ISO8601
    ✓ Numeric timestamps
    ✓ Weather datasets
    ✓ Stock datasets
    ✓ IoT datasets
    ✓ Missing values
    ✓ Duplicate timestamps
    """

    if series is None:
        return pd.Series(dtype="datetime64[ns]")

    s = series.copy()

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            if getattr(s.dt, "tz", None) is not None:
                return s.dt.tz_convert(None)
        except:
            pass
        return s

    # Empty column
    if len(s) == 0:
        return pd.to_datetime(s, errors="coerce")

    ####################################################################
    # Numeric timestamps
    ####################################################################

    if pd.api.types.is_numeric_dtype(s):

        numeric = pd.to_numeric(s, errors="coerce")

        valid = numeric.dropna()

        if len(valid):

            median = valid.median()

            # Excel serial dates
            if 20000 < median < 90000:

                try:
                    parsed = pd.to_datetime(
                        numeric,
                        origin="1899-12-30",
                        unit="D",
                        errors="coerce",
                    )

                    if parsed.notna().mean() > 0.80:
                        return parsed

                except:
                    pass

            # Unix milliseconds
            if median > 1e11:

                try:

                    parsed = pd.to_datetime(
                        numeric,
                        unit="ms",
                        utc=True,
                        errors="coerce",
                    ).dt.tz_localize(None)

                    if parsed.notna().mean() > 0.80:
                        return parsed

                except:
                    pass

            # Unix seconds
            if 1e8 < median < 1e11:

                try:

                    parsed = pd.to_datetime(
                        numeric,
                        unit="s",
                        utc=True,
                        errors="coerce",
                    ).dt.tz_localize(None)

                    if parsed.notna().mean() > 0.80:
                        return parsed

                except:
                    pass

    ####################################################################
    # Mixed strings
    ####################################################################

    s = (
        s.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "NULL": np.nan,
                "null": np.nan,
                "NaT": np.nan,
            }
        )
    )

    # First attempt
    try:

        parsed = pd.to_datetime(
            s,
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_localize(None)

        if parsed.notna().mean() > 0.60:
            return parsed

    except:
        pass

    # Day-first attempt
    try:

        parsed = pd.to_datetime(
            s,
            errors="coerce",
            utc=True,
            dayfirst=True,
            format="mixed",
        ).dt.tz_localize(None)

        if parsed.notna().mean() > 0.60:
            return parsed

    except:
        pass

    # Infer format
    try:

        parsed = pd.to_datetime(
            s,
            errors="coerce",
            utc=True,
            infer_datetime_format=True,
        ).dt.tz_localize(None)

        return parsed

    except:

        return pd.Series(pd.NaT, index=s.index)

def detect_datetime_column(df: pd.DataFrame):
    """
    Automatically detects the best datetime column.

    Returns
    -------
    column_name or None
    """

    if df is None or df.empty:
        return None

    # Common datetime column names
    priority_names = [
        "date",
        "datetime",
        "timestamp",
        "time",
        "day",
        "month",
        "year",
        "created_at",
        "updated_at",
        "order_date",
        "invoice_date",
        "transaction_date",
        "purchase_date",
        "sales_date",
        "weather_date",
        "observation_date",
        "observation_time",
        "measurement_time",
        "recorded_at",
        "event_time",
        "event_date",
        "log_time",
        "log_date",
        "dt"
    ]

    # ---------- STEP 1 ----------
    # Check by column name first

    for keyword in priority_names:

        for col in df.columns:

            if keyword in str(col).lower():

                parsed = safe_to_datetime(df[col])

                success = parsed.notna().mean()

                if success >= 0.60:

                    return col

    # ---------- STEP 2 ----------
    # Score every object/string column

    scores = {}

    for col in df.columns:

        try:

            parsed = safe_to_datetime(df[col])

            success = parsed.notna().mean()

            unique = parsed.nunique()

            score = (
                success * 100
                + min(unique, 100) * 0.05
            )

            scores[col] = score

        except Exception:

            continue

    if len(scores):

        best = max(scores, key=scores.get)

        if scores[best] >= 60:

            return best

    # ---------- STEP 3 ----------
    # Numeric datetime detection
    # Excel serial / Unix timestamps

    for col in df.select_dtypes(include=np.number).columns:

        parsed = safe_to_datetime(df[col])

        success = parsed.notna().mean()

        if success >= 0.80:

            return col

    return None

def detect_target_column(df: pd.DataFrame):
    """
    Detect the best forecasting target column.

    Returns
    -------
    recommended_column
    scores_dictionary
    """

    if df is None or df.empty:
        return None, {}

    scores = {}

    priority_keywords = [
        "sales",
        "revenue",
        "profit",
        "price",
        "temperature",
        "temp",
        "humidity",
        "rain",
        "rainfall",
        "wind",
        "pressure",
        "load",
        "demand",
        "consumption",
        "power",
        "energy",
        "traffic",
        "count",
        "volume",
        "production",
        "stock",
        "close",
        "open",
        "high",
        "low",
        "cases",
        "patients"
    ]

    ignore_keywords = [
        "id",
        "index",
        "serial",
        "zipcode",
        "postal",
        "pin",
        "phone",
        "mobile"
    ]

    numeric_cols = list(df.select_dtypes(include=np.number).columns)

    if not numeric_cols:
        return None, {}

    for col in numeric_cols:

        score = 0

        name = str(col).lower()

        # Ignore ID columns
        if any(x in name for x in ignore_keywords):
            continue

        # Priority names
        if any(x in name for x in priority_keywords):
            score += 40

        series = df[col]

        # Missing values
        missing_ratio = series.isna().mean()
        score += (1 - missing_ratio) * 20

        # Variance
        try:
            variance = series.var()

            if variance > 0:
                score += 15
        except:
            pass

        # Unique values
        unique_ratio = series.nunique() / max(len(series), 1)

        if unique_ratio > 0.30:
            score += 15

        # Continuous values
        if series.nunique() > 20:
            score += 10

        scores[col] = round(score, 2)

    if not scores:
        return None, {}

    best = max(scores, key=scores.get)

    return best, scores

def detect_frequency(df: pd.DataFrame, date_col: str):
    """
    Intelligent frequency detection.

    Returns
    -------
    {
        "code": "D",
        "name": "Daily",
        "seasonal_period": 7,
        "delta": Timedelta(...)
    }
    """

    result = {
        "code": None,
        "name": "Unknown",
        "seasonal_period": None,
        "delta": None,
    }

    if date_col is None:
        return result

    dates = safe_to_datetime(df[date_col])

    dates = dates.dropna().sort_values()

    if len(dates) < 3:
        return result

    # ---------------------------------------------------------
    # Try pandas first
    # ---------------------------------------------------------

    try:

        inferred = pd.infer_freq(dates)

        if inferred is not None:

            result["code"] = inferred

            mapping = {
                "H": ("Hourly", 24),
                "D": ("Daily", 7),
                "W": ("Weekly", 52),
                "M": ("Monthly", 12),
                "MS": ("Monthly", 12),
                "Q": ("Quarterly", 4),
                "QS": ("Quarterly", 4),
                "Y": ("Yearly", 1),
                "YS": ("Yearly", 1),
                "T": ("Minute", 60),
                "min": ("Minute", 60),
            }

            if inferred in mapping:

                result["name"] = mapping[inferred][0]

                result["seasonal_period"] = mapping[inferred][1]

                return result

    except Exception:
        pass

    # ---------------------------------------------------------
    # Manual detection
    # ---------------------------------------------------------

    delta = dates.diff().dropna()

    if len(delta) == 0:
        return result

    median = delta.median()

    result["delta"] = median

    hours = median.total_seconds() / 3600

    days = median.total_seconds() / 86400

    if hours <= 1.1:

        result["code"] = "H"

        result["name"] = "Hourly"

        result["seasonal_period"] = 24

    elif days <= 1.5:

        result["code"] = "D"

        result["name"] = "Daily"

        result["seasonal_period"] = 7

    elif days <= 8:

        result["code"] = "W"

        result["name"] = "Weekly"

        result["seasonal_period"] = 52

    elif 27 <= days <= 32:

        result["code"] = "M"

        result["name"] = "Monthly"

        result["seasonal_period"] = 12

    elif 80 <= days <= 100:

        result["code"] = "Q"

        result["name"] = "Quarterly"

        result["seasonal_period"] = 4

    elif 360 <= days <= 370:

        result["code"] = "Y"

        result["name"] = "Yearly"

        result["seasonal_period"] = 1

    else:

        result["code"] = "IRREGULAR"

        result["name"] = "Irregular"

        result["seasonal_period"] = None

    return result

def analyze_dataset(df: pd.DataFrame):
    """
    Intelligent dataset analyzer.

    Returns
    -------
    dict containing all important dataset information.
    """

    analysis = {}

    if df is None or df.empty:
        return analysis

    # -------------------------------------------------------
    # Basic Information
    # -------------------------------------------------------

    analysis["rows"] = len(df)
    analysis["columns"] = len(df.columns)

    analysis["missing_percent"] = round(
        df.isna().sum().sum()
        / max(df.size, 1)
        * 100,
        2,
    )

    analysis["duplicate_rows"] = int(df.duplicated().sum())

    # -------------------------------------------------------
    # Detect Date Column
    # -------------------------------------------------------

    date_col = detect_datetime_column(df)

    analysis["date_column"] = date_col

    # -------------------------------------------------------
    # Detect Target Column
    # -------------------------------------------------------

    target_col, target_scores = detect_target_column(df)

    analysis["target_column"] = target_col

    analysis["target_scores"] = target_scores

    # -------------------------------------------------------
    # Numeric Columns
    # -------------------------------------------------------

    numeric_cols = list(df.select_dtypes(include=np.number).columns)

    analysis["numeric_columns"] = numeric_cols

    # -------------------------------------------------------
    # Dataset Type
    # -------------------------------------------------------

    dataset_type = "Generic Time Series"

    names = " ".join(df.columns.astype(str)).lower()

    if any(x in names for x in ["temp", "humidity", "wind", "rain", "pressure"]):
        dataset_type = "Weather Dataset"

    elif any(x in names for x in ["sales", "revenue", "profit"]):
        dataset_type = "Sales Dataset"

    elif any(x in names for x in ["close", "open", "volume", "stock"]):
        dataset_type = "Stock Market Dataset"

    elif any(x in names for x in ["sensor", "iot", "device"]):
        dataset_type = "IoT Dataset"

    elif any(x in names for x in ["power", "load", "energy"]):
        dataset_type = "Energy Dataset"

    elif any(x in names for x in ["traffic", "vehicle"]):
        dataset_type = "Traffic Dataset"

    analysis["dataset_type"] = dataset_type

    # -------------------------------------------------------
    # Frequency Detection
    # -------------------------------------------------------

    # -------------------------------------------------------
    # Frequency Detection
    # -------------------------------------------------------

    frequency = detect_frequency(df, date_col)

    analysis["frequency"] = frequency

    # -------------------------------------------------------
    # Trend & Seasonality
    # -------------------------------------------------------

    if (
            date_col is not None
            and target_col is not None
            and date_col in df.columns
            and target_col in df.columns
    ):
        ts_info = detect_trend_and_seasonality(
            df,
            date_col,
            target_col,
        )

        analysis.update(ts_info)

    # -------------------------------------------------------
    # Time Span
    # -------------------------------------------------------

    if date_col is not None:

        dates = safe_to_datetime(df[date_col])

        analysis["start_date"] = dates.min()

        analysis["end_date"] = dates.max()

        analysis["total_days"] = (
            dates.max() - dates.min()
        ).days

    else:

        analysis["start_date"] = None

        analysis["end_date"] = None

        analysis["total_days"] = None

    return analysis

def clean_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
):
    """
    Professional Time Series Cleaning Pipeline

    Performs:
    ----------
    ✓ Datetime conversion
    ✓ Remove invalid timestamps
    ✓ Remove duplicate timestamps
    ✓ Sort chronologically
    ✓ Handle missing target values
    ✓ Infer frequency
    ✓ Fill missing timestamps
    ✓ Time interpolation
    ✓ Forward/Backward fill
    ✓ Remove duplicate observations
    ✓ Reset index

    Returns
    -------
    cleaned_df
    cleaning_report
    """

    report = {
        "original_rows": len(df),
        "rows_removed": 0,
        "duplicates_removed": 0,
        "missing_dates_filled": 0,
        "missing_values_filled": 0,
        "frequency": None,
    }

    df = df.copy()

    # ---------------------------------------------------------
    # Datetime Conversion
    # ---------------------------------------------------------

    df[date_col] = safe_to_datetime(df[date_col])

    before = len(df)

    df = df[df[date_col].notna()].copy()

    report["rows_removed"] += before - len(df)

    # ---------------------------------------------------------
    # Remove duplicate timestamps
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(subset=date_col)

    report["duplicates_removed"] = before - len(df)

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    df = df.sort_values(date_col)

    # ---------------------------------------------------------
    # Set Datetime Index
    # ---------------------------------------------------------

    df = df.set_index(date_col)

    # ---------------------------------------------------------
    # Detect Frequency
    # ---------------------------------------------------------

    freq_info = detect_frequency(
        df.reset_index(),
        date_col,
    )

    report["frequency"] = freq_info["name"]

    freq = freq_info["code"]

    # ---------------------------------------------------------
    # Fill Missing Dates
    # ---------------------------------------------------------

    if freq not in [None, "IRREGULAR"]:

        try:

            full_index = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=freq,
            )

            report["missing_dates_filled"] = (
                len(full_index) - len(df)
            )

            df = df.reindex(full_index)

            df.index.name = date_col

        except Exception:

            pass

    # ---------------------------------------------------------
    # Handle Missing Target Values
    # ---------------------------------------------------------

    if value_col in df.columns:

        before = df[value_col].isna().sum()

        try:

            df[value_col] = df[value_col].interpolate(
                method="time",
                limit_direction="both",
            )

        except Exception:

            df[value_col] = df[value_col].interpolate()

        df[value_col] = (
            df[value_col]
            .ffill()
            .bfill()
        )

        after = df[value_col].isna().sum()

        report["missing_values_filled"] = before - after

    # ---------------------------------------------------------
    # Remove duplicate rows
    # ---------------------------------------------------------

    df = df.drop_duplicates()

    # ---------------------------------------------------------
    # Reset Index
    # ---------------------------------------------------------

    df = df.reset_index()

    report["final_rows"] = len(df)

    return df, report

def detect_trend_and_seasonality(
        df: pd.DataFrame,
        date_col: str,
        value_col: str
):
    """
    Detect trend, seasonality and data characteristics.

    Returns
    -------
    Dictionary containing:

    trend_direction
    trend_strength
    seasonality
    seasonal_strength
    noise_level
    recommended_model
    """

    result = {}

    if df.empty:
        return result

    try:

        series = (
            df[[date_col, value_col]]
            .dropna()
            .copy()
        )

        series = series.sort_values(date_col)

        y = series[value_col].astype(float)

        x = np.arange(len(y))

        # --------------------------------------------------
        # TREND
        # --------------------------------------------------

        slope, intercept = np.polyfit(x, y, 1)

        trend_std = np.std(intercept + slope * x)

        data_std = np.std(y)

        if data_std == 0:
            trend_strength = 0
        else:
            trend_strength = trend_std / data_std

        if slope > 0:

            direction = "Increasing"

        elif slope < 0:

            direction = "Decreasing"

        else:

            direction = "Flat"

        result["trend_direction"] = direction

        result["trend_strength"] = round(
            float(trend_strength),
            3,
        )

        # --------------------------------------------------
        # SEASONALITY
        # --------------------------------------------------

        freq = detect_frequency(df, date_col)

        period = freq["seasonal_period"]

        result["seasonal_period"] = period

        seasonality = "Unknown"

        seasonal_strength = 0

        if (
            period is not None
            and len(y) >= period * 2
        ):

            try:

                decomposition = seasonal_decompose(
                    y,
                    model="additive",
                    period=period,
                    extrapolate_trend="freq",
                )

                seasonal = decomposition.seasonal

                residual = decomposition.resid

                seasonal_strength = (
                    np.nanstd(seasonal)
                    /
                    np.nanstd(y)
                )

                if seasonal_strength > 0.60:

                    seasonality = "Strong"

                elif seasonal_strength > 0.30:

                    seasonality = "Moderate"

                else:

                    seasonality = "Weak"

            except Exception:

                pass

        result["seasonality"] = seasonality

        result["seasonal_strength"] = round(
            float(seasonal_strength),
            3,
        )

        # --------------------------------------------------
        # Noise
        # --------------------------------------------------

        rolling = y.rolling(
            window=max(5, len(y)//20),
            center=True
        ).mean()

        noise = np.nanstd(y - rolling)

        result["noise_level"] = round(
            float(noise),
            3,
        )

        # --------------------------------------------------
        # Recommended Model
        # --------------------------------------------------

        if (
            direction != "Flat"
            and seasonality == "Strong"
        ):

            model = "SARIMA"

        elif direction != "Flat":

            model = "ARIMA"

        elif seasonality == "Strong":

            model = "Exponential Smoothing"

        else:

            model = "Prophet"

        result["recommended_model"] = model

        return result

    except Exception:

        return {}

def prepare_forecast_data(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Prepare dataset for forecasting.

    Returns
    -------
    cleaned dataframe
    metadata
    """

    metadata = {}

    df = df.copy()

    # ---------------------------------------
    # Clean Dataset
    # ---------------------------------------

    df, cleaning_report = clean_time_series(
        df,
        date_col,
        value_col,
    )

    metadata["cleaning"] = cleaning_report

    # ---------------------------------------
    # AI Analysis
    # ---------------------------------------

    analysis = analyze_dataset(df)

    metadata["analysis"] = analysis

    # ---------------------------------------
    # Datetime Index
    # ---------------------------------------

    df = df.set_index(date_col)

    # ---------------------------------------
    # Keep only target
    # ---------------------------------------

    df = df[[value_col]]

    # ---------------------------------------
    # Float conversion
    # ---------------------------------------

    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    df = df.dropna()

    # Reduce extremely large datasets while preserving temporal structure
    if len(df) > 50000:
        step = max(1, len(df) // 50000)

        df = df.iloc[::step].copy()

    # ---------------------------------------
    # Final validation
    # ---------------------------------------

    if len(df) < 20:
        raise ValueError(
            "Dataset is too small for reliable forecasting."
        )

    return df, metadata

def split_train_test(
        df: pd.DataFrame,
        target_col: str,
):
    """
    Adaptive train-test split.
    """

    n = len(df)

    if n < 100:
        test_size = 0.30

    elif n < 1000:
        test_size = 0.20

    elif n < 10000:
        test_size = 0.15

    else:
        test_size = 0.10

    split_index = int(n * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df

def calculate_forecast_metrics(
        actual,
        predicted,
):
    """
    Calculate forecasting metrics.
    """

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted)
            /
            np.maximum(
                np.abs(actual),
                1e-9,
            )
        )
    ) * 100

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 2),
        "R2": round(float(r2), 4),
    }

def train_forecasting_models(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        models_to_train: list = None,
):
    """
    Train and evaluate forecasting models.

    Returns
    -------
    {
        "ARIMA": {
            "model": fitted_model,
            "forecast": forecast,
            "metrics": {...}
        },
        ...
    }
    """

    if models_to_train is None:

        models_to_train = [
            "Naive",
            "Moving Average",
            "ARIMA",
            "SARIMA",
            "Holt-Winters",
        ]

    results = {}

    y_train = train_df[target_col]

    y_test = test_df[target_col]

    forecast_steps = len(test_df)

    # Create progress tracking
    total_models = len(models_to_train)
    completed = 0

    # =====================================================
    # Naive Forecast
    # =====================================================

    if "Naive" in models_to_train:
        try:
            with st.spinner(f"⏳ Training: Naive (1/{total_models})"):
                forecast = np.repeat(
                    y_train.iloc[-1],
                    forecast_steps,
                )

                metrics = calculate_forecast_metrics(
                    y_test,
                    forecast,
                )

                results["Naive"] = {

                    "model": None,

                    "forecast": forecast,

                    "metrics": metrics,
                }
                completed += 1

        except Exception:

            pass

    # =====================================================
    # Moving Average Forecast
    # =====================================================

    if "Moving Average" in models_to_train:

        try:
            with st.spinner(f"⏳ Training: Moving Average (2/{total_models})"):
                window = min(10, max(3, len(y_train)//10))

                avg = y_train.tail(window).mean()

                forecast = np.repeat(
                    avg,
                    forecast_steps,
                )

                metrics = calculate_forecast_metrics(
                    y_test,
                    forecast,
                )

                results["Moving Average"] = {

                    "model": None,

                    "forecast": forecast,

                    "metrics": metrics,
                }
                completed += 1

        except Exception:

            pass

        # =====================================================
        # ARIMA
        # =====================================================

    if "ARIMA" in models_to_train:

        try:
            with st.spinner(f"⏳ Training: ARIMA (3/{total_models})..."):
                results["ARIMA"] = train_arima_model(
                    train_df,
                    test_df,
                    target_col,
                )
                completed += 1

        except Exception:

            pass

    # =====================================================
    # SARIMA
    # =====================================================

    if "SARIMA" in models_to_train:

        try:
            with st.spinner(f"⏳ Training: SARIMA (4/{total_models})..."):
                results["SARIMA"] = train_sarima_model(
                    train_df,
                    test_df,
                    train_df.index.name,
                    target_col,
                )
                completed += 1

        except Exception:

            pass

    # =====================================================
    # Holt-Winters
    # =====================================================

    if "Holt-Winters" in models_to_train:

        try:
            with st.spinner(f"⏳ Training: Holt-Winters (5/{total_models})..."):
                results["Holt-Winters"] = train_holt_winters_model(
                    train_df,
                    test_df,
                    train_df.index.name,
                    target_col,
                )
                completed += 1

        except Exception:

            pass

    return results

def train_arima_model(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
):
    """
    Train ARIMA using automatic parameter search.

    Returns
    -------
    {
        "model": fitted_model,
        "forecast": forecast,
        "metrics": {...},
        "order": (p,d,q)
    }
    """

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # ---------------------------------------------
    # Automatic differencing recommendation
    # ---------------------------------------------

    stationarity = check_stationarity(
        train_df,
        target_col,
    )

    recommended_d = stationarity["recommended_d"]

    best_model = None
    best_order = None
    best_forecast = None
    best_metrics = None

    best_rmse = float("inf")

    # ---------------------------------------------------
    # Adaptive ARIMA Search Space
    # ---------------------------------------------------

    n = len(train_df)

    if n < 1000:

        max_p = 3
        max_q = 3

    elif n < 10000:

        max_p = 2
        max_q = 2

    else:

        max_p = 1
        max_q = 1

    for p in range(max_p + 1):

        for d in [recommended_d]:

            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue

                try:

                    model = ARIMA(
                        y_train,
                        order=(p, d, q),
                    )

                    fitted = model.fit()

                    forecast = fitted.forecast(
                        steps=len(y_test)
                    )

                    metrics = calculate_forecast_metrics(
                        y_test,
                        forecast,
                    )

                    rmse = metrics["RMSE"]

                    if rmse < best_rmse:

                        best_rmse = rmse

                        best_model = fitted

                        best_order = (p, d, q)

                        best_forecast = forecast

                        best_metrics = metrics

                except Exception:

                    continue

    if best_model is None:

        raise RuntimeError(
            "ARIMA could not be trained on this dataset."
        )

    return {

        "model": best_model,

        "forecast": best_forecast,

        "metrics": best_metrics,

        "order": best_order,

        "stationarity": stationarity,
    }

def train_sarima_model(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str,
        target_col: str,
        max_p: int = 2,
        max_q: int = 2,
        max_P: int = 1,
        max_Q: int = 1,
):
    """
    Train SARIMA with automatic seasonal parameter selection.

    Returns
    -------
    {
        "model": fitted_model,
        "forecast": forecast,
        "metrics": {...},
        "order": (...),
        "seasonal_order": (...),
        "stationarity": {...}
    }
    """

    y_train = (
        pd.to_numeric(
            train_df[target_col],
            errors="coerce"
        )
        .dropna()
    )

    y_test = (
        pd.to_numeric(
            test_df[target_col],
            errors="coerce"
        )
        .dropna()
    )

    if len(y_train) < 30:
        raise ValueError(
            "Dataset too small for SARIMA."
        )

    # ----------------------------------------
    # Stationarity
    # ----------------------------------------

    stationarity = check_stationarity(
        train_df,
        target_col,
    )

    d = stationarity["recommended_d"]

    # ----------------------------------------
    # Seasonality
    # ----------------------------------------

    seasonality = detect_seasonality(
        train_df.reset_index(),
        date_col,
        target_col,
    )

    m = seasonality["seasonal_period"]

    if m is None:
        raise ValueError(
            "Seasonality not detected."
        )

    best_model = None
    best_forecast = None
    best_metrics = None
    best_order = None
    best_seasonal = None

    best_rmse = float("inf")
    n = len(train_df)

    if n < 1000:

        max_p = 2
        max_q = 2
        max_P = 1
        max_Q = 1

    elif n < 10000:

        max_p = 1
        max_q = 1
        max_P = 1
        max_Q = 1

    else:

        max_p = 1
        max_q = 1
        max_P = 1
        max_Q = 0

    for p in range(max_p + 1):

        for q in range(max_q + 1):

            for P in range(max_P + 1):

                for Q in range(max_Q + 1):

                    if (
                        p == 0
                        and d == 0
                        and q == 0
                        and P == 0
                        and Q == 0
                    ):
                        continue

                    try:

                        model = SARIMAX(
                            y_train,
                            order=(p, d, q),
                            seasonal_order=(P, d, Q, m),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )

                        fitted = model.fit(
                            disp=False
                        )

                        forecast = fitted.forecast(
                            steps=len(y_test)
                        )

                        metrics = calculate_forecast_metrics(
                            y_test,
                            forecast,
                        )

                        rmse = metrics["RMSE"]

                        if rmse < best_rmse:

                            best_rmse = rmse

                            best_model = fitted

                            best_forecast = forecast

                            best_metrics = metrics

                            best_order = (
                                p,
                                d,
                                q,
                            )

                            best_seasonal = (
                                P,
                                d,
                                Q,
                                m,
                            )

                    except Exception:

                        continue

    if best_model is None:

        raise RuntimeError(
            "SARIMA training failed."
        )

    return {

        "model": best_model,

        "forecast": best_forecast,

        "metrics": best_metrics,

        "order": best_order,

        "seasonal_order": best_seasonal,

        "stationarity": stationarity,

        "seasonality": seasonality,
    }

def train_holt_winters_model(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str,
        target_col: str,
):
    """
    Train Holt-Winters Exponential Smoothing.

    Automatically detects trend and seasonality.

    Returns
    -------
    {
        model,
        forecast,
        metrics,
        trend,
        seasonal,
        seasonal_period
    }
    """

    y_train = (
        pd.to_numeric(
            train_df[target_col],
            errors="coerce"
        )
        .dropna()
    )

    y_test = (
        pd.to_numeric(
            test_df[target_col],
            errors="coerce"
        )
        .dropna()
    )

    if len(y_train) < 20:
        raise ValueError(
            "Dataset too small."
        )

    seasonality = detect_seasonality(
        train_df.reset_index(),
        date_col,
        target_col,
    )

    trend = detect_trend(
        train_df.reset_index(),
        date_col,
        target_col,
    )

    period = seasonality["seasonal_period"]

    if period is None:
        period = 12

    trend_component = (
        "add"
        if trend["trend_direction"] != "Flat"
        else None
    )

    seasonal_component = (
        "add"
        if seasonality["seasonality"] in [
            "Moderate",
            "Strong",
        ]
        else None
    )

    model = ExponentialSmoothing(
        y_train,
        trend=trend_component,
        seasonal=seasonal_component,
        seasonal_periods=period,
    )

    fitted = model.fit(
        optimized=True,
    )

    forecast = fitted.forecast(
        len(y_test)
    )

    metrics = calculate_forecast_metrics(
        y_test,
        forecast,
    )

    return {

        "model": fitted,

        "forecast": forecast,

        "metrics": metrics,

        "trend": trend,

        "seasonality": seasonality,

        "trend_component": trend_component,

        "seasonal_component": seasonal_component,
    }

def generate_future_forecast(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        model_name: str,
        forecast_steps: int,
):
    """
    Generate future forecast using the selected model.

    Returns
    -------
    {
        "future_dates": ...,
        "forecast": ...,
        "lower": ...,
        "upper": ...
    }
    """

    df = df.copy()

    df[date_col] = safe_to_datetime(df[date_col])

    df = (
        df.dropna(subset=[date_col, value_col])
          .sort_values(date_col)
    )

    series = pd.to_numeric(
        df[value_col],
        errors="coerce",
    ).dropna()

    freq_info = detect_frequency(df, date_col)

    freq = freq_info["code"]

    if freq in [None, "IRREGULAR"]:
        freq = "D"

    # Normalize offset codes for pandas 2.x compatibility.
    freq_map = {
        "m": "ME",
        "M": "ME",
        "q": "QE",
        "Q": "QE",
        "y": "YE",
        "Y": "YE",
        "w": "W",
        "W": "W",
        "d": "D",
        "D": "D",
        "h": "h",
        "H": "h",
        "ms": "MS",
        "MS": "MS",
        "qs": "QS",
        "QS": "QS",
        "ys": "YS",
        "YS": "YS",
        "t": "min",
        "T": "min",
    }

    freq = freq_map.get(str(freq), str(freq))

    future_dates = pd.date_range(
        start=df[date_col].iloc[-1],
        periods=forecast_steps + 1,
        freq=freq,
    )[1:]

    forecast = None
    lower = None
    upper = None

    # =====================================================
    # ARIMA
    # =====================================================

    if model_name == "ARIMA":

        stationarity = check_stationarity(
            df,
            value_col,
        )

        d = stationarity["recommended_d"]

        best_aic = float("inf")
        best_model = None

        for p in range(4):

            for q in range(4):

                if p == 0 and d == 0 and q == 0:
                    continue

                try:

                    model = ARIMA(
                        series,
                        order=(p, d, q),
                    )

                    fitted = model.fit()

                    if fitted.aic < best_aic:

                        best_aic = fitted.aic
                        best_model = fitted

                except Exception:
                    continue

        if best_model is None:
            raise RuntimeError("Unable to fit ARIMA.")

        prediction = best_model.get_forecast(
            steps=forecast_steps
        )

        forecast = prediction.predicted_mean.values

        ci = prediction.conf_int()

        lower = ci.iloc[:, 0].values

        upper = ci.iloc[:, 1].values

    # =====================================================
    # SARIMA
    # =====================================================

    elif model_name == "SARIMA":

        stationarity = check_stationarity(
            df,
            value_col,
        )

        seasonality = detect_seasonality(
            df,
            date_col,
            value_col,
        )

        d = stationarity["recommended_d"]

        m = seasonality["seasonal_period"]

        if m is None:
            m = 12

        model = SARIMAX(
            series,
            order=(1, d, 1),
            seasonal_order=(1, d, 1, m),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(disp=False)

        prediction = fitted.get_forecast(
            steps=forecast_steps
        )

        forecast = prediction.predicted_mean.values

        ci = prediction.conf_int()

        lower = ci.iloc[:, 0].values

        upper = ci.iloc[:, 1].values

    # =====================================================
    # Holt-Winters
    # =====================================================

    elif model_name == "Holt-Winters":

        seasonality = detect_seasonality(
            df,
            date_col,
            value_col,
        )

        trend = detect_trend(
            df,
            date_col,
            value_col,
        )

        period = seasonality["seasonal_period"]

        if period is None:
            period = 12

        model = ExponentialSmoothing(
            series,
            trend="add" if trend["trend_direction"] != "Flat" else None,
            seasonal="add" if seasonality["seasonality"] != "None" else None,
            seasonal_periods=period,
        )

        fitted = model.fit(
            optimized=True
        )

        forecast = fitted.forecast(
            forecast_steps
        ).values

    # =====================================================
    # Moving Average
    # =====================================================

    elif model_name == "Moving Average":

        window = min(
            10,
            max(
                3,
                len(series) // 10,
            ),
        )

        avg = series.tail(window).mean()

        forecast = np.repeat(
            avg,
            forecast_steps,
        )

    # =====================================================
    # Naive
    # =====================================================

    else:

        forecast = np.repeat(
            series.iloc[-1],
            forecast_steps,
        )

    return {

        "future_dates": future_dates,

        "forecast": np.asarray(forecast),

        "lower": lower,

        "upper": upper,
    }

def select_best_forecasting_model(results: dict):
    """
    Compare all trained forecasting models and rank them.

    Parameters
    ----------
    results : dict
        Output from train_forecasting_models()

    Returns
    -------
    {
        "best_model": ...,
        "leaderboard": DataFrame,
        "results": ...
    }
    """

    if not results:
        raise ValueError("No forecasting models were successfully trained.")

    leaderboard = []

    for model_name, model_info in results.items():

        metrics = model_info.get("metrics", {})

        leaderboard.append({

            "Model": model_name,

            "RMSE": metrics.get("RMSE", np.inf),

            "MAE": metrics.get("MAE", np.inf),

            "MAPE": metrics.get("MAPE", np.inf),

            "R2": metrics.get("R2", -np.inf),
        })

    leaderboard = pd.DataFrame(leaderboard)

    # -------------------------------------------------
    # Ranking
    # -------------------------------------------------

    leaderboard["RMSE Rank"] = leaderboard["RMSE"].rank(method="min")

    leaderboard["MAE Rank"] = leaderboard["MAE"].rank(method="min")

    leaderboard["MAPE Rank"] = leaderboard["MAPE"].rank(method="min")

    leaderboard["R2 Rank"] = (
        leaderboard["R2"]
        .rank(method="min", ascending=False)
    )

    leaderboard["Overall Score"] = (

        leaderboard["RMSE Rank"]

        + leaderboard["MAE Rank"]

        + leaderboard["MAPE Rank"]

        + leaderboard["R2 Rank"]

    )

    leaderboard = leaderboard.sort_values(
        "Overall Score"
    ).reset_index(drop=True)

    leaderboard.index += 1

    best_model = leaderboard.iloc[0]["Model"]

    return {

        "best_model": best_model,

        "leaderboard": leaderboard,

        "results": results,
    }
@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def forecast_pipeline(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Complete AI Forecasting Pipeline.

    Returns
    -------
    Dictionary containing

    • Clean data

    • AI analysis

    • Trained models

    • Leaderboard

    • Best model
    """

    # -----------------------------------------
    # Prepare
    # -----------------------------------------

    forecast_df, metadata = prepare_forecast_data(
        df,
        date_col,
        value_col,
    )

    # -----------------------------------------
    # Train/Test Split
    # -----------------------------------------

    train_df, test_df = split_train_test(
        forecast_df,
        value_col,
    )

    # -----------------------------------------
    # Train Models
    # -----------------------------------------

    model_results = train_forecasting_models(
        train_df,
        test_df,
        value_col,
    )

    # -----------------------------------------
    # Select Best
    # -----------------------------------------

    comparison = select_best_forecasting_model(
        model_results
    )

    metadata["analysis"]["recommended_model"] = comparison["best_model"]
    metadata["analysis"]["recommendation_reason"] = (
        f"Best model selected from training results: {comparison['best_model']}."
        if "analysis" in metadata and metadata["analysis"]
        else "Best model selected from training results."
    )

    comparison["metadata"] = metadata

    comparison["train"] = train_df

    comparison["test"] = test_df

    return comparison

def detect_trend(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Detect long-term trend characteristics.

    Returns
    -------
    {
        trend_direction,
        trend_type,
        slope,
        intercept,
        r2,
        trend_strength,
        confidence,
    }
    """

    result = {
        "trend_direction": "Unknown",
        "trend_type": "Unknown",
        "slope": 0.0,
        "intercept": 0.0,
        "r2": 0.0,
        "trend_strength": "Unknown",
        "confidence": 0.0,
    }

    try:

        if df.empty:
            return result

        data = (
            df[[date_col, value_col]]
            .dropna()
            .copy()
            .sort_values(date_col)
        )

        if len(data) < 10:
            return result

        y = data[value_col].astype(float).values

        x = np.arange(len(y))

        regression = linregress(x, y)

        slope = regression.slope
        intercept = regression.intercept

        r2 = regression.rvalue ** 2

        result["slope"] = float(slope)
        result["intercept"] = float(intercept)
        result["r2"] = round(float(r2), 4)

        # ---------------------------------------------------
        # Direction
        # ---------------------------------------------------

        tolerance = np.std(y) * 0.0005

        if slope > tolerance:

            direction = "Increasing"

        elif slope < -tolerance:

            direction = "Decreasing"

        else:

            direction = "Flat"

        result["trend_direction"] = direction

        # ---------------------------------------------------
        # Trend Strength
        # ---------------------------------------------------

        if r2 >= 0.85:

            strength = "Very Strong"

        elif r2 >= 0.65:

            strength = "Strong"

        elif r2 >= 0.40:

            strength = "Moderate"

        elif r2 >= 0.20:

            strength = "Weak"

        else:

            strength = "Very Weak"

        result["trend_strength"] = strength

        # ---------------------------------------------------
        # Trend Type
        # ---------------------------------------------------

        linear_pred = intercept + slope * x

        linear_rmse = np.sqrt(
            np.mean(
                (y - linear_pred) ** 2
            )
        )

        positive = np.all(y > 0)

        if positive:

            try:

                log_y = np.log(y)

                exp_reg = linregress(x, log_y)

                exp_pred = np.exp(
                    exp_reg.intercept +
                    exp_reg.slope * x
                )

                exp_rmse = np.sqrt(
                    np.mean(
                        (y - exp_pred) ** 2
                    )
                )

                if exp_rmse < linear_rmse * 0.90:

                    trend_type = "Exponential"

                else:

                    trend_type = "Linear"

            except Exception:

                trend_type = "Linear"

        else:

            trend_type = "Linear"

        result["trend_type"] = trend_type

        # ---------------------------------------------------
        # Confidence
        # ---------------------------------------------------

        confidence = min(
            100,
            max(
                0,
                r2 * 100,
            ),
        )

        result["confidence"] = round(confidence, 1)

        return result

    except Exception:

        return result

def detect_seasonality(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Detect seasonality using decomposition + autocorrelation.

    Returns
    -------
    {
        seasonality,
        seasonal_period,
        seasonal_strength,
        acf_peak,
        confidence
    }
    """

    result = {
        "seasonality": "Unknown",
        "seasonal_period": None,
        "seasonal_strength": 0.0,
        "acf_peak": 0.0,
        "confidence": 0.0,
    }

    try:

        if df.empty:
            return result

        data = (
            df[[date_col, value_col]]
            .dropna()
            .copy()
            .sort_values(date_col)
        )

        if len(data) < 20:
            return result

        y = data[value_col].astype(float)

        # --------------------------------------------
        # Frequency
        # --------------------------------------------

        freq_info = detect_frequency(df, date_col)

        period = freq_info["seasonal_period"]

        result["seasonal_period"] = period

        if period is None:
            return result

        if len(y) < period * 2:
            return result

        # --------------------------------------------
        # Seasonal Decomposition
        # --------------------------------------------

        decomposition = seasonal_decompose(
            y,
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )

        seasonal = decomposition.seasonal

        strength = (
            np.nanstd(seasonal)
            /
            max(np.nanstd(y), 1e-9)
        )

        result["seasonal_strength"] = round(
            float(strength),
            3,
        )

        # --------------------------------------------
        # Autocorrelation
        # --------------------------------------------

        acf_values = acf(
            y,
            nlags=min(period * 3, len(y) // 2),
            fft=True,
        )

        peak = np.max(
            np.abs(
                acf_values[period:]
            )
        )

        result["acf_peak"] = round(
            float(peak),
            3,
        )

        # --------------------------------------------
        # Seasonality Classification
        # --------------------------------------------

        score = (
            strength * 0.6 +
            peak * 0.4
        )

        if score >= 0.70:

            label = "Strong"

        elif score >= 0.45:

            label = "Moderate"

        elif score >= 0.25:

            label = "Weak"

        else:

            label = "None"

        result["seasonality"] = label

        result["confidence"] = round(
            min(score * 100, 100),
            1,
        )

        return result

    except Exception:

        return result

def check_stationarity(
        df: pd.DataFrame,
        value_col: str,
):
    """
    Check stationarity using Augmented Dickey-Fuller Test.

    Returns
    -------
    {
        stationary,
        p_value,
        recommended_d,
        confidence,
        interpretation
    }
    """

    result = {

        "stationary": False,

        "p_value": None,

        "recommended_d": 1,

        "confidence": 0,

        "interpretation": "Unknown",
    }

    try:

        series = (
            pd.to_numeric(
                df[value_col],
                errors="coerce",
            )
            .dropna()
        )

        if len(series) < 20:

            return result

        adf = adfuller(series)

        statistic = adf[0]

        pvalue = adf[1]

        result["p_value"] = round(
            float(pvalue),
            5,
        )

        if pvalue < 0.05:

            result["stationary"] = True

            result["recommended_d"] = 0

            result["interpretation"] = (
                "Series is stationary."
            )

        else:

            diff1 = series.diff().dropna()

            pvalue1 = adfuller(diff1)[1]

            if pvalue1 < 0.05:

                result["recommended_d"] = 1

            else:

                result["recommended_d"] = 2

            result["stationary"] = False

            result["interpretation"] = (
                "Series requires differencing."
            )

        confidence = max(
            0,
            min(
                100,
                (1 - pvalue) * 100,
            ),
        )

        result["confidence"] = round(
            confidence,
            1,
        )

        result["adf_statistic"] = round(
            float(statistic),
            4,
        )

        return result

    except Exception:

        return result

def analyze_signal_quality(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Analyze overall signal quality for forecasting.

    Returns
    -------
    {
        noise_level,
        signal_strength,
        signal_to_noise_ratio,
        missing_percent,
        outlier_percent,
        data_quality,
        forecast_difficulty,
        confidence
    }
    """

    result = {
        "noise_level": 0.0,
        "signal_strength": 0.0,
        "signal_to_noise_ratio": 0.0,
        "missing_percent": 0.0,
        "outlier_percent": 0.0,
        "data_quality": "Unknown",
        "forecast_difficulty": "Unknown",
        "confidence": 0.0,
    }

    try:

        if df.empty:
            return result

        data = (
            df[[date_col, value_col]]
            .dropna()
            .copy()
            .sort_values(date_col)
        )

        if len(data) < 10:
            return result

        y = data[value_col].astype(float)

        # -------------------------------------------------
        # Missing %
        # -------------------------------------------------

        missing_percent = (
            df[value_col].isna().mean() * 100
        )

        result["missing_percent"] = round(
            float(missing_percent),
            2,
        )

        # -------------------------------------------------
        # Outlier %
        # -------------------------------------------------

        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((y < lower) | (y > upper)).sum()

        outlier_percent = (
            outliers / len(y)
        ) * 100

        result["outlier_percent"] = round(
            float(outlier_percent),
            2,
        )

        # -------------------------------------------------
        # Noise
        # -------------------------------------------------

        smooth = (
            y.rolling(
                window=max(5, len(y)//25),
                center=True,
                min_periods=1,
            )
            .mean()
        )

        noise = np.std(y - smooth)

        signal = np.std(smooth)

        result["noise_level"] = round(
            float(noise),
            3,
        )

        result["signal_strength"] = round(
            float(signal),
            3,
        )

        snr = signal / max(noise, 1e-9)

        result["signal_to_noise_ratio"] = round(
            float(snr),
            3,
        )

        # -------------------------------------------------
        # Data Quality
        # -------------------------------------------------

        score = 100

        score -= missing_percent * 0.6

        score -= outlier_percent * 0.8

        if snr < 1:
            score -= 30

        elif snr < 2:
            score -= 15

        score = max(0, min(score, 100))

        if score >= 90:
            quality = "Excellent"

        elif score >= 75:
            quality = "Good"

        elif score >= 60:
            quality = "Fair"

        elif score >= 40:
            quality = "Poor"

        else:
            quality = "Very Poor"

        result["data_quality"] = quality

        # -------------------------------------------------
        # Forecast Difficulty
        # -------------------------------------------------

        if score >= 90:

            difficulty = "Easy"

        elif score >= 70:

            difficulty = "Moderate"

        elif score >= 50:

            difficulty = "Hard"

        else:

            difficulty = "Very Hard"

        result["forecast_difficulty"] = difficulty

        result["confidence"] = round(score, 1)

        return result

    except Exception:

        return result

def detect_trend_and_seasonality(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
):
    """
    Master AI Time Series Intelligence Engine

    Combines:
    - Trend Detection
    - Seasonality Detection
    - Signal Quality Analysis

    Returns one complete intelligence dictionary.
    """

    analysis = {}

    if (
        df is None
        or df.empty
        or date_col not in df.columns
        or value_col not in df.columns
    ):
        return analysis

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------

    trend = detect_trend(
        df,
        date_col,
        value_col,
    )

    analysis.update(trend)

    # ---------------------------------------------------------
    # Seasonality
    # ---------------------------------------------------------

    seasonality = detect_seasonality(
        df,
        date_col,
        value_col,
    )

    analysis.update(seasonality)

    # ---------------------------------------------------------
    # Signal Quality
    # ---------------------------------------------------------

    quality = analyze_signal_quality(
        df,
        date_col,
        value_col,
    )

    analysis.update(quality)

    # ---------------------------------------------------------
    # AI Recommended Model
    # ---------------------------------------------------------

    trend_strength = analysis.get("trend_strength", "")
    seasonality_strength = analysis.get("seasonality", "")
    snr = analysis.get("signal_to_noise_ratio", 0)
    frequency = detect_frequency(df, date_col)

    recommended = "ARIMA"
    reason = []

    if (
        seasonality_strength == "Strong"
        and trend_strength in ["Strong", "Very Strong"]
    ):
        recommended = "SARIMA"
        reason.append("Strong trend and strong seasonality detected.")

    elif (
        seasonality_strength == "Strong"
        and trend_strength in ["Weak", "Very Weak"]
    ):
        recommended = "Exponential Smoothing"
        reason.append("Seasonality dominates the series.")

    elif (
        trend_strength in ["Strong", "Very Strong"]
        and seasonality_strength == "None"
    ):
        recommended = "ARIMA"
        reason.append("Strong trend with little seasonality.")

    elif frequency["name"] in [
        "Hourly",
        "Daily",
        "Weekly",
        "Monthly",
    ]:
        recommended = "Prophet"
        reason.append("Regular calendar frequency detected.")

    if snr < 1:
        reason.append("High noise detected; forecast confidence may be lower.")

    analysis["recommended_model"] = recommended
    analysis["recommendation_reason"] = " ".join(reason)

    # ---------------------------------------------------------
    # Forecast Readiness Score
    # ---------------------------------------------------------

    score = 100

    score -= analysis.get("missing_percent", 0) * 0.5
    score -= analysis.get("outlier_percent", 0) * 0.7

    if snr < 1:
        score -= 25
    elif snr < 2:
        score -= 10

    score = max(0, min(score, 100))

    analysis["forecast_readiness"] = round(score, 1)

    if score >= 90:
        analysis["forecast_grade"] = "A+"

    elif score >= 80:
        analysis["forecast_grade"] = "A"

    elif score >= 70:
        analysis["forecast_grade"] = "B"

    elif score >= 60:
        analysis["forecast_grade"] = "C"

    else:
        analysis["forecast_grade"] = "Needs Improvement"

    return analysis

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
        parsed = safe_to_datetime(df[col])
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

        # ======================================================
        # AI Cleaning Pipeline
        # ======================================================

        df, cleaning_report = clean_time_series(
            df,
            date_col,
            value_col,
        )

        with st.expander("🧹 Data Cleaning Report", expanded=False):
            st.json(cleaning_report)

        # ------------------------------------------------------

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

    st.markdown("#### 🤖 AI Forecasting")

    try:
        # Show progress during forecast pipeline
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.info("🔄 Step 1/4: Preparing data...")
        progress_bar.progress(10)

        forecast_df = (
            df.reset_index()
            .copy()
        )

        status_text.info("🔄 Step 2/4: Training models...")
        progress_bar.progress(40)

        forecast_info = forecast_pipeline(
            df.reset_index(),
            date_col,
            value_col,
        )

        status_text.info("🔄 Step 3/4: Evaluating performance...")
        progress_bar.progress(70)

        # Clear progress indicators
        progress_bar.progress(100)
        status_text.success("✅ Forecast complete!")

    except Exception as e:

        st.error(f"Forecast initialization failed: {e}")

        return

    analysis = forecast_info["metadata"]["analysis"]

    st.markdown("## 🤖 AI Forecast Summary")

    c1, c2, c3, c4 = st.columns(4)

    frequency = analysis.get("frequency", {})

    c1.metric(
        "Dataset",
        analysis.get("dataset_type", "Unknown")
    )

    c2.metric(
        "Frequency",
        frequency.get("name", "Unknown")
    )

    c3.metric(
        "Recommended",
        forecast_info.get("best_model", analysis.get("recommended_model", "Unknown"))
    )

    c4.metric(
        "Readiness",
        f"{analysis.get('forecast_readiness', 0):.1f}%"
    )

    leaderboard = forecast_info["leaderboard"]

    st.markdown("## 🏆 Model Leaderboard")

    st.dataframe(
        leaderboard,
        width='stretch',
        hide_index=False
    )

    train = forecast_info["train"]

    test = forecast_info["test"]

    best_model = forecast_info["best_model"]

    models = [
        "🤖 Auto (Recommended)",
        "ARIMA",
        "SARIMA",
        "Holt-Winters",
        "Moving Average",
        "Naive",
    ]

    model_choice = st.selectbox(
        "Forecast Model",
        models,
    )

    forecast_periods = st.slider("Forecast periods (future steps)", 1, 365, 30, key="forecast_periods")

    if model_choice == "🤖 Auto (Recommended)":

        selected_model = best_model

    else:

        selected_model = model_choice

    selected_result = forecast_info["results"][selected_model]

    metrics = selected_result["metrics"]

    model_name = selected_model

    future_result = generate_future_forecast(
        df.reset_index(),
        date_col,
        value_col,
        selected_model,
        forecast_periods,
    )

    forecast = future_result["forecast"]

    future_dates = future_result["future_dates"]

    lower = future_result["lower"]

    upper = future_result["upper"]

    st.info(
        f"""
    ### 🤖 AI Recommendation

    **Selected Model**

    ✅ **{selected_model}**

    Forecast Readiness:
    **{analysis.get('forecast_readiness', 0):.1f}%**

    Reason:

    {analysis.get('recommendation_reason', 'No recommendation available.')}
    """
    )

    if lower is not None:
        lower = np.asarray(lower)[:len(forecast)]

    if upper is not None:
        upper = np.asarray(upper)[:len(forecast)]

    future_dates = future_dates[:len(forecast)]

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

    st.markdown("### 📈 Forecast Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "RMSE",
        f"{metrics['RMSE']:.3f}"
    )

    m2.metric(
        "MAE",
        f"{metrics['MAE']:.3f}"
    )

    m3.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%"
    )

    m4.metric(
        "R²",
        f"{metrics['R2']:.3f}"
    )

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

    with st.expander("ℹ️ Getting Started - Complete Guide to Time Series Analysis", expanded=False):
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

    # ============================================================
    # Automatic Datetime Detection
    # ============================================================

    datetime_cols = list(df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns)

    if not datetime_cols:

        detected_date = detect_datetime_column(df)

        if detected_date is not None:
            df[detected_date] = safe_to_datetime(df[detected_date])

            # Remove invalid timestamps
            df = df[df[detected_date].notna()].copy()

            # Sort chronologically
            df = df.sort_values(detected_date)

            # Remove duplicate timestamps
            df = df.drop_duplicates(subset=detected_date)

            # Reset index
            df = df.reset_index(drop=True)

            datetime_cols = [detected_date]

            session_manager.set_data(section, "df", df)

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)

    if datetime_cols and numeric_cols:
        with st.expander("📅 Time Series Setup", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                default_date = detect_datetime_column(df)

                if default_date in datetime_cols:
                    default_idx = datetime_cols.index(default_date)
                else:
                    default_idx = 0

                date_col = st.selectbox(
                    "Select Date Column",
                    datetime_cols,
                    index=default_idx,
                    key="date_col"
                )
            with col2:

                recommended_target, target_scores = detect_target_column(df)

                if recommended_target is not None:

                    st.info(
                        f"""
            ### 🤖 AI Recommendation

            **Recommended Target Column**

            ✅ **{recommended_target}**
            """
                    )

                    with st.expander("🤖 AI Target Analysis"):

                        ranking = (
                            pd.DataFrame(
                                {
                                    "Column": list(target_scores.keys()),
                                    "Score": list(target_scores.values())
                                }
                            )
                            .sort_values("Score", ascending=False)
                            .reset_index(drop=True)
                        )

                        ranking.index += 1

                        st.dataframe(
                            ranking,
                            width='stretch',
                            hide_index=False
                        )

                        st.caption(
                            """
            Recommendation is based on

            • Missing Values

            • Variance

            • Number of Unique Values

            • Continuous Data

            • Column Name Intelligence
            """
                        )

                    default_index = numeric_cols.index(recommended_target)

                else:

                    default_index = 0

                value_col = st.selectbox(
                    "🎯 Select Target Column",
                    numeric_cols,
                    index=default_index,
                    key="value_col"
                )

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