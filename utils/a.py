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