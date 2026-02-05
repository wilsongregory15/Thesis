import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import glob
import os
import joblib
from datetime import timedelta
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

st.set_page_config(
    page_title="Thesis: Macro-Economic Forecasting",
    page_icon="🎓",
    layout="wide"
)

# 📖 DATA DICTIONARY
COVARIATE_DEFINITIONS = {
    "none": "Baseline model using only historical price data (No external factors).",
    "domestic": "🏠 Domestic Set: Includes all local economic indicators (Inflation, GDP, CPI, Unemployment).",
    "global": "🌍 Global Set: Includes all external market indicators (USD, Brent, WTI).",
    "all": "🌐 All Indicators: A comprehensive combination of both Domestic and Global variables.",
    "inflation": "🎈 Inflation: The rate at which the general level of prices is rising.",
    "gdp": "💰 GDP: Gross Domestic Product.",
    "cpi": "🛒 CPI: Consumer Price Index.",
    "unemployment": "👥 Unemployment: The percentage of the labor force that is jobless.",
    "usd": "💵 USD: The exchange rate of the local currency against the US Dollar.",
    "brent": "🛢️ Brent: Brent Crude Oil price (Global benchmark).",
    "wti": "🛢️ WTI: West Texas Intermediate Crude Oil."
}

# 📖 SCENARIO DEFINITIONS
SCENARIO_DETAILS = {
    "Short-Term Sprint": {
        "desc": "⚡ **Short-Term Sprint (30 → 1)**",
        "explanation": "The model takes **30 days** of historical data to predict the next **1 day**. This mimics a day-trading strategy focused on immediate reactions."
    },
    "Medium-Term Pace": {
        "desc": "🏃 **Medium-Term Pace (60 → 5)**",
        "explanation": "The model takes **60 days** of historical data to predict the next **5 days**. This mimics a weekly planning strategy (predicting one full business week)."
    },
    "Long-Term Marathon": {
        "desc": "🔭 **Long-Term Marathon (90 → 10)**",
        "explanation": "The model takes **90 days** of historical data to predict the next **10 days**. This mimics a trend-following strategy (predicting two full business weeks)."
    }
}

# ⚠️ UPDATE PATHS BELOW TO MATCH YOUR ACTUAL FOLDERS
RESULTS_FOLDERS = {
    "TSMixer": BASE_DIR / "results/tsmixer_results",
    "TFT": BASE_DIR / "results/tft_results",
    "RandomForest": BASE_DIR / "results/rf_results"
}

RF_MODEL_PATH = BASE_DIR / "models/rf_models_final"
OOS_DATA_PATH = BASE_DIR / "data/final_data_saved_model"

TARGET_FILE_MAP = {
    "IHSG": "ihsg_final.csv",
    "LQ45": "lq45_final.csv",
    "KOMPAS100": "kompas100_final.csv"
}

SCENARIO_PARAMS = {
    "Short-Term Sprint": {"window": 10, "horizon": 1},
    "Medium-Term Pace": {"window": 60, "horizon": 5},
    "Long-Term Marathon": {"window": 90, "horizon": 10}
}

COVARIATE_COL_MAP = {
    "Domestic":     ["close_inflation", "close_gdp", "close_cpi", "close_unemployment"],
    "Global":       ["close_usd", "close_brent", "close_wti"],
    "All":          ["close_inflation", "close_gdp", "close_cpi", "close_unemployment", "close_usd", "close_brent", "close_wti"],
    "Inflation":    ["close_inflation"],
    "GDP":          ["close_gdp"],
    "CPI":          ["close_cpi"],
    "Unemployment": ["close_unemployment"],
    "USD":          ["close_usd"],
    "Brent":        ["close_brent"],
    "WTI":          ["close_wti"],
    "None":         []
}

@st.cache_data
def load_oos_data(target):
    file = TARGET_FILE_MAP[target]
    df = pd.read_csv(os.path.join(OOS_DATA_PATH, file))
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

def load_rf_model(target, scenario, covariate):
    pattern = f"*{target}*{scenario.replace(' ', '_')}*{covariate}*.joblib"
    matches = glob.glob(os.path.join(RF_MODEL_PATH, pattern))
    return joblib.load(matches[0]) if matches else None

def generate_oos_predictions_darts(
    model_bundle,
    oos_df,
    target_col,
    covariate_cols
):
    # 1. Unpack bundle
    model = model_bundle["model"]
    scaler = model_bundle["target_scaler"]
    differencer = model_bundle["differencer"]
    cov_scaler = model_bundle["covariate_scaler"]
    window = model_bundle["window"]
    horizon = model_bundle["horizon"]

    # 2. Prepare full dataset starting from Jan 1, 2025
    oos_start = pd.Timestamp("2025-01-01")
    oos_full = oos_df[oos_df['date'] >= oos_start].sort_values('date').reset_index(drop=True)
    
    total_days = len(oos_full)
    
    # We need at least (window + 1 + 1) days for first prediction
    # +1 for differencing, +1 for first forecast step
    min_required = window + 2
    
    if total_days < min_required:
        st.warning(f"Not enough OOS data. Need at least {min_required} days, have {total_days}")
        return pd.DataFrame()
    
    # 3. Storage for predictions (1-step-ahead only)
    all_predictions = []
    
    # 4. Rolling window loop - slide by 1 day each time
    # We stop when we can't form a complete window anymore
    max_start_idx = total_days - (window + 1 + 1)  # +1 diff, +1 forecast
    
    for start_idx in range(max_start_idx + 1):
        # Define window for this iteration
        end_input_idx = start_idx + window + 1  # +1 for differencing
        
        # Slice target (actual data only)
        target_slice = oos_full.iloc[start_idx:end_input_idx][["date", target_col]].copy()
        
        # For covariates, we need to extend by `horizon` to satisfy Darts requirements
        # But we'll only use the first prediction
        end_cov_idx = min(end_input_idx + horizon, total_days)
        cov_slice = oos_full.iloc[start_idx:end_cov_idx][["date"] + covariate_cols].copy() if covariate_cols else None
        
        # Build TimeSeries for target
        target_ts = TimeSeries.from_dataframe(
            target_slice,
            time_col="date",
            value_cols=target_col,
            fill_missing_dates=True,
            freq="B"
        )
        target_ts = fill_missing_values(target_ts)
        
        # Build TimeSeries for covariates
        cov_ts = None
        if covariate_cols and cov_slice is not None:
            cov_ts = TimeSeries.from_dataframe(
                cov_slice,
                time_col="date",
                value_cols=covariate_cols,
                fill_missing_dates=True,
                freq="B"
            )
            cov_ts = fill_missing_values(cov_ts)
            cov_ts = cov_scaler.transform(cov_ts)
        
        # Apply transformations
        ts_log = target_ts.map(np.log)
        ts_log_diff = differencer.transform(ts_log)
        ts_scaled = scaler.transform(ts_log_diff)
        
        # Predict (still predict full horizon, but we'll only take step 1)
        try:
            forecast_scaled = model.predict(
                n=horizon,
                series=ts_scaled,
                past_covariates=cov_ts
            )
            
            # Inverse transform
            forecast_diff_log = scaler.inverse_transform(forecast_scaled)
            log_returns = forecast_diff_log.values().flatten()
            
            last_log_price = np.log(target_slice[target_col].iloc[-1])
            log_prices = last_log_price + np.cumsum(log_returns)
            prices = np.exp(log_prices)
            
            # Only take the FIRST prediction (1-step-ahead)
            all_predictions.append({
                "Date": forecast_scaled.time_index[0],
                "Predicted": prices[0]
            })
            
        except Exception as e:
            # If prediction fails for this window, skip it
            continue
    
    return pd.DataFrame(all_predictions)

# ==========================================
# 2. DATA LOADING ENGINE (ROBUST FILENAME PARSING)
# ==========================================
@st.cache_data
def load_all_results(folders_dict):
    df_list = []
    for model_name, folder_path in folders_dict.items():
        if not folder_path.exists():
            continue
            
        all_files = list(folder_path.glob("*.csv"))
        
        for filename in all_files:
            try:
                temp_df = pd.read_csv(filename)
                temp_df['Model'] = model_name
                
                # Extract parts from filename for backup
                # Example: tft_pred_LQ45_Short-Term_Sprint_Global.csv
                base_name = os.path.basename(filename).replace('.csv', '')
                parts = base_name.split('_')

                # 1. Handle Covariates
                if 'covariate_set' in temp_df.columns:
                    temp_df['Covariates'] = temp_df['covariate_set'].fillna('None')
                else:
                    # Fallback: Last part of filename (e.g., 'Global')
                    temp_df['Covariates'] = parts[-1] if len(parts) > 1 else 'Unknown'

                # 2. Handle Target
                if 'target' in temp_df.columns:
                    temp_df['Target'] = temp_df['target'].astype(str).str.upper()
                else:
                    # Fallback: 3rd part of filename (e.g., 'LQ45')
                    if len(parts) >= 3:
                        temp_df['Target'] = parts[2].upper()
                    else:
                        temp_df['Target'] = 'UNKNOWN'

                # 3. Handle Scenario
                if 'scenario' in temp_df.columns:
                    temp_df['Scenario'] = temp_df['scenario']
                else:
                    # Fallback: Parts between Target and Covariate
                    # e.g., 'Short-Term', 'Sprint' -> "Short-Term Sprint"
                    if len(parts) >= 5:
                        scenario_parts = parts[3:-1]
                        temp_df['Scenario'] = " ".join(scenario_parts)
                    else:
                        temp_df['Scenario'] = 'UNKNOWN'

                # 4. Standardize Date/Actual/Predicted
                cols_map = {c.lower(): c for c in temp_df.columns}
                if 'date' in cols_map: temp_df['Date'] = pd.to_datetime(temp_df[cols_map['date']])
                elif 'datetime' in cols_map: temp_df['Date'] = pd.to_datetime(temp_df[cols_map['datetime']])
                
                temp_df['Actual'] = temp_df[cols_map.get('actual', 'actual')]
                temp_df['Predicted'] = temp_df[cols_map.get('predicted', 'predicted')]

                if 'train_flag' in temp_df.columns:
                    temp_df['Stage'] = temp_df['train_flag'].map({1: 'Train', 0: 'Test'})
                else:
                    temp_df['Stage'] = 'Test'

                cols = ['Date', 'Model', 'Target', 'Scenario', 'Covariates', 'Stage', 'Actual', 'Predicted']
                df_list.append(temp_df[cols])
                
            except Exception as e:
                # print(f"⚠️ Skipped {filename}: {e}") # Optional: Uncomment to debug specific files
                continue
                
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

try:
    df = load_all_results(RESULTS_FOLDERS)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ No data found. Please check your folder paths.")
    st.stop()

# ==========================================
# 3. THE STORYLINE: INTRODUCTION
# ==========================================
st.title("MacroX: Stock Market Forecasting Dashboard")

st.markdown("""
### Research Abstract
Most financial models rely on **past prices** to predict **future prices**. This research investigates the predictive power of **exogenous macro-economic variables** (such as Inflation, GDP, and Interest Rates) when applied to **Indonesian financial indices (such as IHSG and LQ45)**.

This dashboard utilizes **state-of-the-art (SOTA)** time series forecasting models (including TSMixer and TFT) to evaluate whether these external signals improve prediction accuracy compared to a univariate baseline.
""")

with st.expander("🔍 Guide: How to use this dashboard"):
    st.markdown("""
    ### 📋 Dashboard Workflow
    **1. Configure the Experiment (Sidebar)**
    * **Select Index:** Choose the financial market you are analyzing (e.g., IHSG).
    * **Select Horizon:** Define the prediction task.
        * ⚡ **Short-Term Sprint:** Predicts **1 Day** ahead using 30 days of historical data.
        * 🏃 **Medium-Term Pace:** Predicts **5 Days** (1 Week) ahead using 60 days of historical data.
        * 🔭 **Long-Term Marathon:** Predicts **10 Days** (2 Weeks) ahead using 90 days of historical data.
    * **Select Model:** Choose the forecasting architecture (e.g., TSMixer).

    **2. Analyze Performance (Section 1)**
    * Check the **Leaderboard** to see which economic variables reduced the error.
    * **Green Bars:** The model became **more accurate** (Lower Error).
    * **Red Bars:** The model became **less accurate** (Higher Error).

    **3. Inspect Forecasts (Section 2)**
    * Select a specific variable (e.g., 'Inflation') from the dropdown.
    * Compare the **Green Line** (Enhanced Model) against the **Red Dotted Line** (Baseline) to visually validate the improvement.
    """)

st.divider()

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Configuration")

target_options = df['Target'].unique()
selected_target = st.sidebar.selectbox(
    "1. Select Financial Index", 
    target_options, 
    index=None, 
    placeholder="Choose Index..."
)

if selected_target:
    scenario_options = df[df['Target'] == selected_target]['Scenario'].unique()
    selected_scenario = st.sidebar.selectbox(
        "2. Select Forecast Horizon", 
        scenario_options, 
        index=None, 
        placeholder="Choose Horizon..."
    )
    
    if selected_scenario in SCENARIO_DETAILS:
        details = SCENARIO_DETAILS[selected_scenario]
        st.sidebar.markdown(f"{details['desc']}")
        st.sidebar.caption(f"{details['explanation']}")
        st.sidebar.divider()
        
else:
    selected_scenario = st.sidebar.selectbox("2. Select Forecast Horizon", [], disabled=True)

if selected_target and selected_scenario:
    model_options = df[(df['Target'] == selected_target) & (df['Scenario'] == selected_scenario)]['Model'].unique()

    model_options_list = list(model_options)

    default_model_index = (
        model_options_list.index("RandomForest")
        if "RandomForest" in model_options_list
        else None
    )

    selected_model = st.sidebar.selectbox(
        "3. Select Forecasting Model (?)",
        model_options_list,
        index=default_model_index,
        placeholder="Choose Model...",
        help="Random Forest achieved the lowest average MAPE across all scenarios and is selected as the best-performing model."
    )
    # selected_model = st.sidebar.selectbox(
    #     "3. Select Forecasting Model", 
    #     model_options, 
    #     index=None, 
    #     placeholder="Choose Model..."
    # )
else:
    selected_model = st.sidebar.selectbox("3. Select Forecasting Model", [], disabled=True)

if not (selected_target and selected_scenario and selected_model):
    st.info("👈 **Please select a Financial Index, Forecast Horizon, and Model from the sidebar to begin the analysis.**")
    st.stop()

st.sidebar.info(f"Analysis Scope: **{selected_model}** predicting **{selected_target}**.")

# ==========================================
# 5. DATA PROCESSING
# ==========================================
subset_df = df[(df['Target'] == selected_target) & (df['Scenario'] == selected_scenario) & (df['Model'] == selected_model)]
baseline_df = subset_df[subset_df['Covariates'] == 'None']

# Calculate Performance
performance_df = (
    subset_df[subset_df['Stage'] == 'Test']
    .groupby('Covariates')[['Actual', 'Predicted']]
    .apply(lambda x: np.mean(np.abs((x['Actual'] - x['Predicted']) / x['Actual'])) * 100)
    .reset_index(name='MAPE')
    .sort_values('MAPE')
)

try:
    baseline_mape = performance_df[performance_df['Covariates'] == 'None']['MAPE'].values[0]
except:
    baseline_mape = 0

performance_df['Improvement'] = baseline_mape - performance_df['MAPE']
performance_df['Color'] = performance_df['Improvement'].apply(lambda x: '#21c354' if x > 0 else '#ff4b4b')

# ==========================================
# 6. DASHBOARD LAYOUT
# ==========================================

# --- SECTION 1: COMPARATIVE ANALYSIS ---
st.header("1. Comparative Performance Analysis")
st.write("Evaluation of error rates (MAPE) across different covariate sets compared to the baseline.")

col_left, col_right = st.columns([1.5, 2]) 

with col_left:
    st.subheader("Experiment Results")
    
    display_df = performance_df[['Covariates', 'MAPE', 'Improvement']].copy()
    
    glossary_text = "📚 **Variable Definitions:**\n\n" + "\n\n".join([f"• {v}" for k,v in COVARIATE_DEFINITIONS.items()])

    st.dataframe(
        display_df[['Covariates', 'MAPE', 'Improvement']].style.format({'MAPE': '{:.5f}%', 'Improvement': '{:.5f}%'}),
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Covariates": st.column_config.TextColumn(
                "Covariate Set (?)", 
                width="medium",
                help=glossary_text 
            ),
            "MAPE": st.column_config.NumberColumn("Error (MAPE)", format="%.5f%%"),
            "Improvement": st.column_config.NumberColumn("Vs. Baseline", format="%.5f%%")
        }
    )
    st.caption("Hover over the **Covariate Set (?)** header to view variable definitions.")

with col_right:
    fig_bar = px.bar(
        performance_df, x='Covariates', y='MAPE', color='Color',
        color_discrete_map="identity", title="Model Error Rate (Lower is Better)",
        labels={'MAPE': 'MAPE (%)'}, 
        text_auto='.5f'
    )
    
    fig_bar.update_traces(
        texttemplate='%{y:.5f}%', 
        textposition='inside', 
        textangle=0
    )
    
    fig_bar.add_hline(
        y=baseline_mape, 
        line_dash="dot", 
        line_color="black",
        annotation_text="Baseline (Univariate)"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.caption("**Legend:** 🟢 **Green** = Performance Improvement | 🔴 **Red** = Performance Degradation")

# --- SECTION 2: FORECAST VISUALIZATION ---
st.divider()
st.header("2. Forecast Visualization & Validation")
st.write("Temporal comparison of the predicted values against actual market prices.")

available_covariates = [c for c in performance_df['Covariates'].unique() if c != 'None']

if not available_covariates:
    st.warning("Only Baseline data available for this selection.")
else:
    compare_covariate = st.selectbox(
        "Select Covariate Set for Visual Inspection", 
        available_covariates,
        index=None,
        placeholder="Choose a dataset to visualize..."
    )
    
    if compare_covariate:
        definition = COVARIATE_DEFINITIONS.get(
            compare_covariate.lower(), 
            COVARIATE_DEFINITIONS.get(compare_covariate, "No definition available.")
        )
        
        st.info(f"ℹ️ **Variable Description:** \n\n {definition}")

        enhanced_df = subset_df[subset_df['Covariates'] == compare_covariate]

        train_tail = baseline_df[baseline_df['Stage'] == 'Train'].sort_values('Date').tail(60)
        test_base = baseline_df[baseline_df['Stage'] == 'Test'].sort_values('Date')
        test_enh = enhanced_df[enhanced_df['Stage'] == 'Test'].sort_values('Date')

        fig_ts = go.Figure()
        
        actual_line = pd.concat([train_tail, test_base])
        fig_ts.add_trace(go.Scatter(
            x=actual_line['Date'], y=actual_line['Actual'], 
            mode='lines', name='Actual Market Price', 
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=test_enh['Date'], y=test_enh['Predicted'], 
            mode='lines', name=f'Model w/ {compare_covariate}', 
            line=dict(color='#21c354', width=3)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=test_base['Date'], y=test_base['Predicted'], 
            mode='lines', name='Baseline Model', 
            line=dict(color='#ff4b4b', dash='dot', width=2)
        ))

        oos_df = load_oos_data(selected_target)

        target_col = f"close_{selected_target.lower()}"
        feature_cols = [c for c in oos_df.columns if c != "date"]

        model_bundle = load_rf_model(
            selected_target,
            selected_scenario,
            compare_covariate
        )

        if model_bundle is not None:
            covariate_cols = COVARIATE_COL_MAP.get(compare_covariate, [])
            covariate_cols = [c for c in covariate_cols if c in oos_df.columns]

            oos_pred_df = generate_oos_predictions_darts(
                model_bundle,
                oos_df,
                target_col,
                covariate_cols
            )

            # Also get the input window (what the model saw)
            window = model_bundle["window"]
            oos_start = pd.Timestamp("2025-01-01")
            oos_dates = oos_df[oos_df['date'] >= oos_start].sort_values('date')
            input_window = oos_dates.head(window + 1)  # +1 for differencing

            # Plot the input window first (the actual data the model sees)
            fig_ts.add_trace(go.Scatter(
                x=input_window['date'],
                y=input_window[target_col],
                mode='lines',
                name='OOS Input Window (2025)',
                line=dict(color='#9467bd', width=2, dash='dot'),
                showlegend=True
            ))

            # Then plot the forecast
            fig_ts.add_trace(go.Scatter(
                x=oos_pred_df['Date'],
                y=oos_pred_df['Predicted'],
                mode='lines',
                name='OOS Forecast (2025)',
                line=dict(color='#f5a623', width=3)
            ))

            # Move the vertical line to the END of the input window (where forecast starts)
            forecast_start = input_window['date'].max().timestamp() * 1000
            fig_ts.add_vline(
                x=forecast_start,
                line_width=2,
                line_dash="dashdot",
                line_color="purple",
                annotation_text="OOS Forecast Start",
                annotation_position="bottom right"
            )

            oos_start = oos_pred_df['Date'].min().timestamp() * 1000
            fig_ts.add_vline(
                x=oos_start,
                line_width=2,
                line_dash="dashdot",
                line_color="purple",
                annotation_text="OOS Testing Data (2025)",
                annotation_position="top right"
            )

        if not test_base.empty:
            split_date_numeric = test_base['Date'].min().timestamp() * 1000
            
            fig_ts.add_vline(
                x=split_date_numeric, 
                line_width=1, 
                line_dash="dash", 
                line_color="red", 
                annotation_text="Forecast Horizon Start", 
                annotation_position="top right"
            )

        fig_ts.update_layout(height=500, title=f"Time Series Comparison: Baseline vs {compare_covariate}", hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Statistical Interpretation")
        
        m_base = baseline_mape
        m_enh = performance_df[performance_df['Covariates'] == compare_covariate]['MAPE'].values[0]
        imp = m_base - m_enh
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Baseline MAPE", f"{m_base:.5f}%")
        kpi2.metric(f"Enhanced MAPE", f"{m_enh:.5f}%", delta=f"{imp:.5f}% Improvement" if imp > 0 else f"{abs(imp):.5f}% Degradation")
        
        status = "✅ Supported" if imp > 0 else "❌ Rejected"
        kpi3.metric("Hypothesis Status", status)

        with st.container(border=True):
            if imp > 0:
                st.success(f"**Observation:** Incorporating **{compare_covariate}** reduced the forecasting error by **{imp:.5f}%**. This suggests that these macro-economic variables contain valid predictive signals for this specific index and timeframe.")
            else:
                st.error(f"**Observation:** Incorporating **{compare_covariate}** increased the forecasting error by **{abs(imp):.5f}%**. This indicates that these variables may have introduced noise or were not relevant for this specific forecast horizon.")
    else:
        st.info("👆 **Please select a covariate set above to see the visualization.**")

st.divider()
st.caption("Thesis Research Dashboard | Macro-Economic Analysis Module | 2024-2025")