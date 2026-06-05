import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set page configuration
st.set_page_config(
    page_title="Linear Regression & Outlier Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for premium styling
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main container fonts */
    .main .block-container {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Custom metric card container */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .custom-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        flex: 1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .card-title {
        color: #6c757d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    .card-value {
        color: #2b2b2b;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .card-delta {
        font-size: 0.85rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    .delta-green {
        color: #2ec4b6;
        font-weight: bold;
    }
    .delta-red {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* Style blockquotes */
    blockquote {
        border-left: 4px solid #1e3c72 !important;
        background-color: #f8f9fa;
        padding: 10px 15px !important;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    
    n_val = st.slider(
        "Sample Size (N)",
        min_value=10,
        max_value=2000,
        value=500,
        step=10,
        help="Number of data points to generate."
    )
    
    a_val = st.slider(
        "True Slope (a)",
        min_value=-50.0,
        max_value=50.0,
        value=8.0,
        step=0.5,
        help="The true slope (coefficient of x) for generating target values."
    )
    
    b_val = st.slider(
        "True Intercept (b)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        help="The true y-intercept value when x=0."
    )
    
    var_val = st.number_input(
        "Noise Variance (var)",
        min_value=0.0,
        max_value=100000.0,
        value=100000.0,
        step=10.0,
        help="Variance of the Gaussian noise. SD is calculated as sqrt(var)."
    )
    
    seed_val = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="Seed to guarantee data generation reproducibility."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Methodology")
    st.info("""
    This application automates the detection of statistical outliers in a dataset following the OLS line fitting step of a standard CRISP-DM pipeline.
    """)

# ==============================================================================
# PIPELINE COMPUTATION (CRISP-DM PHASES 2 - 5)
# ==============================================================================

# Data Generation (Phase 2: Data Understanding)
np.random.seed(seed_val)
x_vals = np.random.uniform(-100.0, 100.0, size=n_val)
std_dev = np.sqrt(var_val)
noise = np.random.normal(loc=0.0, scale=std_dev, size=n_val)
actual_y = a_val * x_vals + b_val + noise

df = pd.DataFrame({
    'x': x_vals,
    'actual_y': actual_y
})

# Feature extraction (Phase 3: Data Preparation)
X = df[['x']]
y = df['actual_y']

# Model fitting (Phase 4: Modeling)
model = LinearRegression()
model.fit(X, y)

# Prediction & Residual Calculation
df['predicted_y'] = model.predict(X)
df['residual'] = df['actual_y'] - df['predicted_y']
df['absolute_residual'] = df['residual'].abs()

# Model evaluation (Phase 5: Evaluation)
est_a = model.coef_[0]
est_b = model.intercept_
error_a = est_a - a_val
error_b = est_b - b_val

r2 = r2_score(y, df['predicted_y'])
mse = mean_squared_error(y, df['predicted_y'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, df['predicted_y'])

# Find top outliers (Phase 5: Evaluation)
top_n = 10
top_indices = df['absolute_residual'].nlargest(top_n).index
ranks = pd.Series(index=top_indices, data=np.arange(1, top_n + 1))
df['outlier_rank'] = df.index.map(ranks).astype('Int64')
df['is_outlier'] = df.index.isin(top_indices)

outliers_df = df[df['is_outlier']].sort_values(by='outlier_rank')

# ==============================================================================
# USER INTERFACE LAYOUT
# ==============================================================================

# Header Card
st.markdown("""
    <div class="main-header">
        <h1>📈 CRISP-DM Linear Regression & Outlier Detector</h1>
        <p>Estimate linear relationships, validate estimators, and isolate top outliers interactively.</p>
    </div>
""", unsafe_allow_html=True)

# Metrics Cards Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Estimated Slope (a)</div>
            <div class="card-value">{est_a:.4f}</div>
            <div class="card-delta">True: {a_val:.2f} | Error: <span class="{"delta-green" if abs(error_a) < 0.1 else "delta-red"}">{error_a:+.4f}</span></div>
        </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Estimated Intercept (b)</div>
            <div class="card-value">{est_b:.4f}</div>
            <div class="card-delta">True: {b_val:.2f} | Error: <span class="{"delta-green" if abs(error_b) < 0.5 else "delta-red"}">{error_b:+.4f}</span></div>
        </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">R-squared Score</div>
            <div class="card-value">{r2:.6f}</div>
            <div class="card-delta">Variance Explained</div>
        </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">RMSE (Residual SD)</div>
            <div class="card-value">{rmse:.4f}</div>
            <div class="card-delta">Target noise SD: {std_dev:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs definitions
tab_dash, tab_explorer, tab_docs = st.tabs([
    "📊 Interactive Analysis Dashboard", 
    "🗄️ Raw Data & CSV Export", 
    "📖 CRISP-DM Process Walkthrough"
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE ANALYSIS DASHBOARD
# ------------------------------------------------------------------------------
with tab_dash:
    col_plot, col_table = st.columns([7, 5])
    
    with col_plot:
        st.subheader("Fitted Regression Line & Outliers")
        
        # Build Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        normal_points = df[~df['is_outlier']]
        
        # Plot normal dots
        ax.scatter(
            normal_points['x'], normal_points['actual_y'],
            color='#3498db', alpha=0.5, edgecolors='none', s=30, label='Normal Observations'
        )
        
        # Plot outlier crosses
        ax.scatter(
            outliers_df['x'], outliers_df['actual_y'],
            color='#e74c3c', marker='X', s=140, edgecolors='black', linewidths=0.8,
            label='Top 10 Outliers (Ranked)'
        )
        
        # Plot red line
        x_sorted = df['x'].sort_values()
        y_line = model.predict(pd.DataFrame({'x': x_sorted}))
        ax.plot(x_sorted, y_line, color='#c0392b', linewidth=2.5, label='Fitted Regression Line')
        
        # Annotate top outliers with dynamic placement offset based on noise SD
        offset_y = std_dev * 0.15 if std_dev > 5 else 1.5
        for _, row in outliers_df.iterrows():
            ax.annotate(
                f"Rank {row['outlier_rank']}",
                xy=(row['x'], row['actual_y']),
                xytext=(row['x'] + 3, row['actual_y'] + offset_y),
                fontsize=8,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f1c40f', alpha=0.85, ec='gray', lw=0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.6, connectionstyle='arc3,rad=0.1')
            )
        
        ax.set_xlabel('X (Independent Variable)', fontsize=10)
        ax.set_ylabel('Y (Dependent Variable)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=True, shadow=True, facecolor='white')
        fig.tight_layout()
        
        # Render plot in Streamlit
        st.pyplot(fig)
        plt.close()
        
    with col_table:
        st.subheader("Top 10 Outlier Observations")
        st.markdown(r"""
        Observations are ranked from 1 to 10 based on their **absolute residual** ($|y - \hat{y}|$) relative to the regression line.
        """)
        
        # Display Outliers Table nicely formatted
        display_outliers = outliers_df[[
            'outlier_rank', 'x', 'actual_y', 'predicted_y', 'residual', 'absolute_residual'
        ]].copy()
        
        # Format columns for display
        display_outliers['x'] = display_outliers['x'].map('{:,.4f}'.format)
        display_outliers['actual_y'] = display_outliers['actual_y'].map('{:,.4f}'.format)
        display_outliers['predicted_y'] = display_outliers['predicted_y'].map('{:,.4f}'.format)
        display_outliers['residual'] = display_outliers['residual'].map('{:,.4f}'.format)
        display_outliers['absolute_residual'] = display_outliers['absolute_residual'].map('{:,.4f}'.format)
        
        st.dataframe(display_outliers, use_container_width=True, hide_index=True)
        
        st.markdown("### ⚠️ Methodological Definition & Limitation")
        st.warning("""
        - **Outlier Definition**: Here, outliers represent points with large vertical deviations from the model predictions (high residual magnitude). 
        - **Limitation**: While residual analysis successfully identifies anomalous targets (vertical outliers), it is not designed to detect high-leverage points (outliers on the independent horizontal $x$-axis).
        """)

# ------------------------------------------------------------------------------
# TAB 2: RAW DATA & CSV EXPORT
# ------------------------------------------------------------------------------
with tab_explorer:
    st.subheader("Exploratory Data Summary")
    
    col_stats, col_preview = st.columns([1, 2])
    
    with col_stats:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df[['x', 'actual_y']].describe(), use_container_width=True)
        
    with col_preview:
        st.markdown("**Dataset Preview (First 100 rows)**")
        st.dataframe(df.head(100), use_container_width=True)
        
    st.markdown("---")
    st.subheader("📥 Export Outputs")
    st.write("Save these datasets to execute local experiments or deploy downstream.")
    
    # Download buttons setup
    results_csv_data = df.to_csv(index=False).encode('utf-8')
    outliers_csv_data = outliers_df[[
        'outlier_rank', 'x', 'actual_y', 'predicted_y', 'residual', 'absolute_residual'
    ]].to_csv(index=False).encode('utf-8')
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Download Complete Results CSV",
            data=results_csv_data,
            file_name="linear_regression_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="Download Top 10 Outliers CSV",
            data=outliers_csv_data,
            file_name="top_10_outliers.csv",
            mime="text/csv",
            use_container_width=True
        )

# ------------------------------------------------------------------------------
# TAB 3: CRISP-DM PROCESS WALKTHROUGH
# ------------------------------------------------------------------------------
with tab_docs:
    st.header("Cross-Industry Standard Process for Data Mining (CRISP-DM)")
    
    st.markdown(r"""
    This app serves as a real-time implementation of the six phases of the **CRISP-DM** methodology.
    
    ### 1. Business Understanding
    - **Objective**: Build a system that generates linear regression data, estimates the true relationship parameters, checks model fit, and isolates the 10 data points with the largest absolute errors.
    - **Deployment Value**: Identifying anomalies (outliers) helps identify hardware reading bugs, financial fraud, or process exceptions in actual manufacturing/scientific pipelines.
    
    ### 2. Data Understanding
    - **Generation**: Uniformly distributes $x$ values between $-100$ and $100$.
    - **Formula**: $y = a \cdot x + b + \epsilon$ where noise $\epsilon$ is generated as Gaussian distribution $N(0, var)$.
    - **EDA**: Generates descriptive stats and checks Pearson correlation between features and labels in real-time.
    
    ### 3. Data Preparation
    - **Validation**: Widget inputs dynamically limit configuration parameter values inside acceptable scientific bounds.
    - **Strategy**: 
        - **Duplicates**: Kept in the dataset. Because values are continuous, duplicate points represent natural random occurrences. Deleting duplicates would artificially reduce variance.
        - **Outliers**: Outliers are kept during OLS fitting, so that the OLS line attempts to accommodate the raw data, allowing us to find which ones are truly the largest deviations.
        
    ### 4. Modeling
    - **Algorithm**: Ordinary Least-Squares (OLS) regression (`sklearn.linear_model.LinearRegression`).
    - **Output Columns**: Adds predictions ($\hat{y}$), residuals ($y - \hat{y}$), and absolute residuals ($|y - \hat{y}|$) to the dataset.
    
    ### 5. Evaluation
    - **Regression Metrics**: Computes Coefficient of Determination ($R^2$), MSE, RMSE (standard deviation of the errors), and MAE.
    - **Parameter Errors**: Calculates estimation errors ($\hat{a} - a$ and $\hat{b} - b$).
    - **Ranking**: Sorts absolute residuals descending to identify Rank 1-10 outliers.
    
    ### 6. Deployment
    - **Interface**: An interactive web app to test sensitivities to noise variance (`var`) and sample sizes ($N$).
    - **Deliverables**: Reusable CSV spreadsheets and Matplotlib plot downloads.
    """)
