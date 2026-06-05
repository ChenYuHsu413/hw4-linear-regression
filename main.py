import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

# Import modular CRISP-DM functions from CLI script
from regression_analysis import (
    validate_parameters,
    generate_data,
    prepare_data,
    fit_regression,
    evaluate_model,
    find_top_outliers
)

# Initialize FastAPI App
app = FastAPI(
    title="Linear Regression & Outlier Detection API",
    description="A FastAPI backend service for synthetic linear regression modeling and outlier detection under the CRISP-DM methodology.",
    version="1.0.0"
)

# ==============================================================================
# ENDPOINT 1: Landing Dashboard HTML Page
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    """
    Renders the premium HTML landing dashboard loaded from index.html.
    """
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)




# ==============================================================================
# ENDPOINT 2: Model Analysis JSON Results
# ==============================================================================
@app.get("/analyze")
def analyze(
    n: int = Query(500, description="Sample size (dataset size)", ge=10),
    a: float = Query(8.0, description="True slope", ge=-50.0, le=50.0),
    b: float = Query(40.0, description="True intercept", ge=0.0, le=100.0),
    var: float = Query(200.0, description="Gaussian noise variance", ge=0.0, le=100000.0),
    seed: int = Query(42, description="Random seed", ge=0),
    top_n: int = Query(10, description="Number of outliers to rank", ge=1)
):
    """
    Validates parameters, runs OLS linear regression modeling, calculates metrics,
    and returns top N ranked outliers in JSON format.
    """
    try:
        # Phase 3: Data Preparation (Validation)
        validate_parameters(n, a, b, var, seed)
        
        # Phase 2: Data Understanding (Generation)
        df_raw = generate_data(n, a, b, var, seed)
        
        # Phase 3: Data Preparation (Splitting)
        X, y = prepare_data(df_raw)
        
        # Phase 4: Modeling (Fitting)
        model, predicted_y, df_modeled = fit_regression(X, y, df_raw)
        
        # Phase 5: Evaluation
        metrics = evaluate_model(model, df_modeled, a, b)
        outliers_df, df_final = find_top_outliers(df_modeled, top_n=top_n)
        
        # Format outliers for JSON output
        outliers_list = outliers_df[[
            'outlier_rank', 'x', 'actual_y', 'predicted_y', 'residual', 'absolute_residual'
        ]].to_dict(orient="records")
        
        # Convert descriptive statistics summary to dict
        summary_stats = df_final[['x', 'actual_y']].describe().to_dict()
        
        return {
            "status": "success",
            "parameters": {
                "n_samples": n,
                "true_slope": a,
                "true_intercept": b,
                "variance": var,
                "random_seed": seed
            },
            "regression_metrics": metrics,
            "descriptive_statistics": summary_stats,
            f"top_{top_n}_outliers": outliers_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# ENDPOINT 3: Dynamic PNG Plot Streaming
# ==============================================================================
@app.get("/plot")
def plot_chart(
    n: int = Query(500, ge=10),
    a: float = Query(8.0, ge=-50.0, le=50.0),
    b: float = Query(40.0, ge=0.0, le=100.0),
    var: float = Query(200.0, ge=0.0, le=100000.0),
    seed: int = Query(42, ge=0),
    top_n: int = Query(10, ge=1)
):
    """
    Renders and streams a high-resolution PNG plot of the linear model and identified outliers.
    """
    try:
        # Validate and pipeline execution
        validate_parameters(n, a, b, var, seed)
        df_raw = generate_data(n, a, b, var, seed)
        X, y = prepare_data(df_raw)
        model, predicted_y, df_modeled = fit_regression(X, y, df_raw)
        outliers_df, df_final = find_top_outliers(df_modeled, top_n=top_n)
        
        # Draw Matplotlib figure in a non-GUI background mode
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        
        normal_points = df_final[~df_final['is_outlier']]
        
        # Plot normal points
        ax.scatter(
            normal_points['x'], normal_points['actual_y'],
            color='#3498db', alpha=0.5, edgecolors='none', s=25, label='Normal Observations'
        )
        
        # Plot outlier points
        ax.scatter(
            outliers_df['x'], outliers_df['actual_y'],
            color='#e74c3c', marker='X', s=130, edgecolors='black', linewidths=0.8,
            label=f'Top {top_n} Outliers'
        )
        
        # Sort and plot OLS model line
        x_sorted = df_final['x'].sort_values()
        y_line = model.predict(pd.DataFrame({'x': x_sorted}))
        ax.plot(x_sorted, y_line, color='#c0392b', linewidth=2.5, label='Fitted Regression Line')
        
        # Annotate outliers with ranking labels
        std_dev = np.sqrt(var)
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
            
        ax.set_title(f'FastAPI Live Plot: OLS Model & Top {top_n} Outliers (n={n}, var={var:.1f})', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X (Independent Variable)', fontsize=10)
        ax.set_ylabel('Y (Dependent Variable)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=True, shadow=True, facecolor='white')
        fig.tight_layout()
        
        # Write plot image to in-memory binary stream
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)
        
        # Stream the image file response
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
