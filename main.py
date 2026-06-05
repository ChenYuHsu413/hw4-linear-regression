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
    Renders a premium HTML landing dashboard listing available microservice endpoints.
    """
    html_content = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CRISP-DM Regression Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Outfit', sans-serif;
                background-color: #f4f6f9;
                color: #2b2b2b;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 1.5rem 2rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .header h1 {
                margin: 0;
                font-size: 1.8rem;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            .header p {
                margin: 0.2rem 0 0 0;
                font-size: 1rem;
                opacity: 0.9;
            }
            .api-link {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.9rem;
                transition: background-color 0.2s;
            }
            .api-link:hover {
                background-color: rgba(255, 255, 255, 0.35);
            }
            .layout-container {
                display: flex;
                flex: 1;
                max-width: 1600px;
                margin: 1.5rem auto;
                width: 95%;
                gap: 1.5rem;
            }
            .sidebar {
                flex: 0 0 320px;
                background-color: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.04);
                height: fit-content;
            }
            .sidebar h2 {
                font-size: 1.2rem;
                color: #1e3c72;
                margin-top: 0;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 0.5rem;
            }
            .control-group {
                margin-bottom: 1.2rem;
            }
            .control-label {
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                font-weight: 600;
                color: #6c757d;
                margin-bottom: 0.4rem;
            }
            .control-input {
                width: 100%;
                box-sizing: border-box;
            }
            .main-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            .metrics-row {
                display: flex;
                gap: 1rem;
                width: 100%;
            }
            .metric-card {
                background-color: white;
                border-radius: 10px;
                padding: 1.2rem;
                flex: 1;
                box-shadow: 0 4px 10px rgba(0,0,0,0.03);
                border: 1px solid #e9ecef;
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .card-title {
                color: #6c757d;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
                letter-spacing: 0.5px;
            }
            .card-value {
                font-size: 1.6rem;
                font-weight: 700;
                color: #2b2b2b;
            }
            .card-subtext {
                font-size: 0.8rem;
                color: #6c757d;
                margin-top: 0.25rem;
            }
            .card-error-green { color: #2ec4b6; font-weight: bold; }
            .card-error-red { color: #e74c3c; font-weight: bold; }
            
            .dashboard-body {
                display: flex;
                gap: 1.5rem;
            }
            .plot-container {
                flex: 7;
                background-color: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.04);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            .plot-container h3 {
                margin-top: 0;
                color: #1e3c72;
                align-self: flex-start;
            }
            .plot-container img {
                width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            .table-container {
                flex: 5;
                background-color: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.04);
                display: flex;
                flex-direction: column;
            }
            .table-container h3 {
                margin-top: 0;
                color: #1e3c72;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
                margin-top: 0.5rem;
            }
            th, td {
                padding: 0.6rem 0.8rem;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }
            th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            tr:hover {
                background-color: #f1f3f5;
            }
            .rank-badge {
                background-color: #f1c40f;
                color: black;
                font-weight: 700;
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
                font-size: 0.8rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>📈 CRISP-DM Regression & Outlier SPA Dashboard</h1>
                <p>FastAPI backend real-time calculations and dynamic data rendering</p>
            </div>
            <a href="/docs" class="api-link" target="_blank">📖 Open API Docs</a>
        </div>
        
        <div class="layout-container">
            <!-- SIDEBAR -->
            <div class="sidebar">
                <h2>⚙️ Configuration</h2>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>Sample Size (n)</span>
                        <span id="n-val">500</span>
                    </div>
                    <input type="range" id="input-n" class="control-input" min="10" max="2000" step="10" value="500">
                </div>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>True Slope (a)</span>
                        <span id="a-val">8.0</span>
                    </div>
                    <input type="range" id="input-a" class="control-input" min="-50" max="50" step="0.5" value="8">
                </div>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>True Intercept (b)</span>
                        <span id="b-val">40.0</span>
                    </div>
                    <input type="range" id="input-b" class="control-input" min="0" max="100" step="1" value="40">
                </div>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>Noise Variance (var)</span>
                    </div>
                    <input type="number" id="input-var" style="width: 100%; padding: 0.4rem; border: 1px solid #ced4da; border-radius: 4px;" min="0" max="100000" step="10" value="100000">
                </div>
                
                <div class="control-group">
                    <div class="control-label">
                        <span>Random Seed</span>
                    </div>
                    <input type="number" id="input-seed" style="width: 100%; padding: 0.4rem; border: 1px solid #ced4da; border-radius: 4px;" min="0" value="42">
                </div>
            </div>
            
            <!-- MAIN PANEL -->
            <div class="main-content">
                <!-- Metrics row -->
                <div class="metrics-row">
                    <div class="metric-card">
                        <div class="card-title">Estimated Slope (a)</div>
                        <div id="metric-slope" class="card-value">-</div>
                        <div id="metric-slope-sub" class="card-subtext">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="card-title">Estimated Intercept (b)</div>
                        <div id="metric-intercept" class="card-value">-</div>
                        <div id="metric-intercept-sub" class="card-subtext">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="card-title">R-squared Score</div>
                        <div id="metric-r2" class="card-value">-</div>
                        <div class="card-subtext">Variance Explained</div>
                    </div>
                    <div class="metric-card">
                        <div class="card-title">RMSE (Residual SD)</div>
                        <div id="metric-rmse" class="card-value">-</div>
                        <div id="metric-rmse-sub" class="card-subtext">-</div>
                    </div>
                </div>
                
                <!-- Plot and table body -->
                <div class="dashboard-body">
                    <!-- Left: Plot -->
                    <div class="plot-container">
                        <h3>Model Fit & Outliers Visualization</h3>
                        <img id="dashboard-plot" src="" alt="Regression Plot">
                    </div>
                    
                    <!-- Right: Table -->
                    <div class="table-container">
                        <h3>Top 10 Outliers</h3>
                        <table id="outliers-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>X</th>
                                    <th>Actual Y</th>
                                    <th>Residual</th>
                                    <th>Abs Residual</th>
                                </tr>
                            </thead>
                            <tbody id="outliers-tbody">
                                <tr>
                                    <td colspan="5" style="text-align: center; color: #888;">Loading outlier records...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Elements
            const inputN = document.getElementById('input-n');
            const inputA = document.getElementById('input-a');
            const inputB = document.getElementById('input-b');
            const inputVar = document.getElementById('input-var');
            const inputSeed = document.getElementById('input-seed');
            
            const textN = document.getElementById('n-val');
            const textA = document.getElementById('a-val');
            const textB = document.getElementById('b-val');
            
            const metricSlope = document.getElementById('metric-slope');
            const metricSlopeSub = document.getElementById('metric-slope-sub');
            const metricIntercept = document.getElementById('metric-intercept');
            const metricInterceptSub = document.getElementById('metric-intercept-sub');
            const metricR2 = document.getElementById('metric-r2');
            const metricRmse = document.getElementById('metric-rmse');
            const metricRmseSub = document.getElementById('metric-rmse-sub');
            
            const plotImg = document.getElementById('dashboard-plot');
            const outliersTbody = document.getElementById('outliers-tbody');
            
            let debounceTimer = null;
            
            // Format utility
            function f(val, decimals=4) {
                return parseFloat(val).toFixed(decimals);
            }
            
            // Update function
            async function updateDashboard() {
                const n = inputN.value;
                const a = inputA.value;
                const b = inputB.value;
                const v = inputVar.value;
                const s = inputSeed.value;
                
                // Update text values
                textN.innerText = n;
                textA.innerText = f(a, 1);
                textB.innerText = f(b, 1);
                
                // Update plot image src
                const params = `n=${n}&a=${a}&b=${b}&var=${v}&seed=${s}&top_n=10`;
                plotImg.src = `/plot?${params}`;
                
                // Fetch metrics & outliers
                try {
                    const response = await fetch(`/analyze?${params}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        const m = data.regression_metrics;
                        
                        // Populate metrics
                        metricSlope.innerText = f(m.est_slope);
                        const slopeErrSign = m.slope_error >= 0 ? '+' : '';
                        metricSlopeSub.innerHTML = `True: ${f(m.true_slope, 1)} | Error: <span class="${Math.abs(m.slope_error) < 0.1 ? 'card-error-green' : 'card-error-red'}">${slopeErrSign}${f(m.slope_error)}</span>`;
                        
                        metricIntercept.innerText = f(m.est_intercept);
                        const intErrSign = m.intercept_error >= 0 ? '+' : '';
                        metricInterceptSub.innerHTML = `True: ${f(m.true_intercept, 1)} | Error: <span class="${Math.abs(m.intercept_error) < 0.5 ? 'card-error-green' : 'card-error-red'}">${intErrSign}${f(m.intercept_error)}</span>`;
                        
                        metricR2.innerText = f(m.r2_score, 6);
                        metricRmse.innerText = f(m.rmse);
                        
                        const trueNoiseSd = Math.sqrt(v);
                        metricRmseSub.innerText = `Target noise SD: ${f(trueNoiseSd, 2)}`;
                        
                        // Populate table
                        const outliers = data.top_10_outliers;
                        outliersTbody.innerHTML = '';
                        outliers.forEach(row => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td><span class="rank-badge">${row.outlier_rank}</span></td>
                                <td>${f(row.x)}</td>
                                <td>${f(row.actual_y)}</td>
                                <td>${f(row.residual)}</td>
                                <td>${f(row.absolute_residual)}</td>
                            `;
                            outliersTbody.appendChild(tr);
                        });
                    }
                } catch (err) {
                    console.error("Error updating dashboard data:", err);
                }
            }
            
            // Debounce function to prevent hammering API
            function triggerUpdate() {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(updateDashboard, 200);
            }
            
            // Listeners
            [inputN, inputA, inputB, inputVar, inputSeed].forEach(elem => {
                elem.addEventListener('input', triggerUpdate);
                elem.addEventListener('change', triggerUpdate);
            });
            
            // Initial call
            updateDashboard();
        </script>
    </body>
    </html>
    """
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
