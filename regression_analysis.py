import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==============================================================================
# Configurable Parameters
# ==============================================================================
n = 500            # Number of generated data points (integer >= 10)
a = 8              # True slope (numeric between -50 and 50)
b = 40             # True intercept (numeric between 0 and 100)
var = 100000       # Gaussian noise variance (numeric between 0 and 100000)
random_seed = 42   # Random seed for reproducibility (integer)


# ==============================================================================
# CRISP-DM Phase 1: Business Understanding
# ==============================================================================
def describe_business_understanding():
    """
    CRISP-DM Phase 1: Business Understanding
    Prints the project objectives, context, and success criteria.
    """
    print("\n" + "="*80)
    print(" CRISP-DM Phase 1: Business Understanding")
    print("="*80)
    print("Project Analytical Objectives:")
    print("  1. Generate synthetic data following a known linear relationship with noise.")
    print("  2. Estimate the parameters (slope and intercept) using Ordinary Least-Squares (OLS) linear regression.")
    print("  3. Measure the accuracy of the fitted model using regression evaluation metrics.")
    print("  4. Identify and rank the top 10 outlier observations that deviate most from the regression line.")
    print("  5. Produce reusable output files (visual plot and CSV datasets) for downstream deployment.")
    print("\nBusiness Context:")
    print("  Detecting outliers is crucial in data science pipelines to uncover anomalies (such as fraud,")
    print("  equipment failures, or transmission errors) and improve model robustness. This program demonstrates")
    print("  an end-to-end framework using synthetic data where the ground truth parameters are known.")
    print("\nProject Success Criteria:")
    print("  - The linear regression model successfully estimates the true slope (a) and intercept (b).")
    print("  - The model outputs validation metrics (R-squared, MSE, RMSE, MAE).")
    print("  - The top 10 outliers are correctly identified, ranked by absolute residual, and summarized.")
    print("  - High-quality visualization and structured datasets are generated and saved.")
    print("="*80 + "\n")


# ==============================================================================
# CRISP-DM Phase 2: Data Understanding
# ==============================================================================
def generate_data(n_val, a_val, b_val, var_val, seed_val):
    """
    CRISP-DM Phase 2: Data Understanding (Data Generation)
    Generates synthetic linear-regression data with Gaussian noise.
    
    Formula: y = a * x + b + noise
    where x ~ Uniform(-100, 100) and noise ~ Normal(0, sqrt(var))
    """
    np.random.seed(seed_val)
    
    # Generate random x values uniformly distributed between -100 and 100
    x_vals = np.random.uniform(-100.0, 100.0, size=n_val)
    
    # Calculate scale (standard deviation) from variance
    std_dev = np.sqrt(var_val)
    
    # Generate Gaussian noise using standard deviation
    noise = np.random.normal(loc=0.0, scale=std_dev, size=n_val)
    
    # Calculate actual y values using y = a * x + b + noise
    actual_y = a_val * x_vals + b_val + noise
    
    # Store in pandas DataFrame
    df = pd.DataFrame({
        'x': x_vals,
        'actual_y': actual_y
    })
    return df


def understand_data(dataframe):
    """
    CRISP-DM Phase 2: Data Understanding (Exploratory Data Analysis)
    Performs basic descriptive statistical analysis and prints details.
    """
    print("="*80)
    print(" CRISP-DM Phase 2: Data Understanding")
    print("="*80)
    print(f"Dataset Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")
    print("\nFirst 5 Rows of the Generated Dataset:")
    print(dataframe.head().to_string())
    
    print("\nDescriptive Statistics:")
    print(dataframe.describe().to_string())
    
    print("\nMissing-Value Counts:")
    print(dataframe.isnull().sum().to_string())
    
    x_min = dataframe['x'].min()
    x_max = dataframe['x'].max()
    y_min = dataframe['actual_y'].min()
    y_max = dataframe['actual_y'].max()
    
    print(f"\nx boundaries: Min x = {x_min:.4f}, Max x = {x_max:.4f}")
    print(f"y boundaries: Min y = {y_min:.4f}, Max y = {y_max:.4f}")
    
    correlation = dataframe['x'].corr(dataframe['actual_y'])
    print(f"\nPearson correlation between x and actual_y: {correlation:.6f}")
    print("="*80 + "\n")


# ==============================================================================
# CRISP-DM Phase 3: Data Preparation
# ==============================================================================
def validate_parameters(n_val, a_val, b_val, var_val, seed_val):
    """
    CRISP-DM Phase 3: Data Preparation (Parameter Validation)
    Checks types and value limits. Raises TypeError or ValueError on failure.
    """
    # 1. Type validation
    if not isinstance(n_val, int):
        raise TypeError(f"Parameter 'n' must be an integer, got {type(n_val).__name__}.")
    if not isinstance(seed_val, int):
        raise TypeError(f"Parameter 'random_seed' must be an integer, got {type(seed_val).__name__}.")
    if not isinstance(a_val, (int, float)):
        raise TypeError(f"Parameter 'a' must be numeric (int or float), got {type(a_val).__name__}.")
    if not isinstance(b_val, (int, float)):
        raise TypeError(f"Parameter 'b' must be numeric (int or float), got {type(b_val).__name__}.")
    if not isinstance(var_val, (int, float)):
        raise TypeError(f"Parameter 'var' must be numeric (int or float), got {type(var_val).__name__}.")
    
    # 2. Value range validation
    if n_val < 10:
        raise ValueError(f"Parameter 'n' (dataset size) must be at least 10, got {n_val}.")
    if not (-50 <= a_val <= 50):
        raise ValueError(f"Parameter 'a' (slope) must be between -50 and 50 inclusive, got {a_val}.")
    if not (0 <= b_val <= 100):
        raise ValueError(f"Parameter 'b' (intercept) must be between 0 and 100 inclusive, got {b_val}.")
    if not (0 <= var_val <= 100000):
        raise ValueError(f"Parameter 'var' (variance) must be between 0 and 100000 inclusive, got {var_val}.")


def prepare_data(dataframe):
    """
    CRISP-DM Phase 3: Data Preparation
    Validates features and targets, checks for duplicates, and structures data variables.
    """
    print("="*80)
    print(" CRISP-DM Phase 3: Data Preparation")
    print("="*80)
    
    # Prepare the feature matrix X and target variable y
    X = dataframe[["x"]]
    y = dataframe["actual_y"]
    print(f"Split completed: Feature matrix X shape: {X.shape}, Target y shape: {y.shape}")
    
    # Check for missing values and duplicates
    missing_sum = dataframe.isnull().sum().sum()
    duplicate_sum = dataframe.duplicated().sum()
    print(f"Missing values found: {missing_sum}")
    print(f"Duplicate rows found: {duplicate_sum}")
    
    print("\nDuplicate Strategy Explanation:")
    print("  - For this continuous synthetic data, duplicate values are theoretically rare but entirely")
    print("    possible and statistically valid because they represent real random draws. We keep them")
    print("    as-is; discarding duplicates would distort the true probability distribution.")
    
    print("\nOutlier Strategy Explanation:")
    print("  - We do NOT remove outliers during this preparation phase. Removing outliers beforehand")
    print("    would skew the regression modeling and defeat the core project objective, which is to")
    print("    identify and rank outliers based on residuals of the fitted regression line.")
    print("="*80 + "\n")
    
    return X, y


# ==============================================================================
# CRISP-DM Phase 4: Modeling
# ==============================================================================
def fit_regression(X, y, dataframe):
    """
    CRISP-DM Phase 4: Modeling
    Fits an Ordinary Least-Squares Linear Regression model.
    Calculates predicted values, residuals, and absolute residuals, returning updated DataFrame.
    """
    print("="*80)
    print(" CRISP-DM Phase 4: Modeling")
    print("="*80)
    
    # Initialize and fit Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    print("OLS Linear Regression model successfully fitted.")
    print(f"  Estimated Slope (a):     {model.coef_[0]:.6f}")
    print(f"  Estimated Intercept (b): {model.intercept_:.6f}")
    
    # Copy DataFrame to avoid modifying original dataset
    df_modeled = dataframe.copy()
    
    # Add predicted_y, residual, and absolute_residual columns
    df_modeled['predicted_y'] = model.predict(X)
    df_modeled['residual'] = df_modeled['actual_y'] - df_modeled['predicted_y']
    df_modeled['absolute_residual'] = df_modeled['residual'].abs()
    
    print("\nFitted outputs (predicted_y, residual, absolute_residual) added to the DataFrame.")
    print("="*80 + "\n")
    
    return model, df_modeled['predicted_y'], df_modeled


# ==============================================================================
# CRISP-DM Phase 5: Evaluation
# ==============================================================================
def evaluate_model(model, dataframe, true_a, true_b):
    """
    CRISP-DM Phase 5: Evaluation
    Computes performance metrics and estimation errors.
    """
    est_a = model.coef_[0]
    est_b = model.intercept_
    
    # Compute differences
    error_a = est_a - true_a
    error_b = est_b - true_b
    
    # Actual and predicted variables
    actual_y = dataframe['actual_y']
    predicted_y = dataframe['predicted_y']
    
    # Metrics
    r2 = r2_score(actual_y, predicted_y)
    mse = mean_squared_error(actual_y, predicted_y)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_y, predicted_y)
    
    return {
        'true_slope': true_a,
        'est_slope': est_a,
        'slope_error': error_a,
        'true_intercept': true_b,
        'est_intercept': est_b,
        'intercept_error': error_b,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }


def find_top_outliers(dataframe, top_n=10):
    """
    CRISP-DM Phase 5: Evaluation (Outlier detection)
    Identifies the top N outliers by absolute residual.
    Assigns ranks (1 for largest absolute residual) and handles is_outlier indicator.
    """
    df_evaluated = dataframe.copy()
    
    # Find indices of top N largest absolute residuals
    top_indices = df_evaluated['absolute_residual'].nlargest(top_n).index
    
    # Initialize rankings (1 to top_n)
    ranks = pd.Series(index=top_indices, data=np.arange(1, top_n + 1))
    
    # Map ranks to the DataFrame (non-outliers will have NaN / <NA>)
    df_evaluated['outlier_rank'] = df_evaluated.index.map(ranks).astype('Int64')
    df_evaluated['is_outlier'] = df_evaluated.index.isin(top_indices)
    
    # Extract only the outliers and sort them by rank
    outliers_df = df_evaluated[df_evaluated['is_outlier']].sort_values(by='outlier_rank')
    
    return outliers_df, df_evaluated


def print_evaluation_results(metrics, outliers_df):
    """
    CRISP-DM Phase 5: Evaluation (Console printing)
    """
    print("="*80)
    print(" CRISP-DM Phase 5: Evaluation")
    print("="*80)
    
    # Parameter estimation accuracy
    print("Parameter Estimation Accuracy:")
    print(f"  True Slope (a):            {metrics['true_slope']}")
    print(f"  Estimated Slope (a_est):    {metrics['est_slope']:.6f}")
    print(f"  Slope Estimation Error:    {metrics['slope_error']:.6f}")
    print(f"  True Intercept (b):        {metrics['true_intercept']}")
    print(f"  Estimated Intercept (b_est): {metrics['est_intercept']:.6f}")
    print(f"  Intercept Estimation Error: {metrics['intercept_error']:.6f}")
    
    # Statistical validation metrics
    print("\nModel Quality Metrics:")
    print(f"  R-squared Score (Coefficient of Determination): {metrics['r2_score']:.6f}")
    print(f"  Mean Squared Error (MSE):                {metrics['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE):          {metrics['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE):              {metrics['mae']:.4f}")
    
    # Definitions and Limitations
    print("\nDefinitions and Methodological Considerations:")
    print("  - Outlier Definition: In this analysis, an outlier is defined as an observation that exhibits")
    print("    a substantial absolute residual (vertical distance) from the fitted regression line.")
    print("  - Method Limitations: This residual-based method identifies observations with highly anomalous")
    print("    target values (y-dimension anomalies) given their input values. However, it does not")
    print("    specifically flag high-leverage outliers (anomalous values along the independent x-axis).")
    
    # Display Outliers Table
    print(f"\nTop {len(outliers_df)} Outlier Observations (Ordered by Rank):")
    outlier_summary = outliers_df[['outlier_rank', 'x', 'actual_y', 'predicted_y', 'residual', 'absolute_residual']]
    print(outlier_summary.to_string(index=False))
    print("="*80 + "\n")


# ==============================================================================
# CRISP-DM Phase 6: Deployment
# ==============================================================================
def deploy_results(dataframe, outliers_df, model):
    """
    CRISP-DM Phase 6: Deployment
    Generates and saves the final regression and outlier visualization.
    Saves outputs as CSV and PNG files.
    Prints confirmation logs.
    """
    print("="*80)
    print(" CRISP-DM Phase 6: Deployment")
    print("="*80)
    
    # 1. Visualization Setup
    plt.figure(figsize=(12, 8))
    
    # Separate normal data points and outliers for different scatter styles
    normal_points = dataframe[~dataframe['is_outlier']]
    
    # Plot normal observations (lighter blue dots)
    plt.scatter(
        normal_points['x'], normal_points['actual_y'],
        color='#3498db', alpha=0.6, edgecolors='none', s=30, label='Normal Observations'
    )
    
    # Plot outlier observations (large red/orange crosses with border)
    plt.scatter(
        outliers_df['x'], outliers_df['actual_y'],
        color='#e74c3c', marker='X', s=130, edgecolors='black', linewidths=0.8,
        label='Top 10 Outliers (Ranked)'
    )
    
    # Sort x before drawing regression line to avoid line-drawing artifacts
    x_sorted = dataframe['x'].sort_values()
    # Reshape for scikit-learn predict
    y_regression = model.predict(pd.DataFrame({'x': x_sorted}))
    
    # Plot red regression line
    plt.plot(x_sorted, y_regression, color='#c0392b', linewidth=2.5, label='Fitted Regression Line')
    
    # Annotate outlier points with their rank
    for _, row in outliers_df.iterrows():
        plt.annotate(
            f"Rank {row['outlier_rank']}",
            xy=(row['x'], row['actual_y']),
            xytext=(row['x'] + 3, row['actual_y'] + 2.0),  # Slight text offset
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='#f1c40f', alpha=0.8, ec='gray', lw=0.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7, connectionstyle='arc3,rad=0.1')
        )
    
    # Plot Labels and Formatting
    plt.title('Linear Regression Model & Top 10 Outliers (CRISP-DM)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('X (Independent Variable)', fontsize=12)
    plt.ylabel('Y (Dependent Variable)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, shadow=True, facecolor='white')
    
    # Tight layout to avoid cropping annotations
    plt.tight_layout()
    
    # Output file paths
    png_file = 'linear_regression_outliers.png'
    results_csv = 'linear_regression_results.csv'
    outliers_csv = 'top_10_outliers.csv'
    
    # Save visualization
    plt.savefig(png_file, dpi=300)
    plt.close()
    print(f"Saved visualization file to: {os.path.abspath(png_file)}")
    
    # Save datasets
    dataframe.to_csv(results_csv, index=False)
    print(f"Saved complete results DataFrame to: {os.path.abspath(results_csv)}")
    
    outliers_to_save = outliers_df[['outlier_rank', 'x', 'actual_y', 'predicted_y', 'residual', 'absolute_residual']]
    outliers_to_save.to_csv(outliers_csv, index=False)
    print(f"Saved top 10 outliers DataFrame to: {os.path.abspath(outliers_csv)}")
    
    print("\nDeployment completed successfully!")
    print("="*80 + "\n")


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main():
    print("================================================================================")
    print("           CRISP-DM LINEAR REGRESSION AND OUTLIER DETECTION PROGRAM             ")
    print("================================================================================\n")
    
    try:
        # Phase 1: Business Understanding
        describe_business_understanding()
        
        # Parameter validation (Phase 3 logic)
        print("Validating model configuration parameters...")
        validate_parameters(n, a, b, var, random_seed)
        print("Parameters successfully validated.\n")
        
        # Phase 2: Data Understanding (Data generation & EDA)
        print("Generating synthetic regression data...")
        df_raw = generate_data(n, a, b, var, random_seed)
        understand_data(df_raw)
        
        # Phase 3: Data Preparation
        X, y = prepare_data(df_raw)
        
        # Phase 4: Modeling
        model, predicted_y, df_modeled = fit_regression(X, y, df_raw)
        
        # Phase 5: Evaluation
        metrics = evaluate_model(model, df_modeled, a, b)
        outliers_df, df_final = find_top_outliers(df_modeled, top_n=10)
        print_evaluation_results(metrics, outliers_df)
        
        # Phase 6: Deployment
        deploy_results(df_final, outliers_df, model)
        
        print("CRISP-DM workflow executed successfully!")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
