# 📈 CRISP-DM Linear Regression & Outlier Detector

📊 **Interactive Data Science Anomaly Detection App**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🔗 **Streamlit Cloud Demo**: [chenyu-hw4-linear-regression.streamlit.app](https://chenyu-hw4-linear-regression.streamlit.app)

An interactive, beginner-friendly Python project that implements synthetic data generation, Ordinary Least-Squares (OLS) linear regression model fitting, evaluation metrics computation, and residual-based outlier identification. The project is designed strictly according to the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

---

## 📋 Table of Contents
1. [🔍 Project Overview](#-project-overview)
2. [🗺️ CRISP-DM Phase Mapping](#%EF%B8%8F-crisp-dm-phase-mapping)
3. [📁 Project Structure](#-project-structure)
4. [🛠️ Installation & Requirements](#%EF%B8%8F-installation--requirements)
5. [🚀 Execution Guide](#-execution-guide)
6. [🔬 Outlier Identification Methodology](#-outlier-identification-methodology)
7. [⚙️ Default Parameters Configuration](#%EF%B8%8F-default-parameters-configuration)

---

## 🔍 Project Overview

The objective is to model a known linear relationship with noise and isolate the top 10 outlier observations:

$$y = a \cdot x + b + \epsilon$$

where:
* **$x$** is uniformly distributed between $-100$ and $100$.
* **$a$** is the true slope.
* **$b$** is the true intercept.
* **$\epsilon$** is Gaussian noise $N(0, var)$.

We estimate parameters using Ordinary Least-Squares (OLS) linear regression, calculate regression evaluation metrics ($R^2$, MSE, RMSE, MAE), rank observations by their absolute residuals, and visualize the findings.

---

## 🗺️ CRISP-DM Phase Mapping

Our codebase is clearly structured around the 6 CRISP-DM phases:

* **Phase 1: Business Understanding**  
  Clearly defines the project success criteria, business/analytical context of anomaly detection, and goals.
* **Phase 2: Data Understanding**  
  Generates synthetic data, executes basic descriptive Exploratory Data Analysis (EDA), computes Pearson correlation, and logs boundary limits.
* **Phase 3: Data Preparation**  
  Validates user configuration bounds, structures features/targets, checks for duplicates, and outlines outlier handling strategies.
* **Phase 4: Modeling**  
  Fits the Ordinary Least-Squares (OLS) linear model and computes predictions, residuals ($y - \hat{y}$), and absolute residuals ($|y - \hat{y}|$).
* **Phase 5: Evaluation**  
  Evaluates estimated slope and intercept parameters against true parameters, calculates regression metrics, ranks outliers, and prints a formatted summary.
* **Phase 6: Deployment**  
  Deploys results as exportable CSV files, generates high-quality visualizations (PNG), and runs the interactive Streamlit Web App dashboard.

---

## 📁 Project Structure

```text
hw4/
├── .gitignore                   # Ignores generated .csv data and .png plots
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies for Streamlit Cloud
├── regression_analysis.py       # Modular CLI program following CRISP-DM
├── app.py                       # Interactive Streamlit Web application
└── docs/
    ├── log.md                   # Chronological command execution history
    └── 工作報告.md               # Detailed project work report (Traditional Chinese)
```

---

## 🛠️ Installation & Requirements

### 💻 Prerequisites
* Python 3.8 or higher.
* `pip` (Python package manager).

### 📦 Dependency Installation
Install the necessary numerical computation, machine learning, plotting, and dashboard libraries by running:

```bash
pip install numpy pandas matplotlib scikit-learn streamlit
```

---

## 🚀 Execution Guide

### 1. CLI Analysis Script 🖥️
To run the automated CRISP-DM linear regression script and generate output files, execute:

```bash
python regression_analysis.py
```

Upon successful execution, the script prints summary logs directly to the console and generates the following files in the project root:
* 📄 **`linear_regression_results.csv`**: Complete results dataframe with predictions, residuals, and outlier rankings.
* 📄 **`top_10_outliers.csv`**: Separated CSV file featuring only the top 10 ranked outliers.
* 🖼️ **`linear_regression_outliers.png`**: Plot visualizing the OLS model line, data points, and highlighted outliers.

### 2. Streamlit Web App Dashboard 🌐
To run the interactive web application allowing real-time parameter tweaking and visualization, execute:

```bash
streamlit run app.py
```
* Open `http://localhost:8501` in your browser.
* Adjust sliders in the sidebar to inspect regression sensitivities to variance (`var`), intercept (`b`), slope (`a`), and sample size (`n`).
* Download generated CSV results directly from the "Raw Data & CSV Export" tab.

---

## 🔬 Outlier Identification Methodology

* **Outlier Definition**: In this pipeline, an outlier is defined as an observation that exhibits a high absolute residual ($|y - \hat{y}|$) relative to the fitted regression line.
* **Ranking**: The points are ranked from $1$ (largest residual) to $10$.
* **Method Limitation**: This residual-based method effectively identifies anomalous targets on the vertical $y$-axis (y-dimension errors), but does not target high-leverage points (horizontal $x$-axis outliers).

---

## ⚙️ Default Parameters Configuration

Both scripts are pre-configured with the following reproducible defaults:
* **Sample Size ($n$)**: `500`
* **True Slope ($a$)**: `8.0`
* **True Intercept ($b$)**: `40.0`
* **Noise Variance ($var$)**: `100000.0`
* **Random Seed**: `42`
