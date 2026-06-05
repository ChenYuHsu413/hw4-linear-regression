# Command Execution Log

This log records all terminal commands executed during the CRISP-DM linear regression outlier analysis project.

## Agent Terminal Executions

1. **Verify Initial Python Execution (Failed due to missing dependency)**
   ```bash
   python regression_analysis.py
   ```
   *Result*: `ModuleNotFoundError: No module named 'matplotlib'`

2. **Install Project Dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
   *Result*: Successfully installed packages `matplotlib-3.10.9`, `scikit-learn-1.9.0`, etc.

3. **Verify Script Execution (Failed due to Unicode character)**
   ```bash
   python regression_analysis.py
   ```
   *Result*: `UnicodeEncodeError: 'cp950' codec can't encode character '\xb2' in position 43`

4. **Verify Script Execution (Completed with Warnings)**
   ```bash
   python regression_analysis.py
   ```
   *Result*: Completed successfully, but displayed a scikit-learn `UserWarning` about feature names in `predict()`.

5. **Verify Script Execution (Completed Cleanly)**
   ```bash
   python regression_analysis.py
   ```
   *Result*: Completed successfully with zero warnings.

6. **Copy Image to Artifacts Directory (Windows Powershell)**
   ```powershell
   Copy-Item -Path "d:\AI Class ChenYu\AIClass\hw4\linear_regression_outliers.png" -Destination "C:\Users\admin\.gemini\antigravity-ide\brain\27ffa635-6ea6-469e-968f-71d26d0d244b\linear_regression_outliers.png"
   ```

7. **Install Streamlit Dashboard Library**
   ```bash
   pip install streamlit
   ```
   *Result*: Already satisfied.

8. **Start Headless Streamlit Server (First Run - Syntax Warnings)**
   ```bash
   streamlit run app.py --server.headless true
   ```
   *Result*: Started, but outputted LaTeX-based escaping `SyntaxWarning` messages for `\h` and `\c`.

9. **Start Headless Streamlit Server (Second Run - Clean)**
   ```bash
   streamlit run app.py --server.headless true
   ```
   *Result*: Server running cleanly at `http://localhost:8501`.

10. **Re-Run CLI Regression Analysis (New default variance = 100,000)**
    ```bash
    python regression_analysis.py
    ```
    *Result*: Regenerated CSV and PNG outputs with standard variance of 100,000.

11. **Update Output Artifact Image (Windows Powershell)**
    ```powershell
    Copy-Item -Path "d:\AI Class ChenYu\AIClass\hw4\linear_regression_outliers.png" -Destination "C:\Users\admin\.gemini\antigravity-ide\brain\27ffa635-6ea6-469e-968f-71d26d0d244b\linear_regression_outliers.png"
    ```

---

## User Terminal Executions

1. **First CLI Validation Run (Failed)**
   ```powershell
   & C:/Users/admin/AppData/Local/Programs/Python/Python314/python.exe "d:/AI Class ChenYu/AIClass/hw4/regression_analysis.py"
   ```
   *Result*: `ValueError: Parameter 'var' (variance) must be between 0 and 300 inclusive, got 500.`

2. **Second CLI Validation Run (Completed)**
   ```powershell
   & C:/Users/admin/AppData/Local/Programs/Python/Python314/python.exe "d:/AI Class ChenYu/AIClass/hw4/regression_analysis.py"
   ```
   *Result*: Completed successfully (estimated parameters print correctly).

3. **Third CLI Validation Run (Failed)**
   ```powershell
   & C:/Users/admin/AppData/Local/Programs/Python/Python314/python.exe "d:/AI Class ChenYu/AIClass/hw4/regression_analysis.py"
   ```
   *Result*: `ValueError: Parameter 'var' (variance) must be between 0 and 300 inclusive, got 100000.`

4. **Fourth CLI Validation Run (Completed after validation bounds update)**
   ```powershell
   & C:/Users/admin/AppData/Local/Programs/Python/Python314/python.exe "d:/AI Class ChenYu/AIClass/hw4/regression_analysis.py"
   ```
   *Result*: Completed successfully with variance = 100,000.

## FastAPI Endpoint Executions (Added)

1. **Install FastAPI and Uvicorn**
   ```bash
   pip install fastapi uvicorn
   ```
   *Result*: Installed starlette, pydantic, fastapi, uvicorn.

2. **Start FastAPI Uvicorn Server**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   *Result*: Server started on port 8000.

3. **Verify API /analyze Endpoint**
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/analyze?n=50&a=5&b=20&var=10"
   ```
   *Result*: Returns JSON object containing metrics and ranked outliers.

4. **Verify API /plot Image Streaming**
   ```powershell
   Invoke-WebRequest -Uri "http://127.0.0.1:8000/plot?n=50&a=5&b=20&var=10" -OutFile "test_plot.png"
   ```
   *Result*: Downloaded rendered regression PNG.

