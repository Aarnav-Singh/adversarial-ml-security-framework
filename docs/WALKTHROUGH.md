# Verification Walkthrough: ZT-Shield v1.0

Follow these steps to verify the complete functionality of the Adversarial Research Framework.

## 1. Environment Preparation

Ensure all dependencies are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit plotly adversarial-robustness-toolbox joblib joblib joblib
```

## 2. Launch Sequence

Start the research console from the project root:

```bash
streamlit run src/dashboard/app.py
```

## 3. Mandatory Verification Steps

### ✅ A. Operations Center (Live Security Feed)

1. Navigate to the **"🟢 Operations"** tab.
2. Click **"▶ Activate Live Feed"**.
3. **Verification**: Confirm that "ALLOW" (Green) and "BLOCKED" (Red) cards begin appearing in the stream. This confirms the baseline detection loop is active.

### ✅ B. Red Team Labs (Attack Simulation)

1. Navigate to the **"🔴 Red Team"** tab.
2. Click **"🚀 Launch Black-Box HSJ Attack"**.
3. **Verification**: Wait for the "Attack Summary" to appear. Confirm that the **ASR (Attack Success Rate)** is significantly high (e.g., > 60%), proving the Baseline stage's vulnerability.
4. Run a **"📊 Generate Vulnerability heatmap"** and confirm the feature sensitivity is visualized.

### ✅ C. Blue Team Analytics (Defense Evolution)

1. Navigate to the **"🟣 Blue Team"** tab.
2. Expand **"🛠️ Demonstration Controls"** and toggle the manual override to test UI consistency.
3. Click **"🔥 Fortify Model (Retrain on FGM)"**.
4. **Verification**: Wait for the "Model Fortified!" success message.
5. Click **"📊 Run Stage Evolution Analysis"**.
6. **Comparison Proof**: Confirm the bar chart now shows a significant side-by-side performance delta between **Stage 1 (Baseline)** and **Stage 2 (Fortified)**.

### ✅ D. Regression Testing

1. Click **"🔄 CI/CD Regression Test"**.
2. **Verification**: Confirm the status reports **PASSED** if accuracy remains above the 80% threshold.

---
*Follow this procedure for a certified verification of the ZT-Shield system.*
