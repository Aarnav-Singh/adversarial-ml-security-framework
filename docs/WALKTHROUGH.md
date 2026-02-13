# Functionality Walkthrough: Adversarial ML Framework

Follow these steps to verify the complete functionality of the Adversarial ML Security Framework.

## 1. Environment Preparation

Open your terminal in the project root directory and install all required libraries:

```bash
pip install -r requirements.txt
```

## 2. Initialization (Essential for New Installs)

Since models and datasets are not committed to the repository (keeping the repo clean), you must generate them locally before the dashboard can "detect" anything.

### Step A: Generate Benign Traffic

This creates the baseline "normal" behavior for your network.

```bash
python src/simulation/traffic_generator.py
```

### Step B: Simulate Initial Attacks

This creates a combined dataset of benign and malicious traffic for training.

```bash
python src/simulation/attack_generator.py
```

### Step C: Train Security Models

This trains the **Isolation Forest** (anomaly detector) and **Random Forest** (classifier).

```bash
python src/training/core_model.py
```

## 3. Launch the Dashboard

Once your models are saved in the `models/` folder, start the interactive analysis console:

```bash
streamlit run src/dashboard/app.py
```

## 3. Mandatory Verification Steps

### Operations Center (Live Security Feed)

1. Navigate to the **"Operations"** tab.
2. Click **"▶ Activate Live Feed"**.
3. **Verification**: Confirm that "ALLOW" (Green) and "BLOCKED" (Red) cards begin appearing in the stream. This confirms the baseline detection loop is active.

### Red Team Labs (Attack Simulation)

1. Navigate to the **"Red Team"** tab.
2. Click **"Launch Black-Box HSJ Attack"**.
3. **Verification**: Wait for the "Attack Summary" to appear. Confirm that the **ASR (Attack Success Rate)** is significantly high (e.g., > 60%), proving the Baseline stage's vulnerability.
4. Run a **"Generate Vulnerability heatmap"** and confirm the feature sensitivity is visualized.

### Blue Team Analytics (Defense Evolution)

1. Navigate to the **"Blue Team"** tab.
2. Expand **"Demonstration Controls"** and toggle the manual override to test UI consistency.
3. Click **"Fortify Model (Retrain on FGM)"**.
4. **Verification**: Wait for the "Model Fortified!" success message.
5. Click **"Run Stage Evolution Analysis"**.
6. **Comparison Proof**: Confirm the bar chart now shows a significant side-by-side performance delta between **Stage 1 (Baseline)** and **Stage 2 (Fortified)**.

### Regression Testing

1. Click **"CI/CD Regression Test"**.
2. **Verification**: Confirm the status reports **PASSED** if accuracy remains above the 80% threshold.

---
*Follow this procedure for a functionality verification of the system.*
