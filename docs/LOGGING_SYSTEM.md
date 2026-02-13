# Structured Logging and Analysis System

## 📦 Complete Package

This package provides a **structured logging and analysis system** designed for experimentation with Adversarial ML and Zero-Trust inspired security models.

## Feature Overview

The system provides several core capabilities designed for research auditing:

1. Log extraction from operations and attacks (JSON, TXT, MD, CSV formats)
2. Analysis engine that processes logs to understand patterns and suggest improvements

This implementation provides these capabilities:

- Drop-in integration with your existing codebase
- Automated vulnerability detection
- Defense hardening recommendations
- Adversarial retraining dataset generation
- Experimental feedback loop: Attack → Log → Analyze → Fortify → Re-evaluate

---

```text
soc-logging-analytics/
├── log_manager.py                  # Core logging system
├── blue_team_analytics.py          # Analytics & recommendations engine  
├── attack_logging_wrappers.py      # Integration helpers (drop-in replacements)
├── integration_examples.py         # Working examples
├── USAGE_GUIDE.md                  # Complete documentation
├── README.md                       # This file
└── sample_logs/                    # Demo logs from examples
    ├── sessions/                   # Consolidated attack/defense history history
    │   ├── session_*.json          # (Includes full attack/defense context)
    │   └── session_*.csv          
    ├── attacks/                    # Placeholder for raw attack vectors
    └── analytics/                  # Blue team analysis outputs
```

> [!NOTE]
> **Why is `logs/attacks` empty?**  
> To maintain the highest "Blue Team" context, this framework consolidates attack events and corresponding defense results into unified session logs within `logs/sessions`. The `attacks/` directory remains as a provisioned placeholder for raw vector exports or individual attack JSONs should they be required in isolation.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Add Files to Your Project

```bash
# Copy the three main files to your project
cp log_manager.py /path/to/your/project/src/logging/
cp blue_team_analytics.py /path/to/your/project/src/logging/
cp attack_logging_wrappers.py /path/to/your/project/src/logging/
```

### Step 2: Run an Attack with Logging

```python
from src.logging.attack_logging_wrappers import (
    run_blackbox_attack_with_logging,
    create_logged_attack_session
)

# Create logging session
log_mgr = create_logged_attack_session()

# Run attack (automatically logged!)
X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack_with_logging(
    model=rf,
    X_test=X_test,
    y_test=y_test,
    clip_values=clip_values,
    log_manager=log_mgr,
    sample_size=100
)

# Export logs
log_mgr.export_logs(format='json')  # For analytics
log_mgr.export_logs(format='md')    # For documentation
```

### Step 3: Analyze Logs & Get Recommendations

```python
from src.logging.blue_team_analytics import analyze_logs_and_generate_report

# Run complete analysis pipeline
report_path = analyze_logs_and_generate_report(
    log_directory="logs/sessions",
    output_dir="logs/analytics"
)

print(f"Analysis complete! Report: {report_path}")
```

---

## 💡 Key Features

### 1. Multi-Format Log Export

```python
log_mgr.export_logs(format='json')  # Machine-readable
log_mgr.export_logs(format='txt')   # Plain text
log_mgr.export_logs(format='md')    # Markdown (GitHub/docs)
log_mgr.export_logs(format='csv')   # Spreadsheet-friendly
```

### 2. Comprehensive Event Tracking

- Attack events (HSJ, FGM, custom)
- Defense decisions (Allow/Deny)
- Model behavior (predictions, confidence)
- Batch operations
- Session metadata

### 3. Blue Team Analytics

```python
analytics = BlueTeamAnalytics(log_directory="logs/sessions")
analytics.load_all_logs()

# Get insights
attack_patterns = analytics.analyze_attack_patterns()
defense_analysis = analytics.analyze_defense_effectiveness()
vulnerabilities = analytics.identify_vulnerabilities()
hardening = analytics.generate_hardening_config()
```

### 4. Automated Recommendations

Example output:

```json
{
  "confidence_threshold": {
    "current": 0.15,
    "recommended": 0.20,
    "reason": "Attack success rate is 65%"
  },
  "enable_adversarial_retraining": {
    "recommended": true
  },
  "expected_improvements": {
    "robustness_increase": "20-30%",
    "evasion_rate_reduction": "15-25%"
  }
}
```

---

## 🔄 The Complete Workflow

```text
┌─────────────┐
│  RED TEAM   │  Run attacks (HSJ, FGM)
│   ATTACK    │  Log everything
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  BLUE TEAM  │  Analyze patterns
│  ANALYTICS  │  Identify vulnerabilities
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   DEFENSE   │  Apply recommendations
│  HARDENING  │  Retrain with adv. examples
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     RE-     │  Test with same attacks
│  EVALUATE   │  Compare metrics
└─────────────┘
```

---

## 📊 Example Analytics Output

### Vulnerability Report

```text
Vulnerabilities Found: 3

- [HIGH] Attack success rate is 65%, indicating weak defense
- [MEDIUM] Only 45% of attacks are blocked
- [MEDIUM] Model vulnerable to perturbations with epsilon=0.25

Recommendations:
1. Increase confidence threshold to 0.20
2. Enable adversarial retraining
3. Adjust Isolation Forest sensitivity
```

### Executive Summary

```json
{
  "security_posture": "MODERATE",
  "risk_level": "MEDIUM",
  "attack_success_rate": 0.65,
  "defense_block_rate": 0.45,
  "requires_immediate_action": false,
  "top_recommendation": "increase_confidence_threshold"
}
```

---

## 🔧 Integration with Your Codebase

### Option A: Wrapper Functions (Easiest!)

**Before:**

```python
from src.attacks.blackbox import run_blackbox_attack

X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(
    model, X_test, y_test, clip_values, sample_size=100
)
```

**After (with logging):**

```python
from src.logging.attack_logging_wrappers import run_blackbox_attack_with_logging, LogManager

log_mgr = LogManager()

X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack_with_logging(
    model, X_test, y_test, clip_values,
    log_manager=log_mgr,  # ← Only addition!
    sample_size=100
)

log_mgr.export_logs(format='json')
```

### Option B: Manual Logging (Fine Control)

```python
from src.logging.log_manager import LogManager

log_mgr = LogManager()

# Your existing attack code
X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(...)

# Add logging after
log_mgr.log_batch_attack(
    attack_type="HopSkipJump",
    summary_stats={
        "total_samples": len(X_adv),
        "success_rate": calculate_success_rate(X_orig, X_adv),
        "avg_queries": avg_q
    },
    individual_results=[...]
)

log_mgr.export_logs(format='json')
```

---

## 📖 Full Documentation

See `USAGE_GUIDE.md` for:

- ✅ Detailed API reference
- ✅ Advanced usage examples
- ✅ Troubleshooting guide
- ✅ Best practices
- ✅ Custom logging examples

---

## 🧪 Testing

Run the included examples to verify everything works:

```bash
python integration_examples.py
```

This will:

1. Simulate attacks with logging
2. Export logs in all formats
3. Run blue team analysis
4. Generate comprehensive reports
5. Create retraining specifications

Check the logs directory for outputs.

---

## System Strengths

1. **Structured Logging** - Structured event tracking aligned with SOC-style workflows
2. **Research Rigor** - Comprehensive metrics, statistical analysis, reproducibility
3. **Minimal Changes** - Drop-in wrappers mean almost no code changes needed
4. **Automated Insights** - Analytics engine identifies patterns you might miss
5. **Actionable** - Concrete recommendations, not just data dumps
6. **Complete Loop** - Attack → Analyze → Harden → Re-test workflow built-in

---

This implementation provides a stable, structured version of the features discussed during development.

---

## Next Steps

1. Copy files to your project
2. Run integration_examples.py to see it in action
3. Use wrapper functions with your existing attacks
4. Analyze logs after each attack campaign
5. Apply hardening recommendations
6. Re-test and compare metrics

---

## License

Use freely in your project. No restrictions.

---

## Support

See USAGE_GUIDE.md for detailed documentation and troubleshooting.

---

**Built as a learning-focused adversarial ML security framework. Structured for experimentation.**
