# Structured Logging and Analysis System

## Overview

This system provides structured logging and analysis for your adversarial ML experimental framework. It implements a **functional feedback loop**:

```text
Attack → Log → Analyze → Fortify → Re-evaluate
```

## Features

### 1. **Log Manager** (`log_manager.py`)

- Multi-format export (JSON, TXT, Markdown, CSV)
- Structured event logging for attacks and defenses
- Session-based organization
- Automatic timestamp and metadata tracking

### 2. **Blue Team Analytics** (`blue_team_analytics.py`)

- Attack pattern analysis
- Defense effectiveness metrics
- Vulnerability identification
- Automated hardening recommendations
- Adversarial retraining dataset generation
- Comprehensive reporting

### 3. **Integration Wrappers** (`attack_logging_wrappers.py`)

- Drop-in replacements for existing attack functions
- Automatic logging without code changes
- Compatible with all existing modules

---

## Quick Start

### Installation

Place these files in your project:

```text
your-project/
├── src/
│   ├── logging/
│   │   ├── log_manager.py           ← Core logging
│   │   ├── blue_team_analytics.py   ← Analysis engine
│   │   └── attack_logging_wrappers.py ← Integration helpers
```

Or use them directly from `/home/claude/`:

```python
import sys
sys.path.insert(0, '/home/claude')
from log_manager import LogManager
from blue_team_analytics import BlueTeamAnalytics
```

---

## 📖 Usage Examples

### Example 1: Basic Attack Logging

```python
from log_manager import LogManager
from src.attacks.blackbox import run_blackbox_attack
from src.evaluation.runner import load_system_assets

# Initialize logger
log_mgr = LogManager(base_dir="logs")

# Load models
rf, iso, X_test, y_test, X_train, y_train, clip_values = load_system_assets()

# Run attack
X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(
    rf, X_test, y_test, clip_values, sample_size=100
)

# Log the attack
log_mgr.log_batch_attack(
    attack_type="HopSkipJump",
    summary_stats={
        "total_samples": len(X_adv),
        "avg_queries": avg_q,
        "total_queries": total_q,
        "success_rate": 0.65  # Calculate from your results
    },
    individual_results=[
        {"sample_id": i, "queries": int(avg_q)}
        for i in range(len(X_adv))
    ]
)

# Export logs
json_file = log_mgr.export_logs(format='json')
md_file = log_mgr.export_logs(format='md')
csv_file = log_mgr.export_logs(format='csv')

print(f"Logs exported to:")
print(f"  - {json_file}")
print(f"  - {md_file}")
print(f"  - {csv_file}")
```

### Example 2: Using Wrapper Functions (Easiest!)

```python
from attack_logging_wrappers import (
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
    log_manager=log_mgr,  # Pass the logger
    sample_size=100
)

# That's it! Attack is already logged. Just export:
log_mgr.export_logs(format='json')
```

### Example 3: Blue Team Analysis

```python
from blue_team_analytics import BlueTeamAnalytics

# Initialize analytics
analytics = BlueTeamAnalytics(log_directory="logs/sessions")

# Load all logs
analytics.load_all_logs()

# Analyze attack patterns
attack_analysis = analytics.analyze_attack_patterns()
print(f"Most common attack: {attack_analysis['most_common_attack']}")
print(f"Attack success rate: {attack_analysis['success_rate']:.1%}")

# Analyze defense effectiveness
defense_analysis = analytics.analyze_defense_effectiveness()
print(f"Block rate: {defense_analysis['block_rate']:.1%}")

# Identify vulnerabilities
vulns = analytics.identify_vulnerabilities()
print(f"Found {vulns['total_vulnerabilities']} vulnerabilities")

for vuln in vulns['vulnerabilities']:
    print(f"  - [{vuln['severity'].upper()}] {vuln['description']}")

# Get hardening recommendations
hardening = analytics.generate_hardening_config()
print(f"\nRecommended confidence threshold: {hardening['confidence_threshold']['recommended']}")

# Generate comprehensive report
report = analytics.generate_comprehensive_report(
    output_file="logs/analytics/blue_team_report.json"
)
print(f"\nSecurity Posture: {report['executive_summary']['security_posture']}")
print(f"Risk Level: {report['executive_summary']['risk_level']}")
```

### Example 4: Complete Workflow

```python
from log_manager import LogManager
from blue_team_analytics import analyze_logs_and_generate_report
from src.attacks.blackbox import run_blackbox_attack
from src.evaluation.runner import load_system_assets

# === PHASE 1: RED TEAM ===
print("Phase 1: Red Team Attack")

log_mgr = LogManager()
rf, iso, X_test, y_test, _, _, clip_values = load_system_assets()

# Run attack with logging
X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(
    rf, X_test, y_test, clip_values, sample_size=100
)

# Log attack
preds_orig = rf.predict(X_orig)
preds_adv = rf.predict(X_adv)
success = (preds_orig != preds_adv)

log_mgr.log_batch_attack(
    attack_type="HopSkipJump",
    summary_stats={
        "total_samples": len(X_adv),
        "success_rate": float(success.mean()),
        "avg_queries": avg_q,
        "epsilon": 0.2
    },
    individual_results=[{"sample_id": i, "success": bool(success[i])} for i in range(len(X_adv))]
)

log_mgr.export_logs(format='json')

# === PHASE 2: BLUE TEAM ===
print("\nPhase 2: Blue Team Analysis")

report_path = analyze_logs_and_generate_report(
    log_directory="logs/sessions",
    output_dir="logs/analytics"
)

print(f"Analysis complete! Report: {report_path}")

# === PHASE 3: HARDEN DEFENSES ===
print("\nPhase 3: Apply Hardening")

analytics = BlueTeamAnalytics(log_directory="logs/sessions")
analytics.load_all_logs()
analytics.analyze_attack_patterns()
hardening = analytics.generate_hardening_config()

# Apply recommendations (example)
new_threshold = hardening['confidence_threshold']['recommended']
print(f"Updating confidence threshold: {new_threshold}")

# Create retraining dataset
retraining_spec = analytics.create_retraining_dataset()
print(f"Retraining spec created: {retraining_spec}")

# === PHASE 4: RE-EVALUATE ===
print("\nPhase 4: Re-evaluate with Hardened Defenses")
# Run attacks again with updated config and compare results
```

---

## 🔧 Integration with Your Existing Code

### Option A: Minimal Changes (Recommended)

Use wrapper functions as drop-in replacements:

**Before:**

```python
from src.attacks.blackbox import run_blackbox_attack

X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(
    model, X_test, y_test, clip_values, sample_size=100
)
```

**After:**

```python
from attack_logging_wrappers import run_blackbox_attack_with_logging, LogManager

log_mgr = LogManager()

X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack_with_logging(
    model, X_test, y_test, clip_values,
    log_manager=log_mgr,  # ← Only addition
    sample_size=100
)

log_mgr.export_logs(format='json')  # ← Export logs
```

### Option B: Manual Logging (Fine-Grained Control)

Add logging calls after your existing code:

```python
from log_manager import LogManager

log_mgr = LogManager()

# Your existing attack code
X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(...)

# Add logging
preds_orig = model.predict(X_orig)
preds_adv = model.predict(X_adv)

log_mgr.log_batch_attack(
    attack_type="HopSkipJump",
    summary_stats={...},  # Your metrics
    individual_results=[...]  # Your per-sample results
)

log_mgr.export_logs(format='json')
```

---

## 📊 Log File Structure

### JSON Format (Recommended for Analytics)

```json
{
  "metadata": {
    "session_id": "20260213_143022",
    "start_time": "2026-02-13T14:30:22",
    "total_events": 5,
    "attack_count": 3,
    "defense_count": 2
  },
  "logs": [
    {
      "event_id": "20260213_143022_123456",
      "event_type": "attack",
      "attack": {
        "type": "HopSkipJump",
        "epsilon": 0.2,
        "queries": 145,
        "success": true
      },
      "model_behavior": {
        "original_prediction": 0,
        "adversarial_prediction": 1,
        "confidence": 0.63
      },
      "defense": {
        "isolation_flag": true,
        "final_decision": "DENY"
      }
    }
  ]
}
```

### Markdown Format (Human-Readable)

```markdown
# SOC/Red Team Session Log

**Session ID:** `20260213_143022`

## Session Metadata

| Metric | Value |
|--------|-------|
| total_events | 5 |
| attack_count | 3 |

## Event Log

### Event 1: ATTACK

- **Attack Type:** HopSkipJump
- **Parameters:**
  - epsilon: 0.2
  - queries: 145
```

### CSV Format (Spreadsheet-Friendly)

```csv
event_id,event_type,timestamp,attack_type,epsilon,success,defense_decision
20260213_143022_123456,attack,2026-02-13T14:30:22,HopSkipJump,0.2,True,DENY
```

---

## 🎓 Blue Team Analytics Output

### Vulnerability Report Example

```json
{
  "vulnerabilities": [
    {
      "type": "high_attack_success_rate",
      "severity": "high",
      "description": "Attack success rate is 65%, indicating weak defense"
    }
  ],
  "recommendations": [
    {
      "action": "increase_confidence_threshold",
      "description": "Increase threshold to 0.20",
      "expected_impact": "Reduce false negatives by 15-25%"
    }
  ]
}
```

### Hardening Config Example

```json
{
  "confidence_threshold": {
    "current": 0.15,
    "recommended": 0.20,
    "reason": "Attack success rate is 65%"
  },
  "enable_adversarial_retraining": {
    "recommended": true,
    "reason": "High attack success indicates need for robust training"
  }
}
```

### Retraining Dataset Specification

```json
{
  "purpose": "adversarial_retraining",
  "recommended_training_config": {
    "epsilon_values": [0.1, 0.15, 0.2, 0.25, 0.3],
    "attack_types_to_generate": ["HopSkipJump", "FastGradientMethod"],
    "augmentation_ratio": 0.2
  }
}
```

---

## 🔄 Complete Red vs Blue Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│                     RED TEAM PHASE                          │
│  1. Run attacks (HSJ, FGM)                                  │
│  2. Log attack events                                       │
│  3. Export logs (JSON/CSV/MD/TXT)                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    BLUE TEAM PHASE                          │
│  1. Load and parse logs                                     │
│  2. Analyze attack patterns                                 │
│  3. Identify vulnerabilities                                │
│  4. Generate hardening recommendations                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  DEFENSE HARDENING                          │
│  1. Increase confidence threshold                           │
│  2. Adjust Isolation Forest sensitivity                     │
│  3. Generate adversarial retraining dataset                 │
│  4. Retrain models with adversarial examples                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    RE-EVALUATION                            │
│  1. Run same attacks with hardened defenses                 │
│  2. Compare metrics (before/after)                          │
│  3. Verify improvements                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Advanced Usage

### Custom Logging for New Attack Types

```python
log_mgr.log_attack_event(
    attack_type="CustomAttack",
    attack_params={
        "custom_param_1": value1,
        "custom_param_2": value2
    },
    model_behavior={
        "original_prediction": pred_orig,
        "adversarial_prediction": pred_adv,
        "confidence": conf_score
    },
    defense_response={
        "isolation_flag": iso_flag,
        "final_decision": decision
    }
)
```

### Filtering Logs for Analysis

```python
analytics = BlueTeamAnalytics()
analytics.load_logs_from_file("logs/sessions/specific_session.json")

# Access loaded logs
attack_events = [
    log for log in analytics.logs_data
    if log.get('event_type') == 'attack' 
    and log.get('attack', {}).get('type') == 'HopSkipJump'
]
```

### Custom Report Generation

```python
report = analytics.generate_comprehensive_report()

# Access specific sections
attack_patterns = report['attack_patterns']
vulnerabilities = report['vulnerabilities']
executive_summary = report['executive_summary']

# Save custom sections
with open('custom_report.json', 'w') as f:
    json.dump(executive_summary, f, indent=2)
```

---

## 📂 Directory Structure

After running the system, you'll have:

```text
logs/
├── sessions/              # Exported session logs
│   ├── session_20260213_143022.json
│   ├── session_20260213_143022.txt
│   ├── session_20260213_143022.md
│   └── session_20260213_143022.csv
├── attacks/               # Attack-specific logs
└── analytics/             # Blue team analysis outputs
    ├── blue_team_report_20260213_150000.json
    ├── adversarial_samples_20260213_150000.json
    └── retraining_spec_20260213_150000.json
```

---

## ✅ Best Practices

1. **Always export logs after attacks**

   ```python
   log_mgr.export_logs(format='json')  # For analytics
   log_mgr.export_logs(format='md')    # For documentation
   ```

2. **Use wrapper functions for simplicity**
   - Less code
   - Automatic logging
   - Drop-in compatibility

3. **Run blue team analysis regularly**

   ```python
   # After each major attack campaign
   analyze_logs_and_generate_report()
   ```

4. **Apply hardening recommendations**

   ```python
   hardening = analytics.generate_hardening_config()
   # Update config.py with recommendations
   ```

5. **Track improvements over time**
   - Compare successive analysis reports
   - Monitor evasion rate trends
   - Validate defense effectiveness

---

## 🐛 Troubleshooting

### Issue: No logs exported

**Solution:** Check that events were actually logged:

```python
summary = log_mgr.get_session_summary()
print(f"Total events: {summary['total_events']}")
```

### Issue: Analytics finds no data

**Solution:** Verify log directory and file format:

```python
analytics = BlueTeamAnalytics(log_directory="logs/sessions")
num_files = analytics.load_all_logs(pattern="*.json")
print(f"Loaded {num_files} files")
```

### Issue: Import errors

**Solution:** Add path to sys.path:

```python
import sys
sys.path.insert(0, '/home/claude')  # or wherever files are located
```

---

## Directory Management

> [!NOTE]
> **Consolidated History**  
> Attack events are captured within the `logs/sessions/` files rather than `logs/attacks/`. This consolidation provides full "Blue Team" context, linking every attack attempt directly with the defense decision and model confidence scores in a single audit line. The `attacks/` directory is reserved for future raw vector exports.

## Key Benefits

- **Structured logging aligned with SOC-style workflows** - Industry-standard event tracking  
- **Multiple export formats** - JSON, TXT, Markdown, CSV  
- **Automated analysis** - Pattern detection, vulnerability identification  
- **Actionable recommendations** - Concrete defense hardening steps  
- **Adversarial retraining** - Auto-generate retraining datasets  
- **Minimal code changes** - Drop-in wrapper functions  
- **Complete feedback loop** - Attack → Analyze → Harden → Re-test  

---

## License

Included as part of this project.
