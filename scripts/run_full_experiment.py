"""
Comprehensive experiment runner for adversarial ML + Zero-Trust paper.

Produces all numbers required for Tables I–VI and Figures 5–9:
  - Clean accuracy, precision, recall, F1, ROC-AUC (mean ± std, 5 seeds)
  - FGSM / PGD ML evasion success rate (ESR) and ZT bypass rate
    at ε ∈ {0.01, 0.02, 0.05, 0.10, 0.20}
  - Ablation: ML-only / ML+DeviceTrust / ML+GeoRisk / Full ZT
  - Policy decision distribution (ALLOW / STEP_UP_AUTH / RATE_LIMIT / DENY)
  - t-statistic, p-value, Cohen's d  (ML-only vs full ZT bypass rate)
  - Constraint validity rate (% adversarial examples that are protocol-valid)
  - Training-loss curves (for Figure 8)

Usage:
    python scripts/run_full_experiment.py
"""

import os, sys, json, logging, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'experiment.log'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.unsw_nb15 import UNSWNB15Loader
from src.preprocessing.cicids_2017 import CICIDS2017Loader
from src.preprocessing.base import fix_seeds
from src.risk_engine.network_classifier import NetworkRiskClassifier
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.constraints.unsw_nb15_constraints import UNSWNB15Constraints
from src.attacks.constraints.cicids_constraints import CICIDS2017Constraints
from src.policy.zero_trust_engine import ZeroTrustEngine
from src.simulation.context_profiles import generate_attacker_context, generate_legitimate_context

# ─── Constants ────────────────────────────────────────────────────────────────
SEEDS          = [0, 42, 123, 456, 789]
EPSILONS       = [0.01, 0.02, 0.05, 0.10, 0.20]
TRAIN_EPOCHS   = 15
ATTACK_N       = 300     # attack samples for epsilon sweep
ABLATION_N     = 400     # attack samples for ablation
LEGIT_N        = 5000    # legitimate samples for FPR
ABLATION_EPS   = 0.05
CICIDS_MAX     = 150_000 # cap CICIDS rows for tractable run time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')
LOGS_DIR     = os.path.join(PROJECT_ROOT, 'logs')
for d in (RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

DATASET_COLORS = {'UNSW-NB15': '#1f77b4', 'CICIDS-2017': '#ff7f0e'}


# ─── Sampled CICIDS loader ─────────────────────────────────────────────────────

class SampledCICIDS2017Loader(CICIDS2017Loader):
    """CICIDS-2017 loader that caps total rows for tractable experiments."""

    def __init__(self, data_dir='data/cicids-2017', max_rows=CICIDS_MAX):
        super().__init__(data_dir=data_dir)
        self.max_rows = max_rows

    def load_and_preprocess(self, test_size=0.3):
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        logger.info(f"Loading CICIDS-2017 (cap={self.max_rows:,} rows) …")
        frames = []
        rows_loaded = 0
        for f in sorted(csv_files):
            remaining = self.max_rows - rows_loaded
            if remaining <= 0:
                break
            chunk = pd.read_csv(f, low_memory=False, nrows=remaining)
            chunk.columns = chunk.columns.str.strip()
            frames.append(chunk)
            rows_loaded += len(chunk)
            logger.info(f"  {os.path.basename(f)}: {len(chunk):,} rows  (total={rows_loaded:,})")

        df = pd.concat(frames, ignore_index=True)

        # ── Reuse CICIDS2017Loader preprocessing logic ──
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()

        df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)

        current_drop = [c for c in self.drop_cols if c in df.columns] + ['Label']
        df = df.drop(columns=current_drop)

        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df):,} duplicate rows")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].dtype != 'object' and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        zero_var = [c for c in df.columns if df[c].nunique() <= 1]
        df = df.drop(columns=zero_var)
        logger.info(f"Dropped {len(zero_var)} zero-variance columns")

        X = df.drop(columns=['label_binary']).values.astype(np.float32)
        y = df['label_binary'].values.astype(np.int32)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"Class distribution after SMOTE:  {np.bincount(y_train)}")

        return X_train, X_test, y_train, y_test, X_train.shape[1]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, epochs=TRAIN_EPOCHS,
                batch_size=256, device='cpu'):
    """Train model; return (trained_model, [epoch_loss, ...])."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.astype(np.float32))),
        batch_size=batch_size, shuffle=True
    )
    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by.view(-1, 1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        epoch_losses.append(total / len(loader))
    model.eval()
    return model, epoch_losses


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_scores):
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc':   float(roc_auc_score(y_true, y_scores)),
    }

def aggregate_seeds(records, keys):
    agg = {}
    for k in keys:
        vals = [r[k] for r in records]
        agg[k] = {'mean': float(np.mean(vals)),
                  'std':  float(np.std(vals, ddof=1)),
                  'values': vals}
    return agg


# ─── Enhanced ZT decision (4-way output) ──────────────────────────────────────

def zt_4way(ml_risk: float, ctx: dict) -> str:
    """
    ALLOW / STEP_UP_AUTH / RATE_LIMIT / DENY
    Extends the 2-way DENY/ALLOW engine with intermediate tiers.
    """
    device_trust = ctx.get('device_trust', 0.8)
    geo_risk     = ctx.get('geo_risk', 0.2)
    identity_ok  = ctx.get('identity_verified', True)
    resource_s   = ctx.get('resource_sensitivity', 0.5)
    off_hours    = ctx.get('time_of_day', 12) not in range(8, 19)

    if ml_risk > 0.85:                               return 'DENY'
    if device_trust < 0.3:                           return 'DENY'
    if geo_risk > 0.8:                               return 'DENY'
    if resource_s > 0.9 and device_trust < 0.5:     return 'DENY'
    if ml_risk > 0.5  and geo_risk > 0.6:           return 'DENY'

    if ml_risk > 0.50:                               return 'STEP_UP_AUTH'
    if not identity_ok and ml_risk > 0.40:           return 'STEP_UP_AUTH'
    if off_hours and device_trust < 0.50:            return 'STEP_UP_AUTH'

    if ml_risk > 0.30 and geo_risk > 0.55:          return 'RATE_LIMIT'
    if ml_risk > 0.40:                               return 'RATE_LIMIT'

    return 'ALLOW'


# ─── Statistical helpers ──────────────────────────────────────────────────────

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                     (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# ─── Constraint validity ──────────────────────────────────────────────────────

def constraint_validity(x_adv_t, constraint_fn):
    proj  = constraint_fn.project(x_adv_t.clone(), x_adv_t.clone())
    valid = torch.all(torch.abs(x_adv_t - proj) < 1e-5, dim=1).float().mean().item()
    return float(valid)


# ─── Core adversarial helpers ─────────────────────────────────────────────────

def make_evasion_targets(n, device):
    """Target label = 0 (benign) for evasion attacks."""
    return torch.zeros(n, dtype=torch.float32).to(device)


def eval_esr_and_bypass(model, x_adv_t, device, zt_engine, seed):
    """
    Returns (esr, zt_bypass_rate).
      esr         = fraction of adversarial samples predicted benign by ML
      zt_bypass   = fraction allowed by ZT engine (attacker context)
    """
    with torch.no_grad():
        adv_scores = model(x_adv_t.to(device)).cpu().numpy().squeeze()
    adv_preds = (adv_scores > 0.5).astype(int)
    esr = float(np.mean(adv_preds == 0))

    n = len(adv_scores)
    ctxs = generate_attacker_context(n, seed=seed)
    decs = zt_engine.evaluate_batch(adv_scores.tolist(), ctxs)
    bypass = float(np.mean([d.decision == 'ALLOW' for d in decs]))
    return esr, bypass


# ─── Main per-dataset experiment ──────────────────────────────────────────────

def run_experiment(dataset_name, loader_class, constraint_class, data_dir=None):
    logger.info(f"\n{'='*60}\n  DATASET: {dataset_name}\n{'='*60}")

    # ── 1. Load data ──
    loader = loader_class() if data_dir is None else loader_class(data_dir=data_dir)
    X_train, X_test, y_train, y_test, input_dim = loader.load_and_preprocess()
    logger.info(f"Train {X_train.shape}  Test {X_test.shape}  input_dim={input_dim}")
    logger.info(f"Train labels: benign={int((y_train==0).sum())}  attack={int((y_train==1).sum())}")
    logger.info(f"Test  labels: benign={int((y_test==0).sum())}  attack={int((y_test==1).sum())}")

    feat_names  = getattr(loader, 'feature_names', [f'f{i}' for i in range(input_dim)])
    constraints = constraint_class(feat_names)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_test_t = torch.FloatTensor(X_test).to(device)

    # Fixed attack-sample indices (same across seeds for comparability)
    rng_s     = np.random.default_rng(42)
    atk_idx   = np.where(y_test == 1)[0]
    n_atk     = min(ATTACK_N, len(atk_idx))
    chosen    = rng_s.choice(atk_idx, n_atk, replace=False)
    X_atk_np  = X_test[chosen].copy()
    X_atk_t   = torch.FloatTensor(X_atk_np).to(device)
    y_evasion = make_evasion_targets(n_atk, device)

    leg_idx   = np.where(y_test == 0)[0]
    n_leg     = min(LEGIT_N, len(leg_idx))
    leg_chosen = rng_s.choice(leg_idx, n_leg, replace=False)
    X_leg_t   = torch.FloatTensor(X_test[leg_chosen]).to(device)

    zt_full = ZeroTrustEngine()
    zt_ml   = ZeroTrustEngine(enabled_factors=set())

    # ── 2. Multi-seed training + clean metrics ──
    seed_clean  = []
    train_curves = {}

    for seed in SEEDS:
        logger.info(f"  [Seed {seed}] training …")
        fix_seeds(seed)
        m = NetworkRiskClassifier(input_dim=input_dim)
        m, losses = train_model(m, X_train, y_train, device=device)
        train_curves[seed] = losses

        with torch.no_grad():
            sc = m(X_test_t).cpu().numpy().squeeze()
        preds = (sc > 0.5).astype(int)
        met = compute_metrics(y_test, preds, sc)
        met['seed'] = seed
        seed_clean.append(met)
        logger.info(f"    Acc={met['accuracy']:.4f} Prec={met['precision']:.4f} "
                    f"Rec={met['recall']:.4f} F1={met['f1']:.4f} AUC={met['roc_auc']:.4f}")

    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    agg_clean   = aggregate_seeds(seed_clean, metric_keys)

    # ── 3. Train representative model (seed=42) for sweep + ablation ──
    logger.info(f"  Training representative model (seed=42) for adversarial eval …")
    fix_seeds(42)
    model42 = NetworkRiskClassifier(input_dim=input_dim)
    model42, _ = train_model(model42, X_train, y_train, device=device)

    # ── 4. Epsilon sweep ──
    eps_results = []
    for eps in EPSILONS:
        logger.info(f"  Epsilon sweep ε={eps}")
        row = {'epsilon': eps}

        for name, adv_fn in [
            ('FGSM', lambda eps: fgsm_attack(
                model42, device, X_atk_t.clone(), y_evasion, eps, constraints)),
            ('PGD',  lambda eps: pgd_attack(
                model42, device, X_atk_t.clone(), y_evasion, eps,
                alpha=eps/4, num_iter=10, constraint_fn=constraints)),
        ]:
            X_adv = adv_fn(eps)
            esr, zt_bypass = eval_esr_and_bypass(model42, X_adv, device, zt_full, seed=42)
            val = constraint_validity(X_adv, constraints)
            row[name] = {'esr': esr, 'zt_bypass': zt_bypass, 'validity': val}
            logger.info(f"    {name}: ESR={esr:.4f}  ZT_bypass={zt_bypass:.4f}  validity={val:.4f}")

        eps_results.append(row)

    # ── 5. Ablation study ──
    logger.info(f"  Ablation study (FGSM ε={ABLATION_EPS}) …")
    abl_chosen = rng_s.choice(atk_idx, min(ABLATION_N, len(atk_idx)), replace=False)
    X_abl_t    = torch.FloatTensor(X_test[abl_chosen]).to(device)
    y_abl_ev   = make_evasion_targets(len(abl_chosen), device)
    X_abl_adv  = fgsm_attack(model42, device, X_abl_t.clone(), y_abl_ev, ABLATION_EPS, constraints)

    with torch.no_grad():
        abl_scores = model42(X_abl_adv.to(device)).cpu().numpy().squeeze()
    with torch.no_grad():
        leg_scores = model42(X_leg_t).cpu().numpy().squeeze()

    abl_atk_ctx = generate_attacker_context(len(abl_chosen), seed=42)
    abl_leg_ctx = generate_legitimate_context(n_leg, seed=100)

    ablation_results = []
    for config_name, factors in ZeroTrustEngine.ABLATION_CONFIGS.items():
        eng = ZeroTrustEngine(enabled_factors=factors)
        adv_decs  = eng.evaluate_batch(abl_scores.tolist(), abl_atk_ctx)
        leg_decs  = eng.evaluate_batch(leg_scores.tolist(), abl_leg_ctx)
        deny_rate = float(np.mean([d.decision == 'DENY' for d in adv_decs]))
        fpr       = float(np.mean([d.decision == 'DENY' for d in leg_decs]))
        ablation_results.append({
            'config':      config_name,
            'deny_rate':   deny_rate,
            'fpr':         fpr,
            'bypass_rate': 1.0 - deny_rate,
        })
        logger.info(f"    {config_name:20s} deny={deny_rate:.4f}  fpr={fpr:.4f}  bypass={1-deny_rate:.4f}")

    # ── 6. Policy decision distribution (full test set) ──
    logger.info(f"  Policy decision distribution …")
    with torch.no_grad():
        all_scores = model42(X_test_t).cpu().numpy().squeeze()

    leg_ctxs = generate_legitimate_context(int((y_test == 0).sum()), seed=200)
    atk_ctxs = generate_attacker_context(int((y_test == 1).sum()), seed=201)

    policy_counts = {'ALLOW': 0, 'STEP_UP_AUTH': 0, 'RATE_LIMIT': 0, 'DENY': 0}
    li, ai = 0, 0
    for score, lbl in zip(all_scores, y_test):
        ctx = leg_ctxs[li] if lbl == 0 else atk_ctxs[ai]
        if lbl == 0: li += 1
        else:        ai += 1
        policy_counts[zt_4way(float(score), ctx)] += 1

    total = int(len(y_test))
    policy_dist = {k: {'count': v, 'fraction': v / total}
                   for k, v in policy_counts.items()}
    logger.info("    " + "  ".join(f"{k}={v['count']}" for k, v in policy_dist.items()))

    # ── 7. Statistical tests: ML-only vs full ZT bypass (per seed, FGSM ε=0.05) ──
    logger.info(f"  Statistical tests (5 seeds × FGSM ε=0.05) …")
    ml_bypass_list, zt_bypass_list = [], []
    for seed in SEEDS:
        fix_seeds(seed)
        ms = NetworkRiskClassifier(input_dim=input_dim)
        ms, _ = train_model(ms, X_train, y_train, device=device)
        X_adv_s = fgsm_attack(ms, device, X_atk_t.clone(), y_evasion, 0.05, constraints)
        with torch.no_grad():
            sc_s = ms(X_adv_s.to(device)).cpu().numpy().squeeze()
        ctxs_s = generate_attacker_context(n_atk, seed=seed)
        ml_decs = zt_ml.evaluate_batch(sc_s.tolist(), ctxs_s)
        zt_decs = zt_full.evaluate_batch(sc_s.tolist(), ctxs_s)
        ml_bypass_list.append(float(np.mean([d.decision == 'ALLOW' for d in ml_decs])))
        zt_bypass_list.append(float(np.mean([d.decision == 'ALLOW' for d in zt_decs])))
        logger.info(f"    seed={seed}  ml_bypass={ml_bypass_list[-1]:.4f}  zt_bypass={zt_bypass_list[-1]:.4f}")

    t_stat, p_val = stats.ttest_rel(ml_bypass_list, zt_bypass_list)
    d_val = cohens_d(ml_bypass_list, zt_bypass_list)
    stat_tests = {
        'ml_only_bypass': ml_bypass_list,
        'full_zt_bypass': zt_bypass_list,
        't_statistic':    float(t_stat),
        'p_value':        float(p_val),
        'cohens_d':       float(d_val),
    }
    logger.info(f"    t={t_stat:.4f}  p={p_val:.6f}  d={d_val:.4f}")

    return {
        'dataset':            dataset_name,
        'n_train':            int(X_train.shape[0]),
        'n_test':             int(X_test.shape[0]),
        'input_dim':          int(input_dim),
        'clean_per_seed':     seed_clean,
        'clean_aggregate':    agg_clean,
        'epsilon_sweep':      eps_results,
        'ablation':           ablation_results,
        'policy_distribution': policy_dist,
        'stat_tests':         stat_tests,
        'training_curves':    {str(k): v for k, v in train_curves.items()},
    }


# ─── Figure generation ────────────────────────────────────────────────────────

def fig5_baseline(results_list):
    metric_keys   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        for i, res in enumerate(results_list):
            v = res['clean_aggregate'][mk]
            ax.bar(i, v['mean'], yerr=v['std'], capsize=6,
                   color=DATASET_COLORS[res['dataset']], alpha=0.85, width=0.4)
        ax.set_ylim(0, 1.08)
        ax.set_title(ml, fontsize=10)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
    handles = [plt.Rectangle((0,0),1,1,color=DATASET_COLORS[r['dataset']]) for r in results_list]
    fig.legend(handles, [r['dataset'] for r in results_list],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Fig. 5 — Baseline Classifier Performance (mean ± std, 5 seeds)', fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig5_baseline_performance.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {p}")


def fig6_epsilon_sweep(results_list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, atk in enumerate(['FGSM', 'PGD']):
        ax = axes[col]
        for res in results_list:
            eps = [r['epsilon']          for r in res['epsilon_sweep']]
            esr = [r[atk]['esr']         for r in res['epsilon_sweep']]
            byp = [r[atk]['zt_bypass']   for r in res['epsilon_sweep']]
            c   = DATASET_COLORS[res['dataset']]
            ax.plot(eps, esr, 'o-',  color=c, lw=2, label=f"{res['dataset']} ESR")
            ax.plot(eps, byp, 's--', color=c, lw=2, alpha=0.7, label=f"{res['dataset']} ZT Bypass")
        ax.set_title(f'{atk}', fontsize=11)
        ax.set_xlabel('ε'); ax.set_ylabel('Rate'); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Fig. 6 — Epsilon Sweep: ML ESR vs ZT Bypass Rate', fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig6_epsilon_sweep.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {p}")


def fig7_ablation(results_list):
    configs = ['ml_only', 'ml_device', 'ml_geo', 'full']
    clabels = ['ML Only', 'ML+Device', 'ML+Geo', 'Full ZT']
    pairs   = [('deny_rate','Deny Rate'), ('fpr','FPR'), ('bypass_rate','Bypass Rate')]
    x = np.arange(len(configs)); w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col, (mk, ml) in enumerate(pairs):
        ax = axes[col]
        for di, res in enumerate(results_list):
            vals = [next((r[mk] for r in res['ablation'] if r['config']==c), 0) for c in configs]
            ax.bar(x + (di-0.5)*w, vals, w,
                   color=DATASET_COLORS[res['dataset']], alpha=0.85,
                   label=res['dataset'])
        ax.set_title(ml, fontsize=11); ax.set_xticks(x); ax.set_xticklabels(clabels, rotation=15, fontsize=9)
        ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
        if col == 0: ax.legend()
    fig.suptitle('Fig. 7 — Zero-Trust Ablation Study (FGSM ε=0.05)', fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig7_ablation.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {p}")


def fig8_training_curves(results_list):
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]
    for ax, res in zip(axes, results_list):
        for seed, losses in res['training_curves'].items():
            ax.plot(range(1, len(losses)+1), losses, alpha=0.8, label=f'seed={seed}')
        ax.set_title(res['dataset'], fontsize=11)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss (BCE)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Fig. 8 — Training Loss Curves (all 5 seeds)', fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig8_training_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {p}")


def fig9_fgsm_vs_pgd(results_list):
    keys  = ['esr', 'zt_bypass']
    klabs = ['ML Evasion Success Rate', 'ZT Bypass Rate']
    fig, axes = plt.subplots(len(keys), len(results_list),
                              figsize=(6*len(results_list), 4.5*len(keys)))
    if len(results_list) == 1:
        axes = [[axes[r]] for r in range(len(keys))]
    for row, (mk, ml) in enumerate(zip(keys, klabs)):
        for col, res in enumerate(results_list):
            ax  = axes[row][col]
            eps = [r['epsilon']       for r in res['epsilon_sweep']]
            f   = [r['FGSM'][mk]     for r in res['epsilon_sweep']]
            p_  = [r['PGD'][mk]      for r in res['epsilon_sweep']]
            ax.plot(eps, f,  'o-',  lw=2, label='FGSM')
            ax.plot(eps, p_, 's--', lw=2, label='PGD')
            ax.set_title(f"{res['dataset']} — {ml}", fontsize=10)
            ax.set_xlabel('ε'); ax.set_ylabel(ml)
            ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Fig. 9 — FGSM vs PGD: ML ESR and ZT Bypass Rate', fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig9_fgsm_vs_pgd.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {p}")


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(all_results):
    SEP = '=' * 72
    print(f"\n{SEP}\n  PAPER NUMBERS — ALL DATASETS\n{SEP}")
    for res in all_results:
        ds = res['dataset']
        print(f"\n{'─'*72}")
        print(f"  {ds}   train={res['n_train']:,}  test={res['n_test']:,}  features={res['input_dim']}")
        print(f"{'─'*72}")

        # Table I: clean performance
        print("\n  Table I — Clean Performance (mean ± std, 5 seeds)")
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            v = res['clean_aggregate'][k]
            print(f"    {k:12s}: {v['mean']:.4f} ± {v['std']:.4f}")

        # Tables II/III: epsilon sweep
        print("\n  Tables II/III — Epsilon Sweep (ML ESR / ZT Bypass / Constraint Validity)")
        print(f"  {'Atk':<5} {'ε':>5} {'   ESR':>7} {'ZT_Bypass':>10} {'Validity':>9}")
        print(f"  {'-'*41}")
        for row in res['epsilon_sweep']:
            for atk in ['FGSM', 'PGD']:
                v = row[atk]
                print(f"  {atk:<5} {row['epsilon']:>5.2f}  {v['esr']:>6.4f}  "
                      f"{v['zt_bypass']:>9.4f}  {v['validity']:>8.4f}")

        # Table IV: ablation
        print("\n  Table IV — Ablation Study (FGSM ε=0.05)")
        print(f"  {'Config':<22} {'Deny%':>7} {'FPR%':>7} {'Bypass%':>8}")
        print(f"  {'-'*46}")
        for a in res['ablation']:
            print(f"  {a['config']:<22} {a['deny_rate']:>6.4f}  "
                  f"{a['fpr']:>6.4f}  {a['bypass_rate']:>7.4f}")

        # Table V: policy distribution
        print("\n  Table V — Policy Decision Distribution (full test set)")
        for k, v in res['policy_distribution'].items():
            print(f"    {k:<16}: {v['count']:8,}  ({v['fraction']*100:5.2f}%)")

        # Table VI: statistics
        st = res['stat_tests']
        print("\n  Table VI — Statistical Tests (ML-only vs Full ZT, FGSM ε=0.05)")
        print(f"    ML-only bypass / seed: " +
              "  ".join(f"{x:.4f}" for x in st['ml_only_bypass']))
        print(f"    Full ZT bypass / seed: " +
              "  ".join(f"{x:.4f}" for x in st['full_zt_bypass']))
        print(f"    t-statistic : {st['t_statistic']:>10.4f}")
        print(f"    p-value     : {st['p_value']:>12.6f}")
        print(f"    Cohen's d   : {st['cohens_d']:>10.4f}")
    print(f"\n{SEP}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    all_results = []

    # Use absolute paths so the script works regardless of which directory it's run from
    unsw_data_dir   = os.path.join(PROJECT_ROOT, 'data', 'unsw-nb15')
    cicids_data_dir = os.path.join(PROJECT_ROOT, 'data', 'cicids-2017')

    for ds_name, loader_cls, constr_cls, data_dir in [
        ('UNSW-NB15',   UNSWNB15Loader,         UNSWNB15Constraints,   unsw_data_dir),
        ('CICIDS-2017', SampledCICIDS2017Loader, CICIDS2017Constraints, cicids_data_dir),
    ]:
        try:
            res = run_experiment(ds_name, loader_cls, constr_cls, data_dir=data_dir)
            all_results.append(res)
        except Exception as exc:
            logger.error(f"{ds_name} failed: {exc}", exc_info=True)

    if not all_results:
        logger.error("No results — aborting."); return

    # Save JSON
    out = os.path.join(RESULTS_DIR, 'full_experiment_results.json')
    with open(out, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    logger.info(f"Results saved → {out}")

    # Figures
    logger.info("\nGenerating figures …")
    fig5_baseline(all_results)
    fig6_epsilon_sweep(all_results)
    fig7_ablation(all_results)
    fig8_training_curves(all_results)
    fig9_fgsm_vs_pgd(all_results)

    # Print
    print_summary(all_results)
    logger.info("Done.  Check results/ and figures/ directories.")


if __name__ == '__main__':
    main()
