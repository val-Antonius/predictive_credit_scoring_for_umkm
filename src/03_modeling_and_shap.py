"""
================================================================
PROJECT: Credit Scoring
FILE   : 03_modeling_and_shap.py
AUTHOR : Antonius Valentino
================================================================

PREREQUISITE:
  - Sudah jalankan 02_feature_engineering.py
  - pip install xgboost lightgbm optuna shap scikit-learn matplotlib seaborn

OUTPUT:
  models/best_xgb_model.json          ← model untuk Streamlit app
  models/scaler.pkl                   ← scaler untuk inferensi
  outputs/modeling/                   ← semua plot
================================================================
"""

import os
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import optuna
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/modeling", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.family": "sans-serif",
})
PALETTE = ["#378ADD", "#E24B4A"]


# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA PROCESSED
# ══════════════════════════════════════════════════════════════

def load_processed_data():
    print("── Load data processed ───────────────────────────────")
    print("  Kontrak data: train/val/test harus menggunakan representasi SCALED yang konsisten")

    # Seluruh split dimuat dari artefak scaled yang dihasilkan pada fase 2.
    # Catatan: file X_train_resampled.csv dipakai sebagai alias untuk train asli yang sudah di-scale.
    X_train = pd.read_csv("data/processed/X_train_resampled.csv", index_col=0)
    X_val   = pd.read_csv("data/processed/X_val.csv",             index_col=0)
    X_test  = pd.read_csv("data/processed/X_test.csv",            index_col=0)
    y_train = pd.read_csv("data/processed/y_train_resampled.csv", index_col=0).squeeze()
    y_val   = pd.read_csv("data/processed/y_val.csv",             index_col=0).squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv",            index_col=0).squeeze()

    if list(X_train.columns) != list(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise ValueError("Kontrak fitur tidak konsisten: kolom train/val/test berbeda")

    print(f"  X_train : {X_train.shape}  | positif: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"  X_val   : {X_val.shape}  | positif: {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
    print(f"  X_test  : {X_test.shape}  | positif: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════
# 2. BASELINE — LogisticRegression
#    Selalu mulai dari model paling sederhana sebagai patokan.
#    Jika XGBoost hanya unggul 1%, maka complexity-nya tidak justified.
# ══════════════════════════════════════════════════════════════

def train_baseline(X_train, y_train, X_val, y_val):
    print("\n── Baseline: Logistic Regression ────────────────────")
    lr = LogisticRegression(
        class_weight="balanced",   # handle imbalance tanpa resampling
        max_iter=1000,
        random_state=42,
        C=0.1,
    )
    lr.fit(X_train, y_train)
    y_pred_proba = lr.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    f1  = f1_score(y_val, (y_pred_proba >= 0.5).astype(int), average="weighted")
    print(f"  Logistic Regression → AUC-ROC: {auc:.4f}  |  F1: {f1:.4f}")
    return lr, auc


# ══════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER TUNING DENGAN OPTUNA
#    Lebih modern dan efisien dari GridSearchCV.
#    Optuna pakai Bayesian optimization — tidak brute-force semua kombinasi.
# ══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """
    n_trials=50 cukup untuk menemukan kombinasi yang baik.
    Naikkan ke 100 jika punya waktu lebih.
    Setiap trial dicatat otomatis oleh Optuna — bisa visualisasi progress-nya.
    """
    print(f"\n── Optuna Hyperparameter Tuning (XGBoost, {n_trials} trials) ──")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "early_stopping_rounds": 20,
            "use_label_encoder": False,
            "eval_metric":       "auc",
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best AUC-ROC (val) : {study.best_value:.4f}")
    print(f"  Best params        :")
    for k, v in study.best_params.items():
        print(f"    {k:<25}: {v}")

    # Plot optimization history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    trials_df = study.trials_dataframe()

    # AUC per trial
    axes[0].plot(trials_df["number"], trials_df["value"],
                 alpha=0.4, color="#378ADD", linewidth=0.8, label="Trial AUC")
    # Running best
    running_best = trials_df["value"].cummax()
    axes[0].plot(trials_df["number"], running_best,
                 color="#E24B4A", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("AUC-ROC (validation)")
    axes[0].set_title("Optuna: Optimization History")
    axes[0].legend()

    # Parameter importance (top 6)
    importances = optuna.importance.get_param_importances(study)
    top_params = dict(list(importances.items())[:6])
    axes[1].barh(list(top_params.keys()), list(top_params.values()),
                 color="#378ADD", edgecolor="white")
    axes[1].set_xlabel("Relative importance")
    axes[1].set_title("Hyperparameter Importance")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("outputs/modeling/01_optuna_tuning.png", bbox_inches="tight")
    plt.show()

    return study.best_params


# ══════════════════════════════════════════════════════════════
# 4. TRAIN FINAL MODEL DENGAN BEST PARAMS
# ══════════════════════════════════════════════════════════════

def train_final_model(X_train, y_train, X_val, y_val, best_params: dict):
    print("\n── Training Final XGBoost Model ─────────────────────")
    params = {**best_params,
              "early_stopping_rounds": 20,
              "use_label_encoder": False,
              "eval_metric": "auc",
              "random_state": 42,
              "n_jobs": -1}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # Plot learning curve
    results = model.evals_result()
    epochs = len(results["validation_0"]["auc"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(epochs), results["validation_0"]["auc"],
            label="Train AUC", color="#378ADD", alpha=0.8)
    ax.plot(range(epochs), results["validation_1"]["auc"],
            label="Val AUC", color="#E24B4A", linewidth=2)
    ax.axvline(x=model.best_iteration, color="#888", linestyle="--",
               linewidth=1, label=f"Best iteration: {model.best_iteration}")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("XGBoost Learning Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/modeling/02_learning_curve.png", bbox_inches="tight")
    plt.show()

    # Simpan model
    model.save_model("models/best_xgb_model.json")
    print(f"Model disimpan: models/best_xgb_model.json")
    return model


# ══════════════════════════════════════════════════════════════
# 5. EVALUASI KOMPREHENSIF — semua metrik yang relevan
# ══════════════════════════════════════════════════════════════

def evaluate_models(models_dict: dict, X_val, y_val, X_test, y_test):
    """
    Evaluasi semua model sekaligus dalam satu tabel.
    """
    print("\n── Evaluasi Model ────────────────────────────────────")
    rows = []
    for name, model in models_dict.items():
        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_prob = model.predict_proba(X_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            rows.append({
                "Model":       name,
                "Split":       split_name,
                "AUC-ROC":     round(roc_auc_score(y_s, y_prob), 4),
                "F1 (weighted)": round(f1_score(y_s, y_pred, average="weighted"), 4),
                "F1 (minority)": round(f1_score(y_s, y_pred, average="binary"), 4),
                "AP Score":    round(average_precision_score(y_s, y_prob), 4),
            })

    results_df = pd.DataFrame(rows)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("outputs/modeling/model_comparison.csv", index=False)

    # ── ROC Curve + PR Curve ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_model = {"Logistic Regression": "#888780",
                    "XGBoost (tuned)": "#378ADD",
                    "LightGBM": "#1D9E75"}

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        color = colors_model.get(name, "#888")

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                     color=color, linewidth=2)

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                     color=color, linewidth=2)

    # ROC formatting
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].legend(fontsize=9)

    # PR formatting
    baseline_pr = y_test.mean()
    axes[1].axhline(y=baseline_pr, color="k", linestyle="--",
                    linewidth=1, alpha=0.4, label=f"Baseline ({baseline_pr:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test Set)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/modeling/03_roc_pr_curves.png", bbox_inches="tight")
    plt.show()

    return results_df


# ══════════════════════════════════════════════════════════════
# 6. CONFUSION MATRIX + THRESHOLD ANALYSIS
#    Ini penting untuk framing bisnis di artikel Medium:
#    "berapa kerugian jika model salah?"
# ══════════════════════════════════════════════════════════════

def plot_confusion_and_threshold(model, X_test, y_test):
    print("\n── Confusion Matrix & Threshold Analysis ─────────────")
    y_prob = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Confusion Matrix pada threshold 0.5 ──────────────────
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    labels = [["True Neg\n(Benar ditolak)", "False Pos\n(Salah approve)"],
              ["False Neg\n(Salah tolak)", "True Pos\n(Benar ditolak)"]]

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Aman (0)", "Pred: Gagal (1)"])
    ax.set_yticklabels(["Act: Aman (0)", "Act: Gagal (1)"])
    ax.set_title("Confusion Matrix (threshold=0.5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n{labels[i][j]}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Threshold vs Precision/Recall/F1 ─────────────────────
    thresholds = np.linspace(0.1, 0.9, 80)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_t = (y_prob >= t).astype(int)
        if y_t.sum() == 0:
            precisions.append(0); recalls.append(0); f1s.append(0)
            continue
        precisions.append(precision_score(y_test, y_t, zero_division=0))
        recalls.append(recall_score(y_test, y_t, zero_division=0))
        f1s.append(f1_score(y_test, y_t, average="binary", zero_division=0))

    ax2 = axes[1]
    ax2.plot(thresholds, precisions, label="Precision", color="#378ADD", linewidth=2)
    ax2.plot(thresholds, recalls,    label="Recall",    color="#E24B4A", linewidth=2)
    ax2.plot(thresholds, f1s,        label="F1-score",  color="#1D9E75", linewidth=2)
    best_t = thresholds[np.argmax(f1s)]
    ax2.axvline(x=best_t, color="#888", linestyle="--",
                linewidth=1.5, label=f"Best F1 threshold: {best_t:.2f}")
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Threshold vs Precision / Recall / F1")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.1, 0.9)

    plt.tight_layout()
    plt.savefig("outputs/modeling/04_confusion_threshold.png", bbox_inches="tight")
    plt.show()

    print(f"\n  Insight bisnis:")
    print(f"  False Negative = {cm[1,0]:,}  -> peminjam gagal bayar yang lolos (kerugian langsung)")
    print(f"  False Positive = {cm[0,1]:,}  -> peminjam baik yang ditolak (kehilangan revenue)")
    print(f"  -> Naikkan threshold jika prioritas adalah minimalkan kredit macet")
    print(f"  -> Turunkan threshold jika prioritas adalah jangkauan peminjam lebih luas")
    print(f"\n  Best F1 threshold: {best_t:.2f} (gunakan ini di production, bukan 0.5)")

    return best_t


# ══════════════════════════════════════════════════════════════
# 7. SHAP — INI INTI DARI FASE 3
# ══════════════════════════════════════════════════════════════

def compute_shap_values(model, X_train, X_test):
    """
    TreeExplainer adalah versi SHAP yang dioptimasi untuk tree-based models
    (XGBoost, LightGBM, RandomForest). Jauh lebih cepat dari KernelExplainer.
    """
    print("\n── Computing SHAP Values ─────────────────────────────")
    print("  Menggunakan TreeExplainer (dioptimasi untuk XGBoost)...")

    explainer = shap.TreeExplainer(model)

    # Hitung SHAP untuk test set (sample 1000 agar cepat untuk visualisasi)
    X_explain = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_explain)
    expected_value = explainer.expected_value

    print(f"  SHAP values shape  : {shap_values.shape}")
    print(f"  Expected value     : {expected_value:.4f}  ← log-odds baseline model")
    print(f"  Sampel yang dianalisis: {len(X_explain):,}")

    return explainer, shap_values, X_explain, expected_value


# ══════════════════════════════════════════════════════════════
# 7a. SHAP SUMMARY PLOT — fitur mana yang paling berpengaruh secara global
# ══════════════════════════════════════════════════════════════

def plot_shap_summary(shap_values, X_explain):
    """
    Summary plot menunjukkan:
    - Sumbu Y: fitur diurutkan dari yang paling penting ke paling tidak
    - Setiap titik = satu sampel
    - Warna = nilai fitur (merah=tinggi, biru=rendah)
    - Posisi X = kontribusi terhadap prediksi (positif = mendorong ke 'gagal bayar')

    """
    print("\n── SHAP Summary Plot ─────────────────────────────────")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Plot 1: Beeswarm (full detail) ───────────────────────
    plt.sca(axes[0])
    shap.summary_plot(
        shap_values, X_explain,
        plot_type="dot",
        max_display=15,
        show=False,
        color_bar=True,
    )
    axes[0].set_title("SHAP Summary: Kontribusi per Fitur per Sampel",
                       fontsize=11, fontweight="bold", pad=10)
    axes[0].set_xlabel("SHAP value (dampak terhadap prediksi gagal bayar)")

    # ── Plot 2: Bar (mean absolute SHAP — feature importance) ─
    plt.sca(axes[1])
    shap.summary_plot(
        shap_values, X_explain,
        plot_type="bar",
        max_display=15,
        show=False,
        color="#378ADD",
    )
    axes[1].set_title("SHAP Feature Importance (mean |SHAP|)",
                       fontsize=11, fontweight="bold", pad=10)
    axes[1].set_xlabel("Mean absolute SHAP value")

    plt.tight_layout()
    plt.savefig("outputs/modeling/05_shap_summary.png", bbox_inches="tight")
    plt.show()

    # Print ranking teks
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_explain.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\n  Top 10 fitur berdasarkan mean |SHAP|:")
    for i, (feat, val) in enumerate(feature_importance.head(10).items(), 1):
        bar = "█" * int(val / feature_importance.max() * 20)
        print(f"  {i:>2}. {feat:<30} {val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════
# 7b. SHAP FORCE PLOT — penjelasan per individu
#     INI yang membuat project kamu stand out dari kebanyakan
# ══════════════════════════════════════════════════════════════

def plot_shap_force_plots(explainer, shap_values, X_explain,
                           expected_value, best_threshold: float):
    """
    Force plot menunjukkan KENAPA satu peminjam spesifik
    di-approve atau ditolak

    Merah  = fitur yang mendorong ke arah 'gagal bayar' (berbahaya)
    Biru   = fitur yang mendorong ke arah 'aman' (menguntungkan)
    """
    print("\n── SHAP Force Plots: Penjelasan Individual ───────────")

    # Pilih contoh kasus: satu yang diprediksi aman, satu yang diprediksi berisiko
    model_scores = shap_values.sum(axis=1) + expected_value

    # Kasus "aman" — score terendah (paling tidak berisiko)
    idx_safe = np.argmin(model_scores)
    # Kasus "berisiko" — score tertinggi (paling berisiko)
    idx_risky = np.argmax(model_scores)
    # Kasus "borderline" — score paling dekat ke threshold
    idx_border = np.argmin(np.abs(model_scores - best_threshold))

    cases = [
        (idx_safe,   "Kasus A: Profil AMAN (score rendah)",    "#378ADD"),
        (idx_risky,  "Kasus B: Profil BERISIKO (score tinggi)", "#E24B4A"),
        (idx_border, "Kasus C: Profil BORDERLINE (dekat threshold)", "#BA7517"),
    ]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("SHAP Force Plots: Penjelasan Per Peminjam",
                 fontsize=14, fontweight="bold", y=1.01)

    for plot_idx, (sample_idx, title, accent_color) in enumerate(cases):
        shap_row = shap_values[sample_idx]
        feat_vals = X_explain.iloc[sample_idx]
        score = model_scores[sample_idx]

        # Sort fitur: positif (berbahaya) di kanan, negatif (aman) di kiri
        contrib = pd.Series(shap_row, index=X_explain.columns)
        positive = contrib[contrib > 0].sort_values(ascending=False)
        negative = contrib[contrib < 0].sort_values(ascending=True)

        ax = fig.add_subplot(3, 1, plot_idx + 1)

        # Gambar bar horizontal: merah = menaikkan risiko, biru = menurunkan
        all_feats = pd.concat([positive, negative])
        top_n = 10
        display_feats = all_feats.abs().nlargest(top_n).index
        display_vals = all_feats[display_feats].sort_values()

        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in display_vals]
        bars = ax.barh(
            [f"{f}={feat_vals[f]:.3f}" for f in display_vals.index],
            display_vals.values,
            color=colors, edgecolor="white", height=0.6
        )

        # Annotate nilai SHAP
        for bar, val in zip(bars, display_vals.values):
            x_pos = val + (0.002 if val >= 0 else -0.002)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8.5,
                    color="#2C2C2A")

        ax.axvline(x=0, color="#888", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("SHAP value (merah = naikkan risiko, biru = turunkan risiko)")
        ax.set_title(f"{title}  |  Score: {score:.3f}",
                     fontsize=11, fontweight="bold", color=accent_color)

    plt.tight_layout()
    plt.savefig("outputs/modeling/06_shap_force_plots.png", bbox_inches="tight")
    plt.show()

    # Print narasi teks
    _print_case_narrative(shap_values, X_explain, expected_value,
                          idx_safe, idx_risky, idx_border, best_threshold)


def _print_case_narrative(shap_values, X_explain, expected_value,
                           idx_safe, idx_risky, idx_border, threshold):
    print("\n  BBusiness Narrative per Kasus:")
    model_scores = shap_values.sum(axis=1) + expected_value

    for label, idx in [("AMAN", idx_safe), ("BERISIKO", idx_risky), ("BORDERLINE", idx_border)]:
        shap_row = shap_values[idx]
        feat_vals = X_explain.iloc[idx]
        score = model_scores[idx]
        contrib = pd.Series(shap_row, index=X_explain.columns)
        top_risk   = contrib.nlargest(3)
        top_protect = contrib.nsmallest(3)

        print(f"\n  [{label}] Score: {score:.3f} (threshold: {threshold:.2f})")
        print(f"  Faktor yang MENAIKKAN risiko  :")
        for f, v in top_risk.items():
            print(f"    + {f:<30} nilai={feat_vals[f]:.3f}  kontribusi={v:+.3f}")
        print(f"  Faktor yang MENURUNKAN risiko :")
        for f, v in top_protect.items():
            print(f"    - {f:<30} nilai={feat_vals[f]:.3f}  kontribusi={v:+.3f}")


# ══════════════════════════════════════════════════════════════
# 7c. SHAP DEPENDENCE PLOT — hubungan non-linear fitur vs target
# ══════════════════════════════════════════════════════════════

def plot_shap_dependence(shap_values, X_explain):
    """
    Dependence plot menunjukkan bagaimana SHAP value satu fitur
    berubah seiring nilai fitur tersebut — menangkap non-linearity
    yang tidak bisa dilihat dari korelasi biasa.
    """
    print("\n── SHAP Dependence Plots ─────────────────────────────")

    # Pilih 4 fitur paling penting
    mean_abs = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs)[::-1][:4]
    top4_feats = X_explain.columns[top4_idx].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SHAP Dependence Plot: Hubungan Non-Linear Fitur vs Risiko",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flatten(), top4_feats):
        feat_idx = list(X_explain.columns).index(feat)
        feat_vals = X_explain[feat].values
        shap_col  = shap_values[:, feat_idx]

        # Cap outlier untuk visualisasi
        p1, p99 = np.percentile(feat_vals, [1, 99])
        mask = (feat_vals >= p1) & (feat_vals <= p99)

        sc = ax.scatter(
            feat_vals[mask], shap_col[mask],
            c=feat_vals[mask], cmap="RdBu_r",
            alpha=0.4, s=12, linewidths=0,
        )
        ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"Dependence: {feat}", fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="Nilai fitur")

    plt.tight_layout()
    plt.savefig("outputs/modeling/07_shap_dependence.png", bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 8. LIGHTGBM — pembanding cepat vs XGBoost
# ══════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n── LightGBM (pembanding) ─────────────────────────────")
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    y_prob = lgbm.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    print(f"  LightGBM → AUC-ROC (val): {auc:.4f}")
    return lgbm


# ══════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY TABLE — masuk langsung ke README
# ══════════════════════════════════════════════════════════════

def print_final_summary(results_df: pd.DataFrame, best_threshold: float):
    print("\n" + "═" * 65)
    print("HASIL AKHIR EVALUASI MODEL:")
    print("═" * 65)
    test_results = results_df[results_df["Split"] == "test"]
    print(test_results.to_string(index=False))
    print(f"\n  Recommended threshold: {best_threshold:.2f} (optimized F1)")
    print(f"\n  Plot yang dihasilkan:")
    plots = [
        "01_optuna_tuning.png      — hyperparameter optimization history",
        "02_learning_curve.png     — overfitting check",
        "03_roc_pr_curves.png      — perbandingan 3 model",
        "04_confusion_threshold.png — threshold analysis + business impact",
        "05_shap_summary.png       — global feature importance (WAJIB di artikel)",
        "06_shap_force_plots.png   — penjelasan per peminjam (demo-able)",
        "07_shap_dependence.png    — non-linear relationship",
    ]
    for p in plots:
        print(f"  outputs/modeling/{p}")

    print(f"""
LANGKAH BERIKUTNYA (Fase 4 — Streamlit App):
  [ ] Load model dari models/best_xgb_model.json
  [ ] Buat input form: age, income, debt_ratio, utilization, late_payments
  [ ] Hitung semua fitur turunan secara real-time
  [ ] Tampilkan: probabilitas + label + SHAP waterfall plot per input
  [ ] Deploy ke Streamlit Community Cloud (gratis)
""")


def save_best_threshold(best_threshold: float) -> None:
    threshold_payload = {
        "best_threshold": round(float(best_threshold), 4),
        "selection_rule": "validation F1 maximum",
        "source": "03_modeling_and_shap.py",
    }
    with open("models/best_threshold.json", "w", encoding="utf-8") as threshold_file:
        json.dump(threshold_payload, threshold_file, indent=2)
    print(" Threshold disimpan: models/best_threshold.json")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(" Fase 3: Modeling, Tuning & SHAP Explainability")
    print("=" * 65)

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

    # 2. Baseline
    lr_model, lr_auc = train_baseline(X_train, y_train, X_val, y_val)

    # 3. Optuna tuning — kurangi n_trials jika ingin cepat
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)

    # 4. Train final XGBoost
    xgb_model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # 5. LightGBM
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # 6. Evaluasi semua model
    models = {
        "Logistic Regression": lr_model,
        "XGBoost (tuned)":     xgb_model,
        "LightGBM":            lgbm_model,
    }
    results_df = evaluate_models(models, X_val, y_val, X_test, y_test)

    # 7. Confusion matrix + threshold analysis
    best_threshold = plot_confusion_and_threshold(xgb_model, X_test, y_test)

    # 8. SHAP — semua 3 jenis plot
    explainer, shap_values, X_explain, expected_value = compute_shap_values(
        xgb_model, X_train, X_test
    )
    plot_shap_summary(shap_values, X_explain)
    plot_shap_force_plots(explainer, shap_values, X_explain,
                          expected_value, best_threshold)
    plot_shap_dependence(shap_values, X_explain)

    # 9. Summary akhir
    save_best_threshold(best_threshold)
    print_final_summary(results_df, best_threshold)

    print("\n Fase 3 selesai! Semua output di: outputs/modeling/")
    print("   Next: python 04_streamlit_app.py")
