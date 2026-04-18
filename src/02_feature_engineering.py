"""
================================================================
PROJECT: Credit Scoring
FILE   : 02_feature_engineering.py
AUTHOR : Antonius Valentino
================================================================

PREREQUISITE:
  - Sudah jalankan 01_eda_credit_scoring.py
  - pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

OUTPUT:
  data/processed/X_train.csv, X_val.csv, X_test.csv
  data/processed/y_train.csv, y_val.csv, y_test.csv
    data/processed/X_train_resampled.csv  ← alias untuk train asli yang sudah di-scale
    data/processed/y_train_resampled.csv  ← alias untuk label train asli
  outputs/feature_eng/                  ← semua plot
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/feature_eng", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})
PALETTE = ["#378ADD", "#E24B4A"]


# ══════════════════════════════════════════════════════════════
# 1. LOAD & RENAME  (sama seperti EDA, dijadikan fungsi reusable)
# ══════════════════════════════════════════════════════════════

def load_and_rename(path: str = "data/cs-training.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.rename(columns={
        "SeriousDlqin2yrs":                        "target",
        "RevolvingUtilizationOfUnsecuredLines":     "revolving_utilization",
        "age":                                      "age",
        "NumberOfTime30-59DaysPastDueNotWorse":     "late_30_59",
        "DebtRatio":                                "debt_ratio",
        "MonthlyIncome":                            "monthly_income",
        "NumberOfOpenCreditLinesAndLoans":          "open_credit_lines",
        "NumberOfTimes90DaysLate":                  "late_90d",
        "NumberRealEstateLoansOrLines":             "real_estate_loans",
        "NumberOfTime60-89DaysPastDueNotWorse":     "late_60_89",
        "NumberOfDependents":                       "num_dependents",
    }, inplace=True)
    print(f"Loaded: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    return df


# ══════════════════════════════════════════════════════════════
# 2. CLEANING — keputusan berbasis EDA sebelumnya
# ══════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Cleaning ──────────────────────────────────────────")
    df = df.copy()
    original_len = len(df)

    # ── Hapus baris dengan usia tidak masuk akal ──────────────
    # Usia 0 kemungkinan data entry error, usia >95 sangat ekstrem
    mask_age = (df["age"] > 0) & (df["age"] <= 95)
    df = df[mask_age]
    print(f"  Hapus age ≤0 atau >95     : -{original_len - len(df)} baris")

    # ── Cap revolving utilization ─────────────────────────────
    # >1.0 berarti pemakaian melebihi limit — secara finansial masih valid
    # tapi nilai ekstrem (>10, >100) adalah data error
    df["revolving_utilization"] = df["revolving_utilization"].clip(upper=1.5)
    print(f"  Cap revolving_utilization  : clipped ke max 1.5")

    # ── Imputasi missing values ───────────────────────────────
    # monthly_income: median per kelompok umur (lebih informatif dari global median)
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 100],
                             labels=["<30", "30-45", "45-60", "60+"])
    income_median_by_age = df.groupby("age_group", observed=True)["monthly_income"].median()
    global_income_median = df["monthly_income"].median()

    def impute_income(row):
        if pd.isna(row["monthly_income"]):
            return income_median_by_age.get(row["age_group"], global_income_median)
        return row["monthly_income"]

    missing_income = df["monthly_income"].isna().sum()
    df["monthly_income"] = df.apply(impute_income, axis=1)
    print(f"  Impute monthly_income      : {missing_income:,} baris (median per kelompok usia)")

    # num_dependents: mode = 0 (mayoritas tidak punya tanggungan)
    missing_dep = df["num_dependents"].isna().sum()
    df["num_dependents"] = df["num_dependents"].fillna(0)
    print(f"  Impute num_dependents      : {missing_dep:,} baris (mode = 0)")

    # Hapus kolom bantu
    df.drop(columns=["age_group"], inplace=True)

    print(f"  Dataset setelah cleaning   : {len(df):,} baris")
    return df


# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — ini inti Fase 2
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setiap fitur baru harus punya justifikasi bisnis.
    """
    print("\n── Feature Engineering ───────────────────────────────")
    df = df.copy()

    # ── Fitur 1: Total keterlambatan pembayaran ───────────────
    # Rationale: model credit scoring industri sangat memperhatikan
    # riwayat keterlambatan — gabungkan semua horizon waktu
    df["total_late_payments"] = (
        df["late_30_59"] + df["late_60_89"] + df["late_90d"]
    )
    print("  ✚ total_late_payments      = late_30_59 + late_60_89 + late_90d")

    # ── Fitur 2: Weighted late score ──────────────────────────
    # Rationale: keterlambatan 90 hari jauh lebih serius dari 30 hari
    # Beri bobot berbeda mencerminkan severity
    df["weighted_late_score"] = (
        df["late_30_59"] * 1 +
        df["late_60_89"] * 2 +
        df["late_90d"]   * 4   # 4x lebih berat
    )
    print("  ✚ weighted_late_score      = 1×late30 + 2×late60 + 4×late90")

    # ── Fitur 3: Income per dependent ─────────────────────────
    # Rationale: pendapatan Rp5jt dengan 3 tanggungan sangat berbeda
    # dari Rp5jt tanpa tanggungan — discretionary income proxy
    df["income_per_dependent"] = (
        df["monthly_income"] / (df["num_dependents"] + 1)
    )
    print("  ✚ income_per_dependent     = income / (dependents + 1)")

    # ── Fitur 4: Debt-to-income ratio yang lebih akurat ───────
    # Rationale: debt_ratio asli = total monthly debt / gross income
    # Kita buat versi yang lebih interpretable
    df["monthly_debt"] = df["debt_ratio"] * df["monthly_income"]
    df["dti_adjusted"] = df["monthly_debt"] / (df["monthly_income"] + 1)
    print("  ✚ dti_adjusted             = monthly_debt / (income + 1)")

    # ── Fitur 5: Credit utilization bucket (ordinal) ──────────
    # Rationale: industry rule of thumb — <30% baik, 30-70% perhatian, >70% risiko
    df["utilization_risk"] = pd.cut(
        df["revolving_utilization"],
        bins=[-0.01, 0.30, 0.70, 1.00, float("inf")],
        labels=[0, 1, 2, 3]   # 0=rendah, 3=sangat tinggi
    ).astype(int)
    print("  ✚ utilization_risk         = 0(aman) / 1(hati2) / 2(tinggi) / 3(kritis)")

    # ── Fitur 6: Ever late (binary flag) ──────────────────────
    # Rationale: fitur binary sering powerful — pernah vs tidak pernah terlambat
    df["ever_late"] = (df["total_late_payments"] > 0).astype(int)
    print("  ✚ ever_late                = 1 jika pernah terlambat bayar")

    # ── Fitur 7: Credit lines per age ─────────────────────────
    # Rationale: seseorang usia 50 dengan 20 credit lines berbeda dari
    # usia 25 dengan 20 credit lines — normalize by age
    df["credit_lines_per_age"] = (
        df["open_credit_lines"] / df["age"]
    ).round(4)
    print("  ✚ credit_lines_per_age     = open_credit_lines / age")

    # ── Fitur 8: Log transform income ─────────────────────────
    # Rationale: monthly_income sangat right-skewed — log transform
    # mendekati distribusi normal, membantu model linear
    df["log_income"] = np.log1p(df["monthly_income"])
    print("  ✚ log_income               = log(1 + monthly_income)")

    print(f"\n  Total fitur setelah engineering: {df.shape[1] - 1} fitur + 1 target")
    return df


# ══════════════════════════════════════════════════════════════
# 4. VISUALISASI FITUR BARU — untuk artikel Medium
# ══════════════════════════════════════════════════════════════

def plot_engineered_features(df: pd.DataFrame) -> None:
    """Tunjukkan bahwa fitur baru punya separasi yang lebih baik dari fitur asli."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fitur Baru vs Target: Apakah Prediktif?", fontsize=14, fontweight="bold")

    features_to_plot = [
        ("total_late_payments",  "Total keterlambatan pembayaran",  True),
        ("weighted_late_score",  "Weighted late score",              True),
        ("income_per_dependent", "Income per dependent",             False),
        ("log_income",           "Log monthly income",               False),
        ("dti_adjusted",         "DTI adjusted",                     True),
        ("credit_lines_per_age", "Credit lines per age",             False),
    ]

    for ax, (col, title, cap99) in zip(axes.flatten(), features_to_plot):
        for label, color in zip([0, 1], PALETTE):
            data = df[df["target"] == label][col].dropna()
            if cap99:
                data = data[data <= data.quantile(0.99)]
            ax.hist(data, bins=40, alpha=0.6, color=color, density=True,
                    label=f"{'Tidak gagal' if label==0 else 'Gagal bayar'}",
                    edgecolor="none")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        if col == "total_late_payments":
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/feature_eng/01_engineered_features.png", bbox_inches="tight")
    plt.show()
    print(" Plot disimpan: outputs/feature_eng/01_engineered_features.png")


# ══════════════════════════════════════════════════════════════
# 5. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    """
    Split stratified 60/20/20.
    Stratify memastikan proporsi class imbalance
    terjaga di semua split — jangan pernah skip ini.
    """
    print("\n── Train/Val/Test Split ──────────────────────────────")

    FEATURE_COLS = [
        # Fitur asli
        "revolving_utilization", "age", "late_30_59", "debt_ratio",
        "monthly_income", "open_credit_lines", "late_90d",
        "real_estate_loans", "late_60_89", "num_dependents",
        # Fitur baru
        "total_late_payments", "weighted_late_score",
        "income_per_dependent", "monthly_debt", "dti_adjusted",
        "utilization_risk", "ever_late", "credit_lines_per_age", "log_income",
    ]

    X = df[FEATURE_COLS]
    y = df["target"]

    # Split pertama: pisahkan test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Split kedua: dari sisanya, pisahkan val set (20% dari total = 25% dari temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos = y_split.sum()
        total = len(y_split)
        print(f"  {name:<6}: {total:>6,} baris | "
              f"positif: {pos:,} ({pos/total*100:.1f}%) | "
              f"negatif: {total-pos:,} ({(total-pos)/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, FEATURE_COLS


# ══════════════════════════════════════════════════════════════
# 6. HANDLE IMBALANCED CLASS — perbandingan 3 strategi
# ══════════════════════════════════════════════════════════════

def compare_resampling_strategies(X_train: pd.DataFrame,
                                   y_train: pd.Series) -> None:
    """
    Bandingkan 3 strategi resampling secara visual.
    """
    print("\n── Perbandingan Strategi Resampling ──────────────────")

    strategies = {
        "Original\n(no resampling)": (X_train.copy(), y_train.copy()),
    }

    # Strategi 1: SMOTE murni
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    strategies["SMOTE\n(oversample minority)"] = (X_smote, y_smote)

    # Strategi 2: Random undersampling
    rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    strategies["Random Under-\nsampling"] = (X_rus, y_rus)

    # Strategi 3: SMOTETomek (hybrid — SMOTE + hapus Tomek links)
    smotetomek = SMOTETomek(random_state=42)
    X_st, y_st = smotetomek.fit_resample(X_train, y_train)
    strategies["SMOTETomek\n(hybrid)"] = (X_st, y_st)

    # ── Plot perbandingan ─────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Perbandingan Strategi Handle Imbalanced Class", fontsize=13, fontweight="bold")

    colors_bar = ["#378ADD", "#E24B4A"]
    for ax, (name, (_, y_res)) in zip(axes, strategies.items()):
        counts = Counter(y_res)
        total = sum(counts.values())
        bars = ax.bar(["Tidak\ngagal (0)", "Gagal\nbayar (1)"],
                      [counts[0], counts[1]],
                      color=colors_bar, width=0.5, edgecolor="white")
        for bar, val in zip(bars, [counts[0], counts[1]]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=9)
        ratio = counts[0] / counts[1]
        ax.set_title(f"{name}\n(ratio 1:{ratio:.1f})", fontsize=10)
        ax.set_ylabel("Jumlah sampel")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig("outputs/feature_eng/02_resampling_comparison.png", bbox_inches="tight")
    plt.show()

    print("\n  Ringkasan jumlah sampel per strategi:")
    for name, (_, y_res) in strategies.items():
        counts = Counter(y_res)
        label = name.replace("\n", " ")
        print(f"  {label:<35} | total: {sum(counts.values()):>7,} "
              f"| pos: {counts[1]:>6,} | neg: {counts[0]:>6,}")

    return X_smote, y_smote


def explain_smote(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Visualisasi cara kerja SMOTE — 2 fitur untuk kemudahan visualisasi.
    """
    print("\n── Visualisasi Cara Kerja SMOTE ──────────────────────")

    # Pakai 2 fitur yang paling informatif
    feat1, feat2 = "revolving_utilization", "total_late_payments"
    X_2d = X_train[[feat1, feat2]].copy()

    # Cap outlier untuk visualisasi
    X_2d[feat1] = X_2d[feat1].clip(upper=1.5)
    X_2d[feat2] = X_2d[feat2].clip(upper=10)

    # Sebelum SMOTE — ambil sample agar tidak terlalu padat
    sample_idx = X_2d.index[:2000]
    X_sample = X_2d.loc[sample_idx]
    y_sample = y_train.loc[sample_idx]

    # Setelah SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res_2d, y_res_2d = smote.fit_resample(X_sample, y_sample)
    X_res_df = pd.DataFrame(X_res_2d, columns=[feat1, feat2])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cara Kerja SMOTE: Sebelum vs Sesudah", fontsize=13, fontweight="bold")
    titles = ["Sebelum SMOTE (imbalanced)", "Sesudah SMOTE (balanced)"]

    datasets = [
        (X_sample, y_sample),
        (X_res_df, pd.Series(y_res_2d)),
    ]

    for ax, (X_plot, y_plot), title in zip(axes, datasets, titles):
        for label, color, marker, alpha, size in [
            (0, "#378ADD", "o", 0.25, 15),
            (1, "#E24B4A", "^", 0.7, 25),
        ]:
            mask = y_plot == label
            ax.scatter(
                X_plot[feat1][mask], X_plot[feat2][mask],
                c=color, marker=marker, alpha=alpha, s=size,
                label=f"{'Tidak gagal' if label==0 else 'Gagal bayar'} (n={mask.sum():,})"
            )
        ax.set_xlabel("Revolving utilization")
        ax.set_ylabel("Total late payments")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, markerscale=1.5)

    plt.tight_layout()
    plt.savefig("outputs/feature_eng/03_smote_visualization.png", bbox_inches="tight")
    plt.show()

    print("  SMOTE membuat titik SINTETIS baru di antara titik minoritas yang sudah ada.")
    print("  Berbeda dari duplicate. SMOTE interpolasi di antara k-nearest neighbors.")


# ══════════════════════════════════════════════════════════════
# 7. SCALING — RobustScaler untuk data dengan outlier
# ══════════════════════════════════════════════════════════════

def scale_features(X_train, X_val, X_test):
    """
    Gunakan RobustScaler bukan StandardScaler karena data keuangan
    hampir selalu punya outlier ekstrem.
    RobustScaler pakai median & IQR, tidak terpengaruh outlier.

    PENTING: fit HANYA di X_train, transform di semua split.
    Fit di val/test = data leakage!
    """
    print("\n── Feature Scaling ───────────────────────────────────")
    scaler = RobustScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )

    print("  Scaler    : RobustScaler (median + IQR, robust terhadap outlier)")
    print("  Fit pada  : X_train saja ← mencegah data leakage")
    print("  Transform : X_train, X_val, X_test")

    # Simpan scaler ke file fisik.
    # Catatan: XGBoost tidak butuh scaling untuk prediksi (tree-based model).
    # Scaler ini disimpan untuk: (1) kelengkapan pipeline, (2) jika suatu saat
    # kamu menambahkan model linear (LogReg, SVM) ke Streamlit app.
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print("  Saved     : models/scaler.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ══════════════════════════════════════════════════════════════
# 8. SIMPAN SEMUA DATASET
# ══════════════════════════════════════════════════════════════

def save_processed_data(X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         X_train_resampled, y_train_resampled) -> None:
    print("\n── Menyimpan data ────────────────────────────────────")
    files = {
        "data/processed/X_train.csv":            X_train,
        "data/processed/X_val.csv":              X_val,
        "data/processed/X_test.csv":             X_test,
        "data/processed/X_train_resampled.csv":  X_train_resampled,
    }
    targets = {
        "data/processed/y_train.csv":            y_train,
        "data/processed/y_val.csv":              y_val,
        "data/processed/y_test.csv":             y_test,
        "data/processed/y_train_resampled.csv":  y_train_resampled,
    }
    for path, data in {**files, **targets}.items():
        data.to_csv(path, index=True)
        print(f" {path}")
    print("\n  Catatan   : X_train_resampled/y_train_resampled sekarang berisi train asli yang sudah di-scale")
    print("  File ini yang akan di-load oleh 03_modeling.py berikutnya.")


# ══════════════════════════════════════════════════════════════
# 9. PRINT SUMMARY — checklist untuk Fase 3
# ══════════════════════════════════════════════════════════════

def print_phase3_checklist() -> None:
    print("\n" + "═" * 60)
    print("FASE 2 SELESAI — CHECKLIST FASE 3 (Modeling)")
    print("═" * 60)
    print("""
DATA yang tersedia untuk modeling:
    ✔ X_train_resampled  ← gunakan ini untuk training (train asli + SCALED)
    ✔ X_val              ← evaluasi saat hyperparameter tuning (SCALED)
    ✔ X_test             ← evaluasi FINAL, jangan dilihat sebelum model selesai! (SCALED)

Model yang akan dibangun di Fase 3:
  [ ] Baseline : LogisticRegression  (patokan perbandingan)
  [ ] Model 1  : XGBoostClassifier   (utama)
  [ ] Model 2  : LightGBM            (pembanding cepat)
  [ ] Tuning   : Optuna              (hyperparameter search)
  [ ] Explainer: SHAP                (wajib untuk artikel Medium)

Metrik yang digunakan:
  [ ] AUC-ROC          (primary)
  [ ] F1-score         (secondary)
  [ ] Precision-Recall (untuk business threshold analysis)
  [ ] BUKAN accuracy!
""")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(" Fase 2: Feature Engineering & Resampling")
    print("=" * 60)

    # 1. Load data
    df_raw = load_and_rename("data/cs-training.csv")

    # 2. Cleaning
    df_clean = clean_data(df_raw)

    # 3. Feature engineering
    df_feat = engineer_features(df_clean)

    # 4. Visualisasi fitur baru
    plot_engineered_features(df_feat)

    # 5. Split data
    X_train, X_val, X_test, y_train, y_val, y_test, FEATURE_COLS = split_data(df_feat)

    # 6. Resampling tidak dipakai untuk training final pada opsi 2.
    #    Visual perbandingan tetap disimpan sebagai referensi untuk artikel / README.
    explain_smote(X_train, y_train)
    compare_resampling_strategies(X_train, y_train)

    # 7. Scaling — fit hanya di train asli, transform semua
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 8. Simpan
    save_processed_data(
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        X_train_s, y_train
    )

    # 9. Checklist fase berikutnya
    print_phase3_checklist()

    print("\n  Semua plot tersimpan di: outputs/feature_eng/")