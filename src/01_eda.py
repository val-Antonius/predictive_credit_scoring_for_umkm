"""
================================================================
PROJECT: Credit Scoring
FILE   : 01_eda_credit_scoring.py
AUTHOR : Antonius Valentino
DATASET: Give Me Some Credit — Kaggle
         https://www.kaggle.com/c/GiveMeSomeCredit/data
================================================================

STRUKTUR OUTPUT:
  outputs/eda/  → semua plot tersimpan otomatis (.png)
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Setup direktori output ─────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)

# ── Style konsisten untuk semua plot ──────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.family": "sans-serif",
})
PALETTE = ["#378ADD", "#E24B4A"]   # biru = aman, merah = gagal bayar
sns.set_palette(PALETTE)


# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════

def load_data(path: str = "data/cs-training.csv") -> pd.DataFrame:
    """Load dataset dan bersihkan nama kolom."""
    df = pd.read_csv(path, index_col=0)

    # Rename agar lebih mudah dibaca
    rename_map = {
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
    }
    df.rename(columns=rename_map, inplace=True)
    print(f"Data loaded: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    return df


# ══════════════════════════════════════════════════════════════
# 2. RINGKASAN DASAR
# ══════════════════════════════════════════════════════════════

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cetak ringkasan lengkap: dtypes, missing, unik, statistik."""
    print("\n" + "═" * 60)
    print("RINGKASAN DATASET")
    print("═" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Duplikat       : {df.duplicated().sum()}")

    summary = pd.DataFrame({
        "dtype":       df.dtypes,
        "missing":     df.isnull().sum(),
        "missing_%":   (df.isnull().mean() * 100).round(2),
        "unique":      df.nunique(),
        "min":         df.min(),
        "max":         df.max(),
        "mean":        df.mean().round(2),
        "median":      df.median().round(2),
    })

    print("\n Detail per kolom:")
    print(summary.to_string())
    return summary


# ══════════════════════════════════════════════════════════════
# 3. ANALISIS CLASS IMBALANCE — poin penting di Medium/LinkedIn
# ══════════════════════════════════════════════════════════════

def plot_class_imbalance(df: pd.DataFrame) -> None:
    """
    Visualisasi distribusi target.
    tunjukkan seberapa imbalanced data-nya sebelum ke metrik.
    """
    counts = df["target"].value_counts()
    pct = df["target"].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribusi Target: Gagal Bayar vs Tidak", fontsize=14, fontweight="bold", y=1.01)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(["Tidak gagal (0)", "Gagal bayar (1)"],
                  counts.values, color=PALETTE, width=0.5, edgecolor="white")
    for bar, val, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                f"{val:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Jumlah sampel")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Jumlah per kelas")

    # Pie chart
    ax2 = axes[1]
    ax2.pie(counts.values, labels=["Tidak gagal (0)", "Gagal bayar (1)"],
            colors=PALETTE, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax2.set_title("Proporsi kelas")

    plt.tight_layout()
    plt.savefig("outputs/eda/01_class_imbalance.png", bbox_inches="tight")
    plt.show()

    print(f"\n CLASS IMBALANCE RATIO: 1 : {counts[0]/counts[1]:.1f}")
    print("   → Jangan pakai accuracy sebagai metrik utama!")
    print("   → Gunakan AUC-ROC, F1-score, Precision-Recall curve")


# ══════════════════════════════════════════════════════════════
# 4. MISSING VALUES — visualisasi untuk README / artikel
# ══════════════════════════════════════════════════════════════

def plot_missing_values(df: pd.DataFrame) -> None:
    """Heatmap missing values — langsung terlihat polanya."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("Tidak ada missing values.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    pct = (missing / len(df) * 100).round(2)
    bars = ax.barh(missing.index, pct.values, color="#378ADD", edgecolor="white")
    for bar, val in zip(bars, pct.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=10)
    ax.set_xlabel("% missing")
    ax.set_title("Kolom dengan Missing Values")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/eda/02_missing_values.png", bbox_inches="tight")
    plt.show()

    print("\n Missing values ditemukan:")
    for col, pct_val in pct.items():
        strategy = "→ median imputation" if col == "monthly_income" else "→ mode imputation"
        print(f"   {col:<25} {pct_val:.2f}%  {strategy}")


# ══════════════════════════════════════════════════════════════
# 5. DISTRIBUSI FITUR NUMERIK
# ══════════════════════════════════════════════════════════════

def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Histogram per fitur, split by target.
    Ini menunjukkan fitur mana yang punya separasi visual — bukti awal
    bahwa fitur tersebut prediktif.
    """
    numeric_cols = [c for c in df.columns if c != "target"]
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("Distribusi Fitur Numerik (split by target)", fontsize=14, fontweight="bold")

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label, color in zip([0, 1], PALETTE):
            subset = df[df["target"] == label][col].dropna()
            # Cap outlier ekstrem untuk visualisasi saja
            upper = subset.quantile(0.99)
            subset = subset[subset <= upper]
            ax.hist(subset, bins=40, alpha=0.6, color=color,
                    label=f"{'Tidak gagal' if label==0 else 'Gagal bayar'}",
                    density=True, edgecolor="none")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)

    # Sembunyikan axes kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/eda/03_feature_distributions.png", bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 6. OUTLIER ANALYSIS — penting untuk menjelaskan data cleaning
# ══════════════════════════════════════════════════════════════

def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi outlier dengan IQR method.
    """
    print("\n" + "═" * 60)
    print("ANALISIS OUTLIER (IQR Method)")
    print("═" * 60)

    results = []
    for col in df.select_dtypes(include=np.number).columns:
        if col == "target":
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        results.append({
            "feature": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "lower_fence": round(lower, 2),
            "upper_fence": round(upper, 2),
            "n_outliers": n_outliers,
            "outlier_%": round(pct, 2),
        })

    outlier_df = pd.DataFrame(results).sort_values("outlier_%", ascending=False)
    print(outlier_df.to_string(index=False))

    # Highlight kasus spesial
    age_max = df["age"].max()
    income_max = df["monthly_income"].max()
    utilization_max = df["revolving_utilization"].max()
    print(f"\n Kasus anomali yang perlu ditangani:")
    print(f"   age max               = {age_max}  → cap di 95")
    print(f"   monthly_income max    = {income_max:,.0f}  → log transform atau cap")
    print(f"   revolving_utilization max = {utilization_max:.2f}  → cap di 1.0 (>1 tidak masuk akal)")

    return outlier_df


# ══════════════════════════════════════════════════════════════
# 7. CORRELATION MATRIX — untuk section "Key Insights" di Medium
# ══════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap korelasi — tunjukkan fitur mana yang paling berkorelasi dengan target."""
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hanya setengah bawah
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix — Semua Fitur vs Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/eda/04_correlation_matrix.png", bbox_inches="tight")
    plt.show()

    # Print top korelasi dengan target
    target_corr = corr["target"].drop("target").abs().sort_values(ascending=False)
    print("\n Fitur paling berkorelasi dengan target (|r|):")
    for feat, val in target_corr.items():
        bar = "█" * int(val * 20)
        print(f"   {feat:<35} {val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════
# 8. BUSINESS INSIGHT — ini yang masuk ke artikel Medium kamu
# ══════════════════════════════════════════════════════════════

def generate_business_insights(df: pd.DataFrame) -> None:
    """
    Statistik deskriptif yang diframing sebagai insight bisnis.
    """
    print("\n" + "═" * 60)
    print("BUSINESS INSIGHTS")
    print("═" * 60)

    gagal = df[df["target"] == 1]
    aman  = df[df["target"] == 0]

    # Insight 1: usia
    print(f"\n1️. Usia rata-rata peminjam gagal bayar : {gagal['age'].mean():.1f} tahun")
    print(f"   Usia rata-rata peminjam aman        : {aman['age'].mean():.1f} tahun")

    # Insight 2: pendapatan
    print(f"\n2️. Median income gagal bayar  : ${gagal['monthly_income'].median():,.0f}/bulan")
    print(f"   Median income aman         : ${aman['monthly_income'].median():,.0f}/bulan")
    print(f"   → Perbedaan: {aman['monthly_income'].median() - gagal['monthly_income'].median():,.0f}")

    # Insight 3: utilization
    print(f"\n3️.  Rata-rata revolving utilization:")
    print(f"   Gagal bayar : {gagal['revolving_utilization'].mean():.2f}")
    print(f"   Aman        : {aman['revolving_utilization'].mean():.2f}")
    print(f"   → Peminjam gagal bayar punya utilisasi {gagal['revolving_utilization'].mean() / aman['revolving_utilization'].mean():.1f}× lebih tinggi")

    # Insight 4: late payments
    late_cols = ["late_30_59", "late_60_89", "late_90d"]
    print(f"\n4️.  Rata-rata keterlambatan pembayaran (peminjam gagal bayar):")
    for col in late_cols:
        print(f"   {col:<12} : {gagal[col].mean():.2f} kali vs {aman[col].mean():.2f} kali (aman)")

    # Boxplot perbandingan income
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Perbandingan Profil: Gagal Bayar vs Tidak", fontsize=13, fontweight="bold")

    # Income (cap outlier untuk visualisasi)
    income_cap = df["monthly_income"].quantile(0.95)
    ax = axes[0]
    data_plot = [
        aman["monthly_income"].dropna().clip(upper=income_cap).values,
        gagal["monthly_income"].dropna().clip(upper=income_cap).values,
    ]
    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(["Tidak gagal (0)", "Gagal bayar (1)"])
    ax.set_ylabel("Monthly income ($)")
    ax.set_title("Distribusi Pendapatan Bulanan")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Revolving utilization
    ax2 = axes[1]
    util_cap = df["revolving_utilization"].quantile(0.95)
    data_util = [
        aman["revolving_utilization"].clip(upper=util_cap).values,
        gagal["revolving_utilization"].clip(upper=util_cap).values,
    ]
    bp2 = ax2.boxplot(data_util, patch_artist=True, widths=0.5,
                      medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp2["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(["Tidak gagal (0)", "Gagal bayar (1)"])
    ax2.set_ylabel("Revolving Utilization Rate")
    ax2.set_title("Utilisasi Kredit Revolving")

    plt.tight_layout()
    plt.savefig("outputs/eda/05_business_insights.png", bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 9. EDA SUMMARY — cetak checklist untuk fase selanjutnya
# ══════════════════════════════════════════════════════════════

def print_next_steps(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("EDA SELESAI — CHECKLIST FASE BERIKUTNYA")
    print("═" * 60)
    print("""
PREPROCESSING yang harus dilakukan (Fase 2):
  [ ] Impute monthly_income  → median per kelompok umur
  [ ] Impute num_dependents  → mode (0)
  [ ] Cap revolving_utilization di 1.0
  [ ] Cap age di 95, hapus age == 0
  [ ] Log transform monthly_income (skewed)
  [ ] Handle imbalanced class → coba SMOTE + class_weight

FEATURE ENGINEERING yang menjanjikan (Fase 2):
  [ ] debt_to_income_ratio = debt_ratio × monthly_income
  [ ] total_late_payments  = late_30_59 + late_60_89 + late_90d
  [ ] income_per_dependent = monthly_income / (num_dependents + 1)
  [ ] utilization_bucket   = pd.cut(revolving_utilization, bins)

METRIK yang akan digunakan (Fase 3):
  [ ] Primary  : AUC-ROC
  [ ] Secondary: F1-score (weighted), Precision-Recall AUC
  [ ] BUKAN    : Accuracy — menyesatkan di imbalanced data!
""")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ganti path jika dataset kamu di lokasi berbeda
    DATA_PATH = "data/cs-training.csv"

    print("Memulai EDA — Credit Scoring Project")
    print("=" * 60)

    df = load_data(DATA_PATH)
    summary = basic_summary(df)

    plot_class_imbalance(df)
    plot_missing_values(df)
    plot_feature_distributions(df)
    outlier_df = analyze_outliers(df)
    plot_correlation_matrix(df)
    generate_business_insights(df)
    print_next_steps(df)

    print("\n Semua plot tersimpan di: outputs/eda/")
