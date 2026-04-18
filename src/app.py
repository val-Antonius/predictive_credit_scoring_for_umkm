"""
================================================================
PROJECT: Credit Scoring
FILE   : 04_streamlit_app.py
AUTHOR : Antonius Valentino
================================================================

CARA MENJALANKAN LOKAL:
  pip install streamlit shap xgboost plotly pandas numpy joblib
  streamlit run 04_streamlit_app.py

STRUKTUR FOLDER YANG DIBUTUHKAN:
  models/best_xgb_model.json   ← output dari Fase 3
  models/scaler.pkl            ← output dari Fase 2
  requirements.txt
================================================================
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════
# KONSTANTA LOKALISASI
# ══════════════════════════════════════════════════════════════

USD_TO_IDR = 16_000
IDR_TO_USD = 1 / USD_TO_IDR

def idr_to_usd(idr: float) -> float:
    return idr * IDR_TO_USD

def usd_to_idr(usd: float) -> float:
    return usd * USD_TO_IDR

def fmt_idr(idr: float) -> str:
    if idr >= 1_000_000_000:
        return f"Rp {idr/1_000_000_000:.1f} M"
    elif idr >= 1_000_000:
        return f"Rp {idr/1_000_000:.1f} jt"
    elif idr >= 1_000:
        return f"Rp {idr/1_000:.0f} rb"
    return f"Rp {idr:,.0f}"


# ══════════════════════════════════════════════════════════════
# PEMETAAN ISTILAH: nama teknis dalam bahasa Indonesia
# Sumber kebenaran tunggal — dipakai di sidebar, SHAP, tabel
# ══════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "revolving_utilization":  "Beban Limit Kredit",
    "age":                    "Usia Pemohon",
    "late_30_59":             "Tunggakan 30–59 Hari",
    "debt_ratio":             "Rasio Utang",
    "monthly_income":         "Pendapatan Bulanan",
    "open_credit_lines":      "Jumlah Fasilitas Kredit",
    "late_90d":               "Tunggakan 90+ Hari",
    "real_estate_loans":      "Pinjaman Properti / Aset",
    "late_60_89":             "Tunggakan 60–89 Hari",
    "num_dependents":         "Jumlah Tanggungan",
    "total_late_payments":    "Total Riwayat Tunggakan",
    "weighted_late_score":    "Skor Disiplin Pembayaran",
    "income_per_dependent":   "Pendapatan per Tanggungan",
    "monthly_debt":           "Cicilan Utang Bulanan",
    "dti_adjusted":           "Rasio Cicilan vs Pendapatan",
    "utilization_risk":       "Tingkat Risiko Limit",
    "ever_late":              "Pernah Menunggak",
    "credit_lines_per_age":   "Kredit per Usia",
    "log_income":             "Skala Pendapatan (Log)",
}

FEATURE_HINTS = {
    "revolving_utilization":  "Seberapa penuh limit KMK/kartu kredit terpakai. <30% = sehat, >70% = berisiko.",
    "debt_ratio":             "Total cicilan dibagi total pendapatan. Idealnya di bawah 40%.",
    "monthly_income":         "Pendapatan bersih per bulan dari usaha maupun gaji.",
    "total_late_payments":    "Jumlah total kejadian menunggak dari semua pinjaman.",
    "weighted_late_score":    "Tunggakan 90 hari diberi bobot 4× lebih berat dari tunggakan 30 hari.",
    "income_per_dependent":   "Sisa pendapatan setelah dibagi jumlah tanggungan keluarga.",
    "dti_adjusted":           "Porsi pendapatan untuk membayar cicilan. Standar perbankan: maks 30–40%.",
    "weighted_late_score":    "Semakin panjang tunggakan, semakin besar skor negatifnya.",
}

def label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

def hint(feat: str) -> str:
    return FEATURE_HINTS.get(feat, "")


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SHAP Credit Scoring App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #F8F7F4; }
  .block-container { padding: 1.75rem 2.5rem 3rem; max-width: 1120px; }

  .app-header { background: #0F1923; padding: 0.75rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; border-left: 4px solid #1D9E75; }
  .app-header h1 { font-size: 1.65rem; font-weight: 600; letter-spacing: -0.03em; color: white; margin: 0 0 0.3rem; }
  .app-header p  { font-size: 0.85rem; color: #7A9BB5; margin: 0; }
  .app-header .kurs { font-size: 10px; color: #4A7A9B; margin-top: 5px; font-family: 'DM Mono', monospace; }

  .metric-row { display: flex; gap: 10px; margin-bottom: 1.1rem; }
  .metric-card { flex: 1; background: white; border: 0.5px solid #E8E6E0; border-radius: 12px; padding: 0.9rem 1rem; }
  .metric-label { font-size: 10px; font-weight: 500; color: #888780; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 3px; }
  .metric-value { font-size: 1.35rem; font-weight: 600; letter-spacing: -0.02em; color: #0B2032; line-height: 1.2; }
  .metric-sub   { font-size: 10px; color: #B4B2A9; margin-top: 2px; }
  .metric-value.safe   { color: #0F6E56; }
  .metric-value.risk   { color: #993C1D; }
  .metric-value.border { color: #854F0B; }

  .decision-badge { display: inline-block; padding: 3px 12px; border-radius: 99px; font-size: 10px; font-weight: 600; letter-spacing: 0.05em; margin-top: 5px; }
  .badge-approve { background: #E1F5EE; color: #085041; }
  .badge-reject  { background: #FAECE7; color: #712B13; }
  .badge-review  { background: #FAEEDA; color: #633806; }

  .section-title { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888780; margin: 0 0 0.75rem; }

  .shap-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
  .shap-feat { font-size: 12px; color: #2C2C2A; width: 200px; flex-shrink: 0; line-height: 1.35; }
  .shap-feat-sub { font-size: 9px; color: #B4B2A9; font-family: 'DM Mono', monospace; }
  .shap-bar-wrap { flex: 1; height: 15px; background: #F1EFE8; border-radius: 4px; overflow: hidden; }
  .shap-bar { height: 100%; border-radius: 4px; }
  .shap-val { font-size: 10px; font-family: 'DM Mono', monospace; min-width: 48px; text-align: right; }

  .info-box { background: #EEF4FC; border-left: 3px solid #185FA5; border-radius: 0 8px 8px 0; padding: 0.6rem 0.9rem; font-size: 12px; color: #0C447C; margin: 0.65rem 0; }
  .warn-box  { background: #FAEEDA; border-left: 3px solid #854F0B; border-radius: 0 8px 8px 0; padding: 0.6rem 0.9rem; font-size: 12px; color: #633806; margin: 0.65rem 0; }

  section[data-testid="stSidebar"] { background: #0F1923; }
  section[data-testid="stSidebar"] label { color: #C2D0DE !important; font-size: 12px !important; }
  section[data-testid="stSidebar"] .stMarkdown p { color: #7A9BB5 !important; font-size: 11px !important; margin: 0.2rem 0 !important; }
  section[data-testid="stSidebar"] .stMarkdown h3 { color: white !important; font-size: 14px !important; margin: 0.4rem 0 0.3rem 0 !important; }
  section[data-testid="stSidebar"] hr { margin: 0.4rem 0 !important; }
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model("models/best_xgb_model.json")
    return m

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


def load_threshold() -> float:
    try:
        with open("models/best_threshold.json", "r", encoding="utf-8") as threshold_file:
            payload = json.load(threshold_file)
        return float(payload.get("best_threshold", 0.5))
    except FileNotFoundError:
        return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — identik dengan Fase 2
# Input: income sudah dalam USD (dikonversi dari IDR di sidebar)
# ══════════════════════════════════════════════════════════════

def compute_features(inputs: dict) -> pd.DataFrame:
    d = inputs
    revolv  = d["revolving_utilization"]
    age     = d["age"]
    late30  = d["late_30_59"]
    debt_r  = d["debt_ratio"]
    income  = d["monthly_income"]
    ocl     = d["open_credit_lines"]
    late90  = d["late_90d"]
    rel     = d["real_estate_loans"]
    late60  = d["late_60_89"]
    dep     = d["num_dependents"]

    total_late   = late30 + late60 + late90
    weighted     = late30 * 1 + late60 * 2 + late90 * 4
    inc_per_dep  = income / (dep + 1)
    monthly_debt = debt_r * income
    dti_adj      = monthly_debt / (income + 1)
    util_risk    = int(pd.cut([revolv],
                              bins=[-0.01, 0.30, 0.70, 1.00, float("inf")],
                              labels=[0, 1, 2, 3])[0])
    ever_late    = int(total_late > 0)
    cl_per_age   = ocl / age
    log_inc      = np.log1p(income)

    return pd.DataFrame([{
        "revolving_utilization": revolv,
        "age":                   age,
        "late_30_59":            late30,
        "debt_ratio":            debt_r,
        "monthly_income":        income,
        "open_credit_lines":     ocl,
        "late_90d":              late90,
        "real_estate_loans":     rel,
        "late_60_89":            late60,
        "num_dependents":        dep,
        "total_late_payments":   total_late,
        "weighted_late_score":   weighted,
        "income_per_dependent":  inc_per_dep,
        "monthly_debt":          monthly_debt,
        "dti_adjusted":          dti_adj,
        "utilization_risk":      util_risk,
        "ever_late":             ever_late,
        "credit_lines_per_age":  cl_per_age,
        "log_income":            log_inc,
    }])


# ══════════════════════════════════════════════════════════════
# GAUGE
# ══════════════════════════════════════════════════════════════

def make_gauge(prob: float) -> go.Figure:
    pct = round(prob * 100, 1)
    if prob < 0.35:
        color, risk_label = "#0F6E56", "RISIKO RENDAH"
    elif prob < 0.55:
        color, risk_label = "#854F0B", "RISIKO SEDANG"
    else:
        color, risk_label = "#993C1D", "RISIKO TINGGI"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "color": color, "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#D3D1C7", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#F8F7F4", "borderwidth": 0,
            "steps": [
                {"range": [0, 35],   "color": "#E1F5EE"},
                {"range": [35, 55],  "color": "#FAEEDA"},
                {"range": [55, 100], "color": "#FAECE7"},
            ],
            "threshold": {"line": {"color": "#0B2032", "width": 2}, "thickness": 0.75, "value": pct},
        },
        title={
            "text": (f"<b>Probabilitas Gagal Bayar</b><br>"
                     f"<span style='font-size:12px;color:{color};font-weight:600'>{risk_label}</span>"),
            "font": {"size": 13, "family": "DM Sans"},
        },
    ))
    fig.update_layout(height=230, margin=dict(t=55, b=0, l=20, r=20),
                      paper_bgcolor="white", font_family="DM Sans")
    return fig


# ══════════════════════════════════════════════════════════════
# SHAP WATERFALL — label Indonesia + nilai IDR
# ══════════════════════════════════════════════════════════════

def render_shap_waterfall(shap_vals, feature_names, raw_vals, expected_value):
    contrib     = pd.Series(shap_vals, index=feature_names)
    top_idx     = contrib.abs().nlargest(10).index
    contrib_top = contrib[top_idx].sort_values()
    max_abs     = contrib_top.abs().max()
    raw_series  = pd.Series(raw_vals, index=feature_names)

    st.markdown('<p class="section-title">Alasan Keputusan — Faktor Penentu Kelayakan</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '<b>Merah</b> = memperburuk kelayakan &nbsp;|&nbsp; '
        '<b>Hijau</b> = memperkuat kelayakan kredit'
        '</div>',
        unsafe_allow_html=True,
    )

    html_rows = ""
    for feat in contrib_top.index:
        val     = contrib_top[feat]
        raw_val = raw_series[feat]
        bar_pct = abs(val) / max_abs * 100
        color   = "#E24B4A" if val > 0 else "#1D9E75"
        sign    = "+" if val > 0 else ""

        # Format nilai kontekstual
        if feat in ("monthly_income", "income_per_dependent", "monthly_debt"):
            dval = fmt_idr(usd_to_idr(raw_val))
        elif feat in ("revolving_utilization", "debt_ratio", "dti_adjusted"):
            dval = f"{raw_val*100:.1f}%"
        elif feat == "ever_late":
            dval = "Ya" if raw_val >= 1 else "Tidak"
        elif feat == "log_income":
            dval = fmt_idr(usd_to_idr(np.expm1(raw_val)))
        elif feat == "utilization_risk":
            dval = ["Aman", "Perhatian", "Tinggi", "Kritis"][min(int(raw_val), 3)]
        else:
            dval = f"{raw_val:.0f}"

        html_rows += f"""
        <div class="shap-row">
          <div class="shap-feat">{label(feat)}<br>
            <span class="shap-feat-sub">{dval}</span>
          </div>
          <div class="shap-bar-wrap">
            <div class="shap-bar" style="width:{bar_pct:.1f}%;background:{color};opacity:0.85;"></div>
          </div>
          <div class="shap-val" style="color:{color};">{sign}{val:.3f}</div>
        </div>"""

    st.markdown(html_rows, unsafe_allow_html=True)

    baseline_pct = round(expected_value * 100, 1)
    final_pct    = round((expected_value + shap_vals.sum()) * 100, 1)
    st.markdown(
        f'<p style="font-size:11px;color:#B4B2A9;margin-top:10px;">'
        f'Rata-rata populasi: {baseline_pct:.1f}% &nbsp;→&nbsp; '
        f'Prediksi pemohon ini: <b style="color:#2C2C2A">{final_pct:.1f}%</b> '
        f'({final_pct - baseline_pct:+.1f}% dari rata-rata)</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR — input IDR, output USD untuk model
# ══════════════════════════════════════════════════════════════

def build_sidebar() -> dict:
    recommended_threshold = load_threshold()
    with st.sidebar:
        st.markdown("### Profil Pemohon Kredit")
        st.markdown(f"Kurs: **1 USD = Rp {USD_TO_IDR:,}**")
        st.markdown("---")

        st.markdown("**Informasi Pemohon**")
        age = st.slider("Usia (tahun)", 18, 90, 35)
        dep = st.slider("Jumlah tanggungan keluarga", 0, 10, 1)

        st.markdown("---")
        st.markdown("**Kondisi Keuangan**")

        income_idr = st.slider(
            "Pendapatan bersih per bulan (Rp)",
            500_000, 100_000_000, 5_000_000, step=500_000,
            format="Rp %d",
            help="Pendapatan bersih dari usaha + gaji per bulan",
        )
        income_usd = idr_to_usd(income_idr)

        revolv_pct = st.slider(
            "Beban limit kredit (%)", 0, 150, 30, step=1,
            help="Seberapa penuh limit KMK/kartu kredit terpakai. <30% = sehat, >70% = berisiko.",
        )
        debt_pct = st.slider(
            "Rasio cicilan terhadap pendapatan (%)", 0, 100, 35, step=1,
            help="Total cicilan per bulan ÷ pendapatan. Standar bank: maks 30–40%.",
        )
        ocl = st.slider("Jumlah fasilitas kredit aktif", 0, 30, 5,
                        help="KMK, KPR, kartu kredit, koperasi, dll.")
        rel = st.slider("Pinjaman berbasis properti / aset", 0, 10, 1,
                        help="KPR, kredit kendaraan, agunan tanah/bangunan.")

        st.markdown("---")
        st.markdown("**Riwayat Pembayaran**")
        late30 = st.slider("Pernah menunggak 30–59 hari (kali)", 0, 15, 0)
        late60 = st.slider("Pernah menunggak 60–89 hari (kali)", 0, 15, 0)
        late90 = st.slider("Pernah menunggak 90+ hari (kali)",   0, 15, 0)

        st.markdown("---")
        st.markdown("**Pengaturan Keputusan**")
        threshold = st.slider(
            "Ambang batas kelayakan", 0.10, 0.90, recommended_threshold, step=0.01,
            help="Default mengikuti rekomendasi model dari validasi. Naikkan untuk kebijakan lebih ketat, turunkan untuk kebijakan lebih inklusif.",
        )

        st.markdown(
            f"<div style='font-size:10px;color:#7A9BB5;margin-top:6px;line-height:1.5'>"
            f"Rekomendasi model: <b>{recommended_threshold:.2f}</b> (hasil validasi)"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:10px;color:#4A7A9B;margin-top:8px;line-height:1.7'>"
            f"Input model (USD):<br>"
            f"<span style='font-family:monospace'>income = ${income_usd:,.2f}</span><br>"
            f"<span style='font-size:9px;color:#2A5A7A'>dikonversi otomatis dari IDR</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    return {
        "age":                    age,
        "monthly_income":         income_usd,
        "monthly_income_idr":     income_idr,
        "num_dependents":         dep,
        "revolving_utilization":  revolv_pct / 100,
        "debt_ratio":             debt_pct / 100,
        "open_credit_lines":      ocl,
        "real_estate_loans":      rel,
        "late_30_59":             late30,
        "late_60_89":             late60,
        "late_90d":               late90,
        "_threshold":             threshold,
        "_recommended_threshold":  recommended_threshold,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    st.markdown(f"""
    <div class="app-header">
      <h1> SHAP Credit Scoring App </h1>
      <p>Sistem Penilaian Kelayakan Kredit
         &nbsp;·&nbsp; untuk Koperasi, BUMDes, dan Lembaga Keuangan Mikro Indonesia</p>
      <div class="kurs">Model: XGBoost + SHAP &nbsp;·&nbsp;
         Kurs referensi: 1 USD = Rp {USD_TO_IDR:,} &nbsp;·&nbsp; Portfolio Project</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        model     = load_model()
        scaler    = load_scaler()
        explainer = load_explainer(model)
    except FileNotFoundError as e:
        st.error(
            f" File tidak ditemukan: `{e.filename}`\n\n"
            "Jalankan dulu:\n"
            "- `python 02_feature_engineering.py` → `models/scaler.pkl`\n"
            "- `python 03_modeling_and_shap.py`   → `models/best_xgb_model.json`"
        )
        st.stop()

    inputs     = build_sidebar()
    threshold  = inputs.pop("_threshold")
    recommended_threshold = inputs.pop("_recommended_threshold")
    income_idr = inputs.pop("monthly_income_idr")

    # Feature engineering → scaling → prediksi
    X_raw    = compute_features(inputs)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=X_raw.columns)
    prob     = float(model.predict_proba(X_scaled)[0, 1])

    decision  = ("LAYAK" if prob < threshold
                 else "TIDAK LAYAK" if prob > threshold + 0.1
                 else "PERLU KAJIAN")
    badge_cls  = {"LAYAK": "badge-approve", "TIDAK LAYAK": "badge-reject",
                  "PERLU KAJIAN": "badge-review"}[decision]
    metric_cls = {"LAYAK": "safe", "TIDAK LAYAK": "risk",
                  "PERLU KAJIAN": "border"}[decision]

    shap_vals      = explainer.shap_values(X_scaled)[0]
    expected_value = explainer.expected_value

    col_left, col_right = st.columns([1, 1.15], gap="large")

    # ── Kolom kiri ────────────────────────────────────────────
    with col_left:
        st.plotly_chart(make_gauge(prob), use_container_width=True,
                        config={"displayModeBar": False})

        total_late   = inputs["late_30_59"] + inputs["late_60_89"] + inputs["late_90d"]
        monthly_debt = inputs["debt_ratio"] * inputs["monthly_income"]
        dti_pct      = inputs["debt_ratio"] * 100
        revolv_pct   = inputs["revolving_utilization"] * 100

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Keputusan Sistem</div>
            <div class="metric-value {metric_cls}">{decision}</div>
            <span class="decision-badge {badge_cls}">{prob*100:.1f}% risiko gagal bayar</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Riwayat Tunggakan</div>
            <div class="metric-value {'risk' if total_late > 2 else 'safe'}">{total_late}×</div>
            <div class="metric-sub">total kejadian menunggak</div>
          </div>
        </div>
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Pendapatan Bulanan</div>
            <div class="metric-value">{fmt_idr(income_idr)}</div>
            <div class="metric-sub">pendapatan bersih / bulan</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Cicilan Bulanan</div>
            <div class="metric-value {'risk' if dti_pct > 40 else 'safe'}">{fmt_idr(usd_to_idr(monthly_debt))}</div>
            <div class="metric-sub">Rasio DTI: {dti_pct:.0f}%</div>
          </div>
        </div>
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Beban Limit Kredit</div>
            <div class="metric-value {'risk' if revolv_pct > 70 else 'border' if revolv_pct > 30 else 'safe'}">{revolv_pct:.0f}%</div>
            <div class="metric-sub">{"[ALERT] Kritis >70%" if revolv_pct > 70 else "— Perhatikan" if revolv_pct > 30 else "[OK] Sehat <30%"}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Ambang Batas Aktif</div>
            <div class="metric-value">{threshold:.2f}</div>
            <div class="metric-sub">margin: {abs(prob - threshold):.4f}</div>
          </div>
        </div>
                <div class="info-box">
                    Threshold rekomendasi model: <b>{recommended_threshold:.2f}</b>. Slider di sidebar tetap bisa Anda ubah untuk simulasi risk appetite yang berbeda.
                </div>
        """, unsafe_allow_html=True)

        if decision == "LAYAK":
            st.markdown(
                '<div class="info-box">Profil menunjukkan kapasitas bayar yang baik '
                'dan riwayat kredit yang terjaga. Direkomendasikan untuk diproses lebih lanjut.</div>',
                unsafe_allow_html=True)
        elif decision == "TIDAK LAYAK":
            st.markdown(
                '<div class="warn-box">Risiko gagal bayar tinggi. Periksa faktor merah '
                'di grafik sebelah kanan sebagai dasar pertimbangan penolakan.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warn-box">Skor di zona abu-abu. Disarankan kajian '
                'manual oleh analis kredit sebelum keputusan akhir diberikan.</div>',
                unsafe_allow_html=True)

    # ── Kolom kanan: SHAP ─────────────────────────────────────
    with col_right:
        render_shap_waterfall(
            shap_vals, list(X_scaled.columns),
            X_raw.values[0], expected_value,
        )

    # ── Tabel detail semua indikator ──────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Detail Seluruh Indikator yang Dievaluasi Model</p>',
                unsafe_allow_html=True)

    raw_series = pd.Series(X_raw.values[0], index=X_raw.columns)
    rows = []
    for feat in X_scaled.columns:
        raw_val  = raw_series[feat]
        shap_val = shap_vals[list(X_scaled.columns).index(feat)]

        if feat in ("monthly_income", "income_per_dependent", "monthly_debt"):
            dval = fmt_idr(usd_to_idr(raw_val))
        elif feat in ("revolving_utilization", "debt_ratio", "dti_adjusted"):
            dval = f"{raw_val*100:.1f}%"
        elif feat == "ever_late":
            dval = "Ya" if raw_val >= 1 else "Tidak"
        elif feat == "log_income":
            dval = fmt_idr(usd_to_idr(np.expm1(raw_val)))
        elif feat == "utilization_risk":
            dval = ["Aman", "Perhatian", "Tinggi", "Kritis"][min(int(raw_val), 3)]
        else:
            dval = f"{raw_val:.2f}"

        if shap_val > 0.01:
            kontribusi = "[RISK] Memperburuk kelayakan"
        elif shap_val < -0.01:
            kontribusi = "[BOOST] Memperkuat kelayakan"
        else:
            kontribusi = "[NEUTRAL] Tidak signifikan"

        rows.append({
            "Indikator":     label(feat),
            "Nilai":         dval,
            "Skor SHAP":     round(shap_val, 4),
            "Kontribusi":    kontribusi,
            "Keterangan":    hint(feat) or "—",
        })

    tabel = pd.DataFrame(rows).sort_values("Skor SHAP", key=abs, ascending=False)
    st.dataframe(tabel, use_container_width=True, height=440, hide_index=True)

    st.markdown("---")
    st.markdown(
        f'<p style="font-size:10px;color:#B4B2A9;text-align:center;">'
        f'SHAP Credit Scoring App &nbsp;·&nbsp; Portfolio Project &nbsp;·&nbsp; '
        f'Dataset: Give Me Some Credit (Kaggle) &nbsp;·&nbsp; '
        f'Model: XGBoost + SHAP &nbsp;·&nbsp; Kurs: 1 USD = Rp {USD_TO_IDR:,}'
        f'</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
