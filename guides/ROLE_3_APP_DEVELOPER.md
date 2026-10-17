# ROLE GUIDE — PERSON 3: APP & DEMO LEAD

**Title**: Lead Full-Stack Application Developer & UX Lead  
**Core Mission**: Build the live clinical telemetry dashboard in Streamlit that dazzles judges during the 3-minute pitch. Integrate the real-time CTG strip visualizer, FIGO triage status badges, and SHAP explainability cards.  
**Tech Stack**: Python (`streamlit`, `plotly`, `pandas`, `numpy`)

---

## 1. Quick Checklist & Deliverables

- [ ] **Hour 0–2**: Set up Streamlit environment (`pip install streamlit plotly pandas numpy joblib xgboost`).
- [ ] **Hour 2–6**: Build dual-channel CTG strip chart in Plotly (FHR on top with 110–160 bpm green safety zone; UC contractions on bottom).
- [ ] **Hour 6–10**: Add playback controls (Play / Pause / 10-Minute Window Step slider).
- [ ] **Hour 10–14**: Design FIGO Traffic-Light status cards:
  - 🟢 **Category I**: Reassuring
  - 🟡 **Category II**: Suspicious
  - 🔴 **Category III**: Emergency / Pathological
- [ ] **Hour 14–18**: Connect Person 2's `model_xgb.pkl` or import `NexoGuardModel` to update risk scores in real-time.
- [ ] **Hour 18–22**: Build the SHAP Clinical Feature Attribution visualizer card.
- [ ] **Hour 22+**: Add "Generate Shift Handoff Report" button and lock down the demo to run 100% offline without internet.

---

## 2. Complete Starter Code: `app.py`

Save this file in your project folder as `app.py`. Run it via: `streamlit run app.py`

```python
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration for wide clinical layout
st.set_page_config(
    page_title="NexoGuard AI — Clinical CTG Telemetry",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom medical dashboard styling
st.markdown("""
<style>
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #334155;
        background-color: #1e293b;
        margin-bottom: 12px;
    }
    .badge-cat1 { background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
    .badge-cat2 { background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; }
    .badge-cat3 { background-color: rgba(244, 63, 94, 0.15); border: 1px solid #f43f5e; color: #fb7185; }
</style>
""", unsafe_allow_html=True)

# ----------------- SYNTHETIC PATIENT DATA GENERATOR -----------------
def generate_patient_strip(patient_type="pathological", duration_min=10, hz=4):
    n = duration_min * 60 * hz
    t = np.linspace(0, duration_min, n)
    
    if patient_type == "normal":
        base = 140
        fhr = base + 5 * np.sin(t * 1.2) + np.random.normal(0, 2.5, n)
        # 2 Benign accelerations (>15 bpm for 20s)
        fhr[400:500] += 18
        fhr[1600:1700] += 20
        uc = 12 + 35 * np.maximum(0, np.sin(t * 0.8))**4
        meta = {"cat": "Category I", "status": "Normal (Reassuring)", "risk": 0.04, "color": "badge-cat1", "stv": 5.8, "lag": 0}
        factors = ["High Beat-to-Beat STV (5.8 ms)", "Normal Baseline (140 bpm)", "Accelerations Present (+2)"]
    elif patient_type == "suspicious":
        base = 158
        fhr = base + 2 * np.sin(t * 0.9) + np.random.normal(0, 1.2, n)
        uc = 15 + 45 * np.maximum(0, np.sin(t * 0.8))**4
        # Early deceleration (mirrors contraction peak)
        fhr -= 15 * np.maximum(0, np.sin(t * 0.8))**4
        meta = {"cat": "Category II", "status": "Suspicious (Indeterminate)", "risk": 0.48, "color": "badge-cat2", "stv": 3.2, "lag": 5}
        factors = ["Minimal Variability (STV = 3.2 ms)", "Tachycardia Baseline (158 bpm)", "Absence of Accelerations"]
    else: # pathological
        base = 124
        fhr = base + 1.5 * np.sin(t * 0.5) + np.random.normal(0, 1.0, n)
        uc = 15 + 55 * np.maximum(0, np.sin(t * 0.8))**4
        # LATE DECELERATION: delayed by ~30 seconds after contraction peak
        t_delayed = np.maximum(0, t - 0.5)
        fhr -= 35 * np.maximum(0, np.sin(t_delayed * 0.8))**4
        meta = {"cat": "Category III", "status": "Pathological (Urgent)", "risk": 0.91, "color": "badge-cat3", "stv": 2.4, "lag": 34}
        factors = ["Late Deceleration Lag (Δt = 34s)", "Severe Loss of STV (2.4 ms)", "Deceleration Depth (35 bpm drop)"]
        
    return t, fhr, uc, meta, factors

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("👶 NexoGuard AI")
st.sidebar.caption("Physiology-Informed Fetal Telemetry")

patient_choice = st.sidebar.selectbox(
    "Select Clinical Case:",
    ["Patient 309: Acute Distress (Late Decel)", "Patient 101: Normal Reassuring", "Patient 204: Suspicious Flatline"]
)

if "309" in patient_choice:
    p_type = "pathological"
elif "101" in patient_choice:
    p_type = "normal"
else:
    p_type = "suspicious"

show_ema = st.sidebar.checkbox("Show NexoPrima EMA Baseline", value=True)
show_normal_zone = st.sidebar.checkbox("Highlight FIGO Normal Range (110–160 bpm)", value=True)

# ----------------- MAIN CLINICAL LAYOUT -----------------
st.title("Obstetric Real-Time Telemetry & Decision Support")
st.caption("Synchronized Cardiotocography Analysis • FIGO 2015 3-Tier Classification • NexoPrima Research Track")

t, fhr, uc, meta, factors = generate_patient_strip(p_type)

col_strip, col_triage = st.columns([7, 3])

# --- COLUMN 1: DUAL CHANNEL STRIP CHART ---
with col_strip:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Channel 1: Fetal Heart Rate (bpm)", "Channel 2: Uterine Contractions / TOCO (mmHg)"),
        row_heights=[0.7, 0.3]
    )

    # Normal baseline zone green shading (110 - 160 bpm)
    if show_normal_zone:
        fig.add_hrect(
            y0=110, y1=160, row=1, col=1,
            fillcolor="rgba(16, 185, 129, 0.08)", line_width=0,
            annotation_text="FIGO Normal Zone (110–160 bpm)", annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#10b981"
        )

    # FHR trace
    line_color = "#f43f5e" if p_type == "pathological" else ("#f59e0b" if p_type == "suspicious" else "#10b981")
    fig.add_trace(go.Scatter(x=t, y=fhr, mode="lines", name="FHR", line=dict(color=line_color, width=1.8)), row=1, col=1)

    # NexoPrima EMA Baseline overlay
    if show_ema:
        ema_baseline = pd.Series(fhr).ewm(alpha=0.85).mean()
        fig.add_trace(go.Scatter(x=t, y=ema_baseline, mode="lines", name="NexoPrima Baseline", line=dict(color="#cbd5e1", width=1.5, dash="dash")), row=1, col=1)

    # Contractions trace
    fig.add_trace(go.Scatter(x=t, y=uc, mode="lines", name="TOCO Contractions", line=dict(color="#38bdf8", width=2)), row=2, col=1)

    fig.update_yaxes(title_text="FHR (bpm)", range=[60, 200], row=1, col=1, gridcolor="#334155")
    fig.update_yaxes(title_text="TOCO (mmHg)", range=[0, 100], row=2, col=1, gridcolor="#334155")
    fig.update_xaxes(title_text="Time (Minutes)", row=2, col=1, gridcolor="#334155")

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=40, b=30),
        plot_bgcolor="#090d16",
        paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- COLUMN 2: TRIAGE & SHAP EXPLAINABILITY ---
with col_triage:
    st.markdown(f"""
    <div class="metric-card {meta['color']}">
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">FIGO 2015 Clinical Triage</div>
        <div style="font-size: 26px; font-weight: 900; margin-top: 4px;">{meta['cat']}</div>
        <div style="font-size: 13px; font-weight: 600;">{meta['status']}</div>
        <div style="margin-top: 14px; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
            <div><strong>Hypoxia Risk:</strong> {meta['risk']*100:.1f}%</div>
            <div><strong>STV:</strong> {meta['stv']} ms {'(Critical < 3.0)' if meta['stv'] < 3.0 else '(Healthy)'}</div>
            <div><strong>Contraction-Decel Lag:</strong> {meta['lag']}s {'(⚠️ Late Decel)' if meta['lag'] > 20 else ''}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">
            SHAP Clinical Feature Attribution
        </div>
        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">Top physiological drivers of this triage:</div>
    """, unsafe_allow_html=True)

    for factor in factors:
        st.markdown(f"""
        <div style="background-color: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 12px; margin-bottom: 6px; border: 1px solid #334155;">
            {factor}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("📄 Generate Clinical Handoff Report", use_container_width=True):
        st.success(f"Report generated for {patient_choice.split(':')[0]}! Ready for obstetrician handoff.")
```

---

## 3. What to Hand Off & Coordinate

1. **For Person 4 (Pitch Lead)**: Provide a clean browser window running `streamlit run app.py`. Practice toggling between Patient 101 and Patient 309 while Person 4 gives the pitch script.
2. **For Person 2 (ML Lead)**: If Person 2 finishes `model_xgb.pkl`, replace the synthetic `generate_patient_strip()` function with Person 2's `model.predict_window()`!

---

## 4. Stuck? How to Ask AI

Copy the content of `guides/PROJECT_MASTER_CONTEXT_FOR_AI.md` into your AI prompt, and add:
```
I am Person 3 (App Lead). 
I am building the Streamlit dashboard in app.py.
Here is my current code: [paste code]
I want to [e.g. make the strip chart scroll smoothly / add a gauge indicator / style the alert cards].
Please write the updated Streamlit and Plotly code.
```
