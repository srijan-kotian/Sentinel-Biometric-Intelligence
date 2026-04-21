import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_autorefresh import st_autorefresh

# 1. OPTIMIZED REFRESH (2s)
st_autorefresh(interval=2000, key="biometric_refresh")

# 2. PRO CONFIG
st.set_page_config(page_title="Sentinel OS | Biometric Portal", layout="wide", initial_sidebar_state="collapsed")

# 3. LOTTIE ASSETS (Smooth AI Animations)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m6cu9msc.json") # Scanning animation

# 4. HARDWARE-ACCELERATED CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #05070a; }
    
    /* Smooth Transition Cards */
    div.stMetric {
        background: rgba(16, 20, 24, 0.6);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 20px !important;
        transition: all 0.3s ease-in-out;
    }
    div.stMetric:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    
    /* Remove flickering scrollbars */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }

    /* Glassmorphism Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: 0.3s;
    }
    
    h1, h2, h3 { letter-spacing: -0.05rem; }
    </style>
    """, unsafe_allow_html=True)

# 5. HIGH-PERFORMANCE DATA ENGINE
@st.cache_data(ttl=1) # Cache data for 1 second to prevent file-locking lag
def get_telemetry_data():
    file_path = "recognition_analytics.csv"
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Confidence'] = (1 - df['Distance']) * 100
    # Data Normalization
    df['Name'] = df['Name'].apply(lambda x: "Srijan S Kotian" if "SRIJAN" in x.upper() else x)
    return df

# --- INTERFACE ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("🛰️ SENTINEL OS | BIOMETRIC NODE")
    st.markdown("`FAST-API INFERENCE ENGINE` | `LATENT SPACE: STABLE` | `LOCATION: UDUPI_IN` ")
with header_col2:
    if lottie_ai: st_lottie(lottie_ai, height=100, key="ai_icon")

df = get_telemetry_data()

if df is not None:
    # 6. SYSTEM HEALTH CALCULATIONS
    avg_conf = df['Confidence'].mean()
    std_dev = df['Confidence'].std()
    
    # METRIC GRID
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AVG CONFIDENCE", f"{round(avg_conf, 1)}%", delta=f"{len(df)} SAMPLES")
    m2.metric("SIGNAL STABILITY", f"{round(100-(std_dev*3),1)}%", delta_color="normal")
    m3.metric("LATENT NOISE (σ)", f"{round(std_dev, 2)}")
    m4.metric("ENGINE UPTIME", f"{round((df['Timestamp'].max()-df['Timestamp'].min()).total_seconds()/60, 1)}m")

    # 7. INTERACTIVE PLOTLY ENGINE (Zero Jitter)
    tab1, tab2 = st.tabs(["📊 SYSTEM TELEMETRY", "📅 AUDIT TRAIL"])
    
    with tab1:
        # Confidence Trend with Range Slider
        fig = px.area(df.tail(100), x="Timestamp", y="Confidence", 
                     color_discrete_sequence=['#58a6ff'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="#c9d1d9", margin=dict(l=0, r=0, t=20, b=0),
            height=300, xaxis_title="", yaxis_title="Score (%)"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Distribution Histogram
        dist_fig = px.histogram(df, x="Confidence", nbins=20, marginal="rug",
                               color_discrete_sequence=['#238636'])
        dist_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color="#c9d1d9", height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(dist_fig, use_container_width=True)

    with tab2:
        df['Date'] = df['Timestamp'].dt.date
        summary = df.groupby(['Date', 'Name']).agg(
            Check_In=('Timestamp', 'min'),
            Check_Out=('Timestamp', 'max'),
            Samples=('Confidence', 'count')
        ).reset_index()
        summary['Duration'] = (summary['Check_Out'] - summary['Check_In']).astype(str).str.split('.').str[0]
        
        st.dataframe(summary.sort_values(by='Date', ascending=False), use_container_width=True)

else:
    st.warning("⚠️ CRITICAL: Inference stream offline. Waiting for telemetry packet...")

st.markdown("---")
st.caption("v2.6-PRO | HARDWARE ACCELERATED RENDERING | DESIGNED FOR SRIJAN S KOTIAN")