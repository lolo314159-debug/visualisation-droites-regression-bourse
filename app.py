import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Indices Européens Automatisés", layout="wide")

# --- 1. RÉCUPÉRATION AUTOMATIQUE DES COMPOSANTS ---
@st.cache_data(ttl=86400)
def get_index_components(index_name):
    try:
        if index_name == "CAC 40 (France)":
            url = "https://en.wikipedia.org/wiki/CAC_40"
            df_wiki = pd.read_html(url)[4]
            return {row['Company']: row['Ticker'] + ".PA" for _, row in df_wiki.iterrows()}
        
        elif index_name == "DAX (Allemagne)":
            url = "https://en.wikipedia.org/wiki/DAX"
            df_wiki = pd.read_html(url)[4]
            return {row['Company']: row['Ticker'] for _, row in df_wiki.iterrows()}

        elif index_name == "EURO STOXX 50":
            url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
            df_wiki = pd.read_html(url)[4]
            # Mapping manuel pour STOXX (les suffixes varient : .AS, .DE, .PA...)
            return {row['Name']: row['Ticker'] for _, row in df_wiki.iterrows()}

        elif index_name == "IBEX 35 (Espagne)":
            url = "https://en.wikipedia.org/wiki/IBEX_35"
            df_wiki = pd.read_html(url)[2]
            return {row['Company']: row['Ticker'] + ".MC" for _, row in df_wiki.iterrows()}
            
    except Exception as e:
        st.error(f"Erreur de lecture Wikipedia : {e}")
        return {"LVMH": "MC.PA"}

# --- 2. INTERFACE ---
st.sidebar.title("🌍 Sélection")
idx_choice = st.sidebar.selectbox("Choisir l'indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)"])
stock_dict = get_index_components(idx_choice)
stock_name = st.sidebar.selectbox("Choisir la valeur", list(stock_dict.keys()))
symbol = stock_dict[stock_name]

@st.cache_data
def load_data(s):
    data = yf.download(s, start="2000-01-01")
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    return data.dropna()

df = load_data(symbol)

# --- 3. CALCULS ET GRAPHIQUES ---
if not df.empty:
    prices = df['Close'].values.flatten().astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    std_dev = np.std(y_log - y_pred_log)
    
    # CAGR
    try:
        y_count = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (pow(prices[-1] / prices[0], 1/y_count) - 1) * 100
        cagr_str = f"{cagr:.2f}%"
    except: cagr_str = "N/A"

    st.subheader(f"{stock_name} ({symbol})")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR", cagr_str)
    m2.metric("Fiabilité (R²)", f"{model.score(x, y_log):.4(f)}")
    m3.metric("Dernier Prix", f"{prices[-1]:.2f} €")

    tab1, tab2 = st.tabs(["📉 Échelle Log", "📈 Échelle Arithmétique"])

    def make_fig(log_mode):
        fig = go.Figure()
        y_trend = np.exp(y_pred_log)
        # Zone Sigma
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log+2*std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log-2*std_dev), fill='tonexty', fillcolor='rgba(255,215,0,0.1)', line=dict(width=0), name="Canal Sigma"))
        # Courbes
        fig.add_trace(go.Scatter(x=df.index, y=prices, name="Prix", line=dict(color='#00D4FF')))
        fig.add_trace(go.Scatter(x=df.index, y=y_trend, name="Tendance", line=dict(color='gold', dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, yaxis_type="log" if log_mode else "linear", margin=dict(l=0,r=0,t=10,b=0))
        return fig

    tab1.plotly_chart(make_fig(True), use_container_width=True)
    tab2.plotly_chart(make_fig(False), use_container_width=True)
else:
    st.warning(f"⚠️ Yahoo Finance ne trouve pas {symbol}. Essayez une autre valeur.")
