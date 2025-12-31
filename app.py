import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Expert Indices Européens", layout="wide")

# --- 1. FONCTION DE RÉCUPÉRATION AVEC IDENTITÉ (USER-AGENT) ---
@st.cache_data(ttl=86400)
def get_index_components(index_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    urls = {
        "CAC 40 (France)": "https://en.wikipedia.org/wiki/CAC_40",
        "DAX (Allemagne)": "https://en.wikipedia.org/wiki/DAX",
        "EURO STOXX 50": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
        "IBEX 35 (Espagne)": "https://en.wikipedia.org/wiki/IBEX_35",
        "FTSE 100 (UK)": "https://en.wikipedia.org/wiki/FTSE_100_Index"
    }
    
    try:
        url = urls[index_name]
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            all_tables = pd.read_html(response.read())
            
        # Sélection de la table et formatage des tickers selon l'indice
        if index_name == "CAC 40 (France)":
            df_wiki = all_tables[4]
            return {row['Company']: row['Ticker'] + ".PA" for _, row in df_wiki.iterrows()}
        elif index_name == "DAX (Allemagne)":
            df_wiki = all_tables[4]
            return {row['Company']: row['Ticker'] for _, row in df_wiki.iterrows()}
        elif index_name == "EURO STOXX 50":
            df_wiki = all_tables[4]
            return {row['Name']: row['Ticker'] for _, row in df_wiki.iterrows()}
        elif index_name == "IBEX 35 (Espagne)":
            df_wiki = all_tables[2]
            return {row['Company']: row['Ticker'] + ".MC" for _, row in df_wiki.iterrows()}
        elif index_name == "FTSE 100 (UK)":
            df_wiki = all_tables[4]
            return {row['Company']: row['EPIC'] + ".L" for _, row in df_wiki.iterrows()}
            
    except Exception as e:
        st.error(f"Erreur d'accès aux données : {e}")
        return {"LVMH": "MC.PA"}

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🌍 Marchés Européens")
idx_choice = st.sidebar.selectbox("Indices", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
stock_dict = get_index_components(idx_choice)
stock_name = st.sidebar.selectbox("Actions", sorted(list(stock_dict.keys())))
symbol = stock_dict[stock_name]

@st.cache_data
def load_data(s):
    data = yf.download(s, start="2000-01-01")
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    return data.dropna()

df = load_data(symbol)

# --- 3. CALCULS ET VISUALISATION ---
if not df.empty and len(df) > 10:
    prices = df['Close'].values.flatten().astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    std_dev = np.std(y_log - y_pred_log)
    
    # Statistiques
    y_count = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (pow(prices[-1] / prices[0], 1/y_count) - 1) * 100 if y_count > 0 else 0

    st.subheader(f"Analyse Graphique : {stock_name} ({symbol})")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR", f"{cagr:.2f}%")
    c2.metric("Corrélation (R²)", f"{model.score(x, y_log):.4f}")
    c3.metric("Dernier Prix", f"{prices[-1]:.2f}")

    tab1, tab2 = st.tabs(["📉 Échelle Logarithmique", "📈 Échelle Arithmétique"])

    def create_plot(is_log):
        fig = go.Figure()
        y_trend = np.exp(y_pred_log)
        # Bandes Sigma
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log+2*std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log-2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.12)', line=dict(width=0), name="Canal de Tendance"))
        # Prix et Régression
        fig.add_trace(go.Scatter(x=df.index, y=prices, name="Cours", line=dict(color='#00D4FF', width=1.8)))
        fig.add_trace(go.Scatter(x=df.index, y=y_trend, name="Régression", line=dict(color='gold', dash='dash')))
        
        fig.update_layout(template="plotly_dark", height=480, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0))
        return fig

    tab1.plotly_chart(create_plot(True), use_container_width=True)
    tab2.plotly_chart(create_plot(False), use_container_width=True)
else:
    st.warning(f"Données Yahoo Finance indisponibles pour {symbol}.")
