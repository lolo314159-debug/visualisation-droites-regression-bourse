import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Analyse Boursière Pro", layout="wide")

# --- 1. RÉCUPÉRATION DES COMPOSANTS ---
@st.cache_data(ttl=86400)
def get_index_components(index_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    indices = {
        "CAC 40 (France)": {"url": "https://en.wikipedia.org/wiki/CAC_40", "suffix": ".PA", "table_idx": 4, "bench": "^FCHI"},
        "DAX (Allemagne)": {"url": "https://en.wikipedia.org/wiki/DAX", "suffix": ".DE", "table_idx": 4, "bench": "^GDAXI"},
        "EURO STOXX 50": {"url": "https://en.wikipedia.org/wiki/EURO_STOXX_50", "suffix": "", "table_idx": 4, "bench": "^STOXX50E"},
        "IBEX 35 (Espagne)": {"url": "https://en.wikipedia.org/wiki/IBEX_35", "suffix": ".MC", "table_idx": 2, "bench": "^IBEX"},
        "FTSE 100 (UK)": {"url": "https://en.wikipedia.org/wiki/FTSE_100_Index", "suffix": ".L", "table_idx": 4, "bench": "^FTSE"}
    }
    
    try:
        info = indices[index_name]
        req = urllib.request.Request(info["url"], headers=headers)
        with urllib.request.urlopen(req) as response:
            df_wiki = pd.read_html(response.read())[info["table_idx"]]
        
        # Nettoyage intelligent des tickers
        col_ticker = 'Ticker' if 'Ticker' in df_wiki.columns else ('EPIC' if 'EPIC' in df_wiki.columns else df_wiki.columns[1])
        col_name = 'Company' if 'Company' in df_wiki.columns else ('Name' if 'Name' in df_wiki.columns else df_wiki.columns[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            t = str(row[col_ticker]).strip()
            # On n'ajoute le suffixe que s'il n'est pas déjà présent
            if info["suffix"] and not t.endswith(info["suffix"]):
                t += info["suffix"]
            stocks[row[col_name]] = t
        return stocks, info["bench"]
    except Exception as e:
        st.error(f"Erreur : {e}")
        return {"LVMH": "MC.PA"}, "^FCHI"

# --- 2. INTERFACE ---
st.sidebar.title("🌍 Marchés")
idx_choice = st.sidebar.selectbox("Indices", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
stock_dict, bench_ticker = get_index_components(idx_choice)
stock_name = st.sidebar.selectbox("Valeurs", sorted(list(stock_dict.keys())))
symbol = stock_dict[stock_name]

@st.cache_data
def load_data(s, b):
    # Téléchargement de l'action + l'indice de référence
    data = yf.download([s, b], start="2000-01-01")['Close']
    return data.dropna()

df_all = load_data(symbol, bench_ticker)

# --- 3. CALCULS ET GRAPHIQUES ---
if not df_all.empty and symbol in df_all.columns:
    df = df_all[[symbol]].rename(columns={symbol: 'Close'})
    prices = df['Close'].values.astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    std_dev = np.std(y_log - y_pred_log)
    
    # CAGR
    y_count = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (pow(prices[-1] / prices[0], 1/y_count) - 1) * 100

    st.subheader(f"{stock_name} vs {idx_choice}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR", f"{cagr:.2f}%")
    c2.metric("R² (Qualité)", f"{model.score(x, y_log):.4f}")
    c3.metric("Prix Actuel", f"{prices[-1]:.2f}")

    tab1, tab2 = st.tabs(["📉 Échelle Log", "📈 Échelle Arithmétique"])

    def create_plot(is_log):
        fig = go.Figure()
        # Canal Sigma
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log+2*std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log-2*std_dev), fill='tonexty', fillcolor='rgba(255,215,0,0.1)', line=dict(width
