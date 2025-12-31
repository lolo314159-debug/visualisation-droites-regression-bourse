import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Analyse Boursière Avancée", layout="wide")

# --- 1. RÉCUPÉRATION DES COMPOSANTS (WIKIPEDIA) ---
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
            all_tables = pd.read_html(response.read())
            df_wiki = all_tables[info["table_idx"]]
        
        # Identification automatique des colonnes
        col_ticker = next((c for c in df_wiki.columns if c in ['Ticker', 'EPIC', 'Symbol']), df_wiki.columns[1])
        col_name = next((c for c in df_wiki.columns if c in ['Company', 'Name', 'Constituent']), df_wiki.columns[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            t = str(row[col_ticker]).strip().split(':')[ -1] # Nettoie les formats "PAR:MC"
            if info["suffix"] and not t.endswith(info["suffix"]):
                t += info["suffix"]
            stocks[str(row[col_name])] = t
        return stocks, info["bench"]
    except Exception as e:
        st.error(f"Erreur Wikipedia : {e}")
        return {"LVMH": "MC.PA"}, "^FCHI"

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🌍 Marchés Européens")
idx_choice = st.sidebar.selectbox("1. Choisir l'indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
stock_dict, bench_ticker = get_index_components(idx_choice)
stock_name = st.sidebar.selectbox("2. Choisir la valeur", sorted(list(stock_dict.keys())))
symbol = stock_dict[stock_name]

@st.cache_data
def load_data(s, b):
    # Téléchargement de l'action + l'indice de référence
    data = yf.download([s, b], start="2000-01-01")['Close']
    return data.dropna()

df_all = load_data(symbol, bench_ticker)

# --- 3. CALCULS ET GRAPHIQUES ---
if not df_all.empty and symbol in df_all.columns:
    prices = df_all[symbol].values.astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    std_dev = np.std(y_log - y_pred_log)
    
    # CAGR
    years = (df_all.index[-1] - df_all.index[0]).days / 365.25
    cagr = (pow(prices[-1] / prices[0], 1/years) - 1) * 100 if years > 0 else 0

    st.subheader(f"Analyse : {stock_name} vs {idx_choice}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR moyen", f"{cagr:.2f}%")
    m2.metric("R² (Fiabilité Log)", f"{model.score(x, y_log):.4f}")
    m3.metric("Dernier Prix", f"{prices[-1]:.2f} €")

    tab1, tab2 = st.tabs(["📉 Échelle Logarithmique", "📈 Échelle Arithmétique"])

    def create_fig(is_log):
        fig = go.Figure()
        # Canal Sigma
        y_u2 = np.exp(y_pred_log + 2*std_dev)
        y_l2 = np.exp(y_pred_log - 2*std_dev)
        
        fig.add_trace(go.Scatter(x=df_all.index, y=y_u2, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df_all.index, y=y_l2, fill='tonexty', fillcolor='rgba(255, 215, 0, 0.1)', line=dict(width=0), name="Canal Sigma"))
        
        # Indice de référence (normalisé au prix de départ de l'action)
        bench_prices = df_all[bench_ticker].values
        ratio = prices[0] / bench_prices[0]
        fig.add_trace(go.Scatter(x=df_all.index, y=bench_prices * ratio, name="Indice (base action)", line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')))
        
        # Action et Régression
        fig.add_trace(go.Scatter(x=df_all.index, y=prices, name="Prix Action", line=dict(color='#00D4FF', width=2)))
        fig.add_trace(go.Scatter(x=df_all.index, y=np.exp(y_pred_log), name="Tendance Log", line=dict(color='gold', dash='dash')))
        
        fig.update_layout(template="plotly_dark", height=500, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0))
        return fig

    tab1.plotly_chart(create_fig(True), use_container_width=True)
    tab2.plotly_chart(create_fig(False), use_container_width=True)
else:
    st.error(f"Données Yahoo Finance indisponibles pour {symbol}.")
