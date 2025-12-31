import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Analyse Statistique Expert", layout="wide")

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
            all_tables = pd.read_html(response.read())
            df_wiki = all_tables[info["table_idx"]]
        
        col_ticker = next((c for c in df_wiki.columns if c in ['Ticker', 'EPIC', 'Symbol']), df_wiki.columns[1])
        col_name = next((c for c in df_wiki.columns if c in ['Company', 'Name', 'Constituent']), df_wiki.columns[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            t = str(row[col_ticker]).strip().split(':')[-1]
            if info["suffix"] and not t.endswith(info["suffix"]):
                t += info["suffix"]
            stocks[str(row[col_name])] = t
        return stocks, info["bench"]
    except Exception as e:
        st.error(f"Erreur Wikipedia : {e}")
        return {"LVMH": "MC.PA"}, "^FCHI"

# --- 2. FONCTION DE CALCUL STATISTIQUE ---
def analyze_stock(prices):
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    r2 = model.score(x, y_log)
    
    # Volatilité annualisée (basée sur les rendements log)
    returns = np.diff(y_log)
    volatility = np.std(returns) * np.sqrt(252) * 100
    
    return r2, volatility, y_pred_log

# --- 3. INTERFACE LATÉRALE (FILTRES) ---
st.sidebar.title("🛠 Configuration")
idx_choice = st.sidebar.selectbox("1. Choisir l'indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
r2_filter = st.sidebar.slider("2. Filtre R² minimum (Qualité)", 0.0, 1.0, 0.70, 0.05)

stock_dict, bench_ticker = get_index_components(idx_choice)

# --- 4. CHARGEMENT ET FILTRAGE ---
@st.cache_data
def load_and_filter(stocks, bench):
    # On télécharge les prix de clôture récents pour filtrer par R2 rapidement
    all_data = yf.download(list(stocks.values()), period="5y", interval="1d")['Close']
    valid_stocks = {}
    
    for name, ticker in stocks.items():
        if ticker in all_data.columns:
            s_prices = all_data[ticker].dropna().values
            if len(s_prices) > 100:
                r2, _, _ = analyze_stock(s_prices)
                if r2 >= r2_filter:
                    valid_stocks[name] = ticker
    return valid_stocks

with st.spinner("Filtrage des actions selon le R²..."):
    filtered_dict = load_and_filter(stock_dict, bench_ticker)

if not filtered_dict:
    st.warning(f"Aucune action ne correspond à un R² de {r2_filter}. Essayez de baisser le filtre.")
    st.stop()

stock_name = st.sidebar.selectbox("3. Choisir la valeur filtrée", sorted(list(filtered_dict.keys())))
symbol = filtered_dict[stock_name]

# --- 5. ANALYSE DÉTAILLÉE ---
@st.cache_data
def get_full_data(s, b):
    return yf.download([s, b], start="2000-01-01")['Close'].dropna()

df_all = get_full_data(symbol, bench_ticker)

if not df_all.empty:
    prices = df_all[symbol].values.astype(float)
    r2, vol, y_pred_log = analyze_stock(prices)
    std_dev = np.std(np.log(prices) - y_pred_log)
    
    # Valeurs actuelles
    current_p = prices[-1]
    theo_p = np.exp(y_pred_log[-1])
    s1_up, s1_down = np.exp(y_pred_log[-1] + std_dev), np.exp(y_pred_log[-1] - std_dev)
    s2_up, s2_down = np.exp(y_pred_log[-1] + 2*std_dev), np.exp(y_pred_log[-1] - 2*std_dev)
    diff_theo = ((current_p / theo_p) - 1) * 100

    # --- AFFICHAGE ---
    st.title(f"📊 {stock_name} ({symbol})")
    
    # Dashboard de métriques
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fiabilité (R²)", f"{r2:.4f}")
    m2.metric("Volatilité Ann.", f"{vol:.2f} %")
    m3.metric("Prix Actuel", f"{current_p:.2f} €")
    m4.metric("Position / Moyenne", f"{diff_theo:+.2f}%")

    # Tableau des seuils
    st.markdown("### 🎯 Niveaux de Régression (Prix)")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Support -2σ", f"{s2_down:.2f} €")
    t2.metric("Support -1σ", f"{s1_down:.2f} €")
    t3.metric("Moyenne", f"{theo_p:.2f} €")
    t4.metric("Résistance +1σ", f"{s1_up:.2f} €")
    t5.metric("Résistance +2σ", f"{s2_up:.2f} €")

    # --- GRAPHIQUE ---
    fig = go.Figure()
    dates = df_all.index
    y_trend = np.exp(y_pred_log)
    
    # Enveloppes Sigma
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log + 2*std_dev), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.05)', line=dict(width=0), name="+/- 2σ"))
    
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Cours", line=dict(color='#00D4FF', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=y_trend, name="Tendance", line=dict(color='gold', width=1, dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)
