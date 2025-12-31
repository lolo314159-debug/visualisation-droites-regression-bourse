import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Analyse Statistique Expert", layout="wide")

# --- 1. RÉCUPÉRATION DES COMPOSANTS (Source Wikipedia) ---
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
        
        stocks = {str(row[col_name]): str(row[col_ticker]).strip().split(':')[-1] for _, row in df_wiki.iterrows()}
        # Ajout du suffixe si manquant
        for name in stocks:
            if info["suffix"] and not stocks[name].endswith(info["suffix"]):
                stocks[name] += info["suffix"]
        return stocks, info["bench"]
    except Exception as e:
        st.error(f"Erreur Wikipedia : {e}")
        return {"LVMH": "MC.PA"}, "^FCHI"

# --- 2. FONCTION DE CALCUL ---
def get_metrics(prices_series):
    prices = prices_series.dropna().values.astype(float)
    if len(prices) < 50: return 0.0, 0.0, None
    
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    r2 = model.score(x, y_log)
    
    returns = np.diff(y_log)
    vol = np.std(returns) * np.sqrt(252) * 100
    return r2, vol, y_pred_log

# --- 3. BARRE LATÉRALE ---
st.sidebar.title("🛠 Filtres Stratégiques")
idx_choice = st.sidebar.selectbox("1. Indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
r2_min = st.sidebar.slider("2. R² minimum (Qualité)", 0.0, 1.0, 0.80, 0.05)

# Chargement initial des composants
base_stocks, bench_ticker = get_index_components(idx_choice)

# --- 4. FILTRAGE DYNAMIQUE ---
@st.cache_data(ttl=3600)
def get_filtered_list(stocks_dict, r2_threshold):
    tickers = list(stocks_dict.values())
    # Téléchargement rapide des 2 dernières années pour le filtrage
    data = yf.download(tickers, period="2y", interval="1d", progress=False)['Close']
    
    valid_names = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            r2, _, _ = get_metrics(data[ticker])
            if r2 >= r2_threshold:
                valid_names.append(name)
    return sorted(valid_names)

with st.sidebar:
    st.write("---")
    with st.spinner("Analyse des R² en cours..."):
        filtered_names = get_filtered_list(base_stocks, r2_min)
    
    if filtered_names:
        stock_name = st.selectbox(f"3. Valeurs éligibles ({len(filtered_names)})", filtered_names)
        symbol = base_stocks[stock_name]
    else:
        st.error("Aucune action ne dépasse ce seuil.")
        st.stop()

# --- 5. ANALYSE DÉTAILLÉE ---
if symbol:
    df_full = yf.download(symbol, start="2000-01-01", progress=False)['Close']
    if isinstance(df_full, pd.DataFrame): df_full = df_full.iloc[:, 0]
    
    r2, vol, y_pred_log = get_metrics(df_full)
    std_dev = np.std(np.log(df_full.values) - y_pred_log)
    
    # Valeurs actuelles
    curr = df_full.iloc[-1]
    theo = np.exp(y_pred_log[-1])
    diff = ((curr / theo) - 1) * 100

    # Affichage des métriques
    st.title(f"📊 {stock_name} ({symbol})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fiabilité (R²)", f"{r2:.4f}")
    col2.metric("Volatilité Ann.", f"{vol:.2f} %")
    col3.metric("Prix Actuel", f"{curr:.2f} €")
    col4.metric("Position / Droite", f"{diff:+.2f}%")

    # Seuils Sigma
    st.markdown("---")
    s1_u, s1_d = np.exp(y_pred_log[-1] + std_dev), np.exp(y_pred_log[-1] - std_dev)
    s2_u, s2_d = np.exp(y_pred_log[-1] + 2*std_dev), np.exp(y_pred_log[-1] - 2*std_dev)
    
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Support -2σ", f"{s2_d:.2f}")
    t2.metric("Support -1σ", f"{s1_d:.2f}")
    t3.metric("Théorique", f"{theo:.2f}")
    t4.metric("Résistance +1σ", f"{s1_u:.2f}")
    t5.metric("Résistance +2σ", f"{s2_u:.2f}")

    # Graphique
    fig = go.Figure()
    y_trend = np.exp(y_pred_log)
    
    fig.add_trace(go.Scatter(x=df_full.index, y=np.exp(y_pred_log + 2*std_dev), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_full.index, y=np.exp(y_pred_log - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.05)', line=dict(width=0), name="Canal 2σ"))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full.values, name="Cours", line=dict(color='#00D4FF', width=2)))
    fig.add_trace(go.Scatter(x=df_full.index, y=y_trend, name="Régression", line=dict(color='gold', dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log", margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)
