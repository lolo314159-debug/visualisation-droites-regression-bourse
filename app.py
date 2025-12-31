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
        
        stocks = {str(row[col_name]): str(row[col_ticker]).strip().split(':')[-1] for _, row in df_wiki.iterrows()}
        for name in stocks:
            if info["suffix"] and not stocks[name].endswith(info["suffix"]):
                stocks[name] += info["suffix"]
        return stocks, info["bench"]
    except Exception as e:
        return {"Air Liquide": "AI.PA"}, "^FCHI"

# --- 2. FONCTIONS STATISTIQUES ---
def get_metrics(prices_series):
    prices = prices_series.dropna().values.astype(float)
    if len(prices) < 50: return 0.0, 0.0, 0.0, None
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    r2 = model.score(x, y_log)
    returns = np.diff(y_log)
    vol = np.std(returns) * np.sqrt(252) * 100
    years = len(prices) / 252
    cagr = (pow(prices[-1] / prices[0], 1/years) - 1) * 100 if years > 0 else 0
    return r2, vol, cagr, y_pred_log

def get_r2_interpretation(v):
    if v > 0.98: return "💎 **Diamant** : Modèle de régularité absolue."
    if v > 0.93: return "🌟 **Exceptionnel** : Croissance prévisible et stable."
    if v > 0.85: return "✅ **Très Bon** : Tendance de fond robuste."
    if v > 0.70: return "⚠️ **Correct** : Présence de cycles ou volatilité."
    return "❌ **Spéculatif** : Faible corrélation temporelle."

# --- 3. FILTRAGE ET INTERFACE ---
st.sidebar.title("⚙️ Filtres")
idx_choice = st.sidebar.selectbox("Indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
r2_min = st.sidebar.slider("R² min (Historique)", 0.0, 1.0, 0.90, 0.01)

base_stocks, bench_ticker = get_index_components(idx_choice)

@st.cache_data(ttl=3600)
def get_strictly_filtered_list(stocks_dict, r2_threshold):
    tickers = list(stocks_dict.values())
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            r2_val, _, _, _ = get_metrics(data[ticker])
            if r2_val >= r2_threshold:
                results.append({"name": name, "r2": r2_val})
    return [item['name'] for item in sorted(results, key=lambda x: x['r2'], reverse=True)]

filtered_names = get_strictly_filtered_list(base_stocks, r2_min)

if not filtered_names:
    st.sidebar.error("Aucun résultat.")
    st.stop()

stock_name = st.sidebar.selectbox(f"Valeurs ({len(filtered_names)})", filtered_names)
symbol = base_stocks[stock_name]

# --- 4. ANALYSE ET AFFICHAGE ---
df_full = yf.download(symbol, start="2000-01-01", progress=False)['Close']
if isinstance(df_full, pd.DataFrame): df_full = df_full.iloc[:, 0]

r2, vol, cagr, y_pred_log = get_metrics(df_full)
std_dev = np.std(np.log(df_full.values) - y_pred_log)
curr, theo = df_full.iloc[-1], np.exp(y_pred_log[-1])

# Header Compact
st.markdown(f"### 📈 {stock_name} ({symbol}) | {get_r2_interpretation(r2)}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("CAGR", f"{cagr:.2f}%")
m2.metric("Volatilité", f"{vol:.1f}%")
m3.metric("R²", f"{r2:.4f}")
m4.metric("Position", f"{((curr/theo)-1)*100:+.1f}%")

# Graphique
tab1, tab2 = st.tabs(["Log", "Linéaire"])

def create_plot(is_log):
    fig = go.Figure()
    y_trend = np.exp(y_pred_log)
    dates = df_full.index
    
    # Zones Sigma
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log + 2*std_dev), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.04)', line=dict(width=0), name="95%"))
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log + std_dev), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=np.exp(y_pred_log - std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.12)', line=dict(width=0), name="68%"))
    
    fig.add_trace(go.Scatter(x=dates, y=df_full.values, name="Prix", line=dict(color='#00D4FF', width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=y_trend, name="Trend", line=dict(color='gold', width=1, dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=450, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=-0.15))
    return fig

tab1.plotly_chart(create_plot(True), use_container_width=True)
tab2.plotly_chart(create_plot(False), use_container_width=True)

# Seuils en dessous
st.markdown("---")
t = st.columns(5)
t[0].caption("Support -2σ"); t[0].write(f"**{np.exp(y_pred_log[-1] - 2*std_dev):.2f}**")
t[1].caption("Support -1σ"); t[1].write(f"**{np.exp(y_pred_log[-1] - std_dev):.2f}**")
t[2].caption("THÉORIQUE"); t[2].write(f"**{theo:.2f}**")
t[3].caption("Résistance +1σ"); t[3].write(f"**{np.exp(y_pred_log[-1] + std_dev):.2f}**")
t[4].caption("Résistance +2σ"); t[4].write(f"**{np.exp(y_pred_log[-1] + 2*std_dev):.2f}**")
