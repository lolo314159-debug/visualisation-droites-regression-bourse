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
        st.error(f"Erreur Wikipedia : {e}")
        return {"Air Liquide": "AI.PA"}, "^FCHI"

# --- 2. FONCTIONS STATISTIQUES ---
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

def get_r2_interpretation(value):
    if value > 0.95: return "🌟 **Exceptionnelle** : Croissance quasi-parfaite."
    if value > 0.90: return "🌟 **Excellente** : Trajectoire très régulière."
    if value > 0.80: return "✅ **Bonne** : Tendance de fond solide."
    if value > 0.65: return "⚠️ **Moyenne** : Volatilité importante."
    return "❌ **Faible** : Le modèle log-linéaire n'est pas adapté."

# --- 3. INTERFACE ET FILTRAGE ---
st.sidebar.title("🛠 Stratégie")
idx_choice = st.sidebar.selectbox("1. Indice", ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)", "FTSE 100 (UK)"])
r2_min = st.sidebar.slider("2. R² minimum", 0.0, 1.0, 0.90, 0.01)

base_stocks, bench_ticker = get_index_components(idx_choice)

@st.cache_data(ttl=3600)
def get_ranked_filtered_list(stocks_dict, r2_threshold):
    tickers = list(stocks_dict.values())
    # Utilisation de 10 ans pour un filtrage précis (ex: Air Liquide vs Pernod)
    data = yf.download(tickers, period="10y", interval="1wk", progress=False)['Close']
    
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            r2_val, _, _ = get_metrics(data[ticker])
            if r2_val >= r2_threshold:
                results.append({"name": name, "r2": r2_val})
    
    # Tri par R2 décroissant
    sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)
    return [item['name'] for item in sorted_results]

with st.sidebar:
    st.write("---")
    with st.spinner("Analyse des tendances..."):
        filtered_names = get_ranked_filtered_list(base_stocks, r2_min)
    
    if filtered_names:
        stock_name = st.selectbox(f"3. Valeurs ({len(filtered_names)})", filtered_names)
        symbol = base_stocks[stock_name]
    else:
        st.error("Aucune action trouvée.")
        st.stop()

# --- 4. ANALYSE DÉTAILLÉE ---
df_full = yf.download(symbol, start="2000-01-01", progress=False)['Close']
if isinstance(df_full, pd.DataFrame): df_full = df_full.iloc[:, 0]

r2, vol, y_pred_log = get_metrics(df_full)
std_dev = np.std(np.log(df_full.values) - y_pred_log)

# Données actuelles
curr = df_full.iloc[-1]
theo = np.exp(y_pred_log[-1])
s1_u, s1_d = np.exp(y_pred_log[-1] + std_dev), np.exp(y_pred_log[-1] - std_dev)
s2_u, s2_d = np.exp(y_pred_log[-1] + 2*std_dev), np.exp(y_pred_log[-1] - 2*std_dev)

# --- 5. AFFICHAGE FINAL ---
st.title(f"🚀 {stock_name} ({symbol})")

# Interprétation R2
st.info(f"**Qualité du modèle (R² = {r2:.4f})** : {get_r2_interpretation(r2)}")

# Métriques
m1, m2, m3, m4 = st.columns(4)
m1.metric("Volatilité Ann.", f"{vol:.2f} %")
m2.metric("Prix Actuel", f"{curr:.2f} €")
m3.metric("Position / Moyenne", f"{((curr/theo)-1)*100:+.2f}%")
m4.metric("Score R²", f"{r2:.4f}")

# Seuils
st.markdown("### 🎯 Niveaux de valorisation actuels")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Support -2σ", f"{s2_d:.2f}")
t2.metric("Support -1σ", f"{s1_d:.2f}")
t3.metric("PRIX THÉORIQUE", f"{theo:.2f}")
t4.metric("Résistance +1σ", f"{s1_u:.2f}")
t5.metric("Résistance +2σ", f"{s2_u:.2f}")

# Onglets Graphiques
tab1, tab2 = st.tabs(["📉 Échelle Logarithmique (Tendance)", "📈 Échelle Arithmétique (Amplitude)"])

def create_plot(is_log):
    fig = go.Figure()
    dates = df_full.index
    y_trend = np.exp(y_pred_log)
    u1, l1 = np.exp(y_pred_log + std_dev), np.exp(y_pred_log - std_dev)
    u2, l2 = np.exp(y_pred_log + 2*std_dev), np.exp(y_pred_log - 2*std_dev)
    
    # Bandes Sigma
    fig.add_trace(go.Scatter(x=dates, y=u2, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=l2, fill='tonexty', fillcolor='rgba(255, 215, 0, 0.05)', line=dict(width=0), name="Zone +/- 2σ (95%)"))
    fig.add_trace(go.Scatter(x=dates, y=u1, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=l1, fill='tonexty', fillcolor='rgba(255, 215, 0, 0.15)', line=dict(width=0), name="Zone +/- 1σ (68%)"))
    
    # Prix et Régression
    fig.add_trace(go.Scatter(x=dates, y=df_full.values, name="Cours réel", line=dict(color='#00D4FF', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=y_trend, name="Régression", line=dict(color='gold', width=1.5, dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log" if is_log else "linear", 
                      margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=-0.1))
    return fig

tab1.plotly_chart(create_plot(True), use_container_width=True)
tab2.plotly_chart(create_plot(False), use_container_width=True)
