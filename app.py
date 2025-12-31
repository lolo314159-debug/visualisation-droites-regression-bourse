import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Terminal Statistique Européen", layout="wide")

# --- 1. CONFIGURATION ET RÉCUPÉRATION ---
@st.cache_data(ttl=86400)
def get_index_components(index_key):
    headers = {'User-Agent': 'Mozilla/5.0'}
    indices_config = {
        "EURO STOXX 50 (Zone Euro)": {"url": "https://en.wikipedia.org/wiki/EURO_STOXX_50", "suffix": "", "table_idx": 4},
        "STOXX Europe 600 (Large Cap)": {"url": "https://en.wikipedia.org/wiki/STOXX_Europe_600", "suffix": "", "table_idx": 4},
        "France (CAC 40)": {"url": "https://en.wikipedia.org/wiki/CAC_40", "suffix": ".PA", "table_idx": 4},
        "Allemagne (DAX 40)": {"url": "https://en.wikipedia.org/wiki/DAX", "suffix": ".DE", "table_idx": 4},
        "Royaume-Uni (FTSE 100)": {"url": "https://en.wikipedia.org/wiki/FTSE_100_Index", "suffix": ".L", "table_idx": 4},
        "Suisse (SMI 20)": {"url": "https://en.wikipedia.org/wiki/Swiss_Market_Index", "suffix": ".SW", "table_idx": 3},
        "Pays-Bas (AEX)": {"url": "https://en.wikipedia.org/wiki/AEX_index", "suffix": ".AS", "table_idx": 3},
        "Belgique (BEL 20)": {"url": "https://en.wikipedia.org/wiki/BEL_20", "suffix": ".BR", "table_idx": 0}
    }
    try:
        info = indices_config[index_key]
        req = urllib.request.Request(info["url"], headers=headers)
        with urllib.request.urlopen(req) as response:
            all_tables = pd.read_html(response.read())
            # Pour le STOXX 600, Wikipedia change souvent l'ordre des tables
            df_wiki = None
            for table in all_tables:
                if any(x in str(table.columns).upper() for x in ['TICKER', 'SYMBOL', 'EPIC']):
                    df_wiki = table
                    break
            if df_wiki is None: df_wiki = all_tables[info["table_idx"]]
        
        cols = df_wiki.columns
        ticker_col = next((c for c in cols if any(x in str(c).upper() for x in ['TICKER', 'SYMBOL', 'EPIC'])), cols[1])
        name_col = next((c for c in cols if any(x in str(c).upper() for x in ['COMPANY', 'NAME', 'CONSTITUENT'])), cols[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            sym = str(row[ticker_col]).split('.')[0].split(':')[0].strip()
            if info["suffix"] and not sym.endswith(info["suffix"]): sym += info["suffix"]
            stocks[str(row[name_col])] = sym
        return stocks
    except Exception as e:
        return {"Air Liquide": "AI.PA"}

# --- 2. FONCTIONS STATISTIQUES ---
def get_metrics(prices_series):
    prices = prices_series.dropna().values.astype(float)
    if len(prices) < 100: return None
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    r2 = model.score(x, y_log)
    cagr = (pow(prices[-1] / prices[0], 1/(len(prices)/252)) - 1) * 100
    vol_hist = np.std(np.diff(y_log)) * np.sqrt(252) * 100
    vol_10y = np.std(np.diff(y_log[-2520:])) * np.sqrt(252) * 100 if len(prices) > 2520 else vol_hist
    return {"r2": r2, "cagr": cagr, "vol_hist": vol_hist, "vol_10y": vol_10y, "y_pred": y_pred_log, "prices": prices}

def get_trading_signal(curr, theo, s1_u, s1_d, s2_u, s2_d):
    if curr <= s2_d: return "🔵 **ACHAT FORT** (-2σ)"
    if curr <= s1_d: return "🟢 **ACHAT** (-1σ)"
    if curr >= s2_u: return "🔴 **VENTE FORTE** (+2σ)"
    if curr >= s1_u: return "🟠 **ALLÉGEMENT** (+1σ)"
    return "⚪ **NEUTRE** (Zone Moyenne)"

# --- 3. INTERFACE ET FILTRAGE ---
st.sidebar.title("🌍 Marchés")
cat = st.sidebar.radio("Catégorie", ["Continentaux", "Par Pays"])
idx_list = ["EURO STOXX 50 (Zone Euro)", "STOXX Europe 600 (Large Cap)"] if cat == "Continentaux" else ["France (CAC 40)", "Allemagne (DAX 40)", "Royaume-Uni (FTSE 100)", "Suisse (SMI 20)", "Pays-Bas (AEX)", "Belgique (BEL 20)"]
idx_choice = st.sidebar.selectbox("Indice", idx_list)
r2_min = st.sidebar.slider("R² min (Filtrage)", 0.0, 1.0, 0.90, 0.01)

base_stocks = get_index_components(idx_choice)

@st.cache_data(ttl=3600)
def get_filtered_list(stocks_dict, r2_threshold):
    tickers = list(stocks_dict.values())
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            stats = get_metrics(data[ticker])
            # On arrondit à 2 décimales pour éviter les faux positifs du type 0.8999
            if stats and round(stats["r2"], 4) >= r2_threshold:
                results.append({"name": name, "r2": stats["r2"]})
    return [item['name'] for item in sorted(results, key=lambda x: x['r2'], reverse=True)]

filtered_names = get_filtered_list(base_stocks, r2_min)

if not filtered_names:
    st.sidebar.error("Aucune valeur trouvée.")
    st.stop()

selected_stock = st.sidebar.selectbox(f"Valeurs ({len(filtered_names)})", filtered_names)
symbol = base_stocks[selected_stock]

# --- 4. AFFICHAGE DE LA PARTIE CENTRALE ---
df = yf.download(symbol, start="2000-01-01", progress=False)['Close']
if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
res = get_metrics(df)

if res:
    std_dev = np.std(np.log(res["prices"]) - res["y_pred"])
    curr, theo = res["prices"][-1], np.exp(res["y_pred"][-1])
    s1_u, s1_d = np.exp(res["y_pred"][-1] + std_dev), np.exp(res["y_pred"][-1] - std_dev)
    s2_u, s2_d = np.exp(res["y_pred"][-1] + 2*std_dev), np.exp(res["y_pred"][-1] - 2*std_dev)
    
    st.header(f"🚀 {selected_stock} ({symbol})")
    st.markdown(f"**Diagnostic :** {get_trading_signal(curr, theo, s1_u, s1_d, s2_u, s2_d)}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR", f"{res['cagr']:.2f}%")
    m2.metric("Vol. 10 ans", f"{res['vol_10y']:.1f}%")
    m3.metric("Vol. 25 ans", f"{res['vol_hist']:.1f}%")
    m4.metric("Score R²", f"{res['r2']:.4f}")
    m5.metric("Position / Moy.", f"{((curr/theo)-1)*100:+.1f}%")

    tab1, tab2 = st.tabs(["📉 Log", "📈 Linéaire"])
    def create_plot(is_log):
        fig = go.Figure()
        d, yp = df.index, res["y_pred"]
        fig.add_trace(go.Scatter(x=d, y=np.exp(yp+2*std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=d, y=np.exp(yp-2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.04)', name="95%"))
        fig.add_trace(go.Scatter(x=d, y=np.exp(yp+std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=d, y=np.exp(yp-std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.12)', name="68%"))
        fig.add_trace(go.Scatter(x=d, y=res["prices"], name="Prix", line=dict(color='#00D4FF', width=1.5)))
        fig.add_trace(go.Scatter(x=d, y=np.exp(yp), name="Trend", line=dict(color='gold', dash='dash')))
        fig.update_layout(template="plotly_dark", height=400, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0))
        return fig
    
    tab1.plotly_chart(create_plot(True), use_container_width=True)
    tab2.plotly_chart(create_plot(False), use_container_width=True)

    st.markdown("---")
    t = st.columns(5)
    t[0].caption("Support -2σ"); t[0].write(f"**{s2_d:.2f}**")
    t[1].caption("Support -1σ"); t[1].write(f"**{s1_d:.2f}**")
    t[2].caption("THÉORIQUE"); t[2].write(f"**{theo:.2f}**")
    t[3].caption("Résistance +1σ"); t[3].write(f"**{s1_u:.2f}**")
    t[4].caption("Résistance +2σ"); t[4].write(f"**{s2_u:.2f}**")
