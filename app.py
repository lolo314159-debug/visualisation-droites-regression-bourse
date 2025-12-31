import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Terminal Statistique Expert", layout="wide")

# --- 1. RÉCUPÉRATION DES COMPOSANTS ---
@st.cache_data(ttl=86400)
def get_index_components(index_name):
    headers = {'User-Agent': 'Mozilla/5.0'}
    indices = {
        "EURO STOXX 50": {"url": "https://en.wikipedia.org/wiki/EURO_STOXX_50", "suffix": "", "table_idx": 4},
        "STOXX Europe 600": {"url": "https://en.wikipedia.org/wiki/STOXX_Europe_600", "suffix": "", "table_idx": 4},
        "CAC 40 (France)": {"url": "https://en.wikipedia.org/wiki/CAC_40", "suffix": ".PA", "table_idx": 4},
        "DAX 40 (Allemagne)": {"url": "https://en.wikipedia.org/wiki/DAX", "suffix": ".DE", "table_idx": 4},
        "FTSE 100 (UK)": {"url": "https://en.wikipedia.org/wiki/FTSE_100_Index", "suffix": ".L", "table_idx": 4},
        "SMI 20 (Suisse)": {"url": "https://en.wikipedia.org/wiki/Swiss_Market_Index", "suffix": ".SW", "table_idx": 3},
        "AEX (Pays-Bas)": {"url": "https://en.wikipedia.org/wiki/AEX_index", "suffix": ".AS", "table_idx": 3},
        "IBEX 35 (Espagne)": {"url": "https://en.wikipedia.org/wiki/IBEX_35", "suffix": ".MC", "table_idx": 2},
        "BEL 20 (Belgique)": {"url": "https://en.wikipedia.org/wiki/BEL_20", "suffix": ".BR", "table_idx": 0}
    }
    try:
        info = indices[index_name]
        req = urllib.request.Request(info["url"], headers=headers)
        with urllib.request.urlopen(req) as response:
            all_tables = pd.read_html(response.read())
            df_wiki = None
            for table in all_tables:
                if any(x in str(table.columns).upper() for x in ['TICKER', 'SYMBOL', 'EPIC']):
                    df_wiki = table
                    break
            if df_wiki is None: df_wiki = all_tables[info["table_idx"]]
        
        col_ticker = next((c for c in df_wiki.columns if any(x in str(c).upper() for x in ['TICKER', 'SYMBOL', 'EPIC'])), df_wiki.columns[1])
        col_name = next((c for c in df_wiki.columns if any(x in str(c).upper() for x in ['COMPANY', 'NAME', 'CONSTITUENT'])), df_wiki.columns[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            sym = str(row[col_ticker]).split('.')[0].split(':')[0].strip().split(' ')[0]
            if info["suffix"] and not sym.endswith(info["suffix"]): sym += info["suffix"]
            stocks[str(row[col_name])] = sym
        return stocks
    except Exception as e:
        st.sidebar.error(f"Erreur d'acquisition : {e}")
        return None

# --- 2. FONCTIONS STATISTIQUES ---
def calc_vol(prices_array):
    returns = np.diff(np.log(np.maximum(prices_array, 0.01)))
    return np.std(returns) * np.sqrt(252) * 100

def get_metrics(prices_series):
    prices = prices_series.dropna().values.astype(float)
    if len(prices) < 100: return None
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x).flatten()
    r2 = model.score(x, y_log)
    years = len(prices) / 252
    cagr = (pow(prices[-1] / prices[0], 1/years) - 1) * 100
    vol_hist = calc_vol(prices)
    vol_10y = calc_vol(prices[-2520:]) if len(prices) > 2520 else vol_hist
    return {"r2": r2, "cagr": cagr, "vol_hist": vol_hist, "vol_10y": vol_10y, "y_pred": y_pred_log, "prices": prices}

# --- 3. INTERFACE ---
st.sidebar.title("⚙️ Paramètres")
cat = st.sidebar.radio("Catégorie", ["Continentaux", "Par Pays"])
idx_list = ["EURO STOXX 50", "STOXX Europe 600"] if cat == "Continentaux" else ["CAC 40 (France)", "DAX 40 (Allemagne)", "FTSE 100 (UK)", "SMI 20 (Suisse)", "AEX (Pays-Bas)", "IBEX 35 (Espagne)", "BEL 20 (Belgique)"]
idx_choice = st.sidebar.selectbox("Indice", idx_list)
r2_min = st.sidebar.slider("R² min (Filtrage)", 0.0, 1.0, 0.90, 0.01)

base_stocks = get_index_components(idx_choice)
if base_stocks is None: st.stop()

@st.cache_data(ttl=3600)
def get_filtered_list(stocks_dict, r2_threshold):
    tickers = list(stocks_dict.values())
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            stats = get_metrics(data[ticker])
            if stats and round(stats["r2"], 4) >= r2_threshold:
                results.append({"name": name, "r2": stats["r2"]})
    return [item['name'] for item in sorted(results, key=lambda x: x['r2'], reverse=True)]

filtered_names = get_filtered_list(base_stocks, r2_min)
if not filtered_names:
    st.sidebar.warning("Aucun résultat.")
    st.stop()

selected_stock = st.sidebar.selectbox(f"Valeurs ({len(filtered_names)})", filtered_names)
symbol = base_stocks[selected_stock]

# --- 4. ANALYSE ET AFFICHAGE ---
df_full = yf.download(symbol, start="2000-01-01", progress=False)['Close']
if isinstance(df_full, pd.DataFrame): df_full = df_full.iloc[:, 0]
res = get_metrics(df_full)

if res:
    # Calcul des zones sigma
    std_dev = np.std(np.log(res["prices"]) - res["y_pred"])
    curr, theo = res["prices"][-1], np.exp(res["y_pred"][-1])
    s1_u, s1_d = np.exp(res["y_pred"][-1] + std_dev), np.exp(res["y_pred"][-1] - std_dev)
    s2_u, s2_d = np.exp(res["y_pred"][-1] + 2*std_dev), np.exp(res["y_pred"][-1] - 2*std_dev)
    
    st.header(f"🚀 {selected_stock} ({symbol})")
    
    # Dashboard Métriques
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR (25 ans)", f"{res['cagr']:.2f}%")
    m2.metric("Vol. 10 ans", f"{res['vol_10y']:.1f}%")
    m3.metric("Vol. 25 ans", f"{res['vol_hist']:.1f}%")
    m4.metric("Fiabilité (R²)", f"{res['r2']:.4f}")
    m5.metric("Position / Moy.", f"{((curr/theo)-1)*100:+.1f}%")

    # Guide d'interprétation
    with st.expander("🔍 Guide d'interprétation des résultats", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Analyse de la Tendance :**")
            if res['r2'] > 0.95: st.info(f"💎 **Qualité Exceptionnelle** : Précision de {res['r2']*100:.1f}%.")
            else: st.info("✅ **Bonne Tendance** : Trajectoire long-terme solide.")
            
            st.markdown("**Risque :**")
            if (res['vol_hist'] - res['vol_10y']) > 5: st.success("📉 **Apaisement** : Plus stable récemment.")
            else
