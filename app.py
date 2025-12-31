import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import urllib.request

st.set_page_config(page_title="Terminal Statistique Européen", layout="wide")

# --- 1. CONFIGURATION ET RÉCUPÉRATION ROBUSTE ---
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
        "Pays-Bas (AEX)": {"url": "https://en.wikipedia.org/wiki/AEX_index", "suffix": ".AS", "table_idx": 3}
    }
    
    try:
        info = indices_config[index_key]
        req = urllib.request.Request(info["url"], headers=headers)
        with urllib.request.urlopen(req) as response:
            all_tables = pd.read_html(response.read())
            # Test de sécurité pour l'index du tableau
            idx = min(info["table_idx"], len(all_tables)-1)
            df_wiki = all_tables[idx]
        
        # Mapping flexible des colonnes
        cols = df_wiki.columns
        ticker_col = next((c for c in cols if any(x in str(c).upper() for x in ['TICKER', 'SYMBOL', 'EPIC'])), cols[1])
        name_col = next((c for c in cols if any(x in str(c).upper() for x in ['COMPANY', 'NAME', 'CONSTITUENT'])), cols[0])
        
        stocks = {}
        for _, row in df_wiki.iterrows():
            sym = str(row[ticker_col]).split('.')[0].split(':')[0].strip()
            if info["suffix"] and not sym.endswith(info["suffix"]):
                sym += info["suffix"]
            stocks[str(row[name_col])] = sym
        return stocks
    except Exception as e:
        st.error(f"Erreur technique : {e}")
        return {"Air Liquide": "AI.PA"}

# --- 2. FONCTION DE CALCUL UNIQUE ---
def compute_all_stats(prices_series):
    prices = prices_series.dropna().values.astype(float)
    if len(prices) < 100: return None
    
    # On force le calcul sur l'historique complet pour la cohérence
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    model = LinearRegression().fit(x, y_log)
    
    return {
        "r2": model.score(x, y_log),
        "y_pred": model.predict(x).flatten(),
        "vol_25y": np.std(np.diff(y_log)) * np.sqrt(252) * 100,
        "cagr": (pow(prices[-1] / prices[0], 1/(len(prices)/252)) - 1) * 100
    }

# --- 3. LOGIQUE DE FILTRAGE STRICTE ---
st.sidebar.title("🌍 Marchés")
cat_choice = st.sidebar.radio("Catégorie", ["Continentaux", "Par Pays"])
indices_list = ["EURO STOXX 50 (Zone Euro)", "STOXX Europe 600 (Large Cap)"] if cat_choice == "Continentaux" else ["France (CAC 40)", "Allemagne (DAX 40)", "Royaume-Uni (FTSE 100)", "Suisse (SMI 20)", "Pays-Bas (AEX)"]

idx_choice = st.sidebar.selectbox("Indice", indices_list)
r2_min = st.sidebar.slider("R² min (Filtrage Strict)", 0.0, 1.0, 0.90, 0.01)

base_stocks = get_index_components(idx_choice)

@st.cache_data(ttl=3600)
def get_filtered_data(stocks_dict, r2_threshold):
    # Téléchargement groupé pour la vitesse
    tickers = list(stocks_dict.values())
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    
    valid_list = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            stats = compute_all_stats(data[ticker])
            if stats and stats["r2"] >= r2_threshold:
                valid_list.append({"name": name, "r2": stats["r2"]})
    
    return sorted(valid_list, key=lambda x: x['r2'], reverse=True)

with st.sidebar:
    st.write("---")
    with st.spinner("Filtrage en cours..."):
        filtered_results = get_filtered_data(base_stocks, r2_min)
        names_only = [item['name'] for item in filtered_results]

if names_only:
    selected_name = st.sidebar.selectbox(f"Valeurs éligibles ({len(names_only)})", names_only)
    symbol = base_stocks[selected_name]
    
    # Affichage final
    df = yf.download(symbol, start="2000-01-01", progress=False)['Close']
    if isinstance(df, pd.DataFrame): df = df.iloc[:, 0]
    
    res = compute_all_stats(df)
    st.header(f"🚀 {selected_name} ({symbol})")
    
    # Validation visuelle du R2
    if res["r2"] < r2_min:
        st.warning(f"Note : Le R² calculé en données quotidiennes ({res['r2']:.4f}) est légèrement inférieur au seuil hebdomadaire.")
    
    # (Ici les colonnes de metrics et graphiques habituels...)
    st.metric("Fiabilité (R²)", f"{res['r2']:.4f}")
    # ... reste du code graphique ...
else:
    st.sidebar.error("Aucune valeur ne correspond.")
