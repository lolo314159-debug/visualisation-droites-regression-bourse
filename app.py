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

r2_range = st.sidebar.slider("Intervalle R²", 0.0, 1.0, (0.85, 1.00), 0.01)
pos_range = st.sidebar.slider("Position / Droite (%)", -100, 100, (-100, 100), 5)

base_stocks = get_index_components(idx_choice)
if base_stocks is None: st.stop()

@st.cache_data(ttl=3600)
def get_filtered_results(stocks_dict, r2_bounds, pos_bounds):
    tickers = list(stocks_dict.values())
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            stats = get_metrics(data[ticker])
            if stats:
                r2_val = round(stats["r2"], 4)
                curr_p = stats["prices"][-1]
                theo_p = np.exp(stats["y_pred"][-1])
                pos_val = ((curr_p / theo_p) - 1) * 100
                if (r2_bounds[0] <= r2_val <= r2_bounds[1]) and (pos_bounds[0] <= pos_val <= pos_bounds[1]):
                    results.append({"name": name, "r2": r2_val})
    return sorted(results, key=lambda x: x['r2'], reverse=True)

filtered_data = get_filtered_results(base_stocks, r2_range, pos_range)
filtered_names = [item['name'] for item in filtered_data]

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
    # Calculs précis
    std_dev = np.std(np.log(res["prices"]) - res["y_pred"])
    curr, theo = res["prices"][-1], np.exp(res["y_pred"][-1])
    z_score = (np.log(curr) - np.log(theo)) / std_dev
    
    s1_u, s1_d = np.exp(res["y_pred"][-1] + std_dev), np.exp(res["y_pred"][-1] - std_dev)
    s2_u, s2_d = np.exp(res["y_pred"][-1] + 2*std_dev), np.exp(res["y_pred"][-1] - 2*std_dev)
    
    st.header(f"🚀 {selected_stock} ({symbol})")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR (25 ans)", f"{res['cagr']:.2f}%")
    m2.metric("Vol. 10 ans", f"{res['vol_10y']:.1f}%")
    m3.metric("Vol. 25 ans", f"{res['vol_hist']:.1f}%")
    m4.metric("Fiabilité (R²)", f"{res['r2']:.4f}")
    m5.metric("Z-Score (Sigma)", f"{z_score:+.2f}")

    # --- GUIDE D'INTERPRÉTATION PRÉCIS ---
    with st.expander("🔍 ANALYSE DE PRÉCISION STATISTIQUE", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🎯 Localisation dans le Canal")
            if abs(z_score) <= 0.25:
                st.info(f"⚖️ **Équilibre Parfait** (Z-Score: {z_score:+.2f})")
                st.write("Le prix est quasi identique à sa valeur théorique. Aucun avantage statistique directionnel. Zone idéale pour du DCA.")
            elif 0.25 < z_score <= 1.0:
                st.warning(f"📈 **Légère Tension** (Z-Score: {z_score:+.2f})")
                st.write("Le titre est 'bien payé' mais reste dans sa zone de fluctuation habituelle.")
            elif 1.0 < z_score <= 1.8:
                st.warning(f"🟠 **Résistance 1σ** (Z-Score: {z_score:+.2f})")
                st.write("Le titre entre dans les 15% des prix les plus chers. Le risque de respiration vers la droite dorée est important.")
            elif z_score > 1.8:
                st.error(f"🔥 **Excès de Confiance (+2σ)** (Z-Score: {z_score:+.2f})")
                st.write("Zone d'euphorie. Un retour à la moyenne est probable à 95%.")
            elif -1.0 <= z_score < -0.25:
                st.success(f"📉 **Légère Décote** (Z-Score: {z_score:+.2f})")
                st.write("L'action glisse sous sa moyenne. Opportunité de renforcement serein.")
            elif -1.8 <= z_score < -1.0:
                st.success(f"🟢 **Opportunité 1σ** (Z-Score: {z_score:+.2f})")
                st.write("L'action est nettement attractive (moins chère que 84% de son historique relatif).")
            elif z_score < -1.8:
                st.error(f"🚨 **Anomalie de Marché (-2σ)** (Z-Score: {z_score:+.2f})")
                st.write("Zone de peur irrationnelle. Force de rappel vers la droite dorée maximale.")

        with col_b:
            st.markdown("### 🧭 Stratégie & Potentiel")
            st.write(f"• **Performance annuelle (CAGR) :** {res['cagr']:.2f}%")
            st.write(f"• **Doublement capital :** ~{72/max(res['cagr'], 1):.1f} ans")
            st.markdown("---")
            if abs(z_score) > 1.5:
                st.markdown("**⚠️ Action Requise :** Le prix touche les bords du canal. Envisager un arbitrage (achat ou vente selon le côté).")
            else:
                st.markdown("**😴 Action Requise :** 'Wait and See'. Le titre est dans son bruit de marché normal.")

    # --- GRAPHIQUES ---
    tab1, tab2 = st.tabs(["📉 Vue Logarithmique", "📈 Vue Linéaire"])
    def create_plot(is_log):
        fig = go.Figure()
        dates, yp = df_full.index, res["y_pred"]
        c2_f, c1_f = 'rgba(255, 215, 0, 0.05)', 'rgba(255, 215, 0, 0.15)'
        line_s = dict(color='rgba(255, 215, 0, 0.2)', width=0.8)

        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp + 2*std_dev), line=line_s, showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp - 2*std_dev), fill='tonexty', fillcolor=c2_f, line=line_s, name="Zone 95% (2σ)"))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp + std_dev), line=line_s, showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp - std_dev), fill='tonexty', fillcolor=c1_f, line=line_s, name="Zone 68% (1σ)"))
        
        fig.add_trace(go.Scatter(x=dates, y=res["prices"], name="Prix", line=dict(color='#00D4FF', width=1.8)))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp), name="Trend", line=dict(color='gold', width=2, dash='dash')))
        
        fig.update_layout(template="plotly_dark", height=450, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=-0.15))
        return fig
    
    tab1.plotly_chart(create_plot(True), use_container_width=True)
    tab2.plotly_chart(create_plot(False), use_container_width=True)

    # Niveaux de prix
    st.markdown("---")
    t = st.columns(5)
    t[0].metric("Support -2σ", f"{s2_d:.2f} €")
    t[1].metric("Support -1σ", f"{s1_d:.2f} €")
    t[2].metric("PRIX THÉORIQUE", f"{theo:.2f} €")
    t[3].metric("Résistance +1σ", f"{s1_u:.2f} €")
    t[4].metric("Résistance +2σ", f"{s2_u:.2f} €")
