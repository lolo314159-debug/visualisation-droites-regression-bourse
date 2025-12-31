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

# --- 3. INTERFACE AVEC FILTRES PAR INTERVALLES ---
st.sidebar.title("⚙️ Paramètres")
cat = st.sidebar.radio("Catégorie", ["Continentaux", "Par Pays"])
idx_list = ["EURO STOXX 50", "STOXX Europe 600"] if cat == "Continentaux" else ["CAC 40 (France)", "DAX 40 (Allemagne)", "FTSE 100 (UK)", "SMI 20 (Suisse)", "AEX (Pays-Bas)", "IBEX 35 (Espagne)", "BEL 20 (Belgique)"]
idx_choice = st.sidebar.selectbox("Indice", idx_list)

# NOUVEAUX FILTRES PAR INTERVALLE
r2_range = st.sidebar.slider("Intervalle R² (Fiabilité)", 0.0, 1.0, (0.85, 1.00), 0.01)
pos_range = st.sidebar.slider("Intervalle Position / Droite (%)", -100, 100, (-100, 100), 5)

base_stocks = get_index_components(idx_choice)
if base_stocks is None: st.stop()

@st.cache_data(ttl=3600)
def get_strictly_filtered_list(stocks_dict, r2_bounds, pos_bounds):
    tickers = list(stocks_dict.values())
    # Download 1wk pour le scan rapide
    data = yf.download(tickers, start="2000-01-01", interval="1wk", progress=False)['Close']
    results = []
    for name, ticker in stocks_dict.items():
        if ticker in data.columns:
            stats = get_metrics(data[ticker])
            if stats:
                r2_val = round(stats["r2"], 4)
                curr_price = stats["prices"][-1]
                theo_price = np.exp(stats["y_pred"][-1])
                pos_val = ((curr_price / theo_price) - 1) * 100
                
                # Vérification des deux intervalles
                if (r2_bounds[0] <= r2_val <= r2_bounds[1]) and (pos_bounds[0] <= pos_val <= pos_bounds[1]):
                    results.append({"name": name, "r2": r2_val})
    return [item['name'] for item in sorted(results, key=lambda x: x['r2'], reverse=True)]

with st.sidebar:
    st.write("---")
    with st.spinner("Filtrage multicritères..."):
        filtered_names = get_strictly_filtered_list(base_stocks, r2_range, pos_range)

if not filtered_names:
    st.sidebar.warning("Aucune valeur trouvée pour ces critères.")
    st.stop()

selected_stock = st.sidebar.selectbox(f"Valeurs ({len(filtered_names)})", filtered_names)
symbol = base_stocks[selected_stock]

# --- 4. ANALYSE ET AFFICHAGE ---
df_full = yf.download(symbol, start="2000-01-01", progress=False)['Close']
if isinstance(df_full, pd.DataFrame): df_full = df_full.iloc[:, 0]
res = get_metrics(df_full)

if res:
    std_dev = np.std(np.log(res["prices"]) - res["y_pred"])
    curr, theo = res["prices"][-1], np.exp(res["y_pred"][-1])
    s1_u, s1_d = np.exp(res["y_pred"][-1] + std_dev), np.exp(res["y_pred"][-1] - std_dev)
    s2_u, s2_d = np.exp(res["y_pred"][-1] + 2*std_dev), np.exp(res["y_pred"][-1] - 2*std_dev)
    
    st.header(f"🚀 {selected_stock} ({symbol})")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CAGR (25 ans)", f"{res['cagr']:.2f}%")
    m2.metric("Vol. 10 ans", f"{res['vol_10y']:.1f}%")
    m3.metric("Vol. 25 ans", f"{res['vol_hist']:.1f}%")
    m4.metric("Fiabilité (R²)", f"{res['r2']:.4f}")
    m5.metric("Position / Moy.", f"{((curr/theo)-1)*100:+.1f}%")

# SECTION INTERPRÉTATION DÉTAILLÉE ET ANALYSE PRÉCISE
    with st.expander("🔍 ANALYSE STRATÉGIQUE DÉTAILLÉE", expanded=True):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 📊 Qualité de la Tendance")
            r2_val = res['r2']
            if r2_val > 0.98:
                st.success(f"**Score : {r2_val:.4f} - Modèle 'Horloge Suisse'**")
                st.write("L'action est d'une régularité absolue. Les écarts à la moyenne sont historiquement très brefs. C'est un profil idéal pour de l'investissement programmé (DCA).")
            elif r2_val > 0.93:
                st.success(f"**Score : {r2_val:.4f} - Tendance Structurelle**")
                st.write("La croissance est solide et prévisible. Le marché respecte très bien le canal de régression. Les signaux σ (Sigma) sont ici très fiables.")
            elif r2_val > 0.85:
                st.info(f"**Score : {r2_val:.4f} - Tendance Validée**")
                st.write("La trajectoire est ascendante mais sujette à des cycles économiques visibles. Attendre impérativement les supports pour entrer.")
            else:
                st.warning(f"**Score : {r2_val:.4f} - Tendance Instable**")
                st.write("La fiabilité statistique est plus faible. Ne pas accorder une confiance aveugle aux objectifs de prix hauts/bas.")

            st.markdown("---")
            st.markdown("### ⚡ Profil de Risque (Volatilité)")
            v10, v25 = res['vol_10y'], res['vol_hist']
            if v10 < v25 * 0.8:
                st.write(f"✅ **Assagissement** : La volatilité actuelle ({v10:.1f}%) est bien inférieure à la moyenne historique ({v25:.1f}%). Le titre devient 'Bon Père de Famille'.")
            elif v10 > v25 * 1.2:
                st.error(f"⚠️ **Nervosité Accrue** : Le titre est beaucoup plus instable ces dernières années ({v10:.1f}%) qu'historiquement. Risque de décrochage brutal.")
            else:
                st.write(f"⚖️ **Risque Constant** : La volatilité est stable autour de {v10:.1f}%. Pas de changement de comportement majeur.")

        with col_b:
            st.markdown("### 🎯 Diagnostic de Prix & Timing")
            pos_moy = ((curr/theo)-1)*100
            
            if curr <= s2_d:
                st.error(f"🚨 **ACHAT FORT (Zone de Capitulation)**")
                st.write(f"Le cours est à {pos_moy:.1f}% de sa moyenne. Statistiquement, le titre est survendu. C'est une zone de rebond historique (95% de probabilité de retour vers le haut).")
            elif curr <= s1_d:
                st.success(f"📉 **OPPORTUNITÉ D'ACHAT (Zone de Décote)**")
                st.write(f"Le cours est sous sa tendance centrale. Le potentiel de hausse pour rejoindre la moyenne est de {abs(pos_moy):.1f}%. Risque de baisse limité.")
            elif curr >= s2_u:
                st.error(f"🔥 **VENTE FORTE (Zone d'Euphorie)**")
                st.write(f"Le titre est en surchauffe totale (+{pos_moy:.1f}% vs moyenne). La probabilité d'une correction imminente vers la droite dorée est de 95%.")
            elif curr >= s1_u:
                st.warning(f"🟠 **PRUDENCE (Zone de Tension)**")
                st.write(f"L'action est chère. Elle se situe en haut de son canal habituel. Un retour vers {theo:.2f} € est probable avant toute nouvelle hausse.")
            else:
                st.info(f"⚪ **ZONE NEUTRE (Prix d'Équilibre)**")
                st.write(f"Le prix actuel est proche de sa valeur théorique ({theo:.2f} €). Le marché est à l'équilibre, il n'y a pas d'avantage statistique à l'achat ou à la vente ici.")

            st.markdown("---")
            st.markdown("### 📈 Potentiel CAGR")
            st.write(f"Si l'action maintient sa tendance de fond, elle génère **{res['cagr']:.2f}%** par an en moyenne. À ce rythme, un capital double tous les **{72/res['cagr']:.1f} ans**.")

    # GRAPHIQUES HARMONISÉS
    tab1, tab2 = st.tabs(["📉 Logarithmique", "📈 Linéaire"])
    def create_plot(is_log):
        fig = go.Figure()
        dates, yp = df_full.index, res["y_pred"]
        c2_fill, c1_fill = 'rgba(255, 215, 0, 0.05)', 'rgba(255, 215, 0, 0.15)'
        line_style = dict(color='rgba(255, 215, 0, 0.2)', width=0.5)
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp + 2*std_dev), line=line_style, showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp - 2*std_dev), fill='tonexty', fillcolor=c2_fill, line=line_style, name="2σ"))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp + std_dev), line=line_style, showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=np.exp(yp - std_dev), fill='tonexty', fillcolor=c1_fill, line=line_style, name="1σ"))
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
    t[2].metric("THÉORIQUE", f"{theo:.2f} €")
    t[3].metric("Résistance +1σ", f"{s1_u:.2f} €")
    t[4].metric("Résistance +2σ", f"{s2_u:.2f} €")
