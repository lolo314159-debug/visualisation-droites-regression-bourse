import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Analyse Indices Automatisée", layout="wide")

# --- 1. FONCTIONS DE RÉCUPÉRATION AUTOMATIQUE DES COMPOSANTS ---

@st.cache_data(ttl=86400) # Mise à jour une fois par jour
def get_index_components(index_name):
    try:
        if index_name == "CAC 40 (France)":
            url = "https://en.wikipedia.org/wiki/CAC_40"
            table = pd.read_html(url)[4] # Le 5ème tableau contient les tickers
            # Nettoyage pour Yahoo Finance (Ticker + .PA)
            tickers = {row['Company']: row['Ticker'] for _, row in table.iterrows()}
            return tickers
        
        elif index_name == "DAX (Allemagne)":
            url = "https://en.wikipedia.org/wiki/DAX"
            table = pd.read_html(url)[4]
            tickers = {row['Company']: row['Ticker'] for _, row in table.iterrows()}
            return tickers

        elif index_name == "EURO STOXX 50":
            url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
            table = pd.read_html(url)[4]
            # Wikipedia utilise souvent les tickers sans extension, on ajoute .AS, .PA, .DE si besoin
            # Pour la démo, on utilise la colonne Ticker brute
            tickers = {row['Name']: row['Ticker'] for _, row in table.iterrows()}
            return tickers

        elif index_name == "IBEX 35 (Espagne)":
            url = "https://en.wikipedia.org/wiki/IBEX_35"
            table = pd.read_html(url)[2]
            tickers = {row['Company']: row['Ticker'] for _, row in table.iterrows()}
            return tickers
            
    except Exception as e:
        st.error(f"Erreur de récupération : {e}")
        return {"LVMH": "MC.PA"} # Valeur par défaut en cas d'erreur

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🌍 Indices Européens")
index_choice = st.sidebar.selectbox("1. Choisir l'indice", 
                                   ["CAC 40 (France)", "DAX (Allemagne)", "EURO STOXX 50", "IBEX 35 (Espagne)"])

# Chargement automatique des valeurs
stock_dict = get_index_components(index_choice)

stock_choice = st.sidebar.selectbox("2. Choisir la valeur", list(stock_dict.keys()))
symbol = stock_dict[stock_choice]

# Correction spécifique pour certains formats Wikipedia
if index_choice == "CAC 40 (France)" and not symbol.endswith(".PA"):
    symbol = symbol.replace(" ", "") + ".PA"

@st.cache_data
def load_stock_data(s):
    data = yf.download(s, start="2000-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()

df = load_stock_data(symbol)

# --- 3. CALCULS ET GRAPHIQUES ---
if not df.empty:
    prices = df['Close'].values.flatten().astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    std_dev = np.std(y_log - y_pred_log)
    
    # Calcul du CAGR
    try:
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (pow(prices[-1] / prices[0], 1/years) - 1) * 100
        cagr_str = f"{cagr:.2f}%"
    except:
        cagr_str = "N/A"

    st.subheader(f"Analyse : {stock_choice} ({symbol})")
    
    # Métriques
    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR", cagr_str)
    m2.metric("R² (Corrélation Log)", f"{model.score(x, y_log):.4f}")
    m3.metric("Dernier Cours", f"{prices[-1]:.2f} €")

    # Onglets
    tab1, tab2 = st.tabs(["Echelle Log", "Echelle Arithmétique"])

    def draw_chart(is_log):
        fig = go.Figure()
        y_trend = np.exp(y_pred_log)
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log + 2*std_dev), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.1)', line=dict(width=0), name="Canal Sigma"))
        fig.add_trace(go.Scatter(x=df.index, y=prices, name="Prix", line=dict(color='#00D4FF', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=y_trend, name="Régression", line=dict(color='gold', dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, yaxis_type="log" if is_log else "linear", margin=dict(l=0,r=0,t=10,b=0))
        return fig

    with tab1: st.plotly_chart(draw_chart(True), use_container_width=True)
    with tab2: st.plotly_chart(draw_chart(False), use_container_width=True)
else:
    st.error("Impossible de charger les données. Le ticker Wikipedia est peut-être différent de Yahoo Finance.")
