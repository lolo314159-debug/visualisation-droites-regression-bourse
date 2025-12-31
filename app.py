import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Analyse Indices Européens", layout="wide")

# --- 1. BASES DE DONNÉES DES INDICES ---
indices_data = {
    "CAC 40 (France)": {
        "ticker": "^FCHI",
        "stocks": {"LVMH": "MC.PA", "Air Liquide": "AI.PA", "TotalEnergies": "TTE.PA", "Sanofi": "SAN.PA", "Airbus": "AIR.PA", "L'Oréal": "OR.PA", "Hermès": "RMS.PA", "Schneider": "SU.PA"}
    },
    "DAX (Allemagne)": {
        "ticker": "^GDAXI",
        "stocks": {"SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE", "Deutsche Telekom": "DTE.DE", "Mercedes-Benz": "MBG.DE"}
    },
    "IBEX 35 (Espagne)": {
        "ticker": "^IBEX",
        "stocks": {"Inditex": "ITX.MC", "Iberdrola": "IBE.MC", "Santander": "SAN.MC", "BBVA": "BBVA.MC"}
    },
    "FTSE MIB (Italie)": {
        "ticker": "FTSEMIB.MI",
        "stocks": {"Enel": "ENEL.MI", "Eni": "ENI.MI", "Ferrari": "RACE.MI", "Intesa Sanpaolo": "ISP.MI"}
    },
    "EURO STOXX 50": {
        "ticker": "^STOXX50E",
        "stocks": {"ASML": "ASML.AS", "LVMH": "MC.PA", "SAP": "SAP.DE", "L'Oréal": "OR.PA", "Sanofi": "SAN.PA"}
    }
}

# --- 2. BARRE LATÉRALE : CHOIX DE L'INDICE ---
st.sidebar.title("🔍 Sélection")
index_choice = st.sidebar.selectbox("1. Choisir l'indice", list(indices_data.keys()))

if 'current_index' not in st.session_state:
    st.session_state.current_index = index_choice

# Bouton de validation pour l'indice
if st.sidebar.button("Valider l'indice"):
    st.session_state.current_index = index_choice

selected_index_info = indices_data[st.session_state.current_index]

# Choix de la valeur au sein de l'indice
stock_choice = st.sidebar.selectbox("2. Choisir la valeur", list(selected_index_info["stocks"].keys()))
symbol = selected_index_info["stocks"][stock_choice]

@st.cache_data
def load_data(s):
    data = yf.download(s, start="2000-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()

df = load_data(symbol)

# --- 3. CALCULS ET AFFICHAGE ---
if not df.empty:
    prices = df['Close'].values.flatten().astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(np.maximum(prices, 0.01))
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    r2 = model.score(x, y_log)
    std_dev = np.std(y_log - y_pred_log)
    
    # Calcul du CAGR
    cagr_str = "N/A"
    try:
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years > 0 and prices[0] > 0:
            cagr_raw = (pow(prices[-1] / prices[0], 1/years) - 1) * 100
            cagr_str = f"{cagr_raw:.2f}%"
    except:
        pass

    # --- AFFICHAGE COMPACT ---
    st.subheader(f"Analyse : {stock_choice} ({symbol})")
    
    # Métriques sur une seule ligne
    m1, m2, m3 = st.columns(3)
    m1.metric("CAGR (Croissance Annuelle)", cagr_str)
    m2.metric("Fiabilité (R²)", f"{r2:.4f}")
    m3.metric("Prix Actuel", f"{prices[-1]:.2f} €")

    # Graphique avec hauteur réduite (500 au lieu de 700)
    fig = go.Figure()

    # Couches Sigma
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log + 2*std_dev), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log - 2*std_dev), fill='tonexty', 
                             fillcolor='rgba(255, 215, 0, 0.1)', line=dict(width=0), name="Bandes +/- 2 Sigma"))

    # Courbes
    fig.add_trace(go.Scatter(x=df.index, y=prices, name="Cours", line=dict(color='#00D4FF', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred_log), name="Régression", line=dict(color='gold', dash='dash')))

    fig.update_layout(
        template="plotly_dark",
        height=500, # Hauteur réduite pour tenir sur l'écran
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Données non disponibles pour ce symbole.")
