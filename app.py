import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Analyse Log-Régression CAC 40", layout="wide")

st.title("📊 Analyse de Régression Logarithmique - CAC 40")

# Liste des tickers du CAC 40 (approximateurs yfinance)
cac40_tickers = {
    "Air Liquide": "AI.PA", "Airbus": "AIR.PA", "Alstom": "ALO.PA", "ArcelorMittal": "MT.AS",
    "AXA": "CS.PA", "BNP Paribas": "BNP.PA", "Bouygues": "EN.PA", "Capgemini": "CAP.PA",
    "Carrefour": "CA.PA", "Crédit Agricole": "ACA.PA", "Danone": "BN.PA", "Dassault Systèmes": "DSY.PA",
    "Edenred": "EDEN.PA", "Engie": "ENGI.PA", "EssilorLuxottica": "EL.PA", "Eurofins Scientific": "ERF.PA",
    "Hermès": "RMS.PA", "Kering": "KER.PA", "L'Oréal": "OR.PA", "LVMH": "MC.PA",
    "Michelin": "ML.PA", "Orange": "ORA.PA", "Pernod Ricard": "RI.PA", "Publicis": "PUB.PA",
    "Renault": "RNO.PA", "Safran": "SAF.PA", "Saint-Gobain": "SGO.PA", "Sanofi": "SAN.PA",
    "Schneider Electric": "SU.PA", "Société Générale": "GLE.PA", "Stellantis": "STLAP.PA",
    "STMicroelectronics": "STMPA.PA", "Teleperformance": "TEP.PA", "Thales": "HO.PA",
    "TotalEnergies": "TTE.PA", "Unibail-Rodamco-Westfield": "URW.PA", "Veolia": "VIE.PA",
    "Vinci": "DG.PA", "Vivendi": "VIV.PA", "Worldline": "WLN.PA"
}

# Sidebar pour la sélection
ticker_name = st.sidebar.selectbox("Choisissez une action :", list(cac40_tickers.keys()))
ticker_symbol = cac40_tickers[ticker_name]

# Chargement des données
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2000-01-01")
    df.dropna(inplace=True)
    return df

data = load_data(ticker_symbol)

if not data.empty:
    # Préparation des données pour la régression
    data['Days'] = np.arange(len(data))
    X = data['Days'].values.reshape(-1, 1)
    y = np.log(data['Close'].values)  # Passage au logarithme

    # Modèle de régression linéaire sur le log des prix
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    std_dev = np.std(y - y_pred)
    
    # Calcul des métriques
    r2 = model.score(X, y)
    
    
# Calcul du CAGR sécurisé
    try:
        start_price = float(data['Close'].iloc[0])
        end_price = float(data['Close'].iloc[-1])
        num_years = (data.index[-1] - data.index[0]).days / 365.25
        
        if start_price > 0 and num_years > 0:
            cagr = (pow(end_price / start_price, 1 / num_years) - 1) * 100
        else:
            cagr = 0.0
    except:
        cagr = 0.0

    # Affichage des statistiques (avec sécurité sur l'affichage)
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR (%)", f"{cagr:.2f}%" if not np.isnan(cagr) else "N/A")
    col2.metric("R² (Fiabilité)", f"{r2:.4f}")
    col3.metric("Prix Actuel", f"{end_price:.2f} €")
    # Transformation inverse pour l'affichage (Exponentielle)
    data['Regression'] = np.exp(y_pred)
    data['Upper_1s'] = np.exp(y_pred + std_dev)
    data['Lower_1s'] = np.exp(y_pred - std_dev)
    data['Upper_2s'] = np.exp(y_pred + 2 * std_dev)
    data['Lower_2s'] = np.exp(y_pred - 2 * std_dev)

    # Affichage des statistiques
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR (%)", f"{cagr:.2f}%")
    col2.metric("R² (Fiabilité)", f"{r2:.4f}")
    col3.metric("Prix Actuel", f"{end_price:.2f} €")

    # Graphique Plotly
    fig = go.Figure()

    # Bandes Sigma
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_2s'], line=dict(color='rgba(255, 0, 0, 0.2)'), name="+2 Sigma"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_2s'], line=dict(color='rgba(255, 0, 0, 0.2)'), fill='tonexty', name="-2 Sigma"))
    
    # Prix et Régression
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Cours de Bourse", line=dict(color='royalblue', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Regression'], name="Régression Log", line=dict(color='gold', dash='dash')))

    fig.update_layout(
        title=f"Analyse Logarithmique de {ticker_name}",
        yaxis_type="log", # Échelle logarithmique
        xaxis_title="Années",
        yaxis_title="Prix (Echelle Log)",
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table de données (optionnel)
    if st.checkbox("Afficher les données brutes"):
        st.write(data.tail())
else:
    st.error("Données non disponibles pour ce ticker.")
