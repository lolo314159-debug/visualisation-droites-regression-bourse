import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Analyse Simple CAC 40", layout="wide")

# Liste des actions
tickers = {
    "LVMH": "MC.PA", "Air Liquide": "AI.PA", "TotalEnergies": "TTE.PA", 
    "Schneider": "SU.PA", "L'Oréal": "OR.PA", "Airbus": "AIR.PA", "Sanofi": "SAN.PA"
}

st.sidebar.title("Menu")
name = st.sidebar.selectbox("Choisir une action", list(tickers.keys()))
symbol = tickers[name]

@st.cache_data
def get_data(ticker):
    try:
        # Téléchargement des données
        df = yf.download(ticker, start="2000-01-01")
        return df.dropna()
    except:
        return pd.DataFrame()

df = get_data(symbol)

if not df.empty and len(df) > 20:
    # 1. Préparation mathématique
    df['Days'] = np.arange(len(df))
    X = df['Days'].values.reshape(-1, 1)
    
    # Transformation Log (on sécurise contre les valeurs <= 0)
    prices = df['Close'].values.astype(float)
    y_log = np.log(np.maximum(prices, 0.01)) 

    # 2. Calcul de la Régression Linéaire sur le Log
    model = LinearRegression().fit(X, y_log)
    y_log_pred = model.predict(X)
    r2 = model.score(X, y_log)
    std_dev = np.std(y_log - y_log_pred)

    # 3. Affichage du score R2
    st.metric("Fiabilité du modèle (R²)", f"{r2:.4f}")

    # 4. Construction du graphique
    fig = go.Figure()
    
    # Conversion inverse (Exponentielle) pour revenir aux prix réels
    trend_line = np.exp(y_log_pred)
    upper_2s = np.exp(y_log_pred + 2 * std_dev)
    lower_2s = np.exp(y_log_pred - 2 * std_dev)
    upper_1s = np.exp(y_log_pred + 1 * std_dev)
    lower_1s = np.exp(y_log_pred - 1 * std_dev)

    # Ajout des zones Sigma
    fig.add_trace(go.Scatter(x=df.index, y=upper_2s, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=lower_2s, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name="Zone 2 Sigma (95%)"))
    
    fig.add_trace(go.Scatter(x=df.index, y=upper_1s, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=lower_1s, fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name="Zone 1 Sigma (68%)"))

    # Prix de clôture
    fig.add_trace(go.Scatter(x=df.index, y=prices, name="Prix Réel", line=dict(color='white', width=1.5)))
    
    # Droite de régression
    fig.add_trace(go.Scatter(x=df.index, y=trend_line, name="Régression Log", line=dict(color='gold', dash='dash')))

    fig.update_layout(
        title=f"Courbe de Régression Logarithmique : {name}",
        yaxis_type="log", # Échelle logarithmique visuelle
        template="plotly_dark",
        height=700,
        xaxis_title="Temps",
        yaxis_title="Prix (Echelle Log)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Impossible d'afficher les données pour ce titre.")
