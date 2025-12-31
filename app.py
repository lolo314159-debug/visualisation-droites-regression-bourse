import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CAC 40 Log Regression", layout="wide")

# Liste simplifiée pour tester la stabilité
tickers = {"LVMH": "MC.PA", "Air Liquide": "AI.PA", "TotalEnergies": "TTE.PA", "Schneider": "SU.PA", "L'Oréal": "OR.PA"}

st.sidebar.title("Configuration")
name = st.sidebar.selectbox("Sélectionner une action", list(tickers.keys()))
symbol = tickers[name]

@st.cache_data
def get_data(ticker):
    df = yf.download(ticker, start="2000-01-01")
    return df.dropna()

df = get_data(symbol)

if not df.empty:
    # Préparation des données
    df['Days'] = np.arange(len(df))
    X = df['Days'].values.reshape(-1, 1)
    # On s'assure que le prix est > 0 pour le log
    y = np.log(df['Close'].values.clip(min=0.01)) 

    # Régression
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    std_dev = np.std(y - y_pred)

    # Calcul CAGR sécurisé
    try:
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (pow(df['Close'].iloc[-1] / df['Close'].iloc[0], 1/years) - 1) * 100
    except:
        cagr = 0.0

    # Affichage
    c1, c2 = st.columns(2)
    c1.metric("CAGR (%)", f"{cagr:.2f}%")
    c2.metric("R² (Fiabilité)", f"{r2:.4f}")

    # Graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Cours", line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred), name="Reg. Log", line=dict(color='gold')))
    
    # Bandes Sigma 2
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred + 2*std_dev), name="+2σ", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred - 2*std_dev), name="-2σ", fill='tonexty', line=dict(width=0)))

    fig.update_layout(yaxis_type="log", template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Erreur lors du chargement des données.")
