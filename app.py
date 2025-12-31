import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Debug CAC 40", layout="wide")

st.title("Analyseur de Régression")

# Liste de test
tickers = {"LVMH": "MC.PA", "Air Liquide": "AI.PA", "TotalEnergies": "TTE.PA"}
symbol = st.sidebar.selectbox("Action", list(tickers.values()))

st.write(f"Tentative de téléchargement pour : **{symbol}**...")

# 1. Téléchargement
df = yf.download(symbol, start="2000-01-01")

if df.empty:
    st.error("❌ Yahoo Finance n'a renvoyé aucune donnée. Vérifiez votre connexion ou le ticker.")
else:
    st.success(f"✅ {len(df)} lignes de données récupérées !")
    
    try:
        # Nettoyage
        df = df[['Close']].dropna()
        prices = df['Close'].values.flatten().astype(float)
        
        # 2. Calculs
        days = np.arange(len(prices)).reshape(-1, 1)
        log_prices = np.log(np.maximum(prices, 0.01))
        
        model = LinearRegression().fit(days, log_prices)
        y_pred = np.exp(model.predict(days))
        std_dev = np.std(log_prices - np.log(y_pred))
        
        st.write(f"Modèle calculé avec succès (R2: {model.score(days, log_prices):.4f})")

        # 3. Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=prices, name="Cours réel", line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=y_pred, name="Régression", line=dict(color='gold', dash='dash')))
        
        # Bandes sigma
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(np.log(y_pred) + 2*std_dev), name="+2σ", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=df.index, y=np.exp(np.log(y_pred) - 2*std_dev), name="-2σ", fill='tonexty', line=dict(width=0), fillcolor='rgba(255,215,0,0.1)'))

        fig.update_layout(yaxis_type="log", template="plotly_dark", height=600)
        
        # Affichage forcé
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erreur pendant le calcul : {e}")
