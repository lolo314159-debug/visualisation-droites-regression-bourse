import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CAC 40 Log Regression", layout="wide")

# Liste étendue pour tester
tickers = {
    "LVMH": "MC.PA", "Air Liquide": "AI.PA", "TotalEnergies": "TTE.PA", 
    "Schneider": "SU.PA", "L'Oréal": "OR.PA", "Airbus": "AIR.PA", "Sanofi": "SAN.PA"
}

st.sidebar.title("Configuration")
name = st.sidebar.selectbox("Sélectionner une action", list(tickers.keys()))
symbol = tickers[name]

@st.cache_data
def get_data(ticker):
    try:
        df = yf.download(ticker, start="2000-01-01")
        return df.dropna()
    except:
        return pd.DataFrame()

df = get_data(symbol)

if not df.empty and len(df) > 10:
    # 1. Préparation des données
    df['Days'] = np.arange(len(df))
    X = df['Days'].values.reshape(-1, 1)
    # Sécurité : on s'assure que le prix est positif
    prices = df['Close'].values.flatten()
    y = np.log(np.where(prices > 0, prices, 0.01)) 

    # 2. Régression
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    std_dev = np.std(y - y_pred)

    # 3. Calcul CAGR ultra-sécurisé
    cagr_val = "N/A"
    try:
        p_start = float(prices[0])
        p_end = float(prices[-1])
        date_start = df.index[0]
        date_end = df.index[-1]
        years = (date_end - date_start).days / 365.25
        
        if years > 0 and p_start > 0:
            cagr_raw = (pow(p_end / p_start, 1/years) - 1) * 100
            cagr_val = f"{cagr_raw:.2f}%"
    except:
        pass

    # 4. Affichage des Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR moyen", cagr_val)
    c2.metric("R² (Corrélation)", f"{r2:.4f}")
    c3.metric("Écart-type (log)", f"{std_dev:.4f}")

    # 5. Graphique
    fig = go.Figure()
    
    # Bandes Sigma 2
    y_upper = np.exp(y_pred + 2*std_dev)
    y_lower = np.exp(y_pred - 2*std_dev)
    
    fig.add_trace(go.Scatter(x=df.index, y=y_upper, line=dict(width=0), showlegend=False, name="+2s"))
    fig.add_trace(go.Scatter(x=df.index, y=y_lower, line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.1)', name="Zone de confiance"))
    
    # Prix et Régression
    fig.add_trace(go.Scatter(x=df.index, y=prices, name="Prix de clôture", line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=np.exp(y_pred), name="Tendance Log", line=dict(color='gold', dash='dash')))

    fig.update_layout(
        title=f"Régression Logarithmique : {name}",
        yaxis_type="log",
        template="plotly_dark",
        height=700,
        xaxis_title="Date",
        yaxis_title="Prix (Échelle Log)"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Données insuffisantes pour cette action.")
