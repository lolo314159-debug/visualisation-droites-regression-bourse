import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Analyse CAC 40", layout="wide")

st.title("📈 Analyse de Régression Logarithmique")

# Liste complète du CAC 40
cac40 = {
    "Air Liquide": "AI.PA", "Airbus": "AIR.PA", "AXA": "CS.PA", "BNP Paribas": "BNP.PA",
    "Danone": "BN.PA", "Hermès": "RMS.PA", "L'Oréal": "OR.PA", "LVMH": "MC.PA",
    "Orange": "ORA.PA", "Sanofi": "SAN.PA", "Schneider Electric": "SU.PA", "TotalEnergies": "TTE.PA",
    "Vinci": "DG.PA", "Safran": "SAF.PA", "EssilorLuxottica": "EL.PA"
}

ticker = st.sidebar.selectbox("Sélectionner une action", list(cac40.keys()))
symbol = cac40[ticker]

@st.cache_data
def load_data(s):
    # On télécharge les données et on s'assure qu'on a bien les colonnes
    data = yf.download(s, start="2000-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()

df = load_data(symbol)

if not df.empty:
    # --- CALCULS ---
    prices = df['Close'].values.flatten().astype(float)
    x = np.arange(len(prices)).reshape(-1, 1)
    y_log = np.log(prices)
    
    model = LinearRegression().fit(x, y_log)
    y_pred_log = model.predict(x)
    std_dev = np.std(y_log - y_pred_log)
    
    # --- GRAPHIQUE ---
    fig = go.Figure()

    # 1. Zone Sigma (Jaune translucide)
    fig.add_trace(go.Scatter(
        x=df.index, y=np.exp(y_pred_log + 2*std_dev),
        line=dict(width=0), showlegend=False, name="Upper 2s"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=np.exp(y_pred_log - 2*std_dev),
        fill='tonexty', fillcolor='rgba(255, 215, 0, 0.15)', 
        line=dict(width=0), name="Bandes +/- 2 Sigma"
    ))

    # 2. Droite de Régression (Pointillés Or)
    fig.add_trace(go.Scatter(
        x=df.index, y=np.exp(y_pred_log),
        name="Régression Log",
        line=dict(color='gold', width=2, dash='dash')
    ))

    # 3. COURS RÉEL (Bleu vif pour être bien visible)
    fig.add_trace(go.Scatter(
        x=df.index, y=prices,
        name="Cours de Bourse",
        line=dict(color='#00D4FF', width=1.5)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=700,
        yaxis_type="log",
        yaxis_title="Prix (Echelle Log)",
        xaxis_title="Années",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Modèle validé avec un R² de {model.score(x, y_log):.4f}")
else:
    st.error("Données non disponibles.")
