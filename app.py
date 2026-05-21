import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score

st.set_page_config(page_title="ECG Clustering Dashboard", layout="wide")

st.title("🫀 Clustering Dashboard for ECG signals.")
st.markdown("Morphological analysis of heartbeats using clustering algorithms.")

@st.cache_data # Evita di ricaricare il file a ogni click dell'utente
def load_data():
    data = np.load("risultati_kmeans.npz")
    return data["x_train"], data["labels"], data["representatives"]

x_train, labels, representatives = load_data()
K = len(representatives)


st.sidebar.header("Visualizzation Settings")
max_curves = st.sidebar.slider("Maximum number of background curves", 10, 200, 50)
show_grid = st.sidebar.checkbox("Show Grid", value=True)

# Layout a schede per confrontare i metodi in futuro
tab1, tab2 = st.tabs(["Euclidean K-Means", "K-Means DTW (soon)"])

with tab1:
    st.header("Euclidean Distance Results")
    st.metric(label="Silhouette Score Globale", value=silhouette_score(x_train, labels))
    
    fig, axes = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=True, sharey=True)
    for k in range(K):
        cluster_signals = x_train[labels == k]
        ax = axes[k]
        
        for signal in cluster_signals[:max_curves]:
            ax.plot(signal, color='darkgray', alpha=0.55, linewidth=0.5)
            
        ax.plot(representatives[k], color='crimson', linewidth=2, label=f'Centroid {k}')
        ax.set_title(f"Cluster {k} ({len(cluster_signals)} heartbeats)", fontsize=11, fontweight='bold', loc='left')
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5, which='both')
        else:
            ax.grid(False, which='both')
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    st.pyplot(fig)