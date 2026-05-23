import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from fastdtw import fastdtw

st.set_page_config(page_title="ECG Clustering Dashboard", layout="wide")

st.title("🫀 Clustering Dashboard for ECG signals.")
st.markdown("Morphological analysis of heartbeats using clustering algorithms.")

@st.cache_data # Evita di ricaricare il file a ogni click dell'utente
def load_data(file_name):
    data = np.load(file_name)
    return data["x_train"], data["labels"], data["representatives"]

x_train, labels, representatives = load_data("risultati_kmeans.npz")
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


x_train, labels, representatives = load_data("risultati_DTW.npz")

with tab2:
    st.header("DTW Distance Results")
    st.metric(label="Silhouette Score Globale", value="0.196")

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
    
    st.subheader("Dynamic Time Warping - Elastic Alignment Visualization")
    target_cluster = st.selectbox("Select Cluster to analize:", options=range(K))
    signals_cluster = x_train[labels == target_cluster]
    target_example = st.selectbox("Select Signal to compare:", options=range(len(signals_cluster)))
    if len(signals_cluster) > 0:
        segnale_esempio = signals_cluster[target_example]
        centroide_esempio = representatives[target_cluster]

        _, path = fastdtw(centroide_esempio, segnale_esempio, dist=lambda x, y: abs(x - y))

        fig, ax = plt.subplots(figsize=(12, 6))
        
        offset = np.max(centroide_esempio) - np.min(segnale_esempio) + 0.5
        
        ax.plot(centroide_esempio + offset, color='crimson', linewidth=2, label='Centroid DBA (upper shifted)')
        ax.plot(segnale_esempio, color='royalblue', linewidth=2, label='Real ECG Signal from Cluster')

        for t_bary, t_sig in path[::4]:
            ax.plot([t_bary, t_sig], [centroide_esempio[t_bary] + offset, segnale_esempio[t_sig]], 
                    color='black', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_title("Elastic Alignment Display (Dynamic Time Warping)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        
    else:
        st.warning(f"The cluster {target_cluster} is empty.")
            