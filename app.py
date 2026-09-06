import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from fastdtw import fastdtw

st.set_page_config(page_title="ECG Clustering Dashboard", layout="wide", page_icon="🫀")

st.title("🫀 ECG Clustering Dashboard")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    ## Morphological Analysis of Heartbeats
    This dashboard allows you to explore clustering results on ECG signals using various algorithms.
    """)

@st.cache_data # Evita di ricaricare il file a ogni click dell'utente
def load_data(file_name):
    data = np.load(file_name)
    return data["x_train"], data["labels"], data["representatives"]

def load_all_data():
    x_train_kmeans, labels_kmeans, representatives_kmeans = load_data("risultati_kmeans.npz")
    x_train_man, labels_man, representatives_man = load_data("risultati_DTW_manhattan.npz")
    x_train_euc, labels_euc, representatives_euc = load_data("risultati_DTW_euclidean.npz")
    return (x_train_kmeans, labels_kmeans, representatives_kmeans), (x_train_man, labels_man, representatives_man), (x_train_euc, labels_euc, representatives_euc)

(x_train, labels, representatives), (x_train_man, labels_man, representatives_man), (x_train_euc, labels_euc, representatives_euc) = load_all_data()
K = len(representatives)


st.sidebar.header("Visualization Settings")
max_curves = st.sidebar.slider("Maximum number of background curves", 10, 200, 50)
show_grid = st.sidebar.checkbox("Show Grid", value=True)

# Layout a schede per confrontare i metodi in futuro
tab1, tab2 = st.tabs(["Euclidean K-Means", "K-Means DTW (soon)"])

with tab1:
    st.header("Euclidean Distance Clustering Results")
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

with tab2:
    st.header("Dynamic Time Warping (DTW) Clustering Results")
    measure = st.selectbox("Select Cluster to analyze:", options=["euclidean", "manhattan"])
    metrics = {"manhattan": "0.196", "euclidean": "0.252"}
    x_train_dtw, labels_dtw, representatives_dtw = (x_train_man.copy(), labels_man.copy(), representatives_man.copy()) if measure == "manhattan" else (x_train_euc.copy(), labels_euc.copy(), representatives_euc.copy())
    metric_dtw = metrics[measure]
    print()

    st.metric(label="Silhouette Score Globale", value=metric_dtw)

    fig, axes = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=True, sharey=True)
    for k in range(K):
        cluster_signals_dtw = x_train_dtw[labels_dtw == k]
        ax = axes[k]
        
        for signal in cluster_signals_dtw[:max_curves]:
            ax.plot(signal, color='darkgray', alpha=0.55, linewidth=0.5)
            
        ax.plot(representatives_dtw[k], color='crimson', linewidth=2, label=f'Centroid {k}')
        ax.set_title(f"Cluster {k} ({len(cluster_signals_dtw)} heartbeats)", fontsize=11, fontweight='bold', loc='left')
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5, which='both')
        else:
            ax.grid(False, which='both')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Dynamic Time Warping - Elastic Alignment Visualization")
    target_cluster_dtw = st.selectbox("Select Cluster to analize:", options=range(K))
    signals_cluster_dtw = x_train_dtw[labels_dtw == target_cluster_dtw]
    target_example_dtw = st.selectbox("Select Signal to compare:", options=range(len(signals_cluster_dtw)))
    if len(signals_cluster_dtw) > 0:
        segnale_esempio_dtw = signals_cluster_dtw[target_example_dtw]
        centroide_esempio_dtw = representatives_dtw[target_cluster_dtw]

        _, path = fastdtw(centroide_esempio_dtw, segnale_esempio_dtw, dist=lambda x, y: abs(x - y))

        fig, ax = plt.subplots(figsize=(12, 6))
        
        offset = np.max(centroide_esempio_dtw) - np.min(segnale_esempio_dtw) + 0.5
        
        ax.plot(centroide_esempio_dtw + offset, color='crimson', linewidth=2, label='Centroid DBA (upper shifted)')
        ax.plot(segnale_esempio_dtw, color='royalblue', linewidth=2, label='Real ECG Signal from Cluster')

        for t_bary, t_sig in path[::4]:
            ax.plot([t_bary, t_sig], [centroide_esempio_dtw[t_bary] + offset, segnale_esempio_dtw[t_sig]], 
                    color='black', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_title("Elastic Alignment Display (Dynamic Time Warping)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        
    else:
        st.warning(f"The cluster {target_cluster_dtw} is empty.")
            
