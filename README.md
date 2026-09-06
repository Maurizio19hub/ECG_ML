# Clustering ECG Data

## Descrizione del Progetto

Questo progetto si concentra sull'analisi di serie temporali di dati ECG (elettrocardiogramma) per identificare possibili cluster che distinguono le patologie dei pazienti. Utilizzando diversi algoritmi di clustering, come K-Means, DTW (Dynamic Time Warping) e Kernel K-Means, il progetto mira a migliorare la precisione nell'identificazione dei cluster.

## Dipendenze

Le seguenti librerie Python sono necessarie per eseguire il progetto:

- `matplotlib`: per la creazione di grafici
- `numpy`: per la manipolazione di vettori
- `pandas`: per la gestione dei dataframes
- `sktime.datasets`: per il caricamento dei dataset dai file .ts
- `sklearn.metrics`: per valutare l'accuratezza dei risultati
- `scipy.optimize`: per l'algoritmo ungherese
- `sklearn.decomposition`: per la riduzione dimensionale con PCA
- `sklearn.manifold`: per la riduzione dimensionale con TSNE
- `fastdtw`: per calcolare la distanza tra serie temporali
- `tslearn.clustering`: per l'algoritmo KernelKMeans
- `seaborn`: per la personalizzazione grafica dei grafici Matplotlib

## Istruzioni per l'Esecuzione

1. Clonare il repository:
   ```bash
   git clone <URL del repository>
   ```

2. Installare le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Eseguire il notebook Jupyter `heartbeat.ipynb` per visualizzare i risultati del clustering.

## Risultati Attesi

Il progetto utilizza diversi metodi di clustering per analizzare i dati ECG. I risultati includono la visualizzazione dei cluster identificati, l'analisi delle distanze intra e inter-cluster, e la valutazione della qualità del clustering tramite metriche come il Silhouette Score e l'Adjusted Rand Index (ARI).

## Autore

Questo progetto è stato sviluppato come parte di una selezione per una borsa di studio. Per ulteriori informazioni, contattare l'autore.
