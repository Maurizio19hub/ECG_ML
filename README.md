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

## Dettagli del Progetto

### 1. Importazione delle Librerie

Il progetto utilizza diverse librerie Python per l'analisi dei dati e il clustering. Le librerie principali includono `matplotlib` per la visualizzazione, `numpy` per la manipolazione dei dati, `pandas` per la gestione dei dataframes, e `sktime.datasets` per il caricamento dei dataset.

### 2. Caricamento dei Dataset

I dataset utilizzati nel progetto sono `ECG5000_TRAIN.ts` e `ECG5000_TEST.ts`. Questi contengono serie temporali di dati ECG, con ogni riga che rappresenta l'evoluzione temporale dei battiti cardiaci. I dataset includono anche etichette per la classificazione.

### 3. Implementazione degli Algoritmi di Clustering

#### K-Means
L'algoritmo K-Means è stato utilizzato per identificare i cluster iniziali nei dati ECG. È stato implementato un metodo per determinare il numero ottimale di cluster utilizzando il metodo del gomito.

#### Dynamic Time Warping (DTW)
Il DTW è stato utilizzato per migliorare l'identificazione dei cluster, tenendo conto delle variazioni temporali nei segnali ECG. L'algoritmo DTW Barycenter Averaging (DBA) è stato implementato per calcolare i centroidi dinamici.

#### Kernel K-Means
Il Kernel K-Means è stato utilizzato per migliorare ulteriormente i risultati del clustering. Questo metodo applica un kernel per trasformare i dati in uno spazio di dimensioni superiori, migliorando la separazione dei cluster.

### 4. Valutazione dei Risultati

I risultati del progetto includono la visualizzazione dei cluster identificati tramite grafici, l'analisi delle distanze intra e inter-cluster, e la valutazione della qualità del clustering tramite metriche come il Silhouette Score e l'Adjusted Rand Index (ARI). Sono state utilizzate tecniche di riduzione dimensionale come PCA e t-SNE per visualizzare i dati in due dimensioni.

### 5. Conclusioni

Il progetto ha dimostrato che l'utilizzo di metodi avanzati di clustering, come DTW e Kernel K-Means, può migliorare significativamente la precisione nell'identificazione dei cluster nei dati ECG. Questi metodi permettono di catturare meglio le variazioni temporali e le caratteristiche non lineari dei segnali ECG.

## Autore

Questo progetto è stato sviluppato come parte di una selezione per una borsa di studio. Per ulteriori informazioni, contattare l'autore.
