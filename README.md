<div align="center">

# 🫀 ECG_ML — Unsupervised Clustering of Heartbeat Signals

### Discovering cardiac pathology patterns in ECG time series using unsupervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 📖 Overview

This project performs **unsupervised clustering** on the [ECG5000 dataset](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000), a collection of **5000 heartbeat time series** extracted from a 20-hour long ECG recording. Each series represents the temporal evolution of a single heartbeat, and is labeled with the corresponding **cardiac pathology class**.

The goal is **not** to use the labels to train a supervised model, but to verify whether purely unsupervised algorithms — starting only from the raw shape of the signals — are able to **rediscover the underlying pathology groups** on their own. This is a realistic and challenging scenario, closer to what happens in real clinical / research settings where labeled data is scarce or expensive to obtain.

Three different clustering strategies are implemented and progressively compared, each addressing a specific limitation of the previous one:

| Step | Algorithm | Distance / Kernel | Key Idea |
|------|-----------|--------------------|----------|
| 1️⃣ | **K-Means** (from scratch) | Euclidean | Baseline, fast, but blind to time-shifts |
| 2️⃣ | **K-Means + DTW / DBA** (from scratch) | Dynamic Time Warping | Elastic alignment handles temporal shifts |
| 3️⃣ | **Kernel K-Means (GAK)** | Global Alignment Kernel | Non-linear separation in a similarity space |

The final part of the notebook rigorously **evaluates and compares** the three methods using clustering metrics, the **Hungarian algorithm** for optimal cluster-to-class matching, and **dimensionality reduction** (PCA & t-SNE) for visual inspection.

An interactive **Streamlit dashboard** (`app.py`) is also provided to explore results visually.

---

## 🧠 The Algorithms — In Depth

### 1. K-Means (Euclidean, implemented from scratch)

A classic **Lloyd's algorithm** implementation, coded manually (no `sklearn.cluster.KMeans`) to fully control the iteration logic:

- Random centroid initialization with a fixed **seed** for reproducibility.
- Iterative **assignment step**: each heartbeat is assigned to the closest centroid using Euclidean distance.
- Iterative **update step**: each centroid is recomputed as the mean of the points in its cluster.
- Convergence check comparing centroids across iterations.
- The optimal number of clusters `K` is chosen with the **Elbow Method**, plotting inertia against `K`.

**Limitation:** the Euclidean distance compares signals *point-by-point in time*. Since heartbeats can be slightly shifted or stretched in time even within the same pathology class, the Euclidean mean tends to "blur" the resulting centroid, producing poor, non-representative cluster shapes.

### 2. Dynamic Time Warping (DTW) + DBA

To overcome the temporal misalignment problem, the project implements a **K-Means variant based on Dynamic Time Warping**:

- **`fastdtw`** is used to compute an approximate DTW distance between two time series, along with the optimal **warping path** that elastically aligns them in time.
- **STEP 1 — Assignment:** every training signal is assigned to the representative (centroid) with the minimum DTW distance.
- **STEP 2 — Update (DBA — DTW Barycenter Averaging):** instead of a simple arithmetic mean, the new centroid is recomputed by:
  1. Aligning every signal of the cluster to the old centroid via its DTW warping path.
  2. Accumulating, for each time-step of the centroid, the values of the aligned points from all cluster members.
  3. Averaging these accumulated values to produce a **barycenter that respects the temporal elasticity** of the signals.
- This process is repeated until convergence (or a max number of iterations), producing centroids that are much more representative of the true heartbeat morphology than the Euclidean ones.
- A dedicated plot visualizes the **elastic alignment** between a centroid and a real signal, drawing the DTW warping path as connecting lines.

### 3. Kernel K-Means with Global Alignment Kernel (GAK)

As a third and more sophisticated approach, **Kernel K-Means** (via `tslearn.clustering.KernelKMeans`) is applied using the **Global Alignment Kernel (GAK)**:

- GAK computes a similarity (kernel) measure between time series that, similarly to DTW, is robust to temporal distortions, but expresses it as a valid positive-definite kernel.
- Clustering is then performed **implicitly in a high-dimensional feature space** induced by the kernel, without ever explicitly computing coordinates in that space (the "kernel trick").
- Unlike the previous methods, kernel K-Means does **not** produce explicit centroids in the original signal space — clusters are defined purely by similarity relationships.

### 4. Evaluation & Interpretability Toolkit

Since labels are available, the notebook goes beyond simple visual inspection and builds a **rigorous evaluation pipeline**:

- **Silhouette Score** & **Calinski-Harabasz Score** — internal metrics measuring cluster cohesion/separation, computed both with Euclidean and DTW-precomputed distance matrices.
- **Adjusted Rand Index (ARI)** — external metric comparing predicted clusters against ground-truth pathology labels on the held-out test set.
- **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) — applied to the confusion matrix between predicted clusters and true classes to find the **optimal cluster ↔ class assignment**, maximizing the matches on the diagonal. This allows a fair computation of **Precision, Recall and F1-score per class**.
- **Confusion matrices** (Seaborn heatmaps) after Hungarian re-labeling.
- **Intra-cluster vs. Inter-cluster distance** boxplots, to visually assess cluster compactness and separation for every method.
- **Dimensionality reduction for visualization:**
  - **PCA** (Principal Component Analysis) — linear projection of the 140-dimensional signals down to 2D.
  - **t-SNE** (t-Distributed Stochastic Neighbor Embedding) — non-linear embedding that better preserves local neighborhood structure, used as a complementary, more expressive 2D visualization.

---

## 📊 Dataset

**ECG5000** — 5000 heartbeats, each represented as a time series of **140 samples**, labeled with 5 pathology classes (heavily imbalanced, with one dominant "normal" class). The dataset is provided already split into:

- `ECG5000_TRAIN.ts`
- `ECG5000_TEST.ts`

and loaded through `sktime.datasets.load_from_tsfile`.

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Data manipulation | `numpy`, `pandas` |
| Dataset loading (`.ts` format) | `sktime` |
| Clustering metrics | `scikit-learn` (`silhouette_score`, `calinski_harabasz_score`, `adjusted_rand_score`, `confusion_matrix`, ...) |
| Optimal cluster-class matching | `scipy.optimize.linear_sum_assignment` (Hungarian Algorithm) |
| Dimensionality reduction | `scikit-learn` (`PCA`, `TSNE`) |
| DTW distance & alignment | `fastdtw` |
| Kernel K-Means (GAK) | `tslearn` |
| Visualization | `matplotlib`, `seaborn` |
| Interactive dashboard | `streamlit` |

---

## 📁 Project Structure

```
ECG_ML/
├── heartbeat.ipynb              # Main notebook: full analysis & algorithms
├── app.py                       # Streamlit interactive dashboard
├── ECG5000_TRAIN.ts             # Training set (time series + labels)
├── ECG5000_TEST.ts              # Test set (time series + labels)
├── risultati_kmeans.npz         # Cached results — Euclidean K-Means
├── risultati_DTW_euclidean.npz  # Cached results — DTW K-Means (euclidean cost)
├── risultati_DTW_manhattan.npz  # Cached results — DTW K-Means (manhattan cost)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Maurizio19hub/ECG_ML.git
cd ECG_ML
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy sktime fastdtw tslearn streamlit
```

### 3. Run the notebook
Open `heartbeat.ipynb` in Jupyter / VS Code and run all cells to reproduce the full analysis, from data loading to the final PCA/t-SNE comparison.

### 4. Launch the interactive dashboard
```bash
streamlit run app.py
```
The dashboard lets you explore, for each clustering method, the resulting clusters, their centroids, the silhouette score, and an interactive visualization of the DTW elastic alignment between a centroid and a chosen real signal.

---

## 📈 Key Takeaways

- **Euclidean K-Means** is fast but structurally limited when signals share the same shape with small time shifts — a very common situation in physiological signals like ECG.
- **DTW + DBA** significantly improves centroid representativeness by aligning signals elastically in time before averaging, at the cost of higher computational complexity.
- **Kernel K-Means with GAK** offers a further, more flexible non-linear similarity measure, trading off interpretability (no explicit centroids) for potentially better separation.
- Combining **internal metrics** (Silhouette, Calinski-Harabasz), **external metrics** (ARI, Hungarian-matched Precision/Recall/F1) and **visual tools** (PCA, t-SNE, confusion matrices) gives a much more trustworthy picture of clustering quality than any single metric alone.

---

<div align="center">

Made with ❤️ and 🫀 data — by **Maurizio**

</div>
