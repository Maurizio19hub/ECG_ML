# ECG Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Unsupervised clustering of ECG heartbeat time series using K-means, Dynamic Time Warping (DTW) with DBA, and Kernel K-means (GAK). The project aims to distinguish different cardiac pathologies from the ECG5000 dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Skills Demonstrated](#key-skills-demonstrated)
- [Contact](#contact)

## Overview
This repository contains a Jupyter notebook (`heartbeat.ipynb`) that performs clustering analysis on the ECG5000 time series dataset. The goal is to group heartbeat signals into clusters that correspond to different cardiac conditions (e.g., normal vs. abnormal) without using the provided labels during training. The analysis compares three clustering approaches:

1. **K-means** with Euclidean distance and the elbow method for selecting the number of clusters.
2. **DTW-based clustering** using Dynamic Time Warping distance and DBA (DTW Barycenter Averaging) for centroid updates.
3. **Kernel K-means** with the Global Alignment Kernel (GAK) implemented via `tslearn`.

The notebook includes comprehensive evaluation using internal metrics (Silhouette, Calinski-Harabasz), external metrics (Adjusted Rand Index on a held-out test set), confusion matrix analysis with Hungarian algorithm mapping, intra/inter cluster distance boxplots, and dimensionality reduction visualizations (PCA and t-SNE).

## Dataset
The **ECG5000** dataset is a subset of the UCR Time Series Archive. It contains 5,000 heartbeat recordings (140 time steps each) split into a training set (`ECG5000_TRAIN.ts`) and a test set (`ECG5000_TEST.ts`). Each recording is labeled with one of five classes representing different pathologies, but for clustering we treat the labels as ground truth only for evaluation.

## Methods
### 1. K-means (Euclidean)
- Centroids are initialized randomly (with a fixed seed for reproducibility).
- The elbow method is used to determine the optimal number of clusters (K=2).
- Assignment step: each time series is assigned to the nearest centroid based on Euclidean distance.
- Update step: centroids are recomputed as the arithmetic mean of the assigned series.

### 2. DTW + DBA
- Distance between time series is computed using **fastdtw** (an efficient approximation of DTW).
- Centroids are updated using **DTW Barycenter Averaging (DBA)**, which aligns each series to the current centroid via the DTW path and averages the aligned points.
- This approach captures temporal misalignments that Euclidean distance cannot handle.

### 3. Kernel K-means (GAK)
- Uses the **Global Alignment Kernel** (GAK) to measure similarity between time series in a high-dimensional feature space.
- Implemented with `tslearn.clustering.KernelKMeans`.
- The kernel bandwidth (`sigma`) is set to 5.0, and the algorithm runs for 20 iterations.

## Evaluation
The clustering quality is assessed through multiple complementary metrics:

- **Internal metrics**: Silhouette score (computed with DTW distance matrix for the DTW method) and Calinski-Harabasz index.
- **External metric**: Adjusted Rand Index (ARI) on the test set, comparing predicted cluster assignments to true labels.
- **Hungarian algorithm**: Automatically maps each cluster to the most likely true class by maximizing the diagonal of the confusion matrix. This mapping is used to compute per-class precision, recall, and F1-score.
- **Intra/inter cluster distances**: Boxplots comparing distances within the same cluster vs. between different clusters.
- **Visualization**: PCA and t-SNE projections of the training data, colored by cluster assignment, to inspect the separation in 2D.

## Results
The notebook outputs the following key findings (exact numbers depend on the run, but the trends are consistent):

- The elbow method suggests **K = 2** clusters, which aligns with the binary nature of the dataset (normal vs. abnormal).
- **DTW + DBA** significantly improves cluster purity compared to Euclidean K-means, as it accounts for temporal warping.
- **Kernel K-means (GAK)** achieves the highest ARI on the test set, indicating the best alignment with true pathology labels.
- The Hungarian mapping reveals that one cluster predominantly contains normal heartbeats, while the other captures various abnormal morphologies.
- PCA and t-SNE plots show clearer separation for the DTW and Kernel methods.

All results are saved in `.npz` files (`risultati_kmeans.npz`, `risultati_DTW_euclidean.npz`) for further analysis.

## Repository Structure
