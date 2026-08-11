# Movies Recommendation Using K-Means Clustering

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

<!-- This project groups users according to their movie genre preferences using K-Means Clustering. Users with similar ratings are placed into the same cluster, and the system recommends similar users from that cluster. -->

## Project Summary

The model studies preferences across these genres:

- Action
- Comedy
- Drama
- Horror
- Romance
- Sci-Fi

After clustering, the user enters a User ID. The program finds that user's cluster and recommends other users who belong to the same group.

## Features

- Preprocesses data with StandardScaler.
- Uses the Elbow Method to select a cluster count.
- Segments users with K-Means.
- Displays cluster centers.
- Uses PCA for two-dimensional visualization.
- Shows clusters with centroids.
- Recommends users with similar preferences.
- Keeps the implementation beginner-friendly.

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- K-Means Clustering
- StandardScaler
- Principal Component Analysis (PCA)

## Project Files

```text
Movies-Recommendation-KMeans/
|-- Movies-Recommendation-kmean.py
|-- movies.csv
|-- requirements.txt
|-- .gitignore
|-- README.md
`-- Images/
    |-- elbow_method.png
    |-- movie_clusters.png
    `-- movie_clusters_centroids.png
```

## Workflow

1. Load the movie preference dataset with Pandas.
2. Select genre rating columns.
3. Standardize the selected features.
4. Use the Elbow Method to calculate WCSS for different K values.
5. Train the K-Means model.
6. Display average preferences for each cluster.
7. Use PCA to visualize clusters in two dimensions.
8. Generate cluster and centroid graphs.
9. Ask for a User ID.
10. Display similar users from the same cluster.

## Concepts Used

- Unsupervised learning
- K-Means clustering
- Feature scaling
- Standardization
- Elbow Method
- PCA
- Data visualization
- Recommendation systems

## Installation

```bash
git clone https://github.com/yourusername/Movies-Recommendation-KMeans.git
cd Movies-Recommendation-KMeans
pip install -r requirements.txt
python movies.py
```

## Output

- Dataset with cluster labels.
- Cluster centers.
- Elbow Method graph.
- PCA cluster visualization.
- PCA cluster visualization with centroids.
- Similar user recommendations.

## Learning Outcomes

This project covers K-Means clustering, unsupervised machine learning, StandardScaler, the Elbow Method, PCA, cluster interpretation, recommendation-system basics, and Matplotlib visualization.
