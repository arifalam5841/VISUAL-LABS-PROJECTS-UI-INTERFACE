# Customer Segmentation Using K-Means Clustering

**Name:** Arif alam  
**Enrollment:** 24111590742

## Overview

This project groups customers into segments with the K-Means Clustering algorithm. It uses annual income and spending score to identify customers with similar spending patterns.

The project also applies the Elbow Method to choose a suitable number of clusters and visualizes the customer groups along with their cluster centroids.

## Objectives

- Perform customer segmentation with K-Means.
- Find the appropriate cluster count using the Elbow Method.
- Visualize customer groups and centroids.
- Understand how unsupervised learning can support business analysis.

## Dataset

Dataset file:

```text
mall_customers.csv
```

Features used:

- Annual Income (k$)
- Spending Score (1-100)

## Workflow

1. Import the required libraries.
2. Load the customer dataset.
3. Select the income and spending-score columns.
4. Standardize the selected features.
5. Use the Elbow Method to compare different K values.
6. Train the K-Means model.
7. Assign a cluster label to each customer.
8. Display cluster centers.
9. Plot customer segments.
10. Plot clusters with centroids.

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Concepts Covered

- Unsupervised learning
- K-Means clustering
- Feature scaling
- Standardization
- Elbow Method
- Cluster centroids
- Data visualization

## Output

- Dataset with assigned cluster labels.
- Elbow Method graph.
- Customer segmentation scatter plot.
- Customer segmentation plot with centroids.
- Cluster center values.

## Project Files

```text
Customer-Segmentation-KMeans/
|-- mall_customers.csv
|-- customer_segmentation.py
|-- requirements.txt
|-- .gitignore
|-- README.md
`-- images/
    |-- elbow_method.png
    |-- customer_clusters.png
    `-- customer_clusters_centroids.png
```

## How to Run

```bash
git clone https://github.com/yourusername/Customer-Segmentation-KMeans.git
cd Customer-Segmentation-KMeans
pip install -r requirements.txt
python customer_segmentation.py
```

## Future Scope

- Use Silhouette Score to select clusters automatically.
- Add interactive plots.
- Include more customer attributes such as age and gender.
- Build a Streamlit web application.
- Export clustered customer data to CSV.
