# Movies Recommendation using K-Means Clustering

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load Dataset
df = pd.read_csv("movies.csv")

print("First 5 Records:\n")
print(df.head())


# Features
X = df[['Action',
        'Comedy',
        'Drama',
        'Horror',
        'Romance',
        'SciFi']]


# Standardization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Elbow Method
wcss = []

for i in range(1, 8):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(7,5))

plt.plot(
    range(1,8),
    wcss,
    marker='o',
    linewidth=2
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)

plt.show()


# Train Model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nDataset with Clusters:\n")
print(df)


# Cluster Centres
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers = pd.DataFrame(
    centers,
    columns=X.columns
)

print("\nCluster Centres")

for i in range(len(centers)):
    print(f"\nCluster {i}")

    print(f"Action   : {centers.loc[i,'Action']:.2f}")
    print(f"Comedy   : {centers.loc[i,'Comedy']:.2f}")
    print(f"Drama    : {centers.loc[i,'Drama']:.2f}")
    print(f"Horror   : {centers.loc[i,'Horror']:.2f}")
    print(f"Romance  : {centers.loc[i,'Romance']:.2f}")
    print(f"SciFi    : {centers.loc[i,'SciFi']:.2f}")


# PCA for Visualization
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

centroids_pca = pca.transform(kmeans.cluster_centers_)


# User Clusters
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['Cluster'],
    cmap='viridis',
    s=80
)

plt.title("Movie User Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()


# User Clusters with Centroids
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['Cluster'],
    cmap='viridis',
    s=80,
    label="Users"
)

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    marker='X',
    c='red',
    s=300,
    label="Centroids"
)

plt.title("Movie User Clusters with Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()


# Recommend Similar Users
user = int(input("\nEnter User ID : "))

if user not in df['UserID'].values:
    print("\nUser ID not found!")

else:

    cluster = df.loc[df['UserID'] == user,'Cluster'].values[0]

    print(f"\nUser {user} belongs to Cluster {cluster}")

    recommend = df[df['Cluster'] == cluster]

    print("\nSimilar Users:\n")

    print(recommend[['UserID','Age', 'Action','Comedy',
                'Drama','Horror','Romance','SciFi']])
