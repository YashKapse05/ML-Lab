"""
ML Lab 9: K-Means Clustering and Hierarchical Clustering
Student : Yash Kapse | Section A | Roll No. 63 | Batch A4
Dataset : Wholesale customers data.csv (place it in the same folder as this file)


"""

# ---------------------------------------------------------------------------
# 1. Import required libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (AgglomerativeClustering, KMeans)
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from IPython.display import display      
except ImportError:
    display = print                          

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# ---------------------------------------------------------------------------
# 2. Load the dataset and explore it
# ---------------------------------------------------------------------------
df = pd.read_csv("Wholesale customers data.csv")

print("First five rows:")
display(df.head())

print("Last five rows:")
display(df.tail())

print("Dataset shape:", df.shape)

print("\nStatistical summary:")
display(df.describe())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nColumns:")
print(df.columns)

print("\nNumber of duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------------------
# 3. Select features
# ---------------------------------------------------------------------------
features = [
    'Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
    'Detergents_Paper', 'Delicassen'
]

X = df[features].copy()
display(X.head())

print("Number of records used for clustering: ", len(X))

# ---------------------------------------------------------------------------
# 4. Exploratory data analysis
# ---------------------------------------------------------------------------
# 4.1 Count plots of the categorical columns
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Channel')
plt.title('Countplot of Channel')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Region')
plt.title('Countplot of Region')
plt.tight_layout()
plt.show()

# 4.2 Histograms of the continuous (spending) features
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen',
                       'Detergents_Paper', 'Delicassen']
df[continuous_features].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Continuous Features', fontsize=16)
plt.tight_layout()
plt.show()

# 4.3 Boxplots to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[continuous_features])
plt.title('Boxplots to Detect Outliers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.4 Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Standardize the features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# ---------------------------------------------------------------------------
# 6. Choose the number of clusters
# ---------------------------------------------------------------------------
# 6.1 Elbow method (WCSS / inertia)
wcss = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

# 6.2 Silhouette score for K = 2 to 10
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"For n_clusters = {k}, the average silhouette_score is : {score:.4f}")

silhouette_table = pd.DataFrame({'n_clusters': k_range,
                                 'silhouette_score': silhouette_scores})
display(silhouette_table)

suggested_k = int(
    silhouette_table.loc[silhouette_table['silhouette_score'].idxmax(), 'n_clusters']
)
print("K suggested by the highest silhouette score", suggested_k)

# ---------------------------------------------------------------------------
# 7. K-Means clustering
# ---------------------------------------------------------------------------
K = suggested_k

kmeans_model = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_scaled)

df_kmeans = df.copy()
df_kmeans["KMeans_Cluster"] = kmeans_labels
print("K-Means Clustering Completed.")
print("\nNumber of customers in each cluster: ")
print(df_kmeans["KMeans_Cluster"].value_counts().sort_index())

print("\nSample records with cluster labels: ")
display(df_kmeans.head())

# ---------------------------------------------------------------------------
# 8. Hierarchical (Agglomerative) clustering - Ward linkage
# ---------------------------------------------------------------------------
hierarchial_model = AgglomerativeClustering(
    n_clusters=K,
    linkage="ward",
    metric="euclidean"
)
hierarchical_labels = hierarchial_model.fit_predict(X_scaled)

df_hierarchical = df.copy()
df_hierarchical["Hierarchical_Cluser"] = hierarchical_labels

print("Hierarchical Clustering Completed.")
print("\nNumber of customers in each cluster: ")
print(df_hierarchical["Hierarchical_Cluser"].value_counts().sort_index())

print("\nSample records with cluster labels: ")
display(df_hierarchical.head())

# ---------------------------------------------------------------------------
# 9. Dendrograms
# ---------------------------------------------------------------------------
Z = linkage(X_scaled, method="ward", metric="euclidean")

# 9.1 Full dendrogram
plt.figure(figsize=(16, 7))
dendrogram(Z, leaf_rotation=90, leaf_font_size=6)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Customer Index")
plt.ylabel("Ward Linkage Distance Diagram")
plt.tight_layout()
plt.show()

# 9.2 Truncated dendrogram (last 30 merged groups)
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True, leaf_rotation=45)
plt.title("Truncated Dendrogram - Last 30 merged groups")
plt.xlabel("Cluster or Group")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 10. PCA - project the clusters to 2 dimensions
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(
    "Variance explained by the 2 PCA components: ",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "KMeans_Cluster": kmeans_labels.astype(str),
    "Hierarchical_Cluster": hierarchical_labels.astype(str)
})

plt.figure(figsize=(9, 6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="KMeans_Cluster",
                palette="tab10", s=70)
plt.title("Customer Segmenrs: K-Means Clustering Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 11. Evaluation function and comparison of both algorithms
# ---------------------------------------------------------------------------
def evaluate_clustering(X_data, labels, model_name):
    return {
        "Algorithm": model_name,
        "Number of Clusters": len(np.unique(labels)),
        "Silhouette Score": silhouette_score(X_data, labels),
        "Davies-Bouldin Score": davies_bouldin_score(X_data, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X_data, labels)
    }

# (Added) use the function to compare K-Means and Hierarchical clustering
evaluation_table = pd.DataFrame([
    evaluate_clustering(X_scaled, kmeans_labels, "K-Means"),
    evaluate_clustering(X_scaled, hierarchical_labels, "Hierarchical (Ward)")
])
print("Clustering evaluation metrics:")
display(evaluation_table.round(4))

cluster_sizes = pd.DataFrame({
    "K-Means": df_kmeans["KMeans_Cluster"].value_counts().sort_index(),
    "Hierarchical": df_hierarchical["Hierarchical_Cluser"].value_counts().sort_index()
}).fillna(0).astype(int)

print("Cluster Sizes: ")
display(cluster_sizes)

# ---------------------------------------------------------------------------
# 12. Profile the K-Means clusters
# ---------------------------------------------------------------------------
kmeans_profile = df[features].copy()
kmeans_profile["Cluster"] = kmeans_labels

print("Average Spending Profile of K-Means Clusters: ")
display(kmeans_profile.groupby("Cluster")[features].mean().round(2))

cluster_distribution = pd.DataFrame({
    "Customer Count": pd.Series(kmeans_labels).value_counts().sort_index()
})
cluster_distribution["Percentage"] = (
    cluster_distribution["Customer Count"] / len(df) * 100
).round(2)

print("K-Means Cluster Distribution: ")
display(cluster_distribution)

# Heatmap of average spending per cluster
KMeans_means = kmeans_profile.groupby("Cluster")[continuous_features].mean()

plt.figure(figsize=(12, 6))
sns.heatmap(KMeans_means, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Average Spending by K-means Cluster")
plt.xlabel("Spending Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# Channel composition inside each cluster
channel_mix = pd.crosstab(df["Channel"], kmeans_labels, normalize="columns") * 100
channel_mix.columns = [f"Cluster {c}" for c in channel_mix.columns]
print("Channel composition within each K-Means cluster (%) : ")
display(channel_mix.round(2))
