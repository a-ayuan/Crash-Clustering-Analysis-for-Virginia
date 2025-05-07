from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
import matplotlib.pyplot as plt

def kmeans_clustering(spatial_data, n_clusters=None):
    """Perform KMeans clustering on spatial data"""
    if n_clusters is None:
        n_clusters = find_optimal_k(spatial_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(spatial_data)

    silhouette_avg = silhouette_score(spatial_data, cluster_labels)
    calinski_score = calinski_harabasz_score(spatial_data, cluster_labels)
    davies_score = davies_bouldin_score(spatial_data, cluster_labels)

    print(f"KMeans clustering results:")
    print(f"- Number of clusters: {n_clusters}")
    print(f"- Silhouette Score: {silhouette_avg:.4f}")
    print(f"- Calinski-Harabasz Index: {calinski_score:.4f}")
    print(f"- Davies-Bouldin Index: {davies_score:.4f}")

    return kmeans, cluster_labels

def find_optimal_k(spatial_data, max_k=20, visualize=False):
    """Find optimal number of clusters for KMeans using Elbow method"""
    inertia = []
    silhouette_scores = []

    K_range = range(2, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(spatial_data)
        inertia.append(kmeans.inertia_)

    try:
        kneedle_inertia = KneeLocator(
            K_range,
            inertia,
            S=1.0,
            curve='convex',
            direction='decreasing'
        )
        optimal_k_inertia = kneedle_inertia.elbow
    except:
        optimal_k_inertia = None
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertia, 'bo-')
        if optimal_k_inertia:
            plt.axvline(x=optimal_k_inertia, color='r', linestyle='--',
                    label=f'Optimal k = {optimal_k_inertia}')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Optimal k by Elbow Method: {optimal_k_inertia}")

    return optimal_k_inertia