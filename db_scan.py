from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

def dbscan_clustering(spatial_data, eps=None, min_samples=5):
    """Perform DBSCAN clustering on spatial data"""
    if eps is None:
        eps = find_optimal_eps(spatial_data, min_samples)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(spatial_data)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"DBSCAN clustering results:")
    print(f"- Number of clusters: {n_clusters}")
    print(f"- Number of noise points: {n_noise} ({n_noise/len(spatial_data)*100:.2f}%)")

    if n_clusters > 1:
        mask = cluster_labels != -1
        if np.sum(mask) > n_clusters:  # Ensure we have enough points
            silhouette_avg = silhouette_score(
                spatial_data[mask],
                cluster_labels[mask]
            )
            print(f"- Silhouette Score (excluding noise): {silhouette_avg:.4f}")

    return dbscan, cluster_labels

def find_optimal_eps(spatial_data, n_neighbors=10, visualize=False):
    """Find optimal epsilon parameter for DBSCAN using k-distance graph"""
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(spatial_data)
    distances = neigh.kneighbors(spatial_data)
    distances = np.sort(distances[:, n_neighbors-1])

    diffs = np.diff(distances)

    try:
        kneedle = KneeLocator(
            range(len(distances)),
            distances,
            S=1.0,
            curve='convex',
            direction='increasing'
        )
        optimal_eps = distances[kneedle.elbow] if kneedle.elbow else None
    except:
        # If KneeLocator fails, use a simple heuristic
        # Find where the distances start increasing rapidly
        acceleration = np.diff(diffs)
        elbow_idx = np.argmax(acceleration) + 1
        optimal_eps = distances[elbow_idx]
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(distances)), distances, 'b-')
        if optimal_eps:
            plt.axhline(y=optimal_eps, color='r', linestyle='-',
                    label=f'Optimal eps = {optimal_eps:.4f}')
        plt.title('K-distance Graph for Optimal Epsilon Selection')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'Distance to {n_neighbors}th nearest neighbor')
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_eps
    
