from load_data import load_data
from data_exploration import explore_data, visualize_data, correlation_features
from data_cleaning import def_geo_features, pipline, get_spatial_data
from db_scan import dbscan_clustering, find_optimal_eps
from data_analysis import analyze_clusters, visualize_cluster_properties
from cluster_mapping import plot_clusters, visualize_clusters_map
from k_means import find_optimal_k, kmeans_clustering

"""Data Exploration"""
load_crash_data = load_data()
crash_data = load_crash_data[load_crash_data['Crash Year'].isin([2024, 2025])] # Filter for recent data

explore_data(crash_data=crash_data)
visualize_data(crash_data=crash_data)
correlation_features(crash_data=crash_data)

"""Data Cleaning"""
crash_data_geo = def_geo_features(crash_data=crash_data)
processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

"""Perform First DBScan"""
first_dbscan, first_cluster_labels = dbscan_clustering(spatial_data=processed_crash_data, eps=0.5, min_samples=10)
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=first_cluster_labels, method_name=first_dbscan)

"""Find Optimal Epsilon and Clustering with both DBScan & K-means"""
optimal_eps = find_optimal_eps(spatial_data=scaled_coords)
print(f"Optimal epsilon value for DBSCAN: {optimal_eps:.4f}")

optimal_k = find_optimal_k(spatial_data=scaled_coords)
print(f"Optimal number of clusters: {optimal_k}")

dbscan, dbscan_labels = dbscan_clustering(spatial_data=scaled_coords, eps=optimal_eps)
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)

"""Visualize Clusters"""
print("\nDBSCAN Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=dbscan_labels, method_name=dbscan)

print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)

kmeans_map = visualize_clusters_map(original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
kmeans_map

"""Cluster Analysis"""
dbscan_stats = analyze_clusters(crash_data_geo=crash_data_geo, cluster_labels=dbscan_labels)
kmeans_stats = analyze_clusters(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

print("DBSCAN Cluster Statistics:")
for cluster, stats in dbscan_stats.items():
    print(f"\nCluster {cluster}:")
    print(f"- Size: {stats['size']} crashes")
    print(f"- Center: ({stats['center'][0]:.4f}, {stats['center'][1]:.4f})")
    print(f"- Radius: {stats['radius']:.4f}")

    if stats['severity_counts']:
        print("- Severity Counts:")
        for sev, count in stats['severity_counts'].items():
            print(f"  - {sev}: {count}")

print("\nK-means Cluster Statistics:")
for cluster, stats in kmeans_stats.items():
    print(f"\nCluster {cluster}:")
    print(f"- Size: {stats['size']} crashes")
    print(f"- Center: ({stats['center'][0]:.4f}, {stats['center'][1]:.4f})")
    print(f"- Radius: {stats['radius']:.4f}")

    if stats['severity_counts']:
        print("- Severity Counts:")
        for sev, count in stats['severity_counts'].items():
            print(f"  - {sev}: {count}")

print("\nDBSCAN Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=dbscan_labels)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)


