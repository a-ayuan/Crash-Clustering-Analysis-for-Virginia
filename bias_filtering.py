from load_data import load_data
from data_cleaning import def_geo_features, pipline, get_spatial_data
from data_analysis import visualize_cluster_properties
from cluster_mapping import plot_clusters
from k_means import find_optimal_k, kmeans_clustering

"""Filter data by Daylight Light Condition"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025]) & load_crash_data['Light Condition'].isin(['2. Daylight']))]

# Pipeline
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter data by Non-Daylight Light Condition"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Light Condition'].isin(['1. Dawn', '3. Dusk',
                                                                         '4. Darkness - Road Lighted',
                                                                         '5. Darkness - Road Not Lighted']))]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Dry Conditions"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Roadway Surface Condition'].isin(['1. Dry'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Relation To Roadway",
                          "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Wet Conditions"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Roadway Surface Condition'].isin(['2. Wet'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Relation To Roadway",
                          "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Intoxicated Conditions"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Alcohol?'].isin(['Yes'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Distracted Conditions"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Distracted?'].isin(['Yes'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Not Wearing Seatbelt"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Unrestrained?'].isin(['Unbelted'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)

"""Filter By Wearing Seatbelt"""
load_crash_data = load_data()
crash_data = load_crash_data[(load_crash_data['Crash Year'].isin([2024, 2025])
                              & load_crash_data['Unrestrained?'].isin(['Belted'])
                              )]

# Get spatial data
crash_data_geo = def_geo_features(crash_data=crash_data, geo_features=["Crash Severity", "Light Condition", "Roadway Surface Condition", "Relation To Roadway",
                          "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                          "Traffic Control Type", "Max Speed Diff",
                          "RoadDeparture Type", "Intersection Analysis", "x", "y"])

processed_crash_data = pipline(crash_data_geo=crash_data_geo)
spatial_data, scaled_coords, original_coords, scaler = get_spatial_data(crash_data_geo=crash_data_geo)

# Find optimal k
optimal_k = find_optimal_k(spatial_data=scaled_coords)

# Perform k means clustering
kmeans, kmeans_labels = kmeans_clustering(spatial_data=scaled_coords, n_clusters=optimal_k)
print("\nKMeans Clustering Results:")
plot_clusters(spatial_data=scaled_coords, original_coords=original_coords, cluster_labels=kmeans_labels, method_name=kmeans)
print("\nKMeans Cluster Analysis:")
visualize_cluster_properties(crash_data_geo=crash_data_geo, cluster_labels=kmeans_labels)