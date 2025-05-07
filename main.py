from load_data import load_data
from data_exploration import explore_data, visualize_data, correlation_features
from data_cleaning import def_geo_features, pipline, get_spatial_data
from db_scan import dbscan_clustering
from data_analysis import analyze_clusters, visualize_cluster_properties
from cluster_mapping import plot_clusters

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



