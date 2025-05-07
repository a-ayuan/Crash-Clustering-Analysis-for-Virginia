import folium
from folium.plugins import HeatMap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

def plot_clusters(spatial_data, original_coords, cluster_labels, method_name):
    """Visualize clustering results"""
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)

    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
    colormap = {}

    for i, label in enumerate(unique_labels):
        if label == -1:
            colormap[label] = 'black'  # Noise points in black
        else:
            colormap[label] = colors[i]

    plt.figure(figsize=(12, 10))

    for label in unique_labels:
        mask = cluster_labels == label
        marker = 'x' if label == -1 else 'o'
        alpha = 0.5 if label == -1 else 0.8
        plt.scatter(
            original_coords.iloc[mask, 0],
            original_coords.iloc[mask, 1],
            c=[colormap[label]],
            marker=marker,
            alpha=alpha,
            s=50 if label == -1 else 70,
            label=f"Cluster {label}" if label != -1 else "Noise"
        )

    if hasattr(method_name, 'cluster_centers_'):
        centers = method_name.cluster_centers_

        if scaler is not None:
            centers = scaler.inverse_transform(centers)

        plt.scatter(
            centers[:, 0], centers[:, 1],
            c='black',
            s=200,
            alpha=0.5,
            marker='*',
            label='Cluster Centers'
        )

    plt.title(f'Spatial Clusters using {type(method_name).__name__}', fontsize=16)
    plt.xlabel('Longitude (x)', fontsize=14)
    plt.ylabel('Latitude (y)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_clusters_map(original_coords, cluster_labels, method_name):
    """Visualize clustering results on an interactive map"""
    mean_lat = original_coords.iloc[:, 1].mean()
    mean_lon = original_coords.iloc[:, 0].mean()

    map_clusters = folium.Map(location=[mean_lat, mean_lon], zoom_start=12,
                            tiles='CartoDB positron')

    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(num_clusters, 1)))

    color_dict = {}
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            color_dict[label] = 'black'
        else:
            color_dict[label] = mcolors.to_hex(colors[color_idx])
            color_idx += 1

    for label in unique_labels:
        fg = folium.FeatureGroup(name=f"Cluster {label}" if label != -1 else "Noise")
        mask = cluster_labels == label
        cluster_points = original_coords[mask]

        for idx, point in cluster_points.iterrows():
            folium.CircleMarker(
                location=[point.iloc[1], point.iloc[0]],
                radius=5,
                color=color_dict[label],
                fill=True,
                fill_color=color_dict[label],
                fill_opacity=0.7,
                popup=f"Cluster: {label}"
            ).add_to(fg)

        fg.add_to(map_clusters)

    folium.LayerControl().add_to(map_clusters)

    heat_data = [[point.iloc[1], point.iloc[0]] for _, point in original_coords.iterrows()]
    HeatMap(heat_data, radius=15).add_to(folium.FeatureGroup(name="Heatmap").add_to(map_clusters))

    return map_clusters