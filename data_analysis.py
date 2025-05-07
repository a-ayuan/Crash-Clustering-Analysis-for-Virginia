import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def analyze_clusters(crash_data_geo, cluster_labels):
    """Analyze the characteristics of each cluster"""
    cluster_df = crash_data_geo.copy()
    cluster_df['cluster'] = cluster_labels

    cluster_stats = {}

    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        cluster_data = cluster_df[cluster_df['cluster'] == cluster]
        cluster_size = len(cluster_data)

        center_x = cluster_data['x'].mean()
        center_y = cluster_data['y'].mean()

        if cluster_size > 1:
            points = cluster_data[['x', 'y']].values
            center = np.array([[center_x, center_y]])
            distances = cdist(points, center, 'euclidean')
            radius = distances.mean()
        else:
            radius = 0

        severity_counts = None
        if 'Crash Severity' in cluster_data.columns:
            severity_counts = cluster_data['Crash Severity'].value_counts().to_dict()

        cluster_stats[cluster] = {
            'size': cluster_size,
            'center': (center_x, center_y),
            'radius': radius,
            'severity_counts': severity_counts
        }

        for col in cluster_data.columns:
            if col in ['x', 'y', 'cluster']:
                continue

            if cluster_data[col].dtype.name in ['object', 'category']:
                cluster_stats[cluster][f'{col}_counts'] = cluster_data[col].value_counts().to_dict()

    return cluster_stats

def visualize_cluster_properties(crash_data_geo, cluster_labels):
    """Visualize properties of clusters"""
    cluster_df = crash_data_geo.copy()
    cluster_df['cluster'] = cluster_labels

    if -1 in cluster_labels:
        cluster_df_filtered = cluster_df[cluster_df['cluster'] != -1]
    else:
        cluster_df_filtered = cluster_df

    if len(np.unique(cluster_df_filtered['cluster'])) > 0:
        plt.figure(figsize=(12, 6))
        cluster_sizes = cluster_df_filtered['cluster'].value_counts().sort_index()
        cluster_sizes.plot(kind='bar', color='skyblue')
        plt.title('Number of Crashes per Cluster', fontsize=16)
        plt.xlabel('Cluster ID', fontsize=14)
        plt.ylabel('Number of Crashes', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        if 'Crash Severity' in crash_data_geo.columns:
            plt.figure(figsize=(14, 8))
            severity_order = ['K', 'A', 'B', 'C', 'O']

            severity_pivot = pd.crosstab(
                cluster_df_filtered['cluster'],
                cluster_df_filtered['Crash Severity']
            )

            for sev in severity_order:
                if sev not in severity_pivot.columns:
                    severity_pivot[sev] = 0

            severity_pivot = severity_pivot[severity_order]

            severity_pivot_pct = severity_pivot.div(severity_pivot.sum(axis=1), axis=0) * 100

            print("\nCrash Severity Distribution by Cluster (Percentage):")
            print(severity_pivot_pct.round(2))


            severity_pivot_pct.plot(kind='bar', stacked=True,
                                   colormap='RdYlGn_r',
                                   figsize=(14, 8))

            plt.title('Crash Severity Distribution by Cluster', fontsize=16)
            plt.xlabel('Cluster ID', fontsize=14)
            plt.ylabel('Percentage', fontsize=14)
            plt.legend(title='Severity', title_fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        cat_columns = [col for col in crash_data_geo.select_dtypes(include=['object', 'category']).columns
                      if col != 'Crash Severity']

        for col in cat_columns[:3]:
            plt.figure(figsize=(15, 8))

            clusters = sorted(cluster_df_filtered['cluster'].unique())

            for i, cluster_id in enumerate(clusters):
                if len(clusters) <= 3:
                    plt.subplot(1, len(clusters), i+1)
                else:
                    plt.subplot(2, (len(clusters)+1)//2, i+1)

                cluster_data = cluster_df_filtered[cluster_df_filtered['cluster'] == cluster_id]
                top_cats = cluster_data[col].value_counts().nlargest(5)

                top_cats.plot(kind='barh', color='skyblue')

                plt.title(f'Cluster {cluster_id}: Top {col}', fontsize=12)
                plt.xlabel('Count', fontsize=10)
                plt.tight_layout()

            plt.suptitle(f'Top {col} Categories by Cluster', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()