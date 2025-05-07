import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(crash_data):
    # Overview of the dataset size
    print("The number of entries is " + str(len(crash_data)))
    print("The number of features is " + str(len(crash_data.columns)))

    # Check the categorical data
    crash_cat_data = crash_data.select_dtypes(exclude=[np.number])
    if (len(crash_cat_data) > 0):
        print(f"There are the following categorical features: ")
    for col in crash_cat_data.columns:
        print(f"- {col}")
    else:
        print("There are no categorical features")

    # Check for missing values
    missing_data = crash_data[crash_data.isnull().any(axis=1)]
    if len(missing_data) > 0:
        print(f"\nThere are {len(missing_data)} entries with missing values in the following columns: ")
    for col in missing_data.columns:
        num_missing_vals = len(missing_data[missing_data[col].isnull()])
        if num_missing_vals > 0:
            print(f"- {col} ({num_missing_vals} missing values)")
    else:
        print("There are no missing values")

    # A preview of the data to understand what values we have to work with
    crash_data.head()
    for col in crash_data.columns:
        print(col)

def visualize_data(crash_data):
    # Crash severity = KABCO
    severity_color_map = {
        "K": "red",      # Fatal
        "A": "orange",   # Suspected serious injury
        "B": "yellow",   # Suspected minor injury
        "C": "green",    # Possible injury
        "O": "blue"      # Property damage only
    }
    crash_data["color"] = crash_data["Crash Severity"].map(severity_color_map)

    # Visualize the crashes to see dense areas - could compare to map for further analysis
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        crash_data["x"], crash_data["y"],
        c=crash_data["color"],
        alpha=0.5,
    )
    plt.title("Traffic Crash Locations in Virginia")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def correlation_features(crash_data):
    """Visualize the correlation between numeric features"""
    columns_to_drop = ["OBJECTID", "Document Nbr"]
    corr_matrix = crash_data.drop(columns=columns_to_drop, axis=1).corr(numeric_only=True)
    print(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Car Crash Features")
    plt.show()