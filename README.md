# Crash Clustering Analysis for Virginia

This project analyzes vehicle crash data in Virginia to identify high-incident areas and examine the environmental and geographic factors contributing to these incidents. Utilizing clustering algorithms like DBSCAN and KMeans, the analysis aims to uncover patterns that can inform traffic safety measures and urban planning.

## Project Objectives

- **Identify High-Incident Areas**: Use clustering techniques to pinpoint regions with a high frequency of crashes.
- **Analyze Contributing Factors**: Examine environmental and geographic variables that correlate with crash occurrences.
- **Evaluate Clustering Performance**: Compare the effectiveness of DBSCAN and KMeans in clustering crash data.

## Features

- **Data Loading and Cleaning**: Scripts to load and preprocess crash data for analysis.
- **Exploratory Data Analysis**: Tools to visualize and understand data distributions and relationships.
- **Bias Filtering**: Methods to identify and mitigate biases in the dataset.
- **Clustering Algorithms**: Implementation of DBSCAN and KMeans for spatial clustering.
- **Cluster Mapping**: Visualization of clusters to interpret spatial patterns.

## Repository Structure

- `load_data.py`: Data loading functions.
- `data_cleaning.py`: Data preprocessing and cleaning routines.
- `data_exploration.py`: Exploratory data analysis tools.
- `bias_filtering.py`: Bias detection and filtering methods.
- `data_analysis.py`: Comprehensive data analysis scripts.
- `db_scan.py`: DBSCAN clustering implementation.
- `k_means.py`: KMeans clustering implementation.
- `cluster_mapping.py`: Cluster visualization and mapping.
- `main.py`: Main script to execute the analysis pipeline.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- geopandas (if spatial data is involved)

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn geopandas
