
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

def def_geo_features(crash_data, geo_features = ["Crash Severity", "Light Condition", "Roadway Surface Condition", "Relation To Roadway",
                            "Roadway Alignment", "Roadway Surface Type", "Roadway Defect", "Intersection Type",
                            "Traffic Control Type", "Max Speed Diff",
                            "RoadDeparture Type", "Intersection Analysis", "x", "y"]):    
    # Taking a look at the columns to see which ones might be the most helpful
    all_columns = crash_data.columns.tolist()
    for col in all_columns:
        print(col)

    crash_data_geo = crash_data[geo_features]

    if crash_data_geo['x'].isnull().any() or crash_data_geo['y'].isnull().any():
            print("Removing entries with missing spatial coordinates")
            crash_data_geo = crash_data_geo.dropna(subset=['x', 'y'])
    
    return crash_data_geo

def pipline(crash_data_geo):
    # Categorical and numeric data separation
    crash_data_num = crash_data_geo.select_dtypes(include=[np.number]).columns.tolist()
    crash_data_cat = crash_data_geo.select_dtypes(exclude=[np.number]).columns.tolist()

    # Separate crash severity and other categorical data
    crash_sev = ["Crash Severity"]
    other_cat_data = [col for col in crash_data_cat if col != "Crash Severity"]


    # Pipe
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, crash_data_num),
        ("sev", OrdinalEncoder(), crash_sev),
        ("cat", OneHotEncoder(handle_unknown="ignore"), other_cat_data) # Avoids issues with unseen categories?
    ])

    processed_crash_data = full_pipeline.fit_transform(crash_data_geo)
    print(processed_crash_data.shape)  # Check transformed data shape

    return processed_crash_data

def get_spatial_data(crash_data_geo):
    # Spatial Data
    numeric_columns = crash_data_geo.select_dtypes(include=[np.number]).columns.tolist()
    
    if "Max Speed Diff" in numeric_columns:
        numeric_columns.remove("Max Speed Diff")
    
    spatial_data = numeric_columns

    # Spatial coords with scaling
    spatial_coords = crash_data_geo[['x', 'y']].copy()

    if spatial_coords.isnull().any().any():
        spatial_coords = spatial_coords.dropna()

    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(spatial_coords)

    return spatial_data, scaled_coords, spatial_coords, scaler


