#---------------------Adaptive Delaunay Topology-based Geographically Weighted Regression ----------------\


#!pip install mgwr # This extension is used for standard GWR
# Other conventional packages like numpy, pandas, etc. are also required
#!pip install tqdm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import pyproj
from tqdm import tqdm


# Reading the CSV dataset
data = pd.read_csv('/content/drive/MyDrive/GWR/Depression.csv')

#Removing Duplicate records
initial_count = data.shape[0]
print(f"Number of records before removing duplications: {initial_count}")

data = data.drop_duplicates()

final_count = data.shape[0]
print(f"Number of records after removing duplications: {final_count}")

# Extraction of Coordinates, Indipendent Variables and Dependent Variable
coords = data.iloc[:, :2].values

# Check if coordinates are in geographic system (longitude, latitude)
if (np.min(coords[:, 0]) >= -180 and np.max(coords[:, 0]) <= 180) and \
   (np.min(coords[:, 1]) >= -90 and np.max(coords[:, 1]) <= 90):

    print("Detected geographic coordinates (longitude, latitude). Converting to projected system...")

    # Determine UTM zone based on data centroid
    lon_center = np.mean(coords[:, 0])
    lat_center = np.mean(coords[:, 1])
    utm_zone = int((lon_center + 180) / 6) + 1
    proj_string = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"

    # Define transformation
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    # Convert all coordinates
    coords_projected = np.array([transformer.transform(lon, lat) for lon, lat in coords])

    # Replace original coords with projected ones
    coords = coords_projected

    print(f"Converted to UTM Zone {utm_zone} projection.")


response = data.iloc[:, 2].values
features = data.iloc[:, 3:].values
feature_names = data.columns[3:]

# Setting of variable neighborhood points
neighbors_range = range(15, 51, 5)

#  AIC Calculation
def compute_aic(y_true, y_pred, num_params):
    residuals = y_true - y_pred
    sse = np.sum(residuals ** 2)
    n = len(y_true)
    if sse == 0:
        sse = 1e-10  # Preventing log(0)
    aic = n * np.log(sse / n) + 2 * num_params
    return aic

#  Adjusted R2 - Calculation
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

best_models = {
    'R2': {'value': -np.inf, 'params': None, 'predictions': None},
    'RMSE': {'value': np.inf, 'params': None, 'predictions': None},
    'AIC': {'value': np.inf, 'params': None, 'predictions': None}
}

best_betas = None


for num_neighbors in neighbors_range:
    y_true, y_pred = [], []
    betas = []

    #for idx in range(len(coords)):
    for idx in tqdm(range(len(coords)), desc=f"Processing {num_neighbors} Neighbors", unit="point"):
        selected_point = coords[idx]
        distances = np.linalg.norm(coords - selected_point, axis=1)
        selected_indices = np.argsort(distances)[:num_neighbors]
        selected_points = coords[selected_indices]
        selected_features = features[selected_indices]
        selected_responses = response[selected_indices]

        delaunay = scipy.spatial.Delaunay(selected_points)
        distance_map = {i: 0 for i in range(len(selected_points))}
        queue = deque([0])
        visited = set(queue)

        while queue:
            current = queue.popleft()
            current_distance = distance_map[current]
            for neighbor in np.unique(delaunay.simplices[np.any(delaunay.simplices == current, axis=1)]):
                if neighbor not in visited:
                    distance_map[neighbor] = current_distance + 1
                    queue.append(neighbor)
                    visited.add(neighbor)

        epsilon = 0.1
        gamma = 2
        dmax = max(distance_map.values()) if distance_map else 1
        weight_map = {i: 1 - (distance_map[i] / (dmax + epsilon)) ** gamma for i in distance_map}

        weights_ = np.array([weight_map[i] for i in range(len(selected_points))])

        X = selected_features
        y = selected_responses

        W = np.diag(weights_)

        if np.linalg.det(X.T @ W @ X) < 1e-10:
            beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        else:
            beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        y_est = features[idx] @ beta

        y_true.append(response[idx])
        y_pred.append(y_est)
        betas.append(beta)

    if best_betas is None:
      best_betas = np.array(betas)

    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, len(y_true), features.shape[1])
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    aic = compute_aic(np.array(y_true), np.array(y_pred), features.shape[1])
    #print(best_betas.shape)


    for metric, value in zip(['R2', 'RMSE', 'AIC'], [r2, rmse, aic]):
        if (metric == 'R2' and value > best_models[metric]['value']) or \
           (metric in ['RMSE', 'AIC'] and value < best_models[metric]['value']):
            best_models[metric] = {'value': value, 'params': (num_neighbors, r2, adj_r2, rmse, mape, aic), 'predictions': np.array(y_pred)}
            best_betas = np.array(betas)

print("\n\n ADT-GWR Results:")
for metric, result in best_models.items():
    print(f'Best model based on {metric}:')
    print(f'Num Neighbors: {result["params"][0]}, R²: {result["params"][1]:.4f}, Adjusted R²: {result["params"][2]:.4f}, '
          f'RMSE: {result["params"][3]:.4f}, MAPE: {result["params"][4]:.4f}, AIC: {result["params"][5]:.4f}')

# Keeping the best model of the ADT-GWR for calculating the residuals
num_neighbors = best_models['R2']['params'][0]
y_pred_delaunay = best_models['R2']['predictions']

# Maintaining the Coeeficient of the Best ADT-GWR model
best_betas_df = pd.DataFrame(best_betas, columns=feature_names)

# Adding the Lat and Long Columns to the result of coefficients and saving the results
best_betas_df.insert(0, 'longitude', data.iloc[:, 0].values)
best_betas_df.insert(1, 'latitude', data.iloc[:, 1].values)
best_betas_df.to_csv('/content/drive/MyDrive/GWR/Depression_Delaunay_GWR_Betas.csv', index=False)
print("-----------------------------------------")



# Implementing the convensional GWR with variable searching bandwidth
print('Running the convensional GWR ...')
bw = Sel_BW(coords, response.reshape(-1, 1), features, kernel='bisquare', fixed=False, spherical=False).search()
gwr_model = GWR(coords, response.reshape(-1, 1), features, bw)
gwr_results = gwr_model.fit()

gwr_feature_names = ['Intercept'] + list(feature_names) 
gwr_betas_df = pd.DataFrame(gwr_results.params, columns=gwr_feature_names)

# Adding the Lat and Long Columns to the result of coefficients and saving the results
gwr_betas_df.insert(0, 'longitude', data.iloc[:, 0].values)
gwr_betas_df.insert(1, 'latitude', data.iloc[:, 1].values)
gwr_betas_df.to_csv('/content/drive/MyDrive/GWR/Depression_Standard_GWR_Betas.csv', index=False)



y_pred_gwr = gwr_model.predict(coords, features).predictions.flatten()
r2_gwr = r2_score(response, y_pred_gwr)
adj_r2_gwr = adjusted_r2(r2_gwr, len(response), features.shape[1])
rmse_gwr = np.sqrt(mean_squared_error(response, y_pred_gwr))
mape_gwr = mean_absolute_percentage_error(response, y_pred_gwr)
aic_gwr = compute_aic(response, y_pred_gwr, features.shape[1])

print(f'Standard GWR Results:')
print(f'R²: {r2_gwr:.4f}, Adjusted R²: {adj_r2_gwr:.4f}, RMSE: {rmse_gwr:.4f}, MAPE: {mape_gwr:.4f}, AIC: {aic_gwr:.4f}')
print("-----------------------------------------")

# OLS Regression 
ols_model = LinearRegression()
ols_model.fit(features, response)
y_pred_ols = ols_model.predict(features)

r2_ols = r2_score(response, y_pred_ols)
adj_r2_ols = adjusted_r2(r2_ols, len(response), features.shape[1])
rmse_ols = np.sqrt(mean_squared_error(response, y_pred_ols))
mape_ols = mean_absolute_percentage_error(response, y_pred_ols)
aic_ols = compute_aic(response, y_pred_ols, features.shape[1])

print(f'OLS Regression Results:')
print(f'R²: {r2_ols:.4f}, Adjusted R²: {adj_r2_ols:.4f}, RMSE: {rmse_ols:.4f}, MAPE: {mape_ols:.4f}, AIC: {aic_ols:.4f}')

print("-----------------------------------------")
print("All models were implemented sucsessfully !")

#Calculating Residuals
residuals_delaunay = response - y_pred_delaunay
residuals_gwr = response - y_pred_gwr
residuals_ols = response - y_pred_ols



# Box plots od Residuals
plt.figure(figsize=(8, 6))
plt.boxplot([residuals_delaunay, residuals_gwr, residuals_ols], labels=['ADT-GWR', 'GWR', 'OLS'])
plt.ylabel('Residuals')
plt.title('Depression Dataset')
plt.grid()
plt.show()
