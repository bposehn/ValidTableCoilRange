from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_min_dists_to_shot_configs(shot_normalized_flux_values: pd.DataFrame, proposed_normalized_flux_configs: pd.DataFrame):
    # For each shot, find the best matching proposed flux config and store it
    closest_match_dists = []
    for i_shot, shot_row in shot_normalized_flux_values.iterrows():
        closest_match_dists.append(np.linalg.norm(proposed_normalized_flux_configs - shot_row[proposed_normalized_flux_configs.columns], axis=1).min())

    return closest_match_dists

def get_clustered_dists_to_shot_configs(n_clusters: int, point_flux_values: pd.DataFrame, point_flux_names: List[str]):
    kmeans = KMeans(n_clusters).fit(shot_flux_configs[point_flux_names])
    clustering_flux_values_df = pd.DataFrame(kmeans.cluster_centers_, columns=point_flux_names)

    return clustering_flux_values_df

if __name__ == '__main__':
    shot_flux_configs = pd.read_csv('data/shot_23000_to_20000_point_flux_configs.csv')
        
    point_flux_nams = ['Psi,Equator', 'Psi,Out.Throat', 'Psi,In.Throat', 'Psi,Upper']

    clustered_flux_df = get_clustered_dists_to_shot_configs(200, shot_flux_configs, point_flux_nams)

    breakpoint()
