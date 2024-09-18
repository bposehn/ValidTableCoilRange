import sys, os
import time

import numpy as np
import pandas as pd

from flux_point_calculator import FluxPointCalculator
from GF_data_tools.fetch_data import get_shot_scalars

COIL_NAMES = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']
QUERY_COIL_NAMES = ['A', 'C', 'D', 'E', 'F']

def get_shot_coil_configs(min_shot: int, max_shot: int, num_shots: int):

    shot_numbers = np.arange(min_shot, max_shot)
    num_shots = len(shot_numbers)
    
    coil_configs = np.empty((num_shots, len(COIL_NAMES)))
    num_in = 0

    used_shot_numbers = []
    for i_shot, shot_number in enumerate(shot_numbers):
        if i_shot%100 == 0:
            print(i_shot)

        try:
            shot_scalars = get_shot_scalars('pi3b', 'default', shot_number)
        except:
            continue

        if 'average_z' not in shot_scalars or 'total_lifetime' not in shot_scalars['average_z']:
            continue

        has_all_query_levels = True
        for query_coil_name in QUERY_COIL_NAMES:
            if query_coil_name not in shot_scalars:
                has_all_query_levels = False
                break

        if not has_all_query_levels:
            continue

        for i_coil, query_coil_name in enumerate(QUERY_COIL_NAMES):        
            coil_configs[num_in, i_coil] = shot_scalars[query_coil_name]['coil_current_at_form']['value']

        used_shot_numbers.append(shot_number)
        num_in += 1

    coil_configs = coil_configs[:num_in]
    return coil_configs, used_shot_numbers

if __name__ == '__main__':
    # arr, shot_numbers = get_shot_coil_configs(19000, 22550, 3000)
    # df = pd.DataFrame(arr, columns=COIL_NAMES)
    # df['Shot'] = shot_numbers
    # df.to_csv('coil_configs.csv', index=False)

    coil_configs = pd.read_csv('data/shot_23000_to_20000_coil_configs.csv')
    fpc = FluxPointCalculator('data/new_points_flux_per_amp.csv')

    flux_values = coil_configs[COIL_NAMES] @ fpc.flux_per_amp_df[COIL_NAMES].T
    flux_values = flux_values.divide(flux_values['Psi,Equator'], axis=0)
    flux_values['Shot'] = coil_configs['Shot']

    flux_values.to_csv('flux_configs.csv', index=False)