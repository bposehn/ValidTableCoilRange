from typing import List

import pandas as pd
import numpy as np

from GF_data_tools.fetch_data import get_shot_scalars

COIL_NAMES = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']
QUERY_COIL_NAMES = ['A', 'C', 'D', 'E', 'F']

from flux_point_calculator import FluxPointCalculator

def get_shot_coil_configs(shots: List[int]):
    coil_configs = []
    for shot_number in shots:
        shot_scalars = get_shot_scalars('pi3b', 'default', shot_number)

        config = {}
        for i_coil, query_coil_name in enumerate(QUERY_COIL_NAMES):        
            config[COIL_NAMES[i_coil]] = shot_scalars[query_coil_name]['coil_current_at_form']['value']

        coil_configs.append(config)

    return coil_configs

if __name__ == '__main__':
    shots = np.arange(20000, 23000)
    coil_configs = get_shot_coil_configs(shots)
    coil_config_df = pd.DataFrame(coil_configs)
    coil_config_df['Shot'] = shots
    
    fpc = FluxPointCalculator('data/flux_per_amp_values.csv')
    flux_configs = []
    for config in coil_configs:
        flux_configs.append(fpc.get_flux_at_points(config))

    df = pd.DataFrame(flux_configs)
    breakpoint()