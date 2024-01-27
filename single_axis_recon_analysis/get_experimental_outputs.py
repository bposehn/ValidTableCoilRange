import random

# from flagships.gs_solver import run_from_yaml_axis_values_and_coil_configs
from GF_data_tools import fetch_data
from flagships.Csharp.csharp_utils import asNetArray, append_gf_recon_dll_path, set_debug_mode

import clr
set_debug_mode(True)
append_gf_recon_dll_path()
clr.AddReference("GFRecon")
from Reconstruction import SharedEquilibriaLUT, TableLocationSteps, SharedViewIndex

import pandas as pd
import numpy as np

def get_experimental_table_axis_values(shot_min: int, shot_max: int):
    table_axis_names = ['NevinsA', 'NevinsC', 'NevinsN', 'beta_pol1', 'psieq_dc', 'psieq_soak', 'CurrentRatio']
    query_layers = ['reconstruction/' + table_axis_name + '_mean' for table_axis_name in table_axis_names]
    query_layers += ['reconstruction/fit_quality']

    num_no_recon_data_shots = 0
    num_bad_quality_shots = 0
    configs = []
    for shot in range(shot_min, shot_max):
        fit_quality_options = {'experiment': 'pi3b',
                    'manifest': 'default', 'shot': shot,
                    'layers': query_layers,
                    'nodisplay': True}
        try:
            data = fetch_data.run_query(fit_quality_options)
        except:
            num_no_recon_data_shots += 1
            continue

        layer_order = [layer[-1] for layer in data['layers']]
        fit_quality_index = layer_order.index('fit_quality')

        fit_qualities = list(data['waves'][fit_quality_index].data)
        # if np.mean(fit_qualities) > 1:
        #     num_bad_quality_shots += 1
        #     continue

        time_step_index = random.choice(range(len(fit_qualities)))
        config = {}
        for table_axis_name in table_axis_names:
            config[table_axis_name] = data['waves'][layer_order.index(table_axis_name + '_mean')][time_step_index]

        configs.append(config)

    print(f'{num_no_recon_data_shots} shots had no recon data\n{num_bad_quality_shots} had unacceptable fit quality')
    return configs

def get_random_table_axis_configs(num_configs, table_dat_path):

    table = SharedEquilibriaLUT(table_dat_path)
    table_row_indices = random.sample(range(table.NumRows), num_configs)

    table_axis_configs = np.empty((num_configs, table.Axes.Count))
    for i_table_row_index, table_row_index in enumerate(table_row_indices):
        for i_axis in range(table.Axes.Count):
            table_axis_configs[i_table_row_index, i_axis] = table.GetValue(table_row_index, i_axis)

    table_axis_names = [table.ColumnNames[i_axis] for i_axis in range(table.Axes.Count)]
    df = pd.DataFrame(columns=table_axis_names, data = table_axis_configs)

    return df

if __name__ == '__main__':
    configs = get_experimental_table_axis_values(18711, 18829)
    breakpoint()