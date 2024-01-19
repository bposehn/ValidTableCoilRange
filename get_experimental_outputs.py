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
    for shot in range(shot_min, shot_max):
        q_options = {'experiment': 'pi3b',
                    'manifest': 'default',
                    'shot': shot,
                    'layers': 'reconstruction/*/*',
                    'nodisplay': True}
        data = fetch_data.run_query(q_options)
    # TODO test this 

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
    # shot_min = 18000
    # shot_max = 
    # shot_range = 
    shared_file = '/home/brendan.posehn@gf.local/tables/pi3b_asbuilt_g486a_emsoak1_2023-05-19_shared'
    df = get_random_table_axis_configs(10, shared_file)
    df.to_csv('data/test_table_axis_configs.csv', index=False)