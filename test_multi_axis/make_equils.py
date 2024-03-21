import os, sys

from typing import Dict, List
import pickle
import random
import numpy as np
import pandas as pd

from flagships.gs_solver.run_from_yaml_axis_values_and_coil_configs import run_from_yaml_axis_values_and_coil_configs
from flagships.post_processing import GenerateColumns

if __name__ == '__main__':
    ta_configs_file = 'scaled_table_d_table_axis_configs_nevins_y.csv'
    ta_configs = pd.read_csv(ta_configs_file)
    ta_configs = ta_configs.round(4)
    
    max_changes = {'Coil_A': 60, 'Coil_C': 100, 'Coil_D': 100, 'PFC_1': 40, 'PFC_2': 15}

    yaml_file = 'pi3b_asbuilt_D_C353_D-313_P154_042_2023-10-12.yaml'
    yaml_dict = GenerateColumns.read_table_def_yaml(yaml_file)

    base_coil_currents = yaml_dict['FEMM_currents']
    del base_coil_currents['Inner Coil']
    del base_coil_currents['Nose Coil']
    del base_coil_currents['PlasmaCurrent']
    del base_coil_currents['PFC_3']

    table_axis_config_list = ta_configs.to_dict('records')

    coil_configs = []
    coil_configs.append(base_coil_currents)

    all_changes_coil_config = base_coil_currents.copy()
    for coil_to_change in max_changes:
        coil_config = base_coil_currents.copy()
        coil_config[coil_to_change] += max_changes[coil_to_change]
        coil_configs.append(coil_config)

        all_changes_coil_config[coil_to_change] += max_changes[coil_to_change]
    coil_configs.append(all_changes_coil_config)

    num_table_axis_configs = len(table_axis_config_list)
    num_coil_configs_per_table_axis_config = len(coil_configs)

    table_axis_configs_to_make = []
    for table_axis_config in table_axis_config_list:
        table_axis_configs_to_make += [table_axis_config]*num_coil_configs_per_table_axis_config 
    coil_configs_to_make = coil_configs*num_table_axis_configs

    pickle.dump({'TA':table_axis_configs_to_make, 'Coils':coil_configs_to_make}, open('multi_axis_cef_test_configs.pickle', 'wb'))

    dc_file = '/home/brendan.posehn@gf.local/dev/gf/flagships/ext_psi_files/pi3/pi3b_as_built_2022-09-16_G486_18425-18433.FEM'
    output_files = run_from_yaml_axis_values_and_coil_configs(yaml_file, table_axis_configs_to_make, coil_configs_to_make,
                                                              'make_equils', force_recalc=False, dc_file=dc_file)
    
    pickle.dump(output_files, open('table_d_multi_axis_test_filenames.pickle', 'wb'))