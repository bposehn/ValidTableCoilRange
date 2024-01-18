from typing import Dict, List
import pickle

import numpy as np
import pandas as pd

from flagships.gs_solver.run_from_yaml_axis_values_and_coil_configs import run_from_yaml_axis_values_and_coil_configs
from flagships.post_processing import GenerateColumns

def get_coil_configs_at_corners_and_on_axes(base_coil_config: Dict[str, float], num_corners, max_fractional_change, max_absolute_change = None, coils_to_ignore: List[str]= None):
    coils_of_interest = [coil_name for coil_name in base_coil_config if coil_name not in coils_to_ignore]
    zero_coils = set([coil_name for coil_name in coils_of_interest if base_coil_config[coil_name] == 0])

    fractional_changes = np.linspace(1, 1+max_fractional_change, 1+num_corners).tolist()[1:]
    absolute_changes = np.linspace(0, max_absolute_change, 1+num_corners).tolist()[1:]
    
    coil_configs = []
    coil_configs.append(base_coil_config)
    for i_corner in range(num_corners):
        corner_config = base_coil_config.copy()
        for coil_name in coils_of_interest:
            if coil_name in zero_coils:
                corner_config[coil_name] = np.round(absolute_changes[i_corner], 2)
            else:
                corner_config[coil_name] = np.round(base_coil_config[coil_name]*fractional_changes[i_corner], 2)

        coil_configs.append(corner_config)
        for coil_name in coils_of_interest:
            single_axis_varied_coil_config = base_coil_config.copy()
            single_axis_varied_coil_config[coil_name] = corner_config[coil_name]
            coil_configs.append(single_axis_varied_coil_config)

    return coil_configs
    
if __name__ == '__main__':
    ta_configs = []
    yaml_file = '/home/brendan.posehn@gf.local/tables/pi3b_asbuilt_A_C362_D-343_2023-10-12.yaml'
    yaml_dict = GenerateColumns.read_table_def_yaml(yaml_file)

    coil_currents = yaml_dict['FEMM_currents']
    del coil_currents['Inner Coil']
    del coil_currents['Nose Coil']
    del coil_currents['PlasmaCurrent']
    del coil_currents['PFC_3']

    coil_configs = get_coil_configs_at_corners_and_on_axes(coil_currents, 10, .3, 120, ['Coil_B', 'PFC_3'])

    table_axis_configs = pd.read_csv('data/test_table_axis_configs.csv')
    table_axis_config_list = []
    for i_row, row in table_axis_configs.iterrows():
        table_axis_config = {}
        for table_axis_name in table_axis_configs.columns:
            table_axis_config[table_axis_name] = row[table_axis_name]
        table_axis_config_list.append(table_axis_config)

    num_table_axis_configs = len(table_axis_config_list)
    num_coil_configs_per_table_axis_config = len(coil_configs)

    table_axis_configs_to_make = []
    for table_axis_config in table_axis_config_list:
        table_axis_configs_to_make += [table_axis_config]*num_coil_configs_per_table_axis_config 
    coil_configs_to_make = coil_configs*num_table_axis_configs

    dc_file = '/home/brendan.posehn@gf.local/dev/gf/flagships/ext_psi_files/pi3/pi3b_as_built_2022-09-16_G486_18425-18433.FEM'
    output_files = run_from_yaml_axis_values_and_coil_configs(yaml_file, table_axis_configs_to_make, coil_configs_to_make,
                                                              'make_equils', force_recalc=False, dc_file=dc_file)


    pickle.dump(output_files, open('data/output_filenames.pickle', 'wb'))