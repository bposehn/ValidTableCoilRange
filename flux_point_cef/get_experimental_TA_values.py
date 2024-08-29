import random
import os
import pickle

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
import bson
import gzip

def get_tables_used_in_recent_shots(shot_max: int, manifest_bson_loc):
    table_uses = {}
    shot = shot_max + 1
    while(shot > 18000):
        if shot %100 == 0:
            print(f'{shot=}')
        shot -= 1

        slice_report_fname = os.path.join(manifest_bson_loc, str(int(shot/1000)*1000), str(shot), 'reconstruction',
                                        'reconstruction', f"{shot}_slice_report_001000us")
        if not os.path.exists(slice_report_fname):
            continue

        with gzip.open(slice_report_fname, 'rb') as f:
            slice_report_data = bson.decode(f.read())
        
        if slice_report_data['physics_model_name'] in table_uses.keys():
            table_uses[slice_report_data['physics_model_name']] += 1
        else:
            table_uses[slice_report_data['physics_model_name']] = 1

    print(table_uses)

def get_experimental_table_axis_values(shot_max: int, num_configs_wanted, manifest_bson_loc, table_name, num_per_shot):
    table_axis_names = ['NevinsA', 'NevinsC', 'NevinsN', 'NevinsY', 'beta_pol1', 'psieq_dc', 'psieq_soak', 'CurrentRatio']
    query_layers = ['reconstruction/' + table_axis_name + '_mean' for table_axis_name in table_axis_names]
    query_layers += ['reconstruction/fit_quality', 'reconstruction/Ishaft_mean']

    num_no_recon_data_shots = 0
    num_bad_quality_shots = 0
    num_wrong_table_shots = 0
    configs = []
    shots_used = []
    shot = shot_max + 1
    while(len(configs) < num_configs_wanted and shot > 20000):
        shot -= 1

        if shot%50 == 0:
            print(f'Shot number {shot}')

        fit_quality_options = {'experiment': 'pi3b',
                    'manifest': 'default', 'shot': shot,
                    'layers': query_layers,
                    'nodisplay': True}
        try:
            data = fetch_data.run_query(fit_quality_options)
        except:
            # print(f'No data for shot {shot}')
            num_no_recon_data_shots += 1
            continue

        layer_order = [layer[-1] for layer in data['layers']]
        fit_quality_index = layer_order.index('fit_quality')
        Ishaft_index = layer_order.index('Ishaft_mean')

        fit_qualities = list(data['waves'][fit_quality_index].data)
        if np.mean(fit_qualities) > 1.1:
            # print(f"Mean fit quality: {np.mean(fit_qualities)}")
            num_bad_quality_shots += 1
            continue

        num_time_steps_per_shot = num_per_shot
        for i in range(num_time_steps_per_shot):
            time_step_index = random.choice(range(len(fit_qualities)))

            if table_name is not None:
                time_step_s = np.round(data['waves'][0].x_axis()[time_step_index], 3)
                slice_report_fname = os.path.join(manifest_bson_loc, str(int(shot/1000)*1000), str(shot), 'reconstruction',
                                                'reconstruction', f"{shot}_slice_report_{str(int(time_step_s*1e6)).zfill(6)}us")
                
                with gzip.open(slice_report_fname, 'rb') as f:
                    slice_report_data = bson.decode(f.read())
                # print(f'Table name: {slice_report_data["physics_model_name"]}')
                if slice_report_data['physics_model_name'] != table_name:
                    num_wrong_table_shots += 1
                    continue

            exp_Ishaft = data['waves'][layer_order.index('Ishaft_mean')][time_step_index]
            table_Ishaft = 1e6
            k_scaling = exp_Ishaft / table_Ishaft

            config = {}
            for table_axis_name in table_axis_names:
                config[table_axis_name] = data['waves'][layer_order.index(table_axis_name + '_mean')][time_step_index]
                if 'psieq' in table_axis_name:
                    config[table_axis_name] /= k_scaling
            shots_used.append(shot)

            if config['psieq_dc'] > 0:
                print(config)

            configs.append(config)
            if len(configs)%50 == 0:
                print(f'{len(configs)} configs found')

    print(f'{num_no_recon_data_shots} shots had no recon data\n{num_bad_quality_shots} had unacceptable fit quality\n{num_wrong_table_shots} used wrong table')
    return configs, shots_used

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
    num_wanted = 5000
    manifest_bson_loc = '/mnt/aurora/bson/pi3b/default'
    newest_shot = 23000
    table_name = None

    configs, shots_used = get_experimental_table_axis_values(newest_shot, num_wanted, manifest_bson_loc, table_name, num_per_shot = 1)
    TA_config_df = pd.DataFrame(configs)
    TA_config_df['Shot'] = shots_used

    