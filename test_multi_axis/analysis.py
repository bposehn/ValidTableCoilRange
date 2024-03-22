import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append('/nfs/home/brendan.posehn/ws/ValidTableCoilRange/single_axis_recon_analysis')

from process_reconstructions import get_all_columns_df_from_recons, organize_df, get_all_cols_sigma_error_data, COIL_CONFIG_INDEX_NAME, TA_CONFIG_INDEX_NAME, QUERY_COLUMN_NAMES

def analyze_superposition_sigma_errors(sigma_error_data, columns, organized_df):
    # sigma error data is entry for each table axis config, for each coil config, for each col sigmas off

    num_coil_multiples = 5
    num_configs_per_coil_multiple = 6 # doesn't include base
    num_non_base_configs_per_ta = num_coil_multiples * num_configs_per_coil_multiple
    
    max_coil_a = organized_df.iloc[-1]['Coil_A']

    for i_coil_multiple in range(num_coil_multiples):
        first_non_base_index = 1 + (i_coil_multiple*num_configs_per_coil_multiple)
        coil_config_indexes = list(range(first_non_base_index, first_non_base_index+num_configs_per_coil_multiple))

        # breakpoint()
        pctg_change = 100 * organized_df.loc[organized_df[COIL_CONFIG_INDEX_NAME] == first_non_base_index].iloc[0]['Coil_A'] / max_coil_a

        single_change_coil_config_indexes = coil_config_indexes[:-1]
        all_changed_coil_config_index = coil_config_indexes[-1]

        individual_column_change_errors = np.zeros_like(sigma_error_data[:, 0, :])
        for single_change_coil_config_index in single_change_coil_config_indexes:
            individual_column_change_errors += abs(sigma_error_data[:, single_change_coil_config_index, :])
        
        all_column_change_errors = abs(sigma_error_data[:, all_changed_coil_config_index, :])

        num_bad = (all_column_change_errors > individual_column_change_errors).sum()
        pctg_bad = num_bad / all_column_change_errors.size 

        print(f'{pctg_change}: {pctg_bad * 100}')

        # breakpoint()

# all recons are done in cef_multi_asxis_test

if __name__ == '__main__':
    num_expected=6050
    # num_non_base_coil_configs_per_coil = 6/5
    num_non_base_coil_configs_per_coil = 30/5

    plt.style.use('dark_background')

    base = 'data/gradual_change_table_d'
    recons_output_loc = os.path.join(base, 'recon_outputs')
    all_cols_df_loc = os.path.join(base, 'recon_results_all_cols.csv')
    organized_df_loc = os.path.join(base, 'organized.csv')
    out_filenames_loc = os.path.join(base, 'out_filenames.pickle')
    sigma_deviance_arr_loc = os.path.join(base, 'sigma_deviance_arr.npy')

    force_solve = False

    if force_solve or not os.path.exists(all_cols_df_loc):
        recons_df = get_all_columns_df_from_recons(recons_output_loc)
        recons_df.to_csv(all_cols_df_loc)
    else:
        recons_df = pd.read_csv(all_cols_df_loc)
        if 'Unnamed: 0' in recons_df.columns:
            recons_df.drop(columns = ['Unnamed: 0'], inplace=True)
        recons_df.set_index('FileName', inplace=True)

    if force_solve or not os.path.exists(organized_df_loc):
        organized_df = organize_df(recons_df, out_filenames_loc, num_expected, num_non_base_coil_configs_per_coil)
        organized_df.to_csv(organized_df_loc)
    else:
        organized_df = pd.read_csv(organized_df_loc)
    
    if force_solve or not os.path.exists(sigma_deviance_arr_loc):
        # Entry for each table axis config, for each coil, for each col sigmas off
        sigma_deviance_arr, columns = get_all_cols_sigma_error_data(organized_df)
        np.save(sigma_deviance_arr_loc, sigma_deviance_arr)
    else:
        sigma_deviance_arr = np.load(sigma_deviance_arr_loc)
        columns = []
        for col in organized_df.columns:
            if col.endswith('_truth'):
                columns.append(col[:-6])

    analyze_superposition_sigma_errors(sigma_deviance_arr, columns, organized_df)