import os, sys
import string
import random
import time
import pickle
import re
import itertools
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

COILS_OF_INTEREST = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1','PFC_2']
QUERY_COLUMN_NAMES = ['WBpolFCnoDC', 'q050', 'NevinsC', 'beta_pol1']
TABLE_D_CONFIG = {  'Coil_A': 0,
                    'Coil_B': 0,
                    'Coil_C': 352.6,
                    'Coil_D': -312.8,
                    'PFC_1': -154.3,
                    'PFC_2': -42.5}

def get_coils_from_filename(filename, coil_names):
    coils = {}
    for coil_name in coil_names:
        start = filename.find(coil_name)
        end_coil_name = start + len(coil_name) + 1
        end_number = filename.find('A', end_coil_name)
        coils[coil_name] = float(filename[end_coil_name:end_number])
    
    return coils

def get_df_from_recons(recons_dir, num_expected):   
    suffixes = ['_mean', '_truth', '_sigma']
    df_column_names = []
    for query_column_name in QUERY_COLUMN_NAMES:
        df_column_names += [query_column_name + suffix for suffix in suffixes]

    other_recon_output_names = ['fit_quality']
    data = np.empty((num_expected, len(df_column_names)+len(other_recon_output_names)))

    pattern = re.compile("^testcase_[0-9]+_[1-5].csv")
    i_row = -1
    filenames = []
    for filepath in os.listdir(recons_dir):
        if len(filenames)%100 == 0:
            print(f'{100*len(filenames)/1000}% processed')
        if pattern.match(filepath):
            i_row += 1
            recon_df = pd.read_csv(os.path.join(recons_dir, filepath))
            for i_column, query_column_name in enumerate(df_column_names + other_recon_output_names):
                data[i_row, i_column] = recon_df[query_column_name].item()
            filenames.append(recon_df['FileName'].item())

    data = data[:len(filenames)]

    interim_df = pd.DataFrame(data = data, columns = df_column_names + other_recon_output_names)
    interim_df['FileName'] = filenames 
    interim_df.set_index('FileName', inplace=True)

    return interim_df

def organize_df(recon_df: pd.DataFrame, out_filenames_pickle_path, num_expected):
    out_filenames = pickle.load(open(out_filenames_pickle_path, 'rb'))

    num_coils = len(COILS_OF_INTEREST)
    num_configs_per_coil = 10 # plus the base!
    num_configs_per_table_axis_config = 1 + (num_coils * num_configs_per_coil)
    data = np.empty((num_expected, len(recon_df.columns) + num_coils + 2)) # 2 for Coil Config Index, TA Config Index
    for i_filename, filename in enumerate(out_filenames):
        filename = os.path.basename(filename)
        i_ta_config = int(i_filename / (num_configs_per_table_axis_config)) 
        i_coil_config = i_filename % (num_configs_per_table_axis_config) #in [0, 50], 0 is base 
        data[i_filename, 0] = i_coil_config
        data[i_filename, 1] = i_ta_config
        coil_values = get_coils_from_filename(filename, COILS_OF_INTEREST)
        for i_coil, coil_name in enumerate(coil_values):
            data[i_filename, 2 + i_coil] = coil_values[coil_name]

        if filename in recon_df.index:
            row = recon_df.loc[filename]
            column_values_for_filename = row.values
        else:
            # equil failed 
            column_values_for_filename = [np.nan]*len(recon_df.iloc[0].values)
        data[i_filename, 2+len(coil_values):] = column_values_for_filename

    column_names = ['Coil Config Index', 'TA Config Index'] + COILS_OF_INTEREST + recon_df.columns.values.tolist()
    organized_df = pd.DataFrame(data = data, columns = column_names)
    organized_df['FileName'] = [os.path.basename(fname) for fname in out_filenames]

    organized_df = organized_df.astype({'Coil Config Index': 'int32', 'TA Config Index': 'int32'})

    return organized_df

def get_truth_outside_sigma_bound(organized_df: pd.DataFrame, sigma_range: float):

    col_type = 'TOSB'
    col_descriptors = [prod[0] + '_' + prod[1] for prod in itertools.product(COILS_OF_INTEREST, QUERY_COLUMN_NAMES)]
    col_names = [col_type + '_' + descriptor for descriptor in col_descriptors]

    ta_config_index_colname = 'TA Config Index'

    num_ta_configs = len(organized_df[ta_config_index_colname].unique())

    data = np.empty((num_ta_configs, len(col_names)))
    
    # Entry for each table axis config, for each coil, for each col 
    filenames = []
    for i_ta_config in organized_df[ta_config_index_colname].unique():
        first_i_ta_config_index = organized_df.where(organized_df[ta_config_index_colname] == i_ta_config).first_valid_index()
        filenames.append(organized_df.iloc[first_i_ta_config_index]['FileName'])
        base_coil_row = pd.DataFrame(data = [organized_df.iloc[first_i_ta_config_index]], columns = organized_df.columns)
        for i_coil_name, coil_name in enumerate(COILS_OF_INTEREST):
            varying_coils_df = organized_df.loc[(organized_df[coil_name] != organized_df.iloc[0][coil_name]) & (organized_df[ta_config_index_colname] == i_ta_config)]
            varying_coils_df = pd.concat([base_coil_row, varying_coils_df]) # this is for this ta_config_idnex
            for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
                truth_values = varying_coils_df[col_name + '_truth']
                recond_values = varying_coils_df[col_name + '_mean']
                recond_sigmas = varying_coils_df[col_name + '_sigma']

                # z_scores = abs((truth_values - recond_values) / recond_sigmas).to_numpy()
                # z_scores = z_scores[~np.isnan(z_scores)]
                # b, m = np.polyfit(range(len(z_scores)), z_scores, 1)
                # slopes.append(m)

                # should also add checking that each one after does (i.e not just a middle coil point, that is concerning)

                # COULD INTERPOLATE THIS ? 
                osb_rows = varying_coils_df.loc[(truth_values < recond_values - sigma_range * recond_sigmas)
                                                      | (truth_values > recond_values + sigma_range * recond_sigmas)]
                
                i_data_column = i_col_name + i_coil_name*len(QUERY_COLUMN_NAMES)

                if len(osb_rows) == 0:
                    data[i_ta_config, i_data_column] = np.nan # Represents coil range was not large enough to get to point when is oor 
                    continue

                lowest_failed_coil_index = min(osb_rows['Coil Config Index'])
                lowest_failed_coil_deviance = varying_coils_df.loc[varying_coils_df['Coil Config Index'] == lowest_failed_coil_index][coil_name] \
                                                - organized_df.iloc[0][coil_name]

                data[i_ta_config, i_data_column] = lowest_failed_coil_deviance
    
    df = pd.DataFrame(data, columns=col_names)

    return df

def plot_truth_outside_sigma_bound(TOSB_df):

    num_columns = 3
    fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        medians = []
        ax = axs[int(i_coil/num_columns), i_coil%num_columns]
        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            vals = TOSB_df['TOSB_' + coil_name + '_' + col_name]
            medians.append(vals.median())
            # ax.hist(vals, label=col_name, histtype='step', color=query_column_colors[i_col_name])
            ax.hist(vals, label=col_name, alpha = .5, color=query_column_colors[i_col_name])
        # ax.axvline(x=min(medians), c=query_column_colors[np.argmin(medians)])
        ax.set_xlabel('Coil Deviation (A)')
        ax.legend()
        ax.set_title(coil_name)
    plt.suptitle(f'Truth Outside of Coil Varied Reconstructed $\mu \pm 1\sigma$\nTable D: {TABLE_D_CONFIG}')
    plt.show()

def plot_test_set_size_analysis(organized_df: pd.DataFrame):
    #for each coil, for each col, show that avg error reaches asymptote

    num_testcases = 195
    num_testcases_per_iteration = np.arange(40, num_testcases+1, 20, dtype=int) # can be used as i_ta_config idxs
    i_ta_configs = np.arange(0, num_testcases, 1, dtype=int)
    random.shuffle(i_ta_configs)
    i_ta_configs_per_iteration = [i_ta_configs[:num_testcases_in_iteration] for num_testcases_in_iteration in num_testcases_per_iteration]
    
    i_coil_configs_to_test = [0, 10, 20, 30, 40, 50] # Base (0) and all extrema in order
    coil_config_names = ['Base'] + [coil + ' Max Change' for coil in COILS_OF_INTEREST]
    
    num_columns = 3
    fig, axs = plt.subplots(int(len(i_coil_configs_to_test)/num_columns), num_columns)
    for i_i_coil_config, i_coil_config_to_test in enumerate(i_coil_configs_to_test):
        coil_config_df = organized_df.loc[organized_df['Coil Config Index'] == i_coil_config_to_test]
        ax = axs[int(i_i_coil_config/num_columns), i_i_coil_config%num_columns]
        for col_name in QUERY_COLUMN_NAMES:
            # if col_name == 'NevinsC':
            #     continue
            vals = []
            for i_ta_configs_in_iteration in i_ta_configs_per_iteration:
                df = coil_config_df.loc[coil_config_df['TA Config Index'].isin(i_ta_configs_in_iteration)]
                df = df.loc[df[col_name + '_sigma'] > 1e-6] # confirmed this is only for high pfc changes

                z_scores = (df[col_name + '_truth'] - df[col_name + '_mean']) / df[col_name + '_sigma'] # if use this, some sigmas are 1e-2, some are 1e-15? 
                # z_scores = (df[col_name + '_truth'] - df[col_name + '_mean']) / df[col_name + '_truth'] # TODO not z scores
                vals.append(np.mean(z_scores[np.isfinite(z_scores)])) # Looks fine with medians but not means, large outliers exist
            ax.plot(num_testcases_per_iteration, vals, label = col_name)
        ax.set_xlabel('Number of testcases')
        ax.set_ylabel('Z Score')
        ax.legend()
        ax.set_title(coil_config_names[i_i_coil_config])

    plt.suptitle('Validate Test Set Size')
    # plt.savefig('plots/validate_test_set_size.png')

    plt.show()

if __name__ == '__main__':
    num_expected=9945

    plt.style.use('dark_background')

    # recons_df = get_df_from_recons('data/recon_results', num_expected)
    # recons_df.to_csv('data/recon_results.csv')

    recons_df = pd.read_csv('data/recon_results.csv')
    recons_df.set_index('FileName', inplace=True)
    organized_df = organize_df(recons_df, 'out_files.pickle', num_expected)
        
    # plot_test_set_size_analysis(organized_df)

    # tosb_df = get_truth_outside_sigma_bound(organized_df, sigma_range=1)
    # tosb_df.to_csv('data/tosb_results.csv')

    tosb_df = pd.read_csv('data/tosb_results.csv')

    plot_truth_outside_sigma_bound(tosb_df)
    breakpoint()