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

COIL_CONFIG_INDEX_NAME = 'Coil Config Index'
TA_CONFIG_INDEX_NAME = 'TA Config Index'
NUM_CONFIGS_PER_COIL = 10

def get_coils_from_filename(filename, coil_names):
    coils = {}
    for coil_name in coil_names:
        start = filename.find(coil_name)
        end_coil_name = start + len(coil_name) + 1
        end_number = filename.find('A', end_coil_name)
        coils[coil_name] = float(filename[end_coil_name:end_number])
    
    return coils

def get_all_columns_df_from_recons(recons_dir):   

    for i_filepath, filepath in enumerate(os.listdir(recons_dir)):
        if i_filepath%100 == 0:
            print(f'{100*i_filepath/10000}% processed')

        recon_df = pd.read_csv(os.path.join(recons_dir, filepath))

        if i_filepath == 0:
            df = recon_df
        else:
            df = pd.concat((df, recon_df), ignore_index=True)

    return df

def get_df_from_recons(recons_dir, num_expected):   
    suffixes = ['_mean', '_truth', '_sigma']
    df_column_names = []
    for query_column_name in QUERY_COLUMN_NAMES:
        df_column_names += [query_column_name + suffix for suffix in suffixes]

    other_recon_output_names = ['fit_quality', 'rxpt_truth']
    data = np.empty((num_expected, len(df_column_names)+len(other_recon_output_names)))

    pattern = re.compile("^testcase_[0-9]+_[1-5].csv")
    i_row = -1
    filenames = []
    for filepath in os.listdir(recons_dir):
        if len(filenames)%100 == 0:
            print(f'{100*len(filenames)/10000}% processed')
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
        legned_loc = 'upper left' if i_i_coil_config != 4 else 'upper right'
        ax.legend(loc='upper_left')
        ax.set_title(coil_config_names[i_i_coil_config])

    plt.suptitle('Validate Test Set Size')
    # plt.savefig('plots/validate_test_set_size.png')

    plt.show()

def plot_sigma_deviance_test_set_size_analysis(data):
    #for each coil, for each col, show that avg error reaches asymptote

    # data Entry for each table axis config, for each coil, for each col sigmas off and sigmas

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
        ax = axs[int(i_i_coil_config/num_columns), i_i_coil_config%num_columns]
        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            vals = []
            for i_ta_configs_in_iteration in i_ta_configs_per_iteration:
                values_at_ta_configs = data[i_ta_configs_in_iteration, i_coil_config_to_test, i_col_name]
                vals.append(np.mean(values_at_ta_configs[np.isfinite(values_at_ta_configs)])) 
            ax.plot(num_testcases_per_iteration, vals, label = col_name)
        ax.set_xlabel('Number of testcases')
        ax.set_ylabel('Mean $\sigma$ Deviance')
        ax.set_title(coil_config_names[i_i_coil_config])

    plt.suptitle('Validate Test Set Size')

    plt.show()

def plot_fq_with_coil_config(organized_df):
    coil_config_indexes = organized_df['Coil Config Index'].unique()
    num_configs_per_coil = 10

    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = 1 + np.arange(i_coil*num_configs_per_coil, (i_coil+1)*num_configs_per_coil)
        print(coil_config_indexes)
        fqs = np.empty(len(coil_config_indexes))
        for i, coil_config_index in enumerate(coil_config_indexes):
            fqs[i] = organized_df.loc[organized_df['Coil Config Index'] == coil_config_index]['fit_quality'].mean()

        plt.plot(fqs, label=coil_name)

    plt.legend()
    plt.xlabel('Coil Config Index')
    plt.ylabel('Mean Recon Fit Quality')
    plt.show()

def get_sigma_error_data(organized_df):
    col_names = QUERY_COLUMN_NAMES

    ta_config_index_colname = 'TA Config Index'
    coil_config_index_colname = 'Coil Config Index'

    num_ta_configs = len(organized_df[ta_config_index_colname].unique())
    num_coil_configs = len(organized_df[coil_config_index_colname].unique())

    data = np.empty((num_ta_configs, num_coil_configs, 2*len(QUERY_COLUMN_NAMES)))
    
    # Entry for each table axis config, for each coil, for each col sigmas off and sigmas
    for i_ta_config in organized_df[ta_config_index_colname].unique():
        single_ta_df = organized_df.loc[organized_df[ta_config_index_colname] == i_ta_config]
        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            truth_values = single_ta_df[col_name + '_truth']
            recond_values = single_ta_df[col_name + '_mean']
            recond_sigmas = single_ta_df[col_name + '_sigma']
            sigmas_off = ((truth_values - recond_values) / recond_sigmas)
            sigmas_off.where(recond_sigmas > 1e-6, np.nan, inplace=True) # TODO is this valid? 
            data[i_ta_config, :, i_col_name] = sigmas_off.values
            data[i_ta_config, :, i_col_name + len(QUERY_COLUMN_NAMES)] = recond_sigmas / truth_values

    return data

def plot_sigma_error(data, coil_increments):

    # Data is for each table axis config, for each coil, for each col 

    num_configs_per_coil = 10
    num_ta_configs = data.shape[0]

    num_columns = 3
    fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
    fig.subplots_adjust(wspace=.4)

    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*num_configs_per_coil + 1, (i_coil+1)*num_configs_per_coil + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)
        ax = axs[int(i_coil/num_columns), i_coil%num_columns]
        ax2 = ax.twinx()
        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            # ax.plot(coil_increments, np.nanmean(data[:, coil_config_indexes, i_col_name], 0), label=col_name, 
            #         color=query_column_colors[i_col_name], marker='o')

            ax2.plot(coil_increments, np.nanmean(data[:, coil_config_indexes, i_col_name+len(QUERY_COLUMN_NAMES)], 0), 
                    color=query_column_colors[i_col_name], linestyle='dashed', label=col_name + ' Mean')

            ax.errorbar(coil_increments - i_col_name, np.nanmean(data[:, coil_config_indexes, i_col_name], 0),
                        yerr=np.nanstd(data[:, coil_config_indexes, i_col_name], 0), label=col_name + ' Mean', color=query_column_colors[i_col_name], marker='o')

        ax.set_xlabel('Coil Increment (A)')
        ax2.set_ylabel('$\sigma_{Recon}$ / Truth')
        ax.set_ylabel('$\sigma$ Deviance')
        ax.set_title(coil_name)
        # ax.set_ylim([0, 160])
    handles_ax1, labels_ax1 = ax.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    last_ax = axs[-1, -1]
    last_ax.legend(handles_ax1 + handles_ax2, labels_ax1 + labels_ax2)
    plt.suptitle('$\sigma$ Deviance = (Truth - $\mu_{Recon}$) / $\sigma_{Recon}$' + f'\nTable D: {TABLE_D_CONFIG}')
    plt.show()

def plot_normalized_limited_diverted_sigma_error(data, coil_increments, organized_df):

    # Data is for each table axis config, for each coil, for each col 

    num_configs_per_coil = 10
    num_ta_configs = data.shape[0]

    num_columns = 3
    fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
    fig.subplots_adjust(wspace=.4)

    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*num_configs_per_coil + 1, (i_coil+1)*num_configs_per_coil + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

        ax = axs[int(i_coil/num_columns), i_coil%num_columns]

        df_at_coils = organized_df.loc[organized_df['Coil Config Index'].isin(coil_config_indexes)]
        limited_ta_config_indexes = set(df_at_coils.loc[df_at_coils['rxpt_truth'] == 0]['TA Config Index'].unique())
        diverted_ta_config_indexes = set(np.arange(len(df_at_coils['TA Config Index'].unique()))) - limited_ta_config_indexes

        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):

            for indexes, state, marker in zip([list(limited_ta_config_indexes), list(diverted_ta_config_indexes)], ['Limited', 'Diverted'], ['o', 'x']):
                base_coil_config_mean_sigma_deviance = np.nanmean(data[indexes, 0, i_col_name])
                ys = np.empty()
                yerrs = []
                for index in indexes:
                    ys.append(data[index, coil_config_indexes, i_col_name])
                    yerrs.append(data[index, coil_config_indexes, i_col_name])

                ax.errorbar(coil_increments - i_col_name*.8, abs(base_coil_config_mean_sigma_deviance - np.nanmean(data[indexes, coil_config_indexes, i_col_name], 0)),
                            yerr=np.nanstd(data[indexes, coil_config_indexes, i_col_name], 0),
                            label=col_name + ' abs(Base Coils Mean - Mean) ' + state, color=query_column_colors[i_col_name], marker=marker)

        ax.set_xlabel('Coil Increment (A)')
        ax.set_ylabel('$\sigma$ Deviance')
        ax.set_title(coil_name)

        if coil_name == 'PFC_1':
            ax.set_ylim([0, 2.5])

        if coil_name == 'PFC_2':
            ax.set_ylim([0, 10])

    handles_ax1, labels_ax1 = ax.get_legend_handles_labels()

    last_ax = axs[-1, -1]
    last_ax.axis('off')
    last_ax.legend(handles_ax1, labels_ax1)
    plt.suptitle('$\sigma$ Deviance = (Truth - $\mu_{Recon}$) / $\sigma_{Recon}$' + f'\nTable D: {TABLE_D_CONFIG}')
    plt.show()

def plot_normalized_sigma_error_quantiles(data, coil_increments, organized_df, plot_fit_qualities=False):

    # Data is for each table axis config, for each coil, for each col 

    num_configs_per_coil = 10
    num_ta_configs = data.shape[0]

    num_columns = 3
    fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
    fig.subplots_adjust(wspace=.4)

    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*num_configs_per_coil + 1, (i_coil+1)*num_configs_per_coil + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

        ax = axs[int(i_coil/num_columns), i_coil%num_columns]
        if plot_fit_qualities:
            ax2 = ax.twinx()
            coil_config_df = organized_df.loc[organized_df['Coil Config Index'].isin(coil_config_indexes)]
            fqs = []
            for i_coil_config_index in coil_config_df['Coil Config Index'].unique():
                fqs.append(np.mean(coil_config_df.loc[coil_config_df['Coil Config Index'] == i_coil_config_index]['fit_quality']))

            ax2.plot(coil_increments, fqs, label = 'Mean Fit Quality', c='m')
            if max(fqs) > 2*fqs[0]:
                ax2.axhline(fqs[0]*2, label='2 * Base Coil Config Fit Quality', c='m', linestyle='dashed')

        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            sigma_deviance_changes = abs(data[:, coil_config_indexes, i_col_name] - data[:, 0, i_col_name][:, np.newaxis])
            quantiles = [0.1, 0.5, .9]
            for quantile, marker in zip(quantiles, ['s', 'o', '+']):
                ax.plot(coil_increments, np.nanquantile(sigma_deviance_changes, quantile, axis=0),
                        label=col_name + f' Quantile(abs(Incremented - Base), {quantile*100}%)', color=query_column_colors[i_col_name], marker=marker)

            # normd_base_coil_config_mean_sigma_deviance = np.nanmedian(data[:, 0, i_col_name])

            # quantiles = [0.1, 0.5, .9]
            # normd_sigma_deviances_at_coil_configs = abs(normd_base_coil_config_mean_sigma_deviance - normd_base_coil_config_mean_sigma_deviance - data[:, coil_config_indexes, i_col_name])
            # for quantile, marker in zip(quantiles, ['s', 'o', '+']):
            #     ax.plot(coil_increments, np.nanquantile(normd_sigma_deviances_at_coil_configs, quantile, axis=0),
            #             label=col_name + f' Quantile(abs(Base Coils Median - All Coils), {quantile*100}%)', color=query_column_colors[i_col_name], marker=marker)

        ax.set_xlabel('Coil Increment (A)')
        ax.set_ylabel('$\sigma$ Deviance')
        ax.set_title(coil_name)

        if plot_fit_qualities:
            ax2.set_ylabel('Fit Quality', c='m')

    handles_ax1, labels_ax1 = ax.get_legend_handles_labels()
    if plot_fit_qualities:
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    else:
        handles_ax2, labels_ax2 = [], []

    last_ax = axs[-1, -1]
    last_ax.axis('off')
    last_ax.legend(handles_ax1+handles_ax2, labels_ax1+labels_ax2)
    plt.suptitle('Testcase-Wise $\sigma$ Deviance = (Truth - $\mu_{Recon}$) / $\sigma_{Recon}$' + f'\nTable D: {TABLE_D_CONFIG}')
    plt.show()

def plot_normalized_sigma_error(data, coil_increments, organized_df, plot_fit_qualities=False):

    # Data is for each table axis config, for each coil, for each col 

    num_configs_per_coil = 10
    num_ta_configs = data.shape[0]

    num_columns = 3
    fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
    fig.subplots_adjust(wspace=.4)

    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*num_configs_per_coil + 1, (i_coil+1)*num_configs_per_coil + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

        ax = axs[int(i_coil/num_columns), i_coil%num_columns]
        if plot_fit_qualities:
            ax2 = ax.twinx()
            coil_config_df = organized_df.loc[organized_df['Coil Config Index'].isin(coil_config_indexes)]
            fqs = []
            for i_coil_config_index in coil_config_df['Coil Config Index'].unique():
                fqs.append(np.mean(coil_config_df.loc[coil_config_df['Coil Config Index'] == i_coil_config_index]['fit_quality']))

            ax2.plot(coil_increments, fqs, label = 'Mean Fit Quality', c='m')
            if max(fqs) > 2*fqs[0]:
                ax2.axhline(fqs[0]*2, label='2 * Base Coil Config Fit Quality', c='m', linestyle='dashed')

        for i_col_name, col_name in enumerate(QUERY_COLUMN_NAMES):
            base_coil_config_mean_sigma_deviance = np.nanmean(data[:, 0, i_col_name])

            # ax.plot(coil_increments, abs(base_coil_config_mean_sigma_deviance - np.nanmean(data[:, coil_config_indexes, i_col_name], 0)),
            #         label=col_name + ' abs(Base Coils Mean - Mean)', color=query_column_colors[i_col_name], marker='o')

            ax.errorbar(coil_increments - i_col_name*.8, abs(base_coil_config_mean_sigma_deviance - np.nanmean(data[:, coil_config_indexes, i_col_name], 0)),
                        yerr=np.nanstd(data[:, coil_config_indexes, i_col_name], 0), label=col_name + ' abs(Base Coils Mean - Mean)', color=query_column_colors[i_col_name], marker='o')

        ax.set_xlabel('Coil Increment (A)')
        ax.set_ylabel('$\sigma$ Deviance')
        ax.set_title(coil_name)

        if coil_name == 'PFC_1':
            ax.set_ylim([0, 2.5])

        if coil_name == 'PFC_2':
            ax.set_ylim([0, 10])

        ax2.set_ylabel('Fit Quality', c='m')

    handles_ax1, labels_ax1 = ax.get_legend_handles_labels()
    if plot_fit_qualities:
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    else:
        handles_ax2, labels_ax2 = [], []

    last_ax = axs[-1, -1]
    last_ax.axis('off')
    last_ax.legend(handles_ax1+handles_ax2, labels_ax1+labels_ax2)
    plt.suptitle('$\sigma$ Deviance = (Truth - $\mu_{Recon}$) / $\sigma_{Recon}$' + f'\nTable D: {TABLE_D_CONFIG}')
    plt.show()

def plot_fqs(organized_df, coil_increments):

    # Data is for each table axis config, for each coil, for each col 
    num_configs_per_coil = 10

    query_column_colors = list(mcolors.BASE_COLORS.keys())[:len(QUERY_COLUMN_NAMES)]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*num_configs_per_coil + 1, (i_coil+1)*num_configs_per_coil + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)
        coil_config_df = organized_df.loc[organized_df['Coil Config Index'].isin(coil_config_indexes)]
        fqs = []
        for i_coil_config_index in coil_config_df['Coil Config Index'].unique():
            fqs.append(np.mean(coil_config_df.loc[coil_config_df['Coil Config Index'] == i_coil_config_index]['fit_quality']))
        plt.plot(coil_increments, fqs, label = coil_name)

    plt.legend()
    plt.xlabel('Coil Increment (A)')
    plt.ylabel('Fit Quality')
    plt.show()

def get_sigma_deviance_df(organized_df):
    cols_of_interest = []
    for col in organized_df.columns:
        if col.endswith('_truth'):
            cols_of_interest.append(col[:-6])

    sigma_deviance_df = pd.DataFrame()
    sigma_deviance_df[COIL_CONFIG_INDEX_NAME] = organized_df[COIL_CONFIG_INDEX_NAME]
    sigma_deviance_df[TA_CONFIG_INDEX_NAME] = organized_df[TA_CONFIG_INDEX_NAME]
    for col in cols_of_interest:
        sigma_deviance_df[col] = ((organized_df[col + '_truth'] - organized_df[col + '_mean']) / organized_df[col + '_sigma'])

    return sigma_deviance_df

def get_all_cols_sigma_error_data(organized_df):
    cols_of_interest = []
    for col in organized_df.columns:
        if col.endswith('_truth'):
            cols_of_interest.append(col[:-6])

    num_ta_configs = len(organized_df[TA_CONFIG_INDEX_NAME].unique())
    num_coil_configs = len(organized_df[COIL_CONFIG_INDEX_NAME].unique())

    data = np.empty((num_ta_configs, num_coil_configs, len(cols_of_interest)))
    
    # Entry for each table axis config, for each coil, for each col sigmas off
    for i_ta_config in organized_df[TA_CONFIG_INDEX_NAME].unique():
        single_ta_df = organized_df.loc[organized_df[TA_CONFIG_INDEX_NAME] == i_ta_config]
        for i_col_name, col_name in enumerate(cols_of_interest):
            truth_values = single_ta_df[col_name + '_truth']
            recond_values = single_ta_df[col_name + '_mean']
            recond_sigmas = single_ta_df[col_name + '_sigma']
            sigmas_off = ((truth_values - recond_values) / recond_sigmas)
            sigmas_off.where(recond_sigmas > 1e-6, np.nan, inplace=True) # TODO is this valid? 
            data[i_ta_config, :, i_col_name] = sigmas_off.values

    return data, cols_of_interest

def plot_all_sigma_deviance_slopes(sigma_deviance_arr, cols_of_interest, coil_increments):
    profile_bases = []
    for col in cols_of_interest:
        if col.endswith('015'):
            profile_bases.append(col[:-3])

    plotted_cols = set()
    # First plot profiles

    num_columns = 3

    num_colors = 21
    cm = plt.get_cmap('gist_rainbow')
    colors=[cm(1.*i/num_colors) for i in range(num_colors)]

    linestyles = ['solid', 'dotted', 'dashed']

    # Data is for each table axis config, for each coil, for each col 
    for profile_base in profile_bases:
        fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
        fig.set_figheight(10)
        fig.set_figwidth(14)
        fig.subplots_adjust(wspace=.4)
        for i_psibar_val, psibar_val in enumerate(np.linspace(0, 100, 21, dtype=int)):
            colname = profile_base + str(psibar_val).zfill(3)
            if colname not in cols_of_interest:
                print(f'{colname} not found')
                continue
            i_colname = cols_of_interest.index(colname)
            plotted_cols.add(colname)
            for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
                coil_config_indexes = np.arange(i_coil*NUM_CONFIGS_PER_COIL + 1, (i_coil+1)*NUM_CONFIGS_PER_COIL + 1)
                coil_config_indexes = np.insert(coil_config_indexes, 0, 0)
                ax = axs[int(i_coil/num_columns), i_coil%num_columns]
                ax.plot(coil_increments, np.nanquantile(sigma_deviance_arr[:, coil_config_indexes, i_colname], .90, 0), label=colname, color=colors[i_psibar_val], linestyle=linestyles[i_psibar_val%3])
                ax.set_xlabel('Coil Increment (A)')
                ax.set_ylabel('$\sigma$ Deviance')
                ax.set_title(coil_name)
        plt.suptitle(f'90% Quantile Curves\nTable D: {TABLE_D_CONFIG}')
        handles, labels = ax.get_legend_handles_labels()
        last_ax = axs[-1, -1]
        last_ax.axis('off')
        last_ax.legend(handles, labels)
        plt.savefig(f'plots/all_column_quantiles/{profile_base}.png')
        plt.clf()

    # TODO order by worst ? 

    unplotted_cols = list(set(cols_of_interest) - plotted_cols)
    i_unplotted_cols = 0
    num_at_once = 20
    while(i_unplotted_cols < len(unplotted_cols)):
        fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
        fig.set_figheight(10)
        fig.set_figwidth(14)
        fig.subplots_adjust(wspace=.4)
        cols_to_plot = unplotted_cols[i_unplotted_cols:i_unplotted_cols + num_at_once]
        i_unplotted_cols += num_at_once
        for colname_i, colname in enumerate(cols_to_plot):
            i_colname = cols_of_interest.index(colname)
            for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
                coil_config_indexes = np.arange(i_coil*NUM_CONFIGS_PER_COIL + 1, (i_coil+1)*NUM_CONFIGS_PER_COIL + 1)
                coil_config_indexes = np.insert(coil_config_indexes, 0, 0)
                ax = axs[int(i_coil/num_columns), i_coil%num_columns]
                ax.plot(coil_increments, np.nanquantile(sigma_deviance_arr[:, coil_config_indexes, i_colname], .90, 0), label=colname, color=colors[colname_i], linestyle=linestyles[colname_i%3])
                ax.set_xlabel('Coil Increment (A)')
                ax.set_ylabel('$\sigma$ Deviance')
                ax.set_title(coil_name)
        plt.suptitle(f'90% Quantile Curves\nTable D: {TABLE_D_CONFIG}')
        handles, labels = ax.get_legend_handles_labels()
        last_ax = axs[-1, -1]
        last_ax.axis('off')
        last_ax.legend(handles, labels)
        plt.savefig(f'plots/all_column_quantiles/{int(i_unplotted_cols/num_at_once)}.png')
        plt.clf()

def get_normd_sigma_deviance_slopes(sigma_deviance_arr, cols_of_interest, coil_increments):

    # slope, residual, last index before 90% quartile greater than 1 
    UPPER_QUANTILE_VALUE = .90
    SIGMA_DEVIANCE_THRESHOLD = 1
    normd_sigma_deviance_slopes = np.empty((len(cols_of_interest), len(COILS_OF_INTEREST)))

    base_deviances = sigma_deviance_arr[:, 0, :]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*NUM_CONFIGS_PER_COIL + 1, (i_coil+1)*NUM_CONFIGS_PER_COIL + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

        deviance_values_along_coil = sigma_deviance_arr[:, coil_config_indexes, :]
        normd_deviances_along_coil = abs(deviance_values_along_coil - base_deviances[:, np.newaxis])

        upper_quantile_contours = np.nanquantile(normd_deviances_along_coil, UPPER_QUANTILE_VALUE, 0)
        for i_col, col in enumerate(cols_of_interest):
            col_upper_quantile_contour = upper_quantile_contours[:, i_col]
            slopes_to_base = col_upper_quantile_contour / coil_increments
            breakpoint()
            normd_sigma_deviance_slopes[i_coil, i_col] = np.nanmax(slopes_to_base)

    return normd_sigma_deviance_slopes


def _old_get_normd_sigma_deviance_slopes(sigma_deviance_arr, cols_of_interest, coil_increments):

    # slope, residual, last index before 90% quartile greater than 1 
    UPPER_QUANTILE_VALUE = .90
    SIGMA_DEVIANCE_THRESHOLD = 1
    normd_sigma_deviance_slopes = np.empty((len(cols_of_interest), 3*len(COILS_OF_INTEREST)))

    base_deviances = sigma_deviance_arr[:, 0, :]
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        coil_config_indexes = np.arange(i_coil*NUM_CONFIGS_PER_COIL + 1, (i_coil+1)*NUM_CONFIGS_PER_COIL + 1)
        coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

        deviance_values_along_coil = sigma_deviance_arr[:, coil_config_indexes, :]
        normd_deviances_along_coil = abs(deviance_values_along_coil - base_deviances[:, np.newaxis])

        upper_quantile_contours = np.nanquantile(normd_deviances_along_coil, UPPER_QUANTILE_VALUE, 0)
        for i_col, col in enumerate(cols_of_interest):
            col_upper_quantile_contour = upper_quantile_contours[:, i_col]
            masked_col_upper_quantile_contour = np.ma.masked_invalid(col_upper_quantile_contour)
            nonnan_mask = np.logical_not(np.isnan(col_upper_quantile_contour))

            # Deal w nans
            if np.max(masked_col_upper_quantile_contour) < SIGMA_DEVIANCE_THRESHOLD:
                last_idx = len(col_upper_quantile_contour) - 1
            elif col_upper_quantile_contour[1] > SIGMA_DEVIANCE_THRESHOLD or nonnan_mask.sum() == 0:
                #First point violates threshold, can't make slope or all nans
                normd_sigma_deviance_slopes[i_col, 3*i_coil] = np.nan
                normd_sigma_deviance_slopes[i_col, 3*i_coil + 1] = np.nan
                normd_sigma_deviance_slopes[i_col, 3*i_coil + 2] = np.nan
                continue
            else:
                last_idx = np.argmax(masked_col_upper_quantile_contour > 1) - 1

            coeffs, residual, _, _, _  = np.polyfit(coil_increments[nonnan_mask][:last_idx+1], col_upper_quantile_contour[nonnan_mask][:last_idx+1], 1, full=True)

            normd_sigma_deviance_slopes[i_col, 3*i_coil] = coeffs[0]
            normd_sigma_deviance_slopes[i_col, 3*i_coil + 1] = residual if not len(residual) == 0 else 0 # Residual is empty if just two points
            normd_sigma_deviance_slopes[i_col, 3*i_coil + 2] = coil_increments[last_idx]

    colnames = []
    for coil_name in COILS_OF_INTEREST:
        colnames += [coil_name + descriptor for descriptor in [' Slope', ' Residual', ' Coil Increment at Threshold']]

    df = pd.DataFrame(normd_sigma_deviance_slopes, columns=colnames)
    df['Column'] = cols_of_interest
    df.set_index('Column', inplace=True)
    return df

def visualize_sigma_deviance_slopes(sigma_deviance_slopes, coil_increments):
    # Hist of slopes
    num_columns = 3

    column_types = [' Slope', ' Residual', ' Coil Increment at Threshold']
    x_axes = ['Sigma Deviance per A', 'Linear Fit Error', 'Coil Increment']
    titles = ['Sigma Deviance Slope', 'Linear Fit Error', 'Coil Increment before 1sig Threshold']

    for descriptor, x_axis_descriptor, title in zip(column_types, x_axes, titles):
        fig, axs = plt.subplots(int(np.ceil(len(COILS_OF_INTEREST)/num_columns)), num_columns)
        fig.set_figheight(10)
        fig.set_figwidth(14)
        fig.subplots_adjust(wspace=.4)
        for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
            ax = axs[int(i_coil/num_columns), i_coil%num_columns]
            if x_axis_descriptor == 'Coil Increment at Threshold':
                ax.hist(sigma_deviance_slopes[coil_name + descriptor], bins=coil_increments)
            else:
                ax.hist(sigma_deviance_slopes[coil_name + descriptor], bins=20)
            ax.set_title(coil_name)
            ax.set_xlabel(x_axis_descriptor)
            ax.set_ylabel('Number of Columns')
        plt.suptitle(title + ' For All Columns\nMax Coil Increment to 1$\sigma$ Deviation')
        plt.savefig(f'plots/{descriptor[1:].replace(" ", "_")}.png')
        plt.clf()

def get_worst_slopes(sigma_deviance_slopes):

    worsts = {}
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        colname =coil_name + ' Slope'
        vals = sigma_deviance_slopes[colname].values
        quantile_value = np.nanquantile(vals, .95)
        bad_cols = sigma_deviance_slopes.loc[sigma_deviance_slopes[colname] > quantile_value].index.tolist()
        worsts[coil_name] = set(bad_cols)

    bad_in_all = worsts['Coil_A']
    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        bad_in_all = bad_in_all.intersection(worsts[coil_name])
    
    print('Worst 5% in all:\n', '\n'.join(bad_in_all), '\n')

    for i_coil, coil_name in enumerate(COILS_OF_INTEREST):
        bad_cols_str = '\n'.join(worsts[coil_name] - bad_in_all)
        print(f'Worst 5pct for {coil_name}: \n{bad_cols_str}\n')

if __name__ == '__main__':
    num_expected=9945

    plt.style.use('dark_background')

    # recons_df = get_df_from_recons('data/2nd_it_same_densities_recon_results/', num_expected)
    # recons_df = get_all_columns_df_from_recons('data/2nd_it_same_densities_recon_results/')
    # recons_df.to_csv('data/recon_results_all_cols.csv')

    # recons_df = pd.read_csv('data/recon_results_all_cols.csv')
    # recons_df.set_index('FileName', inplace=True)
    # organized_df = organize_df(recons_df, 'out_files.pickle', num_expected)

    # organized_df.to_csv('data/organized_df_all_cols.csv')
    organized_df = pd.read_csv('data/organized_df_all_cols.csv')
    coil_increments = organized_df['Coil_A'].unique() # only works for now as Coil A is 0 at base

    # sigma_deviance_arr, cols_of_interest = get_all_cols_sigma_error_data(organized_df)
    # np.save('data/sigma_deviance_arr', sigma_deviance_arr)

    sigma_deviance_arr = np.load('data/sigma_deviance_arr.npy')

    cols_of_interest = []
    for col in organized_df.columns:
        if col.endswith('_truth'):
            cols_of_interest.append(col[:-6])

    # plot_all_sigma_deviance_slopes(sigma_deviance_arr, cols_of_interest, coil_increments)
    sigma_deviance_slopes = get_normd_sigma_deviance_slopes(sigma_deviance_arr, cols_of_interest, coil_increments)
    # breakpoint()
    # get_worst_slopes(sigma_deviance_slopes)
    visualize_sigma_deviance_slopes(sigma_deviance_slopes, coil_increments)

    # # is ta, coil, col
    # data = get_sigma_error_data(organized_df)
    # plot_normalized_sigma_error_quantiles(data, coil_increments, organized_df, plot_fit_qualities=False)