from typing import Dict, List
import pickle
import random
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

from flagships.post_processing import GenerateColumns

def get_coils_from_filename(filename, coil_names):
    coils = {}
    for coil_name in coil_names:
        start = filename.find(coil_name)
        end_coil_name = start + len(coil_name) + 1
        end_number = filename.find('A', end_coil_name)
        coils[coil_name] = float(filename[end_coil_name:end_number])
    
    return coils

def get_truth_outside_sigma_bounds_df(results_dir, base_coil_currents):
    testcase_csvs = os.listdir(results_dir)
    colnames = set(pd.read_csv(os.path.join(results_dir, testcase_csvs[0])).columns) - set('FileName')
    colnames = colnames.tolist()
    differing_coils = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']
    data = np.zeros((len(colnames), len(differing_coils)))
    for filename in os.listdir(results_dir):
        currents = get_coils_from_filename(filename, base_coil_currents.keys())
        file_differing_coils = [coil_name for coil_name in base_coil_currents if not np.isclose(base_coil_currents[coil_name], currents[coil_name])]
        if len(file_differing_coils) != 1:
            breakpoint()

        file_differing_coil = file_differing_coils[0]
        i_differing_coil = differing_coils.index(file_differing_coil)

        df = pd.read_csv(os.path.join(results_dir, filename))
        for i_colname, colname in enumerate(colnames):
            truth = df[colname + '_truth']
            mean = df[colname + '_mean']
            sigma = df[colname + '_sigma']
            if truth > mean + sigma or truth < mean - sigma:
                data[i_colname, i_differing_coil] += 1

    df = pd.DataFrame(data, columns=differing_coils)
    df['Column'] = colnames

    breakpoint()
    return df

def plot_truth_outside_sigma_bounds(truth_outside_sigma_bounds_df):
    num_cols = 3
    num_columns = 3

    coils = truth_outside_sigma_bounds_df.columns

    fig, axs = plt.subplots(int(len(coils)/num_columns), num_columns)
    for i_coil, coil_name in enumerate(coils):
        ax = axs[int(i_coil/num_columns), i_coil%num_columns]
        ax.hist(truth_outside_sigma_bounds_df[coil_name])
        ax.title(coil_name)

if __name__ == '__main__':
    results_dir = ''

    yaml_file = 'pi3b_asbuilt_D_C353_D-313_P154_042_2023-10-12.yaml'
    yaml_dict = GenerateColumns.read_table_def_yaml(yaml_file)

    base_coil_currents = yaml_dict['FEMM_currents']
    # base_coil_currents = {'Coil_A':0, 'Coil_B':0, 'Coil_C':353, 'Coil_D':-313, 'PFC_1':-154, 'PFC_2':-42 }
    del base_coil_currents['Inner Coil']
    del base_coil_currents['Nose Coil']
    del base_coil_currents['PlasmaCurrent']
    del base_coil_currents['PFC_3']

    truth_outside_sigma_bounds_df = get_truth_outside_sigma_bounds_df(results_dir, base_coil_currents)