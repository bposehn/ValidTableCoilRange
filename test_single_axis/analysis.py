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

sys.path.append('/nfs/home/brendan.posehn/ws/ValidTableCoilRange/single_axis_recon_analysis')

from process_reconstructions import *

def truth_inside_bounds(recon_outputs_location, n):

    all_cols = pd.read_csv(os.path.join(recon_outputs_location, '0.csv')).columns
    columns = [col[:-6] for col in all_cols if '_truth' in col]

    data = np.zeros((len(columns)+1, 2))
    df = pd.DataFrame(data, columns=['Pre CEF', 'With CEF'])
    df['Column'] = columns + ['TCs with >= 1 Col OOR']
    df.set_index('Column', inplace=True)

    for i in range(n):
        tc_filepath = os.path.join(recon_outputs_location, str(i)+'.csv')
        tc_data = pd.read_csv(tc_filepath)
        cef_filepath = os.path.join(recon_outputs_location, str(i)+'_cef.csv')
        cef_data = pd.read_csv(cef_filepath)
        cef_data.set_index('Column', inplace=True)
        any_pre_cef_bad_columns = False
        any_with_cef_bad_columns = False
        for col in columns:
            truth = tc_data[col + '_truth'].item()
            cef_sigma = tc_data[col + '_sigma'].item()
            mean = tc_data[col + '_mean'].item()
            cef = cef_data.loc[col].item()
            pre_cef_sigma = cef_sigma / cef

            if np.any(np.isnan([truth, cef_sigma, mean, cef])):
                continue

            if not ((mean - cef_sigma) < truth < (mean + cef_sigma)):
                df.loc[col]['With CEF'] += 1
                any_with_cef_bad_columns = True
            
            if not ((mean - pre_cef_sigma) < truth < (mean + pre_cef_sigma)):
                df.loc[col]['Pre CEF'] += 1
                any_pre_cef_bad_columns = True

        if any_pre_cef_bad_columns:
            df.loc['TCs with >= 1 Col OOR']['Pre CEF'] += 1
        if any_with_cef_bad_columns:
            df.loc['TCs with >= 1 Col OOR']['With CEF'] += 1

    breakpoint()

if __name__ == '__main__':
    n = 1237
    truth_inside_bounds('data/table_d/recon_outputs', n)