import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append('/nfs/home/brendan.posehn/ws/ValidTableCoilRange/single_axis_recon_analysis')

from process_reconstructions import get_all_columns_df_from_recons, organize_df

if __name__ == '__main__':
    num_expected=10050

    plt.style.use('dark_background')

    base = 'data/table_a'
    recons_output_loc = os.path.join(base, 'recon_outputs')
    all_cols_df_loc = os.path.join(base, 'recon_results_all_cols.csv')
    organized_df_loc = os.path.join(base, 'organized.csv')
    out_filenames_loc = os.path.join(base, 'out_filenames.pickle')
    
    recons_df = get_all_columns_df_from_recons(recons_output_loc)
    recons_df.to_csv(recons_df)

    recons_df = pd.read_csv(all_cols_df_loc)
    if 'Unnamed: 0' in recons_df.columns:
        recons_df.drop(columns = ['Unnamed: 0'], inplace=True)
    recons_df.set_index('FileName', inplace=True)

    organized_df = organize_df(recons_df, out_filenames_loc, num_expected)
    organized_df.to_csv(organized_df_loc)