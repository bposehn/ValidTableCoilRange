import os

import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('data/column_data.csv')

    num_coil_configs_per_equil = 60
    num_corners = 10
    num_table_axis_configs = 10
    coils_of_interest = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']
    num_coils_of_interest = len(coils_of_interest)

    corner_differences = np.empty((num_table_axis_configs, len(df.columns)-1, num_corners))
    i_base_row = 0
    for i_table_axis_config in range(num_table_axis_configs):
        for i_column, column_name in enumerate(df.columns[1:]):
            for i_corner in range(num_corners):
                # TODO confirm i_corner_row is calculated correctly
                i_corner_row = 1 + i_table_axis_config*(num_corners*(num_coils_of_interest + 1) + 1) + i_corner*(num_coils_of_interest + 1)
                single_varied_axis_rows = range(i_corner_row+1, i_corner_row+num_coils_of_interest+1)

                single_axis_diffs = 0
                for i_single_varied_axis_row in single_varied_axis_rows:
                    single_axis_diffs += df.iloc[i_single_varied_axis_row][column_name] - df.iloc[i_base_row][column_name]

                print(os.path.basename(df.iloc[i_corner_row]['FileName']))
                for i_single_varied_axis_row in single_varied_axis_rows:
                    print(os.path.basename(df.iloc[i_single_varied_axis_row]['FileName']))
                print(single_varied_axis_rows)

                linearized_column_value = df.iloc[i_base_row][column_name] + single_axis_diffs
                corner_difference = df.iloc[i_corner_row][column_name] - linearized_column_value
                corner_differences[i_table_axis_config, i_column, i_corner] = corner_difference

    np.save('data/corner_differences', corner_differences)
    # (# axes + 1)*num_corners coil configs per equil as well as one at base
