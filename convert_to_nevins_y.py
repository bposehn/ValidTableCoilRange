import sys

import pandas as pd
import numpy as np

from flagships.gs_solver.fs_curves import NevinsCurve

if __name__ == '__main__':
    input_configs_path = sys.argv[1]
    nevins_n_df = pd.read_csv(input_configs_path)
    nevins_y_df = nevins_n_df.copy()
    nevins_y_df.drop(columns=['NevinsN', 'Unnamed: 0'])

    psibar_array = np.linspace(0, 1, 1000)

    nevins_ys = []
    rows_to_delete = []
    for i_row, row in nevins_n_df.iterrows():
        try:
            nevins_curve = NevinsCurve(row['NevinsA'], row['NevinsA'], row['NevinsC'], row['NevinsN'])
            nevins_ys.append(nevins_curve.solve_for_nevins_y(psibar_array))
        except:
            rows_to_delete.append(i_row)

    nevins_y_df.drop(rows_to_delete, inplace=True)
    nevins_y_df['NevinsY'] = nevins_ys
    nevins_y_df.drop(columns=['Unnamed: 0'], inplace=True)
    nevins_y_df.rename(columns={'beta_pol1':'beta_pol1_setpoint'}, inplace=True)
    nevins_y_df.to_csv(input_configs_path[:-4] + '_nevins_y.csv', index=False)