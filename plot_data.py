
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/column_data.csv')
    corner_differences = np.load('data/corner_differences.npy') # configs, columns, corners

    plot_rows = 1
    plot_cols = 4
    fig, axs = plt.subplots(plot_rows, plot_cols)
    fig.tight_layout()
    for i_column_name, column_name in enumerate(df.columns[1:]):
        ax = axs[i_column_name]

        for i_table_axis_config in range(corner_differences.shape[0]):
            ax.plot(range(corner_differences.shape[2]), corner_differences[i_table_axis_config, i_column_name])
        # ax.plot(range(column_differences_for_all_configs.shape[1]), np.mean(column_differences_for_all_configs, axis=0))
        ax.set_xlabel('Coil Config')
        ax.set_ylabel('Corner - (Base + Single Axis Variances) (abs)')
        ax.set_title(column_name)

    # data = np.random.random((10, 3))
    # columns = ("Column I", "Column II", "Column III")
    # the_table = axs[-1, -1].table(cellText=data, colLabels=columns, loc='center')
    
    plt.show()