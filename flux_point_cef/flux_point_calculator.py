from typing import Dict

import numpy as np
import pandas as pd

COIL_NAMES = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']

def get_flux_per_amp_at_points(rs, zs, names, save_loc=None):
    import femm

    femm_coil_names = ['A', 'B', 'C', 'D', 'PFC_1', 'PFC_2']
    data = pd.DataFrame()
    femm.openfemm()

    for femm_coil_name in femm_coil_names:
        ansfile = os.path.join(os.getenv("AURORA_REPOS"), 'reconstruction', 'tools', 'table_tools',
            'coil_current_clustering_tools', 'FEMM_plotting', 'pi3b_femm_1A', f'pi3b_asbuilt_Coil_{femm_coil_name}_1A.ans')
        femm.opendocument(ansfile)

        coil_name = 'Coil_' + femm_coil_name if len(femm_coil_name) == 1 else femm_coil_name
        data[coil_name] = np.array([femm.mo_geta(r*1e3, z*1e3) for r, z in zip(rs, zs)])

    data['r'] = rs
    data['z'] = zs
    data['loc_names'] = names

    femm.closefemm()

    if save_loc is not None:
        data.to_csv(save_loc, index=False)

    return data


class FluxPointCalculator():
    def __init__(self, flux_point_df_loc: str):
        self.flux_per_amp_df = pd.read_csv(flux_point_df_loc)
        self.flux_per_amp_df = self.flux_per_amp_df.set_index('loc_names')

    def get_flux_at_points(self, coil_config: Dict[str, float]):
        coil_names = list(coil_config.keys())
        coil_values = np.array([coil_config[coil_name] for coil_name in coil_names])[np.newaxis].T

        unnormd_point_flux_values: pd.Series = self.flux_per_amp_df[coil_names] @ coil_values
        point_flux_values = unnormd_point_flux_values / unnormd_point_flux_values.loc['Psi,Equator']

        return point_flux_values.to_dict()[0]
    
    def get_fluxes_at_points(self, coil_config_df: pd.DataFrame):
        return coil_config_df[COIL_NAMES] @ self.flux_per_amp_df[COIL_NAMES].T

    def get_coils_from_fluxes(self, flux_values: Dict[str, float]):
        point_names = list(flux_values.keys())
        flux_values = np.array([flux_values[point_name] for point_name in point_names])

        flux_per_amp_values = self.flux_per_amp_df.loc[point_names][COIL_NAMES]
        
        coil_values, _, _, _ = np.linalg.lstsq(flux_per_amp_values, flux_values, rcond=None)
        coil_values /= coil_values.max()

        return {coil_name: coil_values[i_coil] for i_coil, coil_name in enumerate(COIL_NAMES)}
    