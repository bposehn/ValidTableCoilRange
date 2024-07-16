import os, sys
from typing import Dict, List
from copy import deepcopy

import pandas as pd
import numpy as np
import femm

from flagships.femm_tools.run_xfemm import run_fem_file_with_new_properties

FPA_LOC = 'flux_per_amp_values.csv'
COIL_NAMES = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']

class FluxPointCoilErrorFactorTableGenerator():

    def __init__(self, table_axis_values, flux_point_values):
        self.table_axis_values = table_axis_values
        self.flux_point_values = flux_point_values

class FluxPointCalculator():
    def __init__(self, flux_point_df_loc: str):
        self.flux_per_amp_df = pd.read_csv(flux_point_df_loc)
        self.flux_per_amp_df = self.flux_per_amp_df.set_index('loc_names')

    def _get_coil_configs_for_flux_point_values():
        pass

    def _get_flux_at_points(self, coil_config: Dict[str, float]):
        coil_names = list(coil_config.keys())
        coil_values = np.array([coil_config[coil_name] for coil_name in coil_names])[np.newaxis].T

        point_flux_values = self.flux_per_amp_df[coil_names] @ coil_values
        point_flux_values = point_flux_values / point_flux_values.loc['equator']

        return point_flux_values.to_dict()[0]
    
    def _get_coils_from_fluxes(self, flux_values: Dict[str, float]):
        point_names = list(flux_values.keys())
        flux_values = np.array([flux_values[point_name] for point_name in point_names])

        flux_per_amp_values = self.flux_per_amp_df.loc[point_names][COIL_NAMES]
        # flux_per_amp_values.drop(columns=['Coil_A'], inplace=True)
        
        coil_values, _, _, _ = np.linalg.lstsq(flux_per_amp_values, flux_values)
        coil_values /= coil_values.max()

        print(coil_values)
        return {coil_name: coil_values[i_coil] for i_coil, coil_name in enumerate(COIL_NAMES)}

def run_femm(coil_currents):
    fem_properties_base = {'frequency': 0, 'currents': coil_currents}          

    dc_file = os.path.join('/home/brendan.posehn@gf.local/dev/gf/flagships', 'ext_psi_files', 'pi3', 'pi3b_as_built_2022-09-16_G486_18425-18433.FEM')
    return run_fem_file_with_new_properties([fem_properties_base], dc_file, 'ans')

def rerun_fpa():
    fpc = FluxPointCalculator(FPA_LOC)

    fem_properties_base = {'frequency': 0, 'currents': {'Coil_A': 0, 'Coil_B':0, 'Coil_C':0, 'Coil_D':0, 'PFC_1':0, 'PFC_2':0}}

    fem_property_sets = []
    for coil_name in fem_properties_base['currents']:
        fem_properties = deepcopy(fem_properties_base)
        fem_properties['currents'][coil_name] = 1
        fem_property_sets.append(fem_properties)

    breakpoint()
    dc_file = os.path.join('/home/brendan.posehn@gf.local/dev/gf/flagships', 'ext_psi_files', 'pi3', 'pi3b_as_built_2022-09-16_G486_18425-18433.FEM')
    femm_files = run_fem_file_with_new_properties(fem_property_sets, dc_file, 'ans')

    femm.openfemm()
    for femm_file in femm_files:
        femm.opendocument(femm_file)

        flux_vals = {}
        for i, row in fpc.flux_per_amp_df.iterrows():
            flux_vals[row.name] = femm.mo_geta(row['r']*1e3, row['z']*1e3)
        print(femm_file)
        print(flux_vals)

if __name__ == '__main__':
    fpc = FluxPointCalculator(FPA_LOC)

    unnormd_config_fluxes = {'equator': -0.02930246232160905, 'outer_throat': -0.02107526974001225, 'inner_throat': -0.002912227073129654, 'upper': -0.03901617754340687}
    config_fluxes = {key: val / unnormd_config_fluxes['equator'] for key, val in unnormd_config_fluxes.items()}

    coil_values = fpc._get_coils_from_fluxes(config_fluxes)
    flux_values = fpc._get_flux_at_points(coil_values)

    fem_files = run_femm(coil_values)

    femm.openfemm()
    femm.opendocument(fem_files[0])

    flux_vals = {}
    for i, row in fpc.flux_per_amp_df.iterrows():
        flux_vals[row.name] = femm.mo_geta(row['r']*1e3, row['z']*1e3)

    for fluxes in [config_fluxes, flux_vals]:
        for loc in fluxes:
            print(loc, fluxes[loc] / fluxes['equator'])

