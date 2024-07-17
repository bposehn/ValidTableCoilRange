import os, sys
from typing import Dict, List
from copy import deepcopy
import yaml
import pickle
import datetime

import pandas as pd
import numpy as np
import femm
import git

import split
from split.common import RunArgs

from flagships.femm_tools.run_xfemm import run_fem_file_with_new_properties
# from flagships.table_launcher.flagships_table_parallel_worker import FlagshipsT
from flagships.gs_solver.pickled_run_params_parallel_worker import PickledRunParamsParallelWorker, PickledRunParamsParallelWorkerArgs
from flagships.table_launcher.table_completion_checker import TableCompletionChecker
from flagships.gs_solver.fs_flagships import LambdaAndBetaPol1Params
from flagships.gs_solver.fs_curves import NevinsCurve, NevinsCurveYBased

from reconstruction.tools.testcase_tools.csv_tools.testcase_parallel_worker import TestcaseParallelWorker
from reconstruction.tools.testcase_tools.csv_tools.input_profile_json_generator import DensityProfileJsonGenerator

from flux_point_coil_error_factor_equilibria_completion_checker import CoilErrorFactorEquilibriaCompletionChecker
from flux_point_coil_error_factor_testcase_completion_checker import CoilErrorFactorTestcaseCompletionChecker

FPA_LOC = 'flux_per_amp_values.csv'
COIL_NAMES = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1', 'PFC_2']

POINT_FLUX_CONFIG_INDEX_ABBREVIATION = 'PFCI'
TABLE_AXIS_CONFIG_INDEX_ABBREVIATION = 'TACI'

#TODO rename files to be shroter

class FluxPointCoilErrorFactorTableGenerator():
    '''
    
    keyword arguments:
    config_yaml_path - must have:
        path to flux point values csv
        path to table axis values csv
        names and locations of flux points
        which recon table to be based off of (incl table metadata yaml?)
        suffix
        other configs for generating tables
    '''
    def __init__(self, config_yaml_path: str, output_path: str, suffix: str):
        self.output_path = output_path
        self.ans_output_dir = os.path.join(self.output_path, 'ans_files')
        self.equil_output_dir = os.path.join(self.output_path, 'equil')
        self.recon_output_dir = os.path.join(self.output_path, 'recons')
        self.run_params_output_dir = os.path.join(self.output_path, 'run_params')
        self.density_profiles_path = os.path.join(self.output_path, 'density_profiles.json')

        self.suffix = suffix

        with open(config_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.table_axis_configs = pd.read_csv(self.config['table_axis_config_path'])
        self.flux_point_configs = pd.read_csv(self.config['flux_point_values_path'])

        self.num_table_axis_configs = len(self.table_axis_configs)
        self.num_flux_point_configs = len(self.flux_point_configs)

        self.flux_point_calculator = FluxPointCalculator(FPA_LOC)

        table_metadata_file = None # TODO get from table ? 
        with open(table_metadata_file, 'r') as f:
            self.base_table_metadata: Dict = yaml.safe_load(f) 

        self.name = f'CEF_{self.base_table_metadata["table_name"]}_{self.suffix}'
    
        self.fs_job_name = f'FS_{self.name}'
        self.recons_job_name = f'TC_{self.name}'

        self.serialized_path = os.path.join(output_path, f'{self.name}.pickle')
        self.recon_config_path = os.path.join(self.output_path, f'{self.name}_BayesianReconstructionWorkflow.yaml')

        self._coil_configs_at_flux_points: pd.DataFrame = None

        self.flux_point_names = list(self.flux_point_calculator.flux_per_amp_df.index)

    def serialize(self):
        with open(self.serialized_path, 'wb') as f:
            pickle.dump(self)

    # TODO don't love this as it exposes launcher stuff here
    @staticmethod
    def from_argile(argfile_path: str):
        run_args = RunArgs(argfile_path)

        argfile_path = run_args.table
        split_path = argfile_path.split(os.sep)
        serialized_path = os.sep.join(split_path[:-1] + [split_path[-1][3:]]) + '.pickle'
        with open(serialized_path, 'rb') as f:
            return pickle.load(f)

    def launch(self):
        self._generate_equilibria()
        #_perform_reconstructions and _process_testcases done from CompletionCheckers

    def _generate_equilibria(self):
        dc_file_paths = self._generate_dc_files()
        
        for i_dc_file, dc_file_path in enumerate(dc_file_paths):
            for i_table_axis_config, table_axis_config_row in self.table_axis_configs.iterrows():
                equil_name = f'{POINT_FLUX_CONFIG_INDEX_ABBREVIATION}{i_dc_file:05d}_{TABLE_AXIS_CONFIG_INDEX_ABBREVIATION}{i_table_axis_config:05d}.hdf5'

                if 'NevinsN' in table_axis_config_row.columns:
                    lambda_curve = NevinsCurve(table_axis_config_row['NevinsA'], table_axis_config_row['NevinsA'],
                                               table_axis_config_row['NevinsC'], table_axis_config_row['NevinsN'])
                else:
                    lambda_curve = NevinsCurveYBased(table_axis_config_row['NevinsA'], table_axis_config_row['NevinsA'],
                                               table_axis_config_row['NevinsC'], table_axis_config_row['NevinsY'])
                pressure_curve = None

                Ishaft = self.base_table_metadata['Ishaft']
                Ipl = table_axis_config_row['CurrentRatio'] * Ishaft
                psi_lim = 0

                run_params = LambdaAndBetaPol1Params(self.equil_output_dir, equil_name,
                                                     self.base_table_metadata['geom_file'],
                                                     self.base_table_metadata['ext_psi_scale_loc'],
                                                     dc_file_path,
                                                     table_axis_config_row['psieq_dc'],
                                                     self.base_table_metadata['soak_file'],
                                                     table_axis_config_row['psieq_soak'],
                                                     lambda_curve, Ishaft, Ipl,
                                                     table_axis_config_row['beta_pol1_setpoint'], psi_lim,
                                                     expected_opoint=table_axis_config_row['expected_opoint'], 
                                                     pressure_curve=pressure_curve,
                                                     mesh_resolution=self.base_table_metadata['mesh_resolution'])
                
                with open(os.path.join(self.run_params_output_dir, equil_name[:-5] + '.pickle'), 'wb') as f:
                    pickle.dump(run_params, f)

        args = PickledRunParamsParallelWorkerArgs(self.run_params_output_dir, write_to_sql=False,
                                                  output_root=self.equil_output_dir, force_all_skip_none=True)
        launcher = split.Launcher(self.fs_job_name, PickledRunParamsParallelWorker, vars(args),
                                   env_file=os.environ.get("FLAGSHIPS_ENV_FILE"))
        launcher.launch()
        launcher.write_checker(CoilErrorFactorEquilibriaCompletionChecker)

    def _perform_reconstructions(self):

        # need to have same density profiles 
        self._make_density_profiles_json()

        args = {'table': self.recons_job_name, 
                'hdf_dir': self.equil_output_dir,
                'cals_dir': None,
                'fs_table': self.base_table_metadata['table_name'],
                'density_profile_json': self.density_profiles_json_path} # TODO 
        
        launcher = split.Launcher(self.recons_job_name, TestcaseParallelWorker, args)
        launcher.write_checker(CoilErrorFactorTestcaseCompletionChecker)

    def _make_density_profiles_json(self):
        # Need the same density profile for each flux point config as will be comparing them directly
        density_profile_kwargs = {} # TODO perhaps constrain density profiles

        density_profiles = []
        for _ in range(self.num_flux_point_configs):
            density_profiles += [DensityProfileJsonGenerator.get_random_profile()]*self.num_table_axis_configs

        DensityProfileJsonGenerator.save_input_profiles_json(density_profiles, self.density_profiles_path)

    # TODO do before first slurm array started
    # TODO remove if not needed cuz can just copy cals 
    def _make_recon_config(self):
        reconcal_root = os.getenv("RECONCAL_ROOT")
        reconcal_repo = git.Repo(reconcal_root)
        
        def get_user_confirmation(message):
            print(f'{message}: [y/n]')
            choice = input().lower()
            if not (choice == 'y' or choice == 'yes'):
                print('Terminating table launch process.')
                exit()

        # TODO put these into some utils, currently just ripping from table_laucnher.py 
        if reconcal_repo.is_dirty():
            get_user_confirmation(f'{reconcal_root} is dirty. Are you sure you want to continue without committing/stashing?')
        
        commit = reconcal_repo.head.commit.hexsha
        reconcal_repo.remotes.origin.fetch()
        commits_behind = reconcal_repo.iter_commits(f"{commit}..origin/master")
        if len(list(commits_behind)):
            self.get_user_confirmation(f'{reconcal_root} is behind origin/master. Continue anyway?')

        base_recon_config_path = os.path.join(reconcal_root, 'reconstruction_filter_calibration',
                'BayesianReconstructionWorkflow', 'BayesianReconstructionWorkflow', '1-', '-62135596800.yaml')
        with open(base_recon_config_path, 'r') as f:
            base_recon_config = yaml.safe_load(f)

        single_table_recon_config = base_recon_config
        single_table_recon_config['physics_model']['table_cfg']['tables'] = self.base_table_metadata["table_name"]

        with open(self.recon_config_path, 'w') as f:
            yaml.dump(single_table_recon_config, f)

    @property
    def coil_configs_at_flux_points(self):
        coil_configs = []
        if self._coil_configs_at_flux_points is None:
            for i_row, row in self.flux_point_configs.iterrows():
                coil_configs.append(self.flux_point_calculator._get_coils_from_fluxes(row.to_dict()))

            self._coil_configs_at_flux_points = pd.DataFrame.from_dict(coil_configs)

        return self._coil_configs_at_flux_points
    
    def _generate_dc_files(self):
        femm_properties = []
        for i_row, row in self.coil_configs_at_flux_points.iterrows():
            femm_properties.append({'frequency': 0, 'currents': row.to_dict()})

        dc_file_paths = run_fem_file_with_new_properties(femm_properties,
                                dc_file=os.path.join(os.getenv('FS_ROOT'), self.base_table_metadata['dc_file']),
                                output_dir=self.ans_output_dir)

        return dc_file_paths
    
    def _process_testcases(self):
        testcase_results = self._read_testcase_results()
        sigma_deviances, col_names = self._get_sigma_deviances(testcase_results)
        self.normd_sigma_deviance_slopes(sigma_deviances, col_names)

    def _get_normd_sigma_deviance_slopes(self, sigma_deviance_arr, cols_of_interest, coil_increments):

        # TODO need to get working with flux points 

        # slope, residual, last index before 90% quartile greater than 1 
        UPPER_QUANTILE_VALUE = .90
        SIGMA_DEVIANCE_THRESHOLD = 1
        normd_sigma_deviance_slopes = np.empty((len(cols_of_interest), len(self.flux_point_names)))

        base_deviances = sigma_deviance_arr[:, 0, :]

        nan_cols = set()
        for i_flux_point, flux_point_name in enumerate(self.flux_point_names):
            coil_config_indexes = np.arange(i_flux_point*self.num_table_axis_configs + 1,
                                             (i_flux_point+1)*self.num_table_axis_configs + 1)
            coil_config_indexes = np.insert(coil_config_indexes, 0, 0)

            deviance_values_along_coil = sigma_deviance_arr[:, coil_config_indexes, :]
            normd_deviances_along_coil = abs(deviance_values_along_coil - base_deviances[:, np.newaxis])

            upper_quantile_contours = np.nanquantile(normd_deviances_along_coil, UPPER_QUANTILE_VALUE, 0)
            for i_col, col in enumerate(cols_of_interest):
                col_upper_quantile_contour = upper_quantile_contours[:, i_col]

                idxs_geq_threshold = np.where(col_upper_quantile_contour > SIGMA_DEVIANCE_THRESHOLD)[0]
                if len(idxs_geq_threshold) == 0:
                    first_idx_geq_threshold = len(col_upper_quantile_contour)
                else:
                    first_idx_geq_threshold = idxs_geq_threshold[0]

                if first_idx_geq_threshold == 1: 
                    print(col, col_upper_quantile_contour)
                    nan_cols.add(col)
                    normd_sigma_deviance_slopes[i_col, i_flux_point] = col_upper_quantile_contour[0] / coil_increments[0]
                else:
                    slopes_to_base = col_upper_quantile_contour / coil_increments
                    normd_sigma_deviance_slopes[i_col, i_flux_point] = np.nanmax(slopes_to_base[:first_idx_geq_threshold])

        print(f'Columns with nan slope values as even first coil increment caused rise above threshold: {nan_cols}')
        colnames = [coil_name + ' Slope' for coil_name in self.flux_point_names]
        df = pd.DataFrame(normd_sigma_deviance_slopes, columns=colnames)
        return df


    def _get_sigma_deviances(self, testcase_results):
        cols_of_interest = []
        for col in testcase_results.columns:
            if col.endswith('_truth'):
                cols_of_interest.append(col[:-6])

        data = np.empty((self.num_table_axis_configs, self.num_flux_point_configs, len(cols_of_interest)))
        
        # Entry for each table axis config, for each flux config, for each col sigmas off
        for i_ta_config in range(self.num_table_axis_configs):
            single_ta_df = testcase_results.loc[testcase_results[TABLE_AXIS_CONFIG_INDEX_ABBREVIATION] == i_ta_config]
            for i_col_name, col_name in enumerate(cols_of_interest):
                truth_values = single_ta_df[col_name + '_truth']
                recond_values = single_ta_df[col_name + '_mean']
                recond_sigmas = single_ta_df[col_name + '_sigma']
                sigmas_off = ((truth_values - recond_values) / recond_sigmas)
                sigmas_off.where(recond_sigmas > 1e-6, np.nan, inplace=True) # TODO is this valid? 
                data[i_ta_config, :, i_col_name] = sigmas_off.values

        return data, cols_of_interest
        
    def _read_testcase_results(self):
        # TODO likely way faster way to do this 
        testcase_output_files = os.listdir(self.recon_output_dir)

        test_f = testcase_output_files[0]
        PFCI_start = test_f.find(POINT_FLUX_CONFIG_INDEX_ABBREVIATION) + len(POINT_FLUX_CONFIG_INDEX_ABBREVIATION)
        PFCI_end = test_f.find('_', PFCI_start)
        TACI_start = test_f.find(TABLE_AXIS_CONFIG_INDEX_ABBREVIATION) + len(TABLE_AXIS_CONFIG_INDEX_ABBREVIATION)
        TACI_end = test_f.find('.', TACI_start)

        results = []
        point_flux_config_indices = []
        table_axis_config_indices = []
        for f in testcase_output_files:
            point_flux_config_indices.append(int(f[PFCI_start:PFCI_end]))
            table_axis_config_indices.append(int(f[TACI_start:TACI_end]))
            results.append(pd.read_csv(os.path.join(self.recon_output_dir, f)).to_dict())

        testcase_result_df = pd.DataFrame.from_dict(results)
        testcase_result_df[TABLE_AXIS_CONFIG_INDEX_ABBREVIATION] = table_axis_config_indices
        testcase_result_df[POINT_FLUX_CONFIG_INDEX_ABBREVIATION] = point_flux_config_indices

        return testcase_result_df
    


class FluxPointCalculator():
    def __init__(self, flux_point_df_loc: str):
        self.flux_per_amp_df = pd.read_csv(flux_point_df_loc)
        self.flux_per_amp_df = self.flux_per_amp_df.set_index('loc_names')

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

