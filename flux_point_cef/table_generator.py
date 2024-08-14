import os, sys
from typing import Dict, List
from copy import deepcopy
import yaml
import pickle
import datetime
import glob

import pandas as pd
import numpy as np
import femm
import git

import split
from split.common import RunArgs

fs_path = os.path.join(os.path.dirname(os.getenv('FS_ROOT')))
sys.path.insert(0, fs_path)
sys.path.insert(0, os.path.join(os.getenv('FS_ROOT'), 'post_processing'))
print(f"NOTE: Using flagships code from {fs_path}")

from flagships.femm_tools.run_xfemm import run_fem_file_with_new_properties
# from flagships.table_launcher.flagships_table_parallel_worker import FlagshipsT
from flagships.gs_solver.pickled_run_params_parallel_worker import PickledRunParamsParallelWorker, PickledRunParamsParallelWorkerArgs
from flagships.gs_solver.fs_flagships import LambdaAndBetaPol1Params
from flagships.gs_solver.fs_curves import NevinsCurve, NevinsCurveYBased

import equilibria_completion_checker
import testcase_completion_checker

from flux_point_calculator import FluxPointCalculator

FPA_LOC = 'data/flux_per_amp_values.csv' # TODO remove hardcode

POINT_FLUX_CONFIG_INDEX_ABBREVIATION = 'PFCI'
TABLE_AXIS_CONFIG_INDEX_ABBREVIATION = 'TACI'

COILS_TO_IGNORE = ['Inner Coil', 'Nose Coil', 'PFC_3', 'PlasmaCurrent']

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
    def __init__(self, config_yaml_path: str, output_path: str, suffix: str = ''):
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

        self.flux_point_calculator = FluxPointCalculator(FPA_LOC) # TODO define this in the config, this should be in ext_psi_files

        table_metadata_file = self.config['base_table_metadata_path']
        with open(table_metadata_file, 'r') as f:
            self.base_table_metadata: Dict = yaml.safe_load(f) 
        
        self.base_table_coil_currents = self.base_table_metadata['FEMM_currents']
        for coil_name in COILS_TO_IGNORE:
            del self.base_table_coil_currents[coil_name]

        self.base_table_flux_point_config = self.flux_point_calculator.get_flux_at_points(self.base_table_coil_currents)
        self.flux_point_configs = pd.concat((pd.DataFrame(self.base_table_flux_point_config, index=[0]), self.flux_point_configs), ignore_index=True)
        self.CEF_table_to_base_table_flux_point_diffs = self.flux_point_configs - pd.Series(self.base_table_flux_point_config)

        self.num_equil = len(self.table_axis_configs) * len(self.flux_point_configs)
        self.num_table_axis_configs = len(self.table_axis_configs)
        self.num_flux_point_configs = len(self.flux_point_configs)

        self.name = f'CEF_{self.base_table_metadata["description"]}_{self.base_table_metadata["lut_date"]}'
        if suffix != '':
            if suffix[0] == '_': 
                suffix = suffix[1:]
            self.name += f'_{suffix}'
    
        self.fs_job_name = f'FS_{self.name}'
        self.recons_job_name = f'TC_{self.name}'
        self.process_recons_job_name = f'Process_{self.name}'
        self.process_recons_submit_script_name = f'{self.process_recons_job_name}_slurm_submit.sh'

        self.serialized_path = f'{self.name}.pickle'
        self.recon_config_path = os.path.join(self.output_path, f'{self.name}_BayesianReconstructionWorkflow.yaml')

        self._coil_configs_at_flux_points: pd.DataFrame = None

        self._flux_point_configs_diffs_to_base_table: pd.DataFrame = None

        self.flux_point_names = list(self.flux_point_calculator.flux_per_amp_df.index)

        self.flagships_env_file = os.environ.get('FLAGSHIPS_ENV_FILE')
        self.flagships_python_bin = '/home/brendan.posehn/anaconda3/envs/fs_sklearn_env/bin/python3.9' # TODO make a required env var? 

        self.recon_env_file = os.environ.get('RECON_ENV_FILE')
        self.recon_python_bin = '/home/brendan.posehn/aurora_repos/reconstruction/venv/bin/python'

    def serialize(self):
        with open(self.serialized_path, 'wb') as f:
            pickle.dump(self, f)

    # TODO don't love this as it exposes launcher stuff here
    @staticmethod
    def from_argfile(argfile_path: str):
        run_args = RunArgs.read(argfile_path)

        argfile_path = run_args.table
        split_path = argfile_path.split(os.sep)
        serialized_path = os.sep.join(split_path[:-1] + [split_path[-1][3:]]) + '.pickle'
        with open(serialized_path, 'rb') as f:
            gen = pickle.load(f)

        return gen

    def launch(self):
        self._generate_equilibria()
        #_perform_reconstructions and _process_testcases done from CompletionCheckers

    def _generate_equilibria(self):
        dc_file_paths = self._generate_dc_files()

        os.makedirs(self.run_params_output_dir, exist_ok=True)
        os.makedirs(self.equil_output_dir, exist_ok=True)

        fs_root = os.getenv('FS_ROOT')

        for i_dc_file, dc_file_path in enumerate(dc_file_paths):
            for i_table_axis_config, table_axis_config_row in self.table_axis_configs.iterrows():
                equil_name = f'{POINT_FLUX_CONFIG_INDEX_ABBREVIATION}{i_dc_file:05d}_{TABLE_AXIS_CONFIG_INDEX_ABBREVIATION}{i_table_axis_config:05d}.hdf5'

                if 'NevinsN' in table_axis_config_row:
                    lambda_curve = NevinsCurve(table_axis_config_row['NevinsA'], table_axis_config_row['NevinsA'],
                                               table_axis_config_row['NevinsC'], table_axis_config_row['NevinsN'])
                else:
                    lambda_curve = NevinsCurveYBased(table_axis_config_row['NevinsA'], table_axis_config_row['NevinsA'],
                                               table_axis_config_row['NevinsC'], table_axis_config_row['NevinsY'])
                pressure_curve = None

                Ishaft = 1.0e6 # self.base_table_metadata['Ishaft'] # TODO right yaml ? 
                Ipl = table_axis_config_row['CurrentRatio'] * Ishaft
                psi_lim = 0

                run_params = LambdaAndBetaPol1Params(self.equil_output_dir, equil_name,
                                                     os.path.join(fs_root, self.base_table_metadata['geom_file']),
                                                     self.base_table_metadata['ext_psi_scale_loc'],
                                                     dc_file_path,
                                                     table_axis_config_row['psieq_dc'],
                                                     os.path.join(fs_root, self.base_table_metadata['soak_file']),
                                                     table_axis_config_row['psieq_soak'],
                                                     lambda_curve, Ishaft, Ipl,
                                                     table_axis_config_row['beta_pol1_setpoint'], psi_lim,
                                                     expected_opoint=self.base_table_metadata['expected_opoint'], 
                                                     pressure_curve=pressure_curve,
                                                     mesh_resolution=self.base_table_metadata['mesh_resolution'],
                                                     use_csharp_solver=False)
                
                with open(os.path.join(self.run_params_output_dir, equil_name[:-5] + '.pickle'), 'wb') as f:
                    pickle.dump(run_params, f)

        self.serialize()

        args = PickledRunParamsParallelWorkerArgs(self.run_params_output_dir, write_to_sql=False,
                                                  output_root=self.equil_output_dir, force_solve=True, skip_solve=False,
                                                  force_postproc=False, force_all_skip_none=False, skip_postproc=True)
        args.num_jobs = self.num_equil
        launcher = split.Launcher(self.fs_job_name, PickledRunParamsParallelWorker, vars(args),
                                   env_file=self.flagships_env_file, python_bin=self.flagships_python_bin, clear=True)
        launcher.launch()
        launcher.write_checker(equilibria_completion_checker.EquilibriaCompletionChecker, cron_workdir=os.getcwd(), cronlog_file='equilibria_completion_checker_log.txt',
                                checker_env_file=self.recon_env_file, checker_python_bin=self.recon_python_bin)

    def _perform_reconstructions(self):

        # need to have same density profiles for all ta configs at each flux point config
        self._make_density_profiles_json() # TODO confirm things see correct density 

        args = {'table': self.recons_job_name,
                'hdf_dir': self.equil_output_dir,
                'cals_dir': None,
                'fs_table': self.config['recon_table'],
                'density_profile_json': self.density_profiles_path,
                'output_root': self.recon_output_dir,
                'batch_size': 1, # TODO perhaps make it always 1 at the testcase level
                'experiment': 'pi3b',
                'cals_dir': os.path.join(os.getenv('RECONCAL_ROOT'), 'pi3b', 'reconstruction_filter_calibration'),
                'num_workers': 1,
                'num_jobs': self.num_equil,
                } 
        
        from reconstruction.tools.testcase_tools.csv_tools.testcase_parallel_worker import TestcaseParallelWorker

        os.makedirs(self.recon_output_dir, exist_ok=True)

        launcher = split.Launcher(self.recons_job_name, TestcaseParallelWorker, args, env_file=self.recon_env_file, python_bin=self.recon_python_bin, clear=True)
        launcher.launch()
        launcher.write_checker(testcase_completion_checker.TestcaseCompletionChecker, cron_workdir=os.getcwd(), cronlog_file='testcase_completion_checker_log.txt')

    def _make_density_profiles_json(self):
        # Need the same density profile for each flux point config as will be comparing them directly
        density_profile_kwargs = {} # TODO perhaps constrain density profiles

        sys.path.append(os.path.join(os.getenv('AURORA_REPOS'), 'reconstruction'))
        from tools.testcase_tools.csv_tools.input_profile_json_generator import DensityProfileJsonGenerator

        density_profiles = []
        for _ in range(self.num_flux_point_configs):
            density_profiles += [DensityProfileJsonGenerator.get_random_profile(**density_profile_kwargs)]*self.num_table_axis_configs

        DensityProfileJsonGenerator.save_input_profiles_json(density_profiles, self.density_profiles_path)

    @property
    def coil_configs_at_flux_points(self):
        coil_configs = []
        if self._coil_configs_at_flux_points is None:
            for i_row, row in self.flux_point_configs.iterrows():
                coil_configs.append(self.flux_point_calculator.get_coils_from_fluxes(row.to_dict()))

            self._coil_configs_at_flux_points = pd.DataFrame.from_dict(coil_configs)
            self._coil_configs_at_flux_points *= 100

        return self._coil_configs_at_flux_points
    
    def _generate_dc_files(self):
        femm_properties = []
        for i_row, row in self.coil_configs_at_flux_points.iterrows():
            femm_properties.append({'frequency': 0, 'currents': row.to_dict()})

        dc_file_paths = run_fem_file_with_new_properties(femm_properties,
                                dc_file=os.path.join(os.getenv('FS_ROOT'), self.config['base_dc_file_path']),
                                output_dir=self.ans_output_dir)

        return dc_file_paths
    
    def _process_testcases(self):
        testcase_results = self._read_testcase_results()
        testcase_results.to_csv(os.path.join(self.output_path, 'testcase_results.csv'), index=False)
        # testcase_results = pd.read_csv(f'{self.output_path}/testcase_results.csv')
        # # recon_sigma_deviances_to_truth is [table axis config, flux config, col sigmas off]
        recon_sigma_deviances_to_truth, col_names = self._get_recon_sigma_deviances_to_truth(testcase_results)
        pickle.dump((recon_sigma_deviances_to_truth, col_names), open(os.path.join(self.output_path, 'sigma_devs_and_cols.pickle'), 'wb'))
        # recon_sigma_deviances_to_truth, col_names = pickle.load(open('out/sigma_devs_and_cols.pickle', 'rb'))

        # TODO naming is crazy here 
        recon_sigma_deviances_to_truth_difference_to_base_coil_config = \
            self._get_unstructured_recon_sigma_deviances_to_truth_difference_to_base_coil_config(recon_sigma_deviances_to_truth, col_names)

        recon_sigma_deviances_to_truth_difference_to_base_coil_config.to_csv(os.path.join(self.output_path, 'sigma_deviances_to_truth.csv'), index=False)

    # TODO better naming 
    # do this at each point, whereas other way is getting the max at each point 
    def _get_unstructured_recon_sigma_deviances_to_truth_difference_to_base_coil_config(self, recon_sigma_deviances_to_truth, cols_of_interest):
        UPPER_QUANTILE_VALUE = .90 # TODO this should be in config? 

        # for each col, for each flux point config 
        normd_sigma_deviance_slopes = np.empty((recon_sigma_deviances_to_truth.shape[1], len(cols_of_interest)))

        base_deviances = recon_sigma_deviances_to_truth[:, 0, :]

        for i_flux_config in range(1, recon_sigma_deviances_to_truth.shape[1]): 
            deviance_values_at_flux_config = recon_sigma_deviances_to_truth[:, i_flux_config, :] - base_deviances
            absd_deviance_values_at_flux_config = abs(deviance_values_at_flux_config)

            upper_quantile_contours = np.nanquantile(absd_deviance_values_at_flux_config, UPPER_QUANTILE_VALUE, 0)

            normd_sigma_deviance_slopes[i_flux_config] = upper_quantile_contours

        df = pd.DataFrame(normd_sigma_deviance_slopes, columns=cols_of_interest)
        df = pd.concat((self.flux_point_configs, df), axis=1)
        return df

    def _get_recon_sigma_deviances_to_truth(self, testcase_results):
        cols_of_interest = []
        for col in testcase_results.columns:
            if col.endswith('_truth'):
                cols_of_interest.append(col[:-6])

        data = np.empty((self.num_table_axis_configs, self.num_flux_point_configs, len(cols_of_interest)))
        
        # Entry for each table axis config, for each flux config, for each col sigmas off
        for i_ta_config in range(self.num_table_axis_configs):
            for i_fp_config in range(self.num_flux_point_configs):
                single_ta_fp_config_df = testcase_results.loc[(testcase_results[TABLE_AXIS_CONFIG_INDEX_ABBREVIATION] == i_ta_config) & (testcase_results[POINT_FLUX_CONFIG_INDEX_ABBREVIATION] == i_fp_config)]
                if single_ta_fp_config_df.empty:
                    data[i_ta_config, i_fp_config] = np.ones(len(cols_of_interest)) * np.nan
                else:
                    for i_col_name, col_name in enumerate(cols_of_interest):
                        truth_values = single_ta_fp_config_df[col_name + '_truth']
                        recond_values = single_ta_fp_config_df[col_name + '_mean']
                        recond_sigmas = single_ta_fp_config_df[col_name + '_sigma']
                        sigmas_off = ((truth_values - recond_values) / recond_sigmas)
                        sigmas_off.where(recond_sigmas > 1e-6, np.nan, inplace=True) # TODO is this valid? 
                        data[i_ta_config, i_fp_config, i_col_name] = sigmas_off.values

        return data, cols_of_interest
        
    def _read_testcase_results(self):
        testcase_output_files = glob.glob(os.path.join(self.recon_output_dir, 'batch_testcase_outputs/*/testcase_0.csv'))

        test_filename = pd.read_csv(testcase_output_files[0])['FileName'].iloc[0]
        PFCI_start = test_filename.find(POINT_FLUX_CONFIG_INDEX_ABBREVIATION) + len(POINT_FLUX_CONFIG_INDEX_ABBREVIATION)
        PFCI_end = test_filename.find('_', PFCI_start)
        TACI_start = test_filename.find(TABLE_AXIS_CONFIG_INDEX_ABBREVIATION) + len(TABLE_AXIS_CONFIG_INDEX_ABBREVIATION)
        TACI_end = test_filename.find('.', TACI_start)

        results = []
        point_flux_config_indices = []
        table_axis_config_indices = []
        for f in testcase_output_files:
            df = pd.read_csv(f)
            filename = df['FileName'].iloc[0]
            point_flux_config_indices.append(int(filename[PFCI_start:PFCI_end]))
            table_axis_config_indices.append(int(filename[TACI_start:TACI_end]))
            results.append(df.iloc[0].to_dict())

        testcase_result_df = pd.DataFrame.from_dict(results)
        testcase_result_df[TABLE_AXIS_CONFIG_INDEX_ABBREVIATION] = table_axis_config_indices
        testcase_result_df[POINT_FLUX_CONFIG_INDEX_ABBREVIATION] = point_flux_config_indices

        return testcase_result_df
    

if __name__ == '__main__':
    table_generator = FluxPointCoilErrorFactorTableGenerator('cef_table_config.yaml', 'mini')
    table_generator.launch()