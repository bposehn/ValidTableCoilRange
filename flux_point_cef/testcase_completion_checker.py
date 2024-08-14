import sys
import datetime
import subprocess


from split.completion_checker import CompletionChecker
from split.status_column import JobStatus

class TestcaseCompletionChecker(CompletionChecker):
    def __init__(self, argfile_path):
        print('tc completion checker')
        super().__init__(argfile_path)
        self.argfile_path = argfile_path
        self.table_generator = FluxPointCoilErrorFactorTableGenerator.from_argfile(argfile_path)

    def success_behaviour(self):
        super().success_behaviour()
        self.process_testcases()

    def partial_success_behaviour(self):
        super().partial_success_behaviour()
        self.process_testcases()     
    
    def failure_behaviour(self):
        counts_by_status = self.categorize_defined_jobs()
        completed = counts_by_status[[JobStatus.SUCCESS, JobStatus.ACCEPTABLE_ERROR, JobStatus.ERROR]].sum() == self.total_num_jobs
        if (counts_by_status[JobStatus.ERROR] / self.total_num_jobs) > .50:
            super().failure_behaviour()
        elif completed:
            self.partial_success_behaviour()
        else:
            self.still_working_behaviour()

    def process_testcases(self):
        print('Starting processing testcases with slurm id: ')
        contents =  '\n'.join((
                    '#!/bin/bash',
                    '#SBATCH --partition=batch',
                    '#SBATCH --mail-type=NONE',
                    '#SBATCH --nodes=1',
                    '#SBATCH --time=3-00:00:00',
                    '#SBATCH --open-mode=append',
                    '#SBATCH --signal=B:TERM@60',
                    '#SBATCH --requeue',
                    f'#SBATCH --job-name={self.table_generator.process_recons_job_name}',
                    '#SBATCH --output=slurm/slurm-%A.out',
                    '#SBATCH --error=slurm/slurm-%A.err',
                    f'source {self.table_generator.recon_env_file}',
                    f'{self.table_generator.recon_python_bin} run_testcase_processing.py {self.argfile_path}'
                    ))
        print(self.table_generator.process_recons_submit_script_name)
        with open(self.table_generator.process_recons_submit_script_name, 'w') as f:
            f.write(contents)

        result = subprocess.run(f'sbatch {self.table_generator.process_recons_submit_script_name}',
                                 shell=True, capture_output=True, text=True)

if __name__ == '__main__':      
    from table_generator import FluxPointCoilErrorFactorTableGenerator
    completion_checker = TestcaseCompletionChecker(sys.argv[1])
    completion_checker.run()