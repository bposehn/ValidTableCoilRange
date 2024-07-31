import sys
import datetime

from split.completion_checker import CompletionChecker
from split.status_column import JobStatus

class TestcaseCompletionChecker(CompletionChecker):
    def __init__(self, argfile_path):
        print('tc completion checker')
        super().__init__(argfile_path)
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
        print(datetime.datetime.now(), 'Starting processing testcases')
        self.table_generator._process_testcases()

if __name__ == '__main__':      
    from table_generator import FluxPointCoilErrorFactorTableGenerator
    completion_checker = TestcaseCompletionChecker(sys.argv[1])
    completion_checker.run()