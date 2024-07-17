import sys

from split.completion_checker import CompletionChecker
from flux_point_coil_error_factor_table_generator import FluxPointCoilErrorFactorTableGenerator

class CoilErrorFactorTestcaseCompletionChecker(CompletionChecker):
    def __init__(self, argfile_path, table_generator: FluxPointCoilErrorFactorTableGenerator):
        super().__init__(argfile_path)
        self.table_generator = table_generator

    def success_behaviour(self):
        super().success_behaviour()
        self.process_testcases()

    def partial_success_behaviour(self):
        super().partial_success_behaviour()
        self.process_testcases()
    
    def process_testcases(self):
        self.table_generator.process_testcases()

if __name__ == '__main__':      
    completion_checker = CoilErrorFactorTestcaseCompletionChecker(sys.argv[1])
    completion_checker.run()