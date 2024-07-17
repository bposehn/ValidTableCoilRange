import sys

from split.completion_checker import CompletionChecker
from flux_point_coil_error_factor_table_generator import FluxPointCoilErrorFactorTableGenerator

class CoilErrorFactorEquilibriaCompletionChecker(CompletionChecker):
    def __init__(self, argfile_path):
        super().__init__(argfile_path)
        self.table_generator = FluxPointCoilErrorFactorTableGenerator.from_argfile(argfile_path)

    def success_behaviour(self):
        super().success_behaviour()
        self.reconstruct_equilibria()

    def partial_success_behaviour(self):
        super().partial_success_behaviour()
        self.reconstruct_equilibria()
    
    def reconstruct_equilibria(self):
        self.table_generator.reconstruct_equilibria()

if __name__ == '__main__':      
    completion_checker = CoilErrorFactorEquilibriaCompletionChecker(sys.argv[1])
    completion_checker.run()