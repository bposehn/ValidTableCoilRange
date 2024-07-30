import sys

from split.completion_checker import CompletionChecker

class EquilibriaCompletionChecker(CompletionChecker):
    def __init__(self, argfile_path):
        print('equil completion checker')
        super().__init__(argfile_path)
        self.table_generator = FluxPointCoilErrorFactorTableGenerator.from_argfile(argfile_path)

    def success_behaviour(self):
        super().success_behaviour()
        self.reconstruct_equilibria()

    def partial_success_behaviour(self):
        super().partial_success_behaviour()
        self.reconstruct_equilibria()
    
    def reconstruct_equilibria(self):
        self.table_generator._perform_reconstructions()

if __name__ == '__main__':      
    from table_generator import FluxPointCoilErrorFactorTableGenerator
    completion_checker = EquilibriaCompletionChecker(sys.argv[1])
    completion_checker.run()