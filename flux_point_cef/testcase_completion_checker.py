import sys

from split.completion_checker import CompletionChecker

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
    
    def process_testcases(self):
        self.table_generator._process_testcases()

if __name__ == '__main__':      
    from table_generator import FluxPointCoilErrorFactorTableGenerator
    completion_checker = TestcaseCompletionChecker(sys.argv[1])
    completion_checker.run()