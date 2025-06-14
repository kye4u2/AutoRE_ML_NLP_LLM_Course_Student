import os.path

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from labs.lab4.binary_rank import FunctionRankContext, BinaryRankContext

bcc_file_path = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                             "Blackfyre",
                             "test",
                             "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")
def main():
    # Load the bcc file using the VexBinaryContext.load_from_file() method
    vex_binary_context = VexBinaryContext.load_from_file(bcc_file_path)

    # Initialize the FunctionRankContext
    function_rank_context_dict = {
                function_context.start_address: FunctionRankContext.from_function_context(function_context)
                for function_context in
                vex_binary_context.function_context_dict.values()}

    BinaryRankContext._compute_functions_callers(function_rank_context_dict)

    # Display the top 10  functions by number of callers
    print("\nTop 10 functions by number of callers:")


    # check if the num_callers is None then set it to 0
    for function_rank_context in function_rank_context_dict.values():
        if function_rank_context.num_callers is None:
            function_rank_context.num_callers = 0

    # Print  the function name and number of callers by descending order of number of callers
    for function_rank_context in sorted(function_rank_context_dict.values(), key=lambda x: x.num_callers, reverse=True)[:10]:
        print(f" (0x{function_rank_context.start_address:x})  - Function Name: {function_rank_context.name} Number of Callers: {function_rank_context.num_callers}")



if __name__ == "__main__":
    main()