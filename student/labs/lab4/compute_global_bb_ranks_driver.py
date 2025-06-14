import os

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

    # Compute the number of callers for each function
    BinaryRankContext._compute_functions_callers(function_rank_context_dict)

    BinaryRankContext._compute_global_basic_block_ranks(function_rank_context_dict)

    # Display the top 10 basic blocks for all functions
    bb_rank_contexts = [(bb_rank_context, function_rank_context) for function_rank_context in
                        function_rank_context_dict.values()
                        for bb_rank_context in function_rank_context.bb_rank_contexts]

    print("\nTop 10 basic blocks for all functions:")

    for basic_block_rank_context, function_rank_context in sorted(bb_rank_contexts,
                                                                  key=lambda x: x[0].global_rank,
                                                                  reverse=True)[:10]:
        print(f"  - Basic Block Address: 0x{basic_block_rank_context.start_address:x} "
              f"Global Rank: {basic_block_rank_context.global_rank} "
              f"Function Name: {function_rank_context.name} (0x{function_rank_context.start_address:x})")
    print("\n")


if __name__ == "__main__":
    main()