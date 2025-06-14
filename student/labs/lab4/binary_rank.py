import logging
import os
import time
from collections import defaultdict
from queue import Queue
from typing import List, Dict, Optional, Tuple

from blackfyre.common import IRCategory
from blackfyre.datatypes.contexts.vex.vexbbcontext import VexBasicBlockContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext

import logzero
from logzero import logger

from blackfyre.datatypes.contexts.vex.vexinstructcontext import VexInstructionContext
from blackfyre.datatypes.importsymbol import ImportSymbol
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from labs.lab5.lab_5_1.compute_median_prox_weights import _compute_median_proximity_weights

logzero.loglevel(logging.INFO)

# ================================CONSTANTS =========================================
MAX_NUMBER_VISITS_TO_BB = 1

MAX_NUMBER_ANCESTORS = None

ALLOW_CYCLES = True

ONLY_TERMINATE_AT_RETURN = False

MAX_NUM_BASIC_BLOCKS = 15000

MAX_NUMBER_CODE_PATHS = 10000


# ===============================END CONSTANTS =======================================

class BinaryRankTimeoutError(TimeoutError):
    """Exception raised when a timeout occurs during execution."""
    pass


class BasicBlockRankContext(object):

    def __init__(self,
                 start_address: int,
                 end_address: int,
                 local_rank: float = 0,
                 global_rank: float = 0):
        self._start_address = start_address

        self._end_address = end_address

        self._parent_bb_rank_context = None

        self.vex_bb_context = None

        self._local_rank = local_rank

        self._global_rank = global_rank
        deadline: float = float('inf')
        # List of start address of ancestor basic blocks
        self.ancestor_bb_address_list: List[int] = []

    @classmethod
    def from_bb_context(cls,
                        vex_bb_context: VexBasicBlockContext,
                        parent_rank_context: Optional['BasicBlockRankContext'],
                        local_rank: float = 0,
                        global_rank: float = 0):
        basic_block_rank_context = cls(vex_bb_context.start_address,
                                       vex_bb_context.end_address,
                                       local_rank,
                                       global_rank)

        basic_block_rank_context.vex_bb_context = vex_bb_context
        basic_block_rank_context._parent_bb_rank_context = parent_rank_context

        if basic_block_rank_context._parent_bb_rank_context is not None:
            parent_start_address = basic_block_rank_context._parent_bb_rank_context.vex_bb_context.start_address

            # parent's ancestors list + parent
            basic_block_rank_context.ancestor_bb_address_list = list(
                basic_block_rank_context._parent_bb_rank_context.ancestor_bb_address_list +
                [parent_start_address])

        return basic_block_rank_context

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def global_rank(self):
        return self._global_rank

    @global_rank.setter
    def global_rank(self, value):
        self._global_rank = value

    @property
    def start_address(self):
        return self._start_address

    @property
    def end_address(self):
        return self._end_address

    @property
    def address(self):
        return self._start_address

    @property
    def num_ancestors(self):
        return len(self.ancestor_bb_address_list)


class FunctionRankContext(object):
    def __init__(self,
                 vex_function_context: VexFunctionContext,
                 bb_rank_context_dict: Dict[int, BasicBlockRankContext]):

        self._vex_function_context: VexFunctionContext = vex_function_context

        self._bb_rank_context_dict: Dict[int, BasicBlockRankContext] = bb_rank_context_dict

        self._rank: float = 0.0

        self._num_callers: Optional[int] = None

        self._ancestor_dict: Dict[int:int] = {}

    @classmethod
    def _analyze_basic_blocks(cls,
                              vex_function_context: VexFunctionContext,
                              code_paths: List[BasicBlockRankContext], ) -> List[BasicBlockRankContext]:

        bb_rank_contexts: List[BasicBlockRankContext] = []

        basic_block_count_dict: Dict[int, int] = defaultdict(int)

        # Iterate through each basic block
        code_path: BasicBlockRankContext
        for code_path in code_paths:

            # Add the count of the terminating basic block
            basic_block_count_dict[code_path.start_address] += 1

            # Ad the count of the ancestors of the terminating basic block
            for ancestor_bb_address in code_path.ancestor_bb_address_list:
                basic_block_count_dict[ancestor_bb_address] += 1

        # The +1 is added for the terminating basic block
        total_basic_count = sum([basic_block_count_dict[key] for key in basic_block_count_dict]) + 1

        for key in vex_function_context.basic_block_context_dict:

            basic_block_context: VexBasicBlockContext = vex_function_context.basic_block_context_dict[key]

            if total_basic_count == 0:
                logger.info(f"basic block is zero for 0x{key:x}")

            execution_percentage = basic_block_count_dict[basic_block_context.start_address] / total_basic_count

            bb_analyst_context = BasicBlockRankContext.from_bb_context(basic_block_context, None, execution_percentage)

            bb_rank_contexts.append(bb_analyst_context)

        return bb_rank_contexts

    @classmethod
    def generate_code_paths(cls, vex_function_context: VexFunctionContext,
                            code_paths: List[BasicBlockRankContext],
                            bb_rank_context_queue: Queue[BasicBlockRankContext],
                            visits_to_basic_block_dict: Dict[int, int],
                            max_number_code_paths: int = MAX_NUMBER_CODE_PATHS,
                            deadline: float = float('inf')) -> List[BasicBlockRankContext]:

        # Get the function start address
        function_start_address = vex_function_context.start_address

        if function_start_address not in vex_function_context.basic_block_context_dict:
            logger.debug(
                f"Cannot find the basic block that is then entry point of the function"
                f" likely because it is a thunk :{function_start_address:x}")
            return code_paths

        # Get the function's entry basic block context
        entry_bb_context: VexBasicBlockContext = vex_function_context.entry_basic_block_context

        # logger.debug(f"entry IRSB:\n {entry_bb_context.irsb}")

        entry_bb_analysis_context: BasicBlockRankContext = BasicBlockRankContext.from_bb_context(entry_bb_context, None)

        # Push the entry basic block analysis context into the queue
        bb_rank_context_queue.put(entry_bb_analysis_context)

        while not bb_rank_context_queue.empty():

            if time.time() > deadline:
                raise BinaryRankTimeoutError("generate_code_paths timed out")

            if len(code_paths) >= max_number_code_paths:
                logger.debug(f"Exceeded the maximum number of code paths: {max_number_code_paths}")
                break

            # Get the basic block analysis context in queue
            bb_rank_context: BasicBlockRankContext = bb_rank_context_queue.get()

            # Update the count to this basic block by 1
            visits_to_basic_block_dict[bb_rank_context.vex_bb_context.start_address] += 1

            logger.debug(f"Currently evaluating basic block: 0x{bb_rank_context.vex_bb_context.start_address:x}")

            num_bb_added_to_queue = 0
            basic_block_returns = False
            for vex_instruction_context in bb_rank_context.vex_bb_context.vex_instruction_contexts:

                logger.debug(f"Current native instruction : {vex_instruction_context.native_address:x}")

                if vex_instruction_context.category == IRCategory.call:
                    pass

                if vex_instruction_context.category == IRCategory.branch or \
                        vex_instruction_context.category == IRCategory.call:

                    # Retrieve the jump target basic block address
                    jump_target_addr = vex_instruction_context.jump_target_addr

                    # Check added queue would exceed the number of visits to the basic block
                    if MAX_NUMBER_VISITS_TO_BB is not None and visits_to_basic_block_dict[
                        jump_target_addr] + 1 > MAX_NUMBER_VISITS_TO_BB:
                        # Dont add to the queue
                        continue

                    # Check if the added queue would exceed the max number ancestors allowed
                    if MAX_NUMBER_ANCESTORS is not None and bb_rank_context.num_ancestors + 1 > MAX_NUMBER_ANCESTORS:
                        # Don't add to the queue
                        continue

                    #  Check if the jump target address is an ancestor of the current basic block
                    if not ALLOW_CYCLES and jump_target_addr in bb_rank_context.ancestor_bb_address_list:
                        # Don't add this jump target to the queue because it will create a cycle
                        continue

                    if jump_target_addr in vex_function_context.basic_block_context_dict:
                        next_bb_context = vex_function_context.basic_block_context_dict[jump_target_addr]

                        jump_target_bb_analysis_context = BasicBlockRankContext.from_bb_context(next_bb_context,
                                                                                                bb_rank_context)

                        logger.debug(f"Adding basic block jump target to the queue: 0x{jump_target_addr:x}")

                        bb_rank_context_queue.put(jump_target_bb_analysis_context)

                        num_bb_added_to_queue += 1

                elif vex_instruction_context.category == IRCategory.ret:
                    basic_block_returns = True

            # If we haven't added basic blocks to the queue or the block returns,
            # this means this basic block is terminating basic block for this code path
            if (not ONLY_TERMINATE_AT_RETURN and num_bb_added_to_queue == 0) or basic_block_returns:
                bb_address = bb_rank_context.vex_bb_context.start_address
                logger.debug(f"Terminating code path 0x{bb_address:x}")
                # logger.info(bb_analysis_context.vex_bb_context.irsb)
                # bb_analysis_context.display_code_path()

                # Add the terminating basic block analysis context  to the code path liset
                code_paths.append(bb_rank_context)

        return code_paths

    @classmethod
    def from_function_context(cls, vex_function_context: VexFunctionContext,
                              max_number_code_paths: int = MAX_NUMBER_CODE_PATHS,
                              timeout_seconds: Optional[float] = None) -> 'FunctionRankContext':

        start_time = time.time()
        deadline = start_time + timeout_seconds if timeout_seconds is not None else float('inf')

        # Basic Block context queue that stores the next basic block context to be process
        bb_rank_context_queue: Queue[BasicBlockRankContext] = Queue()

        # Key: start address of basic block, value: number of times basic block visited
        visits_to_basic_block_dict: Dict[int, int] = defaultdict(int)

        ancestor_dict = {}

        code_paths: List[BasicBlockRankContext] = []

        num_basic_blocks = vex_function_context.num_basic_blocks

        bb_rank_contexts: List[BasicBlockRankContext] = []

        if vex_function_context.num_basic_blocks <= MAX_NUM_BASIC_BLOCKS or MAX_NUM_BASIC_BLOCKS == 0:
            cls.generate_code_paths(vex_function_context,
                                    code_paths,
                                    bb_rank_context_queue,
                                    visits_to_basic_block_dict,
                                    max_number_code_paths=max_number_code_paths,
                                    deadline=deadline)

            bb_rank_contexts = cls._analyze_basic_blocks(vex_function_context, code_paths)

        else:
            logger.warning(f"Function 0x{vex_function_context.start_address:x} exceeds the max number of "
                           f" basic blocks to perform code path analysis: "
                           f"{vex_function_context.num_basic_blocks} > {MAX_NUM_BASIC_BLOCKS}")

            logger.warning(f"Will default to uniform execution time distribution [i.e. (1/num_basic_blocks) = "
                           f"(1/{num_basic_blocks})] for the basic blocks:")

            num_basic_blocks = vex_function_context.num_basic_blocks

            # Defaulting the execution time to be uniform (i.e. 1/num_basic_block)
            for basic_block_context in vex_function_context.basic_block_contexts:
                execution_percentage = 1.0 / num_basic_blocks

                bb_analyst_context = BasicBlockRankContext.from_bb_context(basic_block_context, None,
                                                                           execution_percentage)

                bb_rank_contexts.append(bb_analyst_context)

        bb_rank_context_dict: Optional[Dict[int, BasicBlockRankContext]] = {
            bb_rank_context.start_address: bb_rank_context
            for bb_rank_context in bb_rank_contexts}

        function_rank_context = cls(vex_function_context, bb_rank_context_dict)

        return function_rank_context

    @property
    def ancestor_dict(self):
        return self._ancestor_dict

    @property
    def bb_rank_contexts(self) -> List[BasicBlockRankContext]:
        return list(self._bb_rank_context_dict.values())

    @property
    def direct_callers(self) -> List[int]:
        return self._vex_function_context.callers

    @property
    def name(self):
        return self._vex_function_context.name

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value

    @property
    def num_callers(self) -> int:
        return self._num_callers

    @num_callers.setter
    def num_callers(self, value):
        self._num_callers = value

    @property
    def start_address(self) -> int:
        return self._vex_function_context.start_address

    @property
    def vex_function_context(self):
        return self._vex_function_context


class BinaryRankContext(object):

    def __init__(self,
                 vex_binary_context: VexBinaryContext,
                 function_rank_context_dict: Dict[int, FunctionRankContext]):

        self._vex_binary_context = vex_binary_context
        self._function_rank_context_dict: Optional[Dict[int, FunctionRankContext]] = function_rank_context_dict

    @classmethod
    def _compute_functions_callers(cls, function_rank_context_dict: Dict[int, FunctionRankContext]) -> None:
        """
        Computes the number of  callers that include the  direct and indirect callers for each function.
        :param function_rank_context_dict: modified in place
        :return: None
        """

        """
        Note: Students will need to implement this function as part of the lab. Students will update the  call
        counter for each function in the function_rank_context_dict.

        The counter should include the direct and indirect callers for each function.

        For indirect calls, increment the count only if the caller has not been counted in direct calls
        FunctionRankContext.num_callers is a property that can be used to set/increment the number of callers
        for a function.

        For additional information, please refer to the BinaryRank lecture slides.
        """

        ### YOUR CODE HERE ###





        ### END YOUR CODE HERE ###

    @classmethod
    def _compute_global_basic_block_ranks(cls, function_rank_context_dict: Dict[int, FunctionRankContext]):
        """
        Computes the global basic block ranks for each basic block in the binary.
        :param function_rank_context_dict: modified in place.
        :return:
        """
        """
        Notes: 
            1) Students will need to implement this function as part of the lab. 

            2) FunctionRankContext has a property bb_rank_contexts that can be used to access the list
            of BasicBlockRankContext objects for a function.

            3) The BasicBlockRankContext.global_rank property should be updated for each basic block, 
            where the global rank is the product of the function rank and the local rank.

            4) The basic block global ranks should be normalized such that the sum of all global ranks is 1.

            5) For additional information, please refer to the BinaryRank lecture slides.
        """

        ### YOUR CODE HERE ###



        ### END YOUR CODE HERE ###

    @classmethod
    def _initialize_function_rank_contexts(cls,
                                           vex_binary_context: VexBinaryContext,
                                           deadline: float = float('inf')) -> Dict[int, FunctionRankContext]:
        function_rank_context_dict = {}
        for function_context in vex_binary_context.function_context_dict.values():
            if time.time() > deadline:
                raise BinaryRankTimeoutError("Initializing function rank contexts timed out")
            # Propagate remaining time as timeout for each function analysis.
            remaining_time = deadline - time.time()
            fr_context = FunctionRankContext.from_function_context(function_context,
                                                                   timeout_seconds=remaining_time)
            function_rank_context_dict[function_context.start_address] = fr_context
        # function_rank_context_dict = {}

        # for function_context in tqdm(vex_binary_context.function_context_dict.values(), desc="Initializing Function Rank Contexts"):
        #     function_rank_context = FunctionRankContext.from_function_context(function_context)
        #     function_rank_context_dict[function_context.start_address] = function_rank_context

        cls._compute_functions_callers(function_rank_context_dict)

        cls._compute_global_basic_block_ranks(function_rank_context_dict)

        return function_rank_context_dict

    @classmethod
    def from_bcc_file_path(cls,
                           bcc_file_path: str,
                           timeout_seconds: Optional[float] = None) -> 'BinaryRankContext':

        """
        Loads a BinaryRankContext object from a BCC file path.
        :param bcc_file_path: The path to the BCC file
        :param timeout_seconds: The maximum number of seconds to run the analysis
        """

        vex_binary_context = VexBinaryContext.load_from_file(bcc_file_path)

        binary_rank_context = cls.from_vex_binary_context(vex_binary_context, timeout_seconds)

        return binary_rank_context

    @classmethod
    def from_vex_binary_context(cls,
                                vex_binary_context: VexBinaryContext,
                                timeout_seconds: Optional[float] = None) -> 'BinaryRankContext':
        """
        Loads a BinaryRankContext object from a VexBinaryContext object.
        :param vex_binary_context:
        :param timeout_seconds: The maximum number of seconds to run the analysis
        :return:
        """

        start_time = time.time()
        deadline = start_time + timeout_seconds if timeout_seconds is not None else float('inf')

        function_rank_context_dict = cls._initialize_function_rank_contexts(vex_binary_context, deadline=deadline)

        binary_rank_context = cls(vex_binary_context, function_rank_context_dict)

        return binary_rank_context

    @property
    def function_rank_context_dict(self) -> Dict[int, FunctionRankContext]:
        return self._function_rank_context_dict

    @property
    def vex_binary_context(self) -> VexBinaryContext:
        return self._vex_binary_context

    @property
    def basic_block_rank_context_dict(self) -> Dict[int, BasicBlockRankContext]:
        basic_block_rank_context_dict: Dict[int, BasicBlockRankContext] = {}

        for function_rank_context in self._function_rank_context_dict.values():
            for basic_block_rank_context in function_rank_context.bb_rank_contexts:
                basic_block_rank_context_dict[basic_block_rank_context.start_address] = basic_block_rank_context

        return basic_block_rank_context_dict

    @property
    def basic_block_rank_contexts(self) -> List[BasicBlockRankContext]:
        basic_block_rank_contexts: List[BasicBlockRankContext] = []

        for function_rank_context in self._function_rank_context_dict.values():
            for basic_block_rank_context in function_rank_context.bb_rank_contexts:
                basic_block_rank_contexts.append(basic_block_rank_context)

        return basic_block_rank_contexts


def compute_uniform_import_ranks(vex_binary_context: VexBinaryContext) -> Dict[str, float]:
    """
    Computes uniform import ranks for each import in the binary.
    Each import is assigned an equal weight such that the sum of all weights equals 1.

    :param vex_binary_context: VexBinaryContext object containing import symbols.
    :return: A dictionary mapping import name to uniform weight.
    """
    import_rank_dict: Dict[str, float] = {}
    import_symbols = vex_binary_context.import_symbols

    if not import_symbols:
        logger.debug("No import symbols found in vex_binary_context")
        return import_rank_dict

    uniform_weight = 1.0 / len(import_symbols)

    for import_symbol in import_symbols:
        import_rank_dict[import_symbol.import_name] = uniform_weight

    return import_rank_dict


def compute_global_import_ranks(binary_rank_context: BinaryRankContext) -> Dict[str, float]:
    """
    Computes the global import ranks for each import in the binary.
    :param binary_rank_context: BinaryRankContext object
    :return: A dictionary of import to global import rank
    """

    bb_rank_contexts: List[BasicBlockRankContext] = binary_rank_context.basic_block_rank_contexts

    vex_binary_context: VexBinaryContext = binary_rank_context.vex_binary_context

    import_symbol_dict: Dict[str, ImportSymbol] = {}
    import_symbol_addr_dict: Dict[int, ImportSymbol] = {}

    for import_symbol in vex_binary_context.import_symbols:
        import_symbol_dict[import_symbol.import_name] = import_symbol

    for import_symbol in vex_binary_context.import_symbols:
        import_symbol_addr_dict[import_symbol.address] = import_symbol

    import_rank_dict: Dict[str, float] = defaultdict(float)
    for bb_rank_context in bb_rank_contexts:

        vex_instruction_context: VexInstructionContext
        for vex_instruction_context in bb_rank_context.vex_bb_context.vex_instruction_contexts:
            if vex_instruction_context.category == IRCategory.call:
                call_target_addr = vex_instruction_context.call_target_addr
                if call_target_addr is not None and call_target_addr in vex_binary_context.function_context_dict:

                    vex_function_context: VexFunctionContext = vex_binary_context.function_context_dict[
                        call_target_addr]

                    if vex_function_context.name in import_symbol_dict:
                        import_rank_dict[vex_function_context.name] += bb_rank_context.global_rank

                elif call_target_addr is not None and call_target_addr in import_symbol_addr_dict:

                    import_symbol = import_symbol_addr_dict[call_target_addr]

                    import_rank_dict[import_symbol.import_name] += bb_rank_context.global_rank

    # Normalize the global import ranks so that it sums to 1
    sum_global_import_ranks = sum([import_rank_dict[key] for key in import_rank_dict])

    if sum_global_import_ranks == 0:
        logger.debug("Sum of global import ranks is 0")

    else:
        import_rank_dict = {key: import_rank_dict[key] / sum_global_import_ranks for key in import_rank_dict}

    return import_rank_dict


def compute_global_strings_ranks(binary_rank_context: BinaryRankContext) -> Dict[str, float]:
    """
    Computes the global strings ranks for each string in the binary.
    :param binary_rank_context: BinaryRankContext object
    :return: A dictionary of string to global string rank
    """

    string_rank_dict: Dict[str, float] = defaultdict(float)  # Will need to modify the values of this dictionary

    ### YOUR CODE HERE ###



    ### END YOUR CODE HERE ###

    return string_rank_dict


def compute_uniform_strings_ranks(vex_binary_context: VexBinaryContext) -> Dict[str, float]:
    """
    Computes uniform string ranks for each string in the binary.
    Each string is assigned an equal weight such that the sum of all weights equals 1.

    :param vex_binary_context: VexBinaryContext object containing string references.
    :return: A dictionary mapping string to uniform weight.
    """
    string_rank_dict: Dict[str, float] = {}
    string_refs = vex_binary_context.string_refs

    if not string_refs:
        logger.debug("No string references found in vex_binary_context")
        return string_rank_dict

    uniform_weight = 1.0 / len(string_refs)

    for string in set(string_refs.values()):
        string_rank_dict[string] = uniform_weight

    return string_rank_dict


from typing import List, Dict, Optional, Union


def compute_median_proximity_weights(
        feature_ranks: Union[List[float], Dict[str, float]]
) -> Optional[Union[List[float], Dict[str, float]]]:
    """
    Calculates and normalizes weights based on proximity to the median rank.
    Supports either a list of floats or a dict mapping keys to floats,
    returning the same type.

    Objective:
    - Compute weights reflecting proximity to the median rank.

    Steps:
    1. Extract values into a NumPy array and initialize alpha (small constant).
    2. If no features have non-zero ranks, return None.
    3. Compute the median of non-zero values.
    4. Compute weights via: weight = alpha / (alpha + |rank - median|).
    5. Normalize weights so they sum to 1.
    6. Map weights back to the original type.

    Parameters:
    - feature_ranks: List[float] or Dict[str, float]

    Returns:
    - List[float] or Dict[str, float] of normalized weights,
      or None if computation is not possible.
    """

    return _compute_median_proximity_weights(feature_ranks)


def main():
    logger.info("Starting BinaryRank")

    bcc_file_path = os.path.join(
        ROOT_PROJECT_FOLDER_PATH,
        "test_bccs",
        "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90f_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc"
    )

    binary_rank_context = BinaryRankContext.from_bcc_file_path(bcc_file_path, timeout_seconds=2)

    import_rank_dict = compute_global_import_ranks(binary_rank_context)

    # Pretty print sorted descending import ranks
    import_rank_dict = dict(sorted(import_rank_dict.items(), key=lambda item: item[1], reverse=True))
    for import_name, rank in import_rank_dict.items():
        logger.info(f"Import: {import_name}, Rank: {rank}")

    string_rank_dict = compute_global_strings_ranks(binary_rank_context)

    # Pretty print sorted descending string ranks
    string_rank_dict = dict(sorted(string_rank_dict.items(), key=lambda item: item[1], reverse=True))
    for string, rank in string_rank_dict.items():
        logger.info(f"String: {string}, Rank: {rank}")


if __name__ == "__main__":
    main()
