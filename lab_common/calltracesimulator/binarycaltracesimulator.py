import logging
import time
import os

# Set logging for this module
from typing import Dict, Type, Optional, List

from blackfyre.datatypes.contexts.binarycontext import BinaryContext
from blackfyre.datatypes.contexts.functioncontext import FunctionContext
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from blackfyre.datatypes.importsymbol import ImportSymbol
from blackfyre.utils import setup_custom_logger
from lab_common.calltracesimulator.functioncalltracesimulator import MAX_NUMBER_ANCESTORS, MAX_NUMBER_VISITS_TO_IRSB, \
    IRSBContext, FunctionCallTraceSimulator
from lab_common.common import ROOT_PROJECT_FOLDER_PATH

logger = setup_custom_logger(os.path.splitext(os.path.basename(__file__))[0])
logging.getLogger("binarycontext").setLevel(logging.WARN)
logging.getLogger("pyvex.lifting.gym.arm_spotter").setLevel(logging.CRITICAL)
logger.setLevel(logging.INFO)
#logging.getLogger("functioncalltracesimulator").setLevel(logging.DEBUG)
DEFAULT_MIN_IMPORTS_IN_CALL_TRACE = 10


class FunctionAnalysisContext(object):

    def __init__(self, function_context: FunctionContext, execution_percentage: float = 0):
        self._function_context = function_context

        self._execution_percentage = execution_percentage

        self._descendent_dict = {}

        self._num_indirect_callees = None

    @property
    def callers(self):
        return self._function_context.callers

    @property
    def callees(self):
        return self._function_context.callees

    @property
    def function_context(self):
        return self._function_context

    @property
    def start_address(self):
        return self._function_context.start_address

    @property
    def execution_percentage(self):
        return self._execution_percentage

    @execution_percentage.setter
    def execution_percentage(self, value):
        self._execution_percentage = value

    @property
    def descendent_dict(self):
        return self._descendent_dict

    @property
    def num_indirect_callees(self):
        return self._num_indirect_callees

    @num_indirect_callees.setter
    def num_indirect_callees(self, value):
        self._num_indirect_callees = value


class BinaryCallTraceSimulator(object):

    def __init__(self, binary_context_file_path: str,
                 min_import_funcs_in_trace: int = DEFAULT_MIN_IMPORTS_IN_CALL_TRACE,
                 max_ancestors: int = MAX_NUMBER_ANCESTORS,
                 max_number_visits_to_irsb: int = MAX_NUMBER_VISITS_TO_IRSB):
        # Path to binary file
        self._binary_context_file_path = binary_context_file_path

        # Create the vex binary context
        self._vex_binary_context: Optional[VexBinaryContext] = None
        # Get binary name
        self._binary_name: Optional[str] = None

        self._binary_sha256_hash: Optional[str] = None

        # Total number of functions that we need to analyze
        self._total_num_functions: Optional[int] = None

        # Stores the current number of functions that have already been analyzed
        self._curr_num_analyzed_functions: int = 0

        # Import symbol map, where the address is the key
        self._import_symbol_map: Dict[int, ImportSymbol] = dict()

        self._min_import_funcs_in_trace = min_import_funcs_in_trace

        self._max_ancestors = max_ancestors

        self._max_number_visits_to_irsb = max_number_visits_to_irsb

        self._initialized: bool = False

        self._call_trace_irsb_contexts: List[IRSBContext] = []

    def initialize(self):

        if self._initialized:
            return

        self._vex_binary_context = VexBinaryContext.load_from_file(self._binary_context_file_path)

        self._binary_name = self._vex_binary_context.name

        self._binary_sha256_hash= self._vex_binary_context.sha256_hash

        self._total_num_functions = self._vex_binary_context.total_functions

        self._init_import_symbol_map()

        self._call_trace_irsb_contexts = self._generate_call_traces()

        self._initialized = True

    def _get_function_with_most_indirect_callees(self) -> FunctionAnalysisContext:

        function_analysis_context_dict: Dict[int, FunctionAnalysisContext] = self._compute_functions_indirect_callers()

        sorted_function_analysis_context_dict = dict(sorted(function_analysis_context_dict.items(),
                                                            key=lambda x: x[1].num_indirect_callees, reverse=True))

        function_analysis_context_most_indirect: Optional[FunctionAnalysisContext] = None

        for key in sorted_function_analysis_context_dict:
            function_analysis_context: FunctionAnalysisContext = sorted_function_analysis_context_dict[key]

            if function_analysis_context_most_indirect is None and function_analysis_context.function_context.size > 50:
                function_analysis_context_most_indirect = function_analysis_context

            logger.debug(
                f"Function (0x{key:x})  {function_analysis_context.function_context.name} : callees {function_analysis_context.num_indirect_callees} ")

        return function_analysis_context_most_indirect

    def _compute_function_indirect_callees(self,
                                           func_analysis_context_dict: Dict[int, FunctionAnalysisContext],
                                           function_analysis_context: FunctionAnalysisContext):

        if function_analysis_context.num_indirect_callees is not None:
            return function_analysis_context.num_indirect_callees, function_analysis_context.descendent_dict.keys()

        number_direct_callees = len(function_analysis_context.callees)

        total_number_indirect_callees = number_direct_callees

        for callee in function_analysis_context.callees:

            if callee in function_analysis_context.descendent_dict:
                continue

            # Prevent recursion
            if callee == function_analysis_context.start_address:
                continue

            if callee not in func_analysis_context_dict:
                # @todo: investigate why this happens
                continue

            # Add descendent address to dictionary
            function_analysis_context.descendent_dict[callee] = 1

            callee_function_analysis_context = func_analysis_context_dict[callee]

            (number_indirect_callees, descendent_list) = self._compute_function_indirect_callees(
                func_analysis_context_dict,
                callee_function_analysis_context)

            total_number_indirect_callees += number_indirect_callees

            for descendent in descendent_list:
                function_analysis_context.descendent_dict[descendent] = 1

        return (total_number_indirect_callees, function_analysis_context.descendent_dict.keys())

    def _compute_functions_indirect_callers(self) -> Dict[int, FunctionAnalysisContext]:

        func_analysis_context_dict: Dict[int, FunctionAnalysisContext] = {
            key: FunctionAnalysisContext(self._vex_binary_context.function_context_dict[key])
            for key in self._vex_binary_context.function_context_dict
            if not self._vex_binary_context.function_context_dict[key].is_thunk}

        for key in func_analysis_context_dict:

            function_analysis_context: FunctionAnalysisContext = func_analysis_context_dict[key]

            if function_analysis_context.num_indirect_callees is None:

                (indirect_callees, ancestor_list) = self._compute_function_indirect_callees(func_analysis_context_dict,
                                                                                            function_analysis_context)

                function_analysis_context.num_indirect_callees = indirect_callees

                for ancestor_addr in ancestor_list:
                    function_analysis_context.descendent_dict[ancestor_addr] = 1

        return func_analysis_context_dict

    def _generate_call_traces(self) -> List[IRSBContext]:

        # We will assume that the function that has the most indirect callees is the entry function for the binary
        entry_function_analysis_context = self._get_function_with_most_indirect_callees()

        entry_function_addr: int = entry_function_analysis_context.start_address

        entry_function_context: VexFunctionContext = self._vex_binary_context.get_function_context(entry_function_addr)

        entry_function_name: str = self._vex_binary_context.get_function_name(entry_function_addr)

        logger.info(f"Generating Call Traces for binary '{self._binary_name}' "
                    f"by starting at the entry point [0x{entry_function_addr:x}] '{entry_function_name}'"
                    f"(number of indirect callees {entry_function_analysis_context.num_indirect_callees})")

        function_call_trace_generator = FunctionCallTraceSimulator(entry_function_context,
                                                                   self._import_symbol_map,
                                                                   self._vex_binary_context,
                                                                   self._min_import_funcs_in_trace,
                                                                   self._max_ancestors,
                                                                   self._max_number_visits_to_irsb)

        call_trace_irsb_contexts = function_call_trace_generator.perform_call_trace_generation()

        #self._print_call_traces(call_trace_irsb_contexts)

        return call_trace_irsb_contexts

    def _print_call_traces(self, call_trace_irsb_contexts: List[IRSBContext], limit: int = 10):

        if limit > 0:
            call_trace_irsb_contexts = call_trace_irsb_contexts[:limit]

        num_call_traces = len(call_trace_irsb_contexts)

        call_trace_irsb_context: IRSBContext
        for index, call_trace_irsb_context in enumerate(call_trace_irsb_contexts):

            logger.info(f"============Call Trace ({index + 1}/{num_call_traces}) ==================")
            for call_trace_index, call_trace_node in enumerate(call_trace_irsb_context.call_trace_list):
                logger.info(f"\t {call_trace_index + 1}) {call_trace_node.pretty_print_str()}")
            logger.info("=================END Call Trace ===================\n")
        pass

    def print_call_traces(self, limit: int = 10):

        self._print_call_traces(self.call_traces, limit)

    def _init_import_symbol_map(self):
        import_symbol: ImportSymbol
        for import_symbol in self._vex_binary_context.import_symbols:
            self._import_symbol_map[import_symbol.address] = import_symbol

    @property
    def binary_context(self) -> VexBinaryContext:
        self.initialize()
        return self._vex_binary_context

    @property
    def call_traces(self) -> List[IRSBContext]:
        self.initialize()
        return self._call_trace_irsb_contexts


def test_harness():
    logger.info("****************Test Harness*****************")

    binary_context_file_path = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                            "lab_datasets",
                                            "lab9",
                                            "sqlite3_03ec84695c5dd60f9837a961b17af09b7415ad8eb830dff4403b35541bcac7db.bcc")

    # Build the disassembly object from an xml file
    binary_call_trace_gen = BinaryCallTraceSimulator(binary_context_file_path,
                                                     max_number_visits_to_irsb=5,
                                                     max_ancestors=300,
                                                     min_import_funcs_in_trace=5)

    # binary_call_trace_gen._get_function_with_most_indirect_callees()

    binary_call_trace_gen.call_traces

    binary_call_trace_gen.print_call_traces(limit=0)


    logger.info("***************END Test Harness****************")


if __name__ == "__main__":
    test_harness()