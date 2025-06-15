import argparse
import os
from collections import defaultdict
from typing import Optional, List, Dict

from logzero import logger

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.calltracesimulator.binarycaltracesimulator import BinaryCallTraceSimulator
from lab_common.calltracesimulator.functioncalltracesimulator import CallTraceNodeInfo, IRSBContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import num_tokens_from_string, MAX_TOKEN_LENGTH
from labs.lab4.binary_rank import BinaryRankContext, BasicBlockRankContext

LAB_9_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab9")

TARGET_BCC = os.path.join(LAB_9_DATASET,
                          "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")
LARGE_TARGET_BCC = os.path.join(LAB_9_DATASET,
                                "sqlite3_03ec84695c5dd60f9837a961b17af09b7415ad8eb830dff4403b35541bcac7db.bcc")
TEMPLATE_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                    "labs",
                                    "lab9",
                                    "lab_9_3")


class CodePathRankBinarySummary(object):

    def __init__(self, bcc_file_path: str):
        self._bcc_file_path = bcc_file_path

        self._binary_rank_context: Optional[BinaryRankContext] = None

        self._vex_binary_context: Optional[VexBinaryContext] = None

        self._bin_call_trace_simulator: Optional[BinaryCallTraceSimulator] = None

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return self._initialized

        self._binary_rank_context = BinaryRankContext.from_bcc_file_path(self._bcc_file_path)

        self._vex_binary_context = self._binary_rank_context.vex_binary_context

        self._bin_call_trace_simulator = BinaryCallTraceSimulator(self._bcc_file_path,
                                                                  max_number_visits_to_irsb=5,
                                                                  max_ancestors=300,
                                                                  min_import_funcs_in_trace=5)

        self._bin_call_trace_simulator.initialize()

        self._initialized = True

        return self._initialized

    def summarize(self):
        self.initialize()

        call_traces: List[IRSBContext] = self._bin_call_trace_simulator.call_traces

        basic_block_rank_context_dict: Dict[
            int, BasicBlockRankContext] = self._binary_rank_context.basic_block_rank_context_dict

        summary = self._summarize_with_ranked_call_traces(self._vex_binary_context,
                                                          call_traces,
                                                          basic_block_rank_context_dict)

        return summary

    def print_call_traces(self, call_trace_dict: Dict[IRSBContext, float], limit: int = 10):

        call_traces = list(call_trace_dict.keys())

        if limit > 0:
            call_traces = call_traces[:limit]

        num_call_traces = len(call_traces)

        call_trace_irsb_context: IRSBContext
        for index, call_trace_irsb_context in enumerate(call_traces):

            rank = call_trace_dict[call_trace_irsb_context]

            logger.info(f"============Call Trace ({index + 1}/{num_call_traces}) [Rank {rank}] ==================")
            for call_trace_index, call_trace_node in enumerate(call_trace_irsb_context.call_trace_list):
                logger.info(f"\t {call_trace_index + 1}) {call_trace_node.pretty_print_str()}")
            logger.info("=================END Call Trace ===================\n")
        pass

    def _summarize_with_ranked_call_traces(self,
                                           vex_binary_context: VexBinaryContext,
                                           call_traces: List[IRSBContext],
                                           basic_block_rank_context_dict: Dict[int, BasicBlockRankContext]) -> str:

        """
        Summarizes the binary's behavior by analyzing prioritized call traces, focusing on the most significant paths
        using BinaryRank.

        :param vex_binary_context: Context object containing detailed information and analysis utilities for the binary.
        :param call_traces: A list of call trace objects, representing sequences of executed instructions or basic blocks.
        :param basic_block_rank_context_dict: Maps basic block addresses to their ranking context, indicating the
                                            importance of each block.

        :return: An HTML-formatted summary highlighting the binary's key behaviors and functionalities based on the
                 prioritized call traces.

        Objective: Efficiently utilize BinaryRank to prioritize call paths for a focused analysis within
                   the LLM's token limit, aiming to unveil the most critical operational sequences of the binary.


         Steps:
        1. Calculate each call trace's rank using the average rank of visited basic blocks.
        2. Normalize all call trace ranks to sum to 1, ensuring a balanced representation.
        3. Sort call traces by descending rank to identify the most relevant paths.
        4. Analyze and incorporate as many top-ranked call traces as possible within the token limit to construct a comprehensive summary.
        5. Craft a prompt for the Large Language Model (LLM) that includes the top-ranked call traces and guides the generation of an HTML report.
        6. Utilize the LLM to generate a detailed HTML report, encapsulating the binary's behavior, intent, and purpose based on the prioritized call traces.

        """
        # Example call trace
        call_trace = call_traces[0]
        call_node_info: CallTraceNodeInfo
        for call_node_info in call_trace.call_trace_list:
            logger.info(f"Call Node Info: {call_node_info.pretty_print_str()}")

        summary = ""
        ### YOUR CODE HERE ###




        ### END YOUR CODE ###

        return summary

    @property
    def vex_binary_context(self) -> VexBinaryContext:
        self.initialize()
        return self._vex_binary_context


def main():
    parser = argparse.ArgumentParser(description="Summarize the Binary.")
    parser.add_argument("-f", "--bcc_file_path",
                        type=str,
                        default=TARGET_BCC,
                        dest="bcc_file_path",
                        help="Path to the BCC (Binary Code Context) file.")

    args = parser.parse_args()

    code_path_rank_binary_summary = CodePathRankBinarySummary(args.bcc_file_path)

    vex_binary_context = code_path_rank_binary_summary.vex_binary_context

    summary_data = code_path_rank_binary_summary.summarize()

    output_file_path = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                    'labs',
                                    'lab9',
                                    'lab_9_4',
                                    'output',
                                    f'binary_summary_{vex_binary_context.name}_{vex_binary_context.sha256_hash}.html')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(summary_data)


if __name__ == "__main__":
    main()
