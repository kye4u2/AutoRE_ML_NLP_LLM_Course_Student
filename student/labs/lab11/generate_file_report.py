import argparse
from  datetime import datetime
import os
from collections import defaultdict
from typing import Optional, List, Dict

from jinja2 import Environment, FileSystemLoader
from logzero import logger

from blackfyre.common import IRCategory
from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from blackfyre.datatypes.contexts.vex.vexinstructcontext import VexInstructionContext
from blackfyre.datatypes.importsymbol import ImportSymbol
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab11.apicalltrace import APICallTrace
from lab_common.labs.lab11.base_generate_file_report import BaseGenerateFileReport
from lab_common.labs.lab11.fileinfo import CrowdsourcedIdsResult, AnalysisResult
from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import num_tokens_from_string, MAX_TOKEN_LENGTH
from labs.lab4.binary_rank import BinaryRankContext, BasicBlockRankContext, compute_global_import_ranks, \
    compute_median_proximity_weights
from labs.lab5.lab_5_1.compute_median_prox_weights import _compute_median_proximity_weights

LAB_11_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab11")

BENIGN_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                "benign",
                                                "2b364c5052c0c8f12f68907551655616d74f2e89f94ad791a93e58c9fd1c8f6c")
SUSPICIOUS_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                    "suspicious",
                                                    "5b246c5d90be3bfcbfcc1eb625a064dcdcb18bd0fe86662a7ed949155e884fae")

MALWARE_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                 "malware",
                                                 "347c3b434e770b38ce06819cad853f93cb139fb4304e40a78933d92f56749d12")

LAB_10_REPORT_FOLDER = os.path.join(ROOT_PROJECT_FOLDER_PATH, "labs", "lab10", "reports")


DEFAULT_SLIDING_WINDOW_SIZE = 30


class GenerateFileReport(BaseGenerateFileReport):

    def __init__(self, file_artifact_folder_path: str, window_size: int):

        super().__init__(file_artifact_folder_path, window_size)

    def _summarize_api_call_trace(self,
                                  api_call_trace: APICallTrace,
                                  bcc_file_path: str,
                                  window_size) -> Optional[str]:

        """
        Objective: Generate a summary of an API call trace to unveil significant behaviors and interactions
        within a binary's execution, providing insights into the binary's key functionalities. This process
        aims to distill the essence of notable API calls and their sequences, thereby reflecting the binary's
        overarching behavior.

        Parameters:
        - api_call_trace (APICallTrace): The API call trace object containing the sequence of API calls along
          with their details such as arguments, return values, and call status.
        - bcc_file_path (str): The file path to the binary code coverage (BCC) file, which includes information
          necessary for computing the median proximity of import functions.
        - window_size (int): The size of the sliding window to use when analyzing segments of the API call
          trace for patterns and behaviors.

        High-Level Steps:
        1. Compute Median Proximity of Import Functions: Analyze the binary's call trace to determine median
           proximity weights for each import function, ranking them based on their significance and proximity to
           critical execution points.

        2. Analyze API Call Patterns Using Sliding Windows: Segment the API call trace with a sliding window,
           assessing patterns and ranking these segments. This step focuses on understanding the dynamic interactions
           and their implications, combining median proximity weights and API call frequencies to highlight relevant
           sequences.

        3. Implement Subsampling for Prompt Integration: To ensure the total prompt does not exceed the MAX_TOKEN_LENGTH,
           apply a subsampling technique based on the average rank of the import functions in each window. Select and
           integrate only the top-ranked windows into the prompt, prioritizing those that provide the most significant
           insights into the binary's behavior.

        4. Craft Prompt for Model Summarization: With the insights from the analysis and subsampled data, prepare a
           detailed prompt for a large language model. This prompt should guide the model to synthesize the key patterns,
           behaviors, and significant API call sequences into a coherent summary.

        5. Generate and Refine Summary: Use the large language model to generate a summary based on the crafted prompt,
           iterating as necessary to ensure the summary accurately reflects the identified behaviors and functionalities
           within the API call trace.

        Returns:
        Optional[str]: A concise and comprehensive summary derived from the API call trace analysis, designed to offer
        clear insights into the binary's operational characteristics and behaviors.

        Note: This function leverages analytical techniques and language model capabilities to distill complex API call
        traces into insightful summaries, aiding in understanding binary functionalities.
        """

        self.initialize()

        api_call_trace_summary: Optional[str] = None

        ### YOUR CODE HERE ###



        ### END YOUR CODE ###

        return api_call_trace_summary

    def _summarize_crowdsource_ids(self, crowdsource_ids_results: List[CrowdsourcedIdsResult]) -> Optional[str]:

        """
        Objective: Analyze and summarize crowdsourced Intrusion Detection System (IDS) alerts to identify and
        understand Indicators of Compromise (IOCs). The function aims to distill key insights from the alerts,
        focusing on specific IOCs like IP addresses, ports, user agent strings, and HTTP headers. It seeks to
        offer a comprehensive analysis that underscores the overarching threats or behaviors indicated by the
        IOCs within the context of cybersecurity.

        Parameters:
        - crowdsource_ids_results (List[CrowdsourcedIdsResult]): A list of crowdsourced IDS alerts, each encapsulating
        details such as alert severity, the specific IOCs involved, and the context of their detection.

        High-Level Steps:
        1. Organize Alerts by Severity: Sort the crowdsourced IDS alerts based on their severity levels to prioritize
        the analysis, ensuring a focus on the most critical alerts first.

        2. Craft Analysis Prompt: Prepare a detailed prompt for a large language model, guiding it to analyze the sorted
        alerts. This prompt should emphasize the need to provide a succinct overview of the primary concerns and patterns,
        a thorough examination of selected alerts, and a conclusion that synthesizes the insights gathered.

        3. Summarize Selected Alerts: Iterate through the sorted list of alerts, incorporating them into the analysis
        prompt as long as the total token count remains within the limit set for model input. This step involves a
        strategic selection of alerts to ensure a comprehensive but concise summary.

        4. Generate and Refine Summary: Use a large language model to generate a summary based on the crafted prompt.
        The process may involve iterative refinement to ensure the summary accurately captures the nuanced understanding
        of the file's activities and intentions as indicated by the IOCs.

        Returns:
        Optional[str]: A concise and comprehensive summary that elucidates the nature of the threats detected and the
        behaviors exhibited by the file, informed by a thorough examination of the crowdsourced IDS alerts.

        Note: This function combines analytical rigor with language model capabilities to extract meaningful insights
        from complex IDS alert data, aiming to enhance the understanding of cybersecurity threats and indicators.
        """

        ids_results_summary: Optional[str] = None

        ### YOUR CODE HERE ###



        ### END YOUR CODE ###

        return ids_results_summary

    def _summarize_scan_results(self, scan_results: Dict[str, AnalysisResult]) -> Optional[str]:
        """
        Objective: Employ a large language model (LLM) to synthesize a comprehensive summary from antivirus scan results,
        highlighting the consensus on potential threats. This process is streamlined into crafting a detailed prompt,
        appending the scan results for context, and then utilizing the LLM to generate a nuanced summary of the findings.

        Parameters:
        - scan_results (Dict[str, AnalysisResult]): A dictionary where keys are scan types and values are AnalysisResult
        objects, detailing each antivirus engine's findings including malware identification and detection methods.

        High-Level Steps:
        1. Craft the Prompt: Start by creating an initial prompt that outlines the objective of the summary, guiding the
        LLM's analysis. This includes a brief overview indicating whether the file is broadly recognized as malicious,
        highlighting any consensus on malware types or families, and noting critical trends in detection methods.

        2. Append Scan Results to the Prompt: Integrate the specific scan results into the prompt, providing the LLM with
        detailed context for its analysis. This involves translating the antivirus findings into a structured format that
        can be easily synthesized by the model, ensuring the prompt is comprehensive yet concise.

        3. Execute the Prompt with the LLM: Use the prepared prompt to instruct the LLM to generate the summary. The model
        will process the provided information, focusing on the synthesis of key insights and consensus among the antivirus
        engines. This step may require iterating on the prompt's details to refine the output, ensuring the summary
        accurately reflects the overarching findings and implications of the scan results.

        Returns:
        Optional[str]: A detailed and insightful summary that encapsulates the antivirus scan results' critical analyses,
        designed to offer a clear understanding of the identified threats and consensus among antivirus engines. This
        summary aims to elucidate the potential risks associated with the analyzed file, leveraging the LLM's capability
        to transform complex datasets into accessible information.

        Note: This function exemplifies the strategic use of a large language model for cybersecurity analysis, highlighting
        the importance of effective prompt crafting and data integration for generating meaningful summaries from complex
        scan results.
        """

        self.initialize()

        scan_results_summary: Optional[str] = None

        ### YOUR CODE HERE ###


        ### END YOUR CODE ###

        return scan_results_summary



def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Generate a report for the file.")
    parser.add_argument("-f", "--file_artifact_folder_path",
                        type=str,
                        default=BENIGN_FILE_ARTIFACT_FOLDER_PATH,
                        help="The path to the file artifact folder.")
    parser.add_argument("-a","--summarize_api_call_trace", action="store_true",
                        help="Summarize the API call trace.")

    parser.add_argument("-c","--summarize_crowdsource_ids", action="store_true",
                        help="Summarize the crowdsource IDS.")

    parser.add_argument("-s","--summarize_scan_results", action="store_true",
                        help="Summarize the scan results.")

    # Generate a report for the file
    parser.add_argument("-r","--generate_report", action="store_true",
                        help="Generate a report for the file.")

    # Window size for the sliding window
    parser.add_argument("-w", "--window_size",
                        type=int,
                        default=DEFAULT_SLIDING_WINDOW_SIZE,
                        help="The window size for the sliding window.")

    # Parse the arguments
    args = parser.parse_args()

    gen_file_report = GenerateFileReport(args.file_artifact_folder_path, window_size=args.window_size)

    if args.summarize_api_call_trace:
        api_call_trace_summary = gen_file_report.summarize_api_call_trace()
        logger.info(f"API Call Trace Summary: {api_call_trace_summary}")

    elif args.summarize_crowdsource_ids:
        crowd_sourced_ids_summary = gen_file_report.summarize_crowdsource_ids()
        logger.info(f"Crowdsource IDS Summary: {crowd_sourced_ids_summary}")

    elif args.summarize_scan_results:
        scan_results_summary = gen_file_report.summarize_scan_results()
        logger.info(f"Scan Results Summary: {scan_results_summary}")

    else:

        gen_file_report.generate_report()


if __name__ == "__main__":
    main()
