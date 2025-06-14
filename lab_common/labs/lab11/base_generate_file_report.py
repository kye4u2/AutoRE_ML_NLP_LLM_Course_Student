import argparse
from  datetime import datetime
import os
from typing import Optional, List, Dict

from jinja2 import Environment, FileSystemLoader
from logzero import logger

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from blackfyre.datatypes.contexts.vex.vexfunctioncontext import VexFunctionContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab11.apicalltrace import APICallTrace
from lab_common.labs.lab11.fileinfo import FileInfo, CrowdsourcedIdsResult, AnalysisResult

from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import num_tokens_from_string, MAX_TOKEN_LENGTH


LAB_11_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab10")

BENIGN_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                "benign",
                                                "2b364c5052c0c8f12f68907551655616d74f2e89f94ad791a93e58c9fd1c8f6c")
SUSPICIOUS_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                    "suspicious",
                                                    "5b246c5d90be3bfcbfcc1eb625a064dcdcb18bd0fe86662a7ed949155e884fae")

MALWARE_FILE_ARTIFACT_FOLDER_PATH = os.path.join(LAB_11_DATASET,
                                                 "malware",
                                                 "347c3b434e770b38ce06819cad853f93cb139fb4304e40a78933d92f56749d12")

LAB_10_REPORT_FOLDER = os.path.join(ROOT_PROJECT_FOLDER_PATH, "labs", "lab11", "reports")

TEMPLATE_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_common","labs", "lab11")

DEFAULT_SLIDING_WINDOW_SIZE = 30


class BaseGenerateFileReport(object):

    def __init__(self,
                 file_artifact_folder_path: str,
                 window_size: int = DEFAULT_SLIDING_WINDOW_SIZE):
        self.file_artifact_folder_path = file_artifact_folder_path

        self._api_call_trace: Optional[APICallTrace] = None

        self._file_info: Optional[FileInfo] = None

        self._bcc_file_path: Optional[str] = None

        self._window_size = window_size

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return self._initialized

        # Initialize the API call trace
        api_call_trace_json_path = os.path.join(self.file_artifact_folder_path, "api_call_trace.json")
        if os.path.exists(api_call_trace_json_path):
            self._api_call_trace = APICallTrace.from_json(api_call_trace_json_path)

        # Initialize the file info json
        file_info_json_path = os.path.join(self.file_artifact_folder_path, "file_info.json")
        if os.path.exists(file_info_json_path):
            self._file_info = FileInfo.from_json_file(file_info_json_path)

        # Get the first file with extension .bcc from the file artifact folder

        for file in os.listdir(self.file_artifact_folder_path):
            if file.endswith(".bcc"):
                self._bcc_file_path = os.path.join(self.file_artifact_folder_path, file)
                break

        self._initialized = True
        return self._initialized

    def summarize_api_call_trace(self):
        self.initialize()
        api_call_trace_summary: Optional[str] = None
        if self._api_call_trace is not None:
            api_call_trace_summary = self._summarize_api_call_trace(self._api_call_trace,
                                                                    self._bcc_file_path,
                                                                    self._window_size)
        return api_call_trace_summary

    def _summarize_api_call_trace(self,
                                  api_call_trace: APICallTrace,
                                  bcc_file_path: str,
                                  window_size) -> Optional[str]:

        raise NotImplementedError("This method needs to be implemented in the derived class.")

    def summarize_crowdsource_ids(self):
        self.initialize()
        crowd_sourced_ids_summary: Optional[str] = None
        if self._file_info is not None:
            crowd_sourced_ids_summary = self._summarize_crowdsource_ids(self._file_info.crowdsourced_ids_results)
        return crowd_sourced_ids_summary

    def _summarize_crowdsource_ids(self, crowdsource_ids_results: List[CrowdsourcedIdsResult]) -> Optional[str]:

        raise NotImplementedError("This method needs to be implemented in the derived class.")

    def summarize_scan_results(self):
        self.initialize()
        scan_results_summary: Optional[str] = None
        if self._file_info is not None:
            scan_results_summary = self._summarize_scan_results(self._file_info.last_analysis_results)
        return scan_results_summary

    def _summarize_scan_results(self, scan_results: Dict[str, AnalysisResult]) -> Optional[str]:
        """
        Summarize the scan results.
        """
        raise NotImplementedError("This method needs to be implemented in the derived class.")



    def generate_report(self):
        """
        Objetive: Generate detailed HTML security reports that consolidate an executive summary with integrated
                  summaries of API call traces, crowdsourced Intrusion Detection System (IDS) results, and antivirus scan findings.
        :return:  The generated report

        Note: The function has been implemented for you. You should not need to modify it. Observing the implementation
        and understanding the logic is important for the lab.
        """

        self.initialize()

        api_call_trace_summary: Optional[str] = None
        if self._api_call_trace is not None:
            api_call_trace_summary = self._summarize_api_call_trace(self._api_call_trace,
                                                                    self._bcc_file_path,
                                                                    self._window_size)
        crowd_sourced_ids_summary: Optional[str] = None
        scan_results_summary: Optional[str] = None
        if self._file_info is not None:
            crowd_sourced_ids_summary = self._summarize_crowdsource_ids(self._file_info.crowdsourced_ids_results)

            scan_results_summary = self._summarize_scan_results(self._file_info.last_analysis_results)

        REPORT_PROMPT = """
        
     Develop an executive summary that delves into the file's threat analysis, spanning 3-5 paragraphs. Use HTML formatting to structure the narrative, ensuring the summary articulates a well-rounded view of the threat level, backed by concrete evidence from the API call trace summary, crowdsourced IDS results, and antivirus scan results. The summary should seamlessly merge insights from these data sources into a coherent argument that leads to your verdict on the file's safety or potential harm. Each paragraph should build upon the last, presenting a logical flow of analysis that culminates in a clear, evidence-based conclusion. Incorporate HTML tags to enhance readability and emphasize key points. The summary should:

    Start with a Strong Opening Verdict: Open with a clear statement of the overall threat level determined for the file, using <strong> to emphasize the conclusion (e.g., malicious, suspicious). Example: <p><strong>Verdict: Comprehensive analysis categorizes the file as a high-risk threat, necessitating immediate attention.</strong></p>

    Evidence from API Call Traces: In the following paragraphs, integrate specific examples of suspicious or malicious activity identified in the API call trace summary. Highlight particular API calls or behavior patterns that signify unauthorized access, data manipulation, or other risk factors, using <em> for emphasis on specific calls or behaviors. Example: <p>Analysis of API call traces reveals <em>repeated unauthorized access attempts</em> to sensitive system areas, indicative of malicious intent.</p>

    Insights from Crowdsourced IDS Results: Discuss the contributions from crowdsourced IDS, highlighting community or expert annotations that corroborate the file's identified threat level. Mention consensus on indicators of compromise or specific threats recognized by the community, using <span> with inline styles for clarity. Example: <p>Community-driven IDS results align, with numerous reports pinpointing the file's association with known <span style="font-weight: bold;">ransomware patterns</span>.</p>

    Confirmation from Antivirus Scan Results: Elaborate on findings from antivirus scans, noting the detection of malware signatures or behaviors. If multiple engines identify the file similarly, cite this as significant evidence supporting the threat level. Utilize <ul> and <li> to list specific malware identifications or consensus points. Example: <p>The consensus among antivirus scans strengthens the verdict, identifying the file as <ul><li>Trojan.GenericKD.12345</li><li>Ransom.Win32.WannaCry</li></ul>.</p>

    Conclusive Synthesis: Close with a synthesis of the evidence, reiterating the file's threat level and the implications for users or systems. This final paragraph should solidify your conclusion, drawing on the highlighted evidence. Example: <p>In conclusion, the amalgamation of API trace anomalies, corroborative crowdsourced intelligence, and antivirus identifications culminates in a definitive assessment of the file as malicious. This conclusion underscores the necessity for stringent security measures to mitigate potential impacts.</p>

    Ensure the executive summary is cohesive, flowing from a strong opening verdict to a detailed presentation of evidence, and culminating in a compelling conclusion. The use of HTML tags should not only structure the content but also emphasize the critical elements of your analysis.
    
      **Do not use ```html or any other code block formatting. **
     
     """


        file_info_str = (f"Meaningful names : {self._file_info.meaningful_name} \n"
                         f"Type: {self._file_info.type}"
                         f"Size: {self._file_info.size}"
                         f"First Seen ITW Date: {self._file_info.first_seen_itw_date}"
                         f" First Submitted Date: {self._file_info.first_submission_date}"
                         f" PE info: {self._file_info.pe_info}"
                         f"Sandbox verdicts : {self._file_info.sandbox_verdicts}")

        prompt = REPORT_PROMPT

        prompt += "\n\n** File Information **\n" + file_info_str
        prompt += "\n\n** API Call Trace Summary **\n" + api_call_trace_summary if api_call_trace_summary else ""
        prompt += "\n\n** CrowdSource IDS Summary **\n" + crowd_sourced_ids_summary if crowd_sourced_ids_summary else ""
        prompt += "\n\n** Scan Results Summary **\n" + scan_results_summary if scan_results_summary else ""

        logger.info(f"Prompt: {prompt}")

        llm_context = llm_completion(prompt)
        report_summary = llm_context.response if llm_context else ""

        logger.info(f"Summary (tokens: {num_tokens_from_string(report_summary)}): {report_summary}")

        # Write the summary to an HTML file
        vex_binary_context = VexBinaryContext.load_from_file(self._bcc_file_path)

        env = Environment(loader=FileSystemLoader(TEMPLATE_FOLDER_PATH))
        template = env.get_template("file_threat_report_template.html")

        # Collect data for rendering
        data_for_template = {
            'file_info': {
                'meaningful_name': self._file_info.meaningful_name,
                # Assuming names is a list and needs to be joined into a string
                'type': vex_binary_context.file_type,
                'size': self._file_info.size,
                'first_seen_itw_date': datetime.fromtimestamp(self._file_info.first_seen_itw_date).strftime('%Y-%m-%d %H:%M:%S'),
                'first_submission_date': datetime.fromtimestamp(self._file_info.first_submission_date).strftime('%Y-%m-%d %H:%M:%S')
,
            },
            'summary_data': {
                'executive_summary': llm_context.response if llm_context else "",
                'api_call_trace_summary': api_call_trace_summary if api_call_trace_summary else "",
                'crowd_sourced_ids_summary': crowd_sourced_ids_summary if crowd_sourced_ids_summary else "",
                'scan_results_summary': scan_results_summary if scan_results_summary else ""
            }
        }

        # Render the template with collected data
        rendered_report = template.render(data_for_template)

        final_report_prompt = f"""
        I have a text report that I need formatted into HTML to enhance its readability without changing the content's meaning. The report should be organized into distinct sections such as the title, executive summary, introduction, methodology, findings or results, discussion, recommendations, conclusion, appendices (if applicable), and references. Please use HTML tags to structure the report accordingly, employing headings, paragraphs, bullet points, tables, and any relevant formatting to make the report visually appealing and easy to navigate. Ensure that:

        The title is enclosed within <h1> tags.
        Subheadings for sections like the executive summary, introduction, etc., are enclosed within <h2> tags, with sub-sections in <h3> tags.
        Paragraphs are enclosed within <p> tags.
        Bullet points are formatted using <ul> for unordered lists and <li> for list items.
        Tables (if any data is presented in tabular form) are created using <table>, with <tr> for table rows, <th> for header cells, and <td> for data cells.
        Use bold <strong> and italic <em> tags sparingly to emphasize key points.
        Links to external references are included using <a href='URL'>link text</a> tags.
        Ensure that the report is structured logically, and the content flows smoothly from one section to the next.
    
        Please convert the provided report text into a well-organized, visually appealing HTML document, enhancing its readability while keeping the semantics intact.
               
        Use CSS to style headings, text, and background for contrast and readability. For example, a light background with dark text, adequate font size, and a readable font family.
        Apply padding and margins to elements to ensure the content is not cramped, enhancing the reading experience.
        
        **Do not use ```html or any other code block formatting. **
        
    
        Here is the report :{rendered_report}
        """

        logger.info("Final Report Prompt: %s", final_report_prompt)

        rendered_report = llm_completion(final_report_prompt)


        # Define the report file path
        report_file_path = os.path.join(LAB_10_REPORT_FOLDER,
                                        f"{vex_binary_context.name}_{vex_binary_context.sha256_hash}_report.html")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(report_file_path), exist_ok=True)

        # Write the rendered report to an HTML file
        with open(report_file_path, "w", encoding='utf-8') as report_file:
            report_file.write(rendered_report.response)

        logger.info(f"Report generated successfully at {report_file_path}")

        return rendered_report.response
