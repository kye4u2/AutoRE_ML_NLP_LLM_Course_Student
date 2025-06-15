import argparse
import json
import os
from dataclasses import dataclass
from logzero import logger

from jinja2 import Environment, FileSystemLoader

from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import LLMContext

LAB_9_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab9")

TARGET_BCC = os.path.join(LAB_9_DATASET,
                          "bison_arm_9409117ee68a2d75643bb0e0a15c71ab52d4e90fa066e419b1715e029bcdc3dd.bcc")
TEMPLATE_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                          "labs",
                                          "lab9",
                                          "lab_9_2")

OUTPUT_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                        "labs",
                                        "lab9",
                                        "lab_9_2",
                                        "output")




@dataclass
class FunctionSummary:
    binary_name: str
    binary_hash: str
    function_name: str
    function_start_address: str
    decompiled_code: str
    summary: str
    predicted_function: str
    reason_for_prediction: str
    confidence: str

    def __str__(self):
        # Define the maximum width for the decompiled code to keep the table tidy
        max_width = 50
        decompiled_preview = (self.decompiled_code[:max_width] + '...') if len(
            self.decompiled_code) > max_width else self.decompiled_code

        # Table header
        table_str = f"{'Field':<25} | {'Value':<50}\n" + "-" * 78 + "\n"

        # Dynamically adding each field to the table
        fields = [
            ("Binary Name", self.binary_name),
            ("Binary Hash", self.binary_hash),
            ("Function Name", self.function_name),
            ("Function Start Address", self.function_start_address),
            ("Decompiled Code (preview)", decompiled_preview),
            ("Summary", self.summary),
            ("Predicted Function", self.predicted_function),
            ("Reason for Prediction", self.reason_for_prediction),
            ("Confidence", self.confidence),
        ]

        for field, value in fields:
            table_str += f"{field:<25} | {value:<50}\n"

        return table_str


def remove_function_prototype(input_string):
    """
    Removes everything before the first "{" in the input string.

    Parameters:
    - input_string (str): The string from which to remove the content.

    Returns:
    - str: Modified string with content before the first "{" removed, if present.
    """
    brace_position = input_string.find("{")
    if brace_position != -1:
        return input_string[brace_position:]
    else:
        return input_string


def summarize_function(vex_function_context, vex_binary_context)-> FunctionSummary:
    """
    Summarizes the decompiled code of a function using a large language model and returns structured data.

    Parameters:
    - vex_function_context: Context object containing function details.
    - vex_binary_context: Context object containing binary details.

    Returns:
    - FunctionSummary: Dataclass instance containing summary information.

    Objective:
        Summarizes the decompiled code of a function using a large language model and returns structured data , where
        the following fields should be returned as a json format: summary, predicted_function_name,
        reason_for_prediction, and confidence.

    Steps:
         1. Construct a prompt string that includes the function's decompiled code and a set of instructions for
            the large language model.
         2. (Done for you) Use the large language model to generate a summary of the function's behavior.
         3. (Done for you) Parse the response from the large language model and return a FunctionSummary object
            containing the summary information.
    """

    SUMMARY_INSTRUCTION = ""
    ### YOUR CODE HERE ###



    ### END YOUR CODE HERE ###

    decompiled_code = remove_function_prototype(vex_function_context.decompiled_code)
    prompt = f"{SUMMARY_INSTRUCTION} {decompiled_code}"
    llm_context: LLMContext = llm_completion(prompt=prompt)
    print(f"LLM response: {llm_context.response}")

    try:
        parsed_response = json.loads(llm_context.response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: You should add additional guidance  such  as "
                     f"'**Do not use ```json or any other code block formatting. The output should be a plain "
                     f"JSON object without any additional text or formatting.**' to the LLM to ensure it returns a valid JSON object.")
        logger.error(f"Response content: {llm_context.response}")
        raise

    return FunctionSummary(
        binary_name=vex_binary_context.name,
        binary_hash=vex_binary_context.sha256_hash,
        function_name=vex_function_context.name,
        function_start_address=f"{vex_function_context.start_address:x}",
        decompiled_code=decompiled_code,
        summary=parsed_response.get("summary", ""),
        predicted_function=parsed_response.get("predicted_function_name", ""),
        reason_for_prediction=parsed_response.get("reason_for_prediction", ""),
        confidence=parsed_response.get("confidence", "")
    )


def main():
    """
    Main function to parse command line arguments and process the function summary.
    """
    # Define the path to the lab dataset and target BCC file

    # Assuming the function address is passed as a command line argument or defined elsewhere
    parser = argparse.ArgumentParser(description="Summarize the decompiled function code.")
    parser.add_argument("-f", "--bcc_file_path",
                        type=str,
                        default=TARGET_BCC,
                        dest="bcc_file_path",
                        help="Path to the BCC (Binary Code Context) file.")
    parser.add_argument("-a", "--address",
                        type=lambda x: int(x, 16),
                        default=0x00044698,
                        dest="address",
                        help="The address of the function to summarize.")

    args = parser.parse_args()
    vex_binary_context = VexBinaryContext.load_from_file(args.bcc_file_path)
    vex_function_context = vex_binary_context.get_function_context(args.address)

    summary_data = summarize_function(vex_function_context, vex_binary_context)

    logger.info(summary_data)

    # Set up Jinja2 for template rendering
    env = Environment(loader=FileSystemLoader(TEMPLATE_FOLDER_PATH))
    template = env.get_template("function_summary_template.html")

    # Render the template with the summary data
    html_output = template.render(summary_data=summary_data.__dict__)

    # Save the rendered HTML to a file
    output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f'function_summary_{vex_function_context.name}'
                                              f'_{vex_function_context.sha_256_hash}.html')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(html_output)

    print(f"HTML report generated: {output_file_path}")

if __name__ == "__main__":
    main()