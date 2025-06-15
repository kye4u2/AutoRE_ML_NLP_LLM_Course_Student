import argparse
import os
import random
from dataclasses import dataclass

from logzero import logger



from blackfyre.datatypes.contexts.vex.vexbinarycontext import VexBinaryContext
from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.llm.client import llm_completion
from lab_common.llm.llm_common import LLMContext, num_tokens_from_string, MAX_TOKEN_LENGTH

LAB_9_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab9")

TARGET_BCC = os.path.join(LAB_9_DATASET,
                          "httpd_0ec25ba58309c9f112c34e7e0395c3093ef5453deeb84e3311e5cddfbcd3f839.bcc")
TEMPLATE_FOLDER_PATH = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                          "labs",
                                          "lab9",
                                          "lab_9_2")

@dataclass
class BinarySummary:
    binary_name: str
    binary_hash: str
    summary: str
    predicted_binary_name: str
    reason_for_prediction: str
    confidence: str



def summarize_strings(vex_binary_context: VexBinaryContext) -> str:
    """
    Summarize the strings found within the binary.

    This function identifies and analyzes the strings present in the binary,
    highlighting potentially significant or interesting strings that could give insights
    into the binary's functionality or behavior.

    Parameters:
    - vex_binary_context (VexBinaryContext): The context object containing all the relevant
      data and methods for binary analysis.

    Returns:
    - str: A summary of the strings found in the binary, formatted as a string.
    """
    summary = ""

    ### YOUR CODE HERE ###


    ### END YOUR CODE ###


    # Extract unique strings from the binary context
    strings_data = list(set([string for string in vex_binary_context.string_refs.values()]))

    # Shuffle the strings to ensure a diverse selection when downsampling
    random.shuffle(strings_data)

    concatenated_strings = ""  # Initialize the string to accumulate the selected strings
    for string in strings_data:
        # Prepare the next potential addition
        next_string = "\n" + string if concatenated_strings else string
        # Check if adding this string exceeds the token limit
        if num_tokens_from_string(STRINGS_SUMMARY_PROMPT + concatenated_strings + next_string) <= MAX_TOKEN_LENGTH:
            concatenated_strings += next_string  # Append the string if within limit
        else:
            break  # Stop if the next string would exceed the token limit

    prompt = f"{STRINGS_SUMMARY_PROMPT}\n{concatenated_strings}"

    llm_context: LLMContext = llm_completion(prompt)
    if llm_context:
        summary = llm_context.response

    # Log the final token count for debugging
    logger.info(f"Final token count: {num_tokens_from_string(summary)}")



    return summary

def summarize_imports(vex_binary_context: VexBinaryContext) -> str:
    """
    Summarize the imports used by the binary.

    This function examines the binary's import statements to identify external libraries
    or modules the binary relies on. Understanding the imports can provide insights into
    the binary's functionality and potential external dependencies.

    Parameters:
    - vex_binary_context (VexBinaryContext): The context object containing all the relevant
      data and methods for binary analysis.

    Returns:
    - str: A summary of the imports used by the binary, formatted as a string.
    """
    summary = ""

    ### YOUR CODE HERE ###





    ### END YOUR CODE ###

    # Extract import symbols and shuffle for random selection
    import_symbols = list(vex_binary_context.import_symbols)
    random.shuffle(import_symbols)

    # Initialize import information string and calculate initial token count
    import_info_data = ""
    total_tokens = num_tokens_from_string(IMPORT_SUMMARY_PROMPT)

    # Iterate over import symbols, adding each until the token limit is reached
    for import_symbol in import_symbols:
        next_import_info = f"{import_symbol.import_name}:{import_symbol.library_name}\n"
        if num_tokens_from_string(IMPORT_SUMMARY_PROMPT + "\n" + import_info_data + next_import_info) <= MAX_TOKEN_LENGTH:
            import_info_data += next_import_info
            total_tokens += num_tokens_from_string(next_import_info)
        else:
            break  # Stop if adding another import would exceed the token limit

    # Generate the final prompt
    prompt = f"{IMPORT_SUMMARY_PROMPT}\n{import_info_data}"
    logger.info(f"Final token count: {total_tokens}")

    # Assuming 'llm_completion' is your function to send the prompt to the LLM and get a response
    llm_context = llm_completion(prompt)
    summary = llm_context.response if llm_context else ""


    return summary

def summarize_functions(vex_binary_context: VexBinaryContext) -> str:
    """
    Summarize the functions defined or used in the binary.

    This function focuses on identifying and summarizing the binary's functions,
    including their names to understand the
    binary's internal logic and operational flow.

    Parameters:
    - vex_binary_context (VexBinaryContext): The context object containing all the relevant
      data and methods for binary analysis.

    Returns:
    - str: A summary of the functions within the binary, formatted as a string.
    """
    summary = ""

    ### YOUR CODE HERE ###



    ### END YOUR CODE ###


def summarize_binary(vex_binary_context: VexBinaryContext):
    """
    This function aims to provide a comprehensive summary of a binary file. It synthesizes various aspects of the binary,
    including its strings, imports, and functions, to give a holistic view of its contents and behavior. The function
    orchestrates the gathering of detailed summaries from each component and compiles them into a single, cohesive summary.

    Parameters:
        vex_binary_context (VexBinaryContext): An instance of VexBinaryContext that contains all the relevant data and
                                               methods to analyze the binary file.

    Return:
        str: A comprehensive summary of the binary, formatted as a string.

    Objective:
        In this lab, students will synthesize a whole-binary summary by analyzing and integrating the binary's strings,
        imports, and functions, learning to navigate and adapt to token limits in language model processing via random
        subsampling. The task will require students to instruct the model to generate a detailed HTML report that
        encapsulates the binary's behavior, intent, and purpose within a firmware/software analysis context.

    Steps:
        1. Implement the 'summarize_strings' function: Students should start by extracting and summarizing the strings found within the binary.
           This involves identifying meaningful strings that could indicate the binary's functionality, logging mechanisms, or embedded secrets.

        2. Implement the 'summarize_imports' function: Next, students should analyze the binary's imports to understand its dependencies,
           external libraries, or APIs it interacts with. This step helps in identifying the binary's external communication and functionality.

        3. Implement the 'summarize_functions' function: This involves identifying and summarizing the binary's functions.
           Students should focus on function names, their purposes, and how they contribute to the binary's overall behavior.

        4. Chain the prompts: Combine summarize_strings, summarize_imports, and summarize_functions into a prompt instructing
          the LLM to produce a comprehensive, HTML-formatted report.

    """

    summary = ""

    ### YOUR CODE HERE ###


    ### END YOUR CODE #




def main():
    parser = argparse.ArgumentParser(description="Summarize the Binary.")
    parser.add_argument("-f", "--bcc_file_path",
                        type=str,
                        default=TARGET_BCC,
                        dest="bcc_file_path",
                        help="Path to the BCC (Binary Code Context) file.")
    parser.add_argument("--summarize_imports", action="store_true",
                        help="Summarize the imports in the binary.")
    parser.add_argument("--summarize_strings", action="store_true",
                        help="Summarize the strings in the binary.")
    parser.add_argument("--summarize_functions", action="store_true",
                        help="Summarize the functions in the binary.")
    parser.add_argument("--summarize_binary", action="store_true",
                        help="Summarize the entire binary.")

    args = parser.parse_args()
    vex_binary_context = VexBinaryContext.load_from_file(args.bcc_file_path)

    if args.summarize_imports:
        summary_data = summarize_imports(vex_binary_context)
    elif args.summarize_strings:
        summary_data = summarize_strings(vex_binary_context)
    elif args.summarize_functions:
        summary_data = summarize_functions(vex_binary_context)
    else:
        args.summarize_binary = True
        summary_data = summarize_binary(vex_binary_context)

    logger.info(summary_data)

    # Write the summary to an HTML file
    if args.summarize_binary:

        # Save the rendered HTML to a file
        output_file_path = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                        'labs',
                                        'lab9',
                                        'lab_9_3',
                                        'output',
                                        f'binary_summary_{vex_binary_context.name}_{vex_binary_context.sha256_hash}.html')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as file:
            file.write(summary_data)

        print(f"HTML report generated: {output_file_path}")

if __name__ == "__main__":
    main()