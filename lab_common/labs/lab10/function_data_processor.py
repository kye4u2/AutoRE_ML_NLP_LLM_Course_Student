import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import List, Optional

from logzero import setup_logger

logger = setup_logger(name=Path(__file__).stem, logfile=f"{Path(__file__).stem}.log", level="INFO")

FUNCTION_SUMMARY_SYSTEM_PROMPT = """
You are an experienced reverse engineer tasked with summarizing the provided code by first stating its overall goal or purpose.
Then, highlight key behaviors, functionalities, or techniques that are particularly relevant.
Aim for two sentences, but use up to four if necessary for complex implementations with extensive capabilities.
Avoid step-by-step descriptions and focus on the big picture and significant details. 
Do not explicitly state the function name; instead, infer its behavior and purpose from the implementation.
"""

FUNCTION_SUMMARY_INSTRUCTION = "Please summarize the following code:\n\n{code}"

@dataclass
class FunctionData:
    binary_id: int
    binary_name: str
    binary_sha256: str
    function_name: str
    function_address: int
    decompiled_code: str
    summary: Optional[str] = None

class FunctionDataSet:
    def __init__(self, data: Optional[List[FunctionData]] = None):
        self._data = data if data is not None else []

    @classmethod
    def load_from_jsonl(cls, jsonl_file: str) -> "FunctionDataSet":
        """
        Load function data from a JSONL file into a FunctionDataSet instance.
        Existing summaries are preserved.
        """
        data_list = []
        file_path = Path(jsonl_file)
        if not file_path.exists():
            logger.error(f"File not found: {jsonl_file}")
            raise FileNotFoundError(f"File not found: {jsonl_file}")

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    function_data = FunctionData(
                        binary_id=entry["binary_id"],
                        binary_name=entry["binary_name"],
                        binary_sha256=entry["binary_sha256"],
                        function_name=entry["function_name"],
                        function_address=entry["function_address"],
                        decompiled_code=entry["decompiled_code"],
                        summary=entry.get("summary")
                    )
                    data_list.append(function_data)
        return cls(data_list)

    def to_alpaca(
        self,
        instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        """
        Convert the dataset of FunctionData objects to the Alpaca format.

        Parameters:
            instruction (Optional[str]): Template for the instruction field.
                Defaults to: "Summarize the function '{function_name}' from binary '{binary_name}'."
            system_prompt (Optional[str]): Template for the system prompt.
                Defaults to: "Binary ID: {binary_id}, SHA256: {binary_sha256}, Function Address: {function_address}"

        Returns:
            List[dict]: A list of dictionaries formatted for Alpaca.
        """
        if instruction is None:
            instruction = FUNCTION_SUMMARY_INSTRUCTION
        if system_prompt is None:
            system_prompt = FUNCTION_SUMMARY_SYSTEM_PROMPT

        alpaca_data = []
        for func in self.data:
            alpaca_data.append({
                "instruction": instruction,
                "input": func.decompiled_code,
                "output": func.summary if func.summary is not None else "",
                "system": system_prompt
            })
        return alpaca_data

    def write_alpaca(
        self,
        file_path: str,
        instruction_template: Optional[str] = None,
        system_template: Optional[str] = None,
    ) -> None:
        """
        Write the dataset in Alpaca format to a file.

        Parameters:
            file_path (str): Path to the output file.
            instruction_template (Optional[str]): Template for the instruction field.
            system_template (Optional[str]): Template for the system prompt.
        """
        alpaca_data = self.to_alpaca(instruction_template, system_template)
        with open(file_path, "w") as f:
            json.dump(alpaca_data, f, indent=2)
        logger.info(f"Alpaca formatted data written to {file_path}")

    def write_jsonl(self, file_path: str) -> None:
        """
        Write the dataset to a JSONL file.

        Parameters:
            file_path (str): Path to the output JSONL file.
        """
        path = Path(file_path)
        with open(path, "w") as f:
            for func in self.data:
                # asdict converts the dataclass instance to a dictionary
                f.write(json.dumps(asdict(func)) + "\n")
        logger.info(f"JSONL data written to {file_path}")

    @property
    def data(self) -> List[FunctionData]:
        return self._data




def main():
    parser = argparse.ArgumentParser(
        description="Process function data JSONL file, count functions with summaries, and optionally write Alpaca formatted data to disk."
    )
    parser.add_argument(
        "-i","--input_jsonl_file",
        type=str,
        help="Path to the input JSONL file containing function data."
    )
    parser.add_argument(
        "-w","--write_alpaca",
        action="store_true",
        help="If specified, write the Alpaca formatted data to disk."
    )
    parser.add_argument(
        "--alpaca_output",
        type=str,
        default=None,
        help="Path to the output file for Alpaca formatted data. If not provided, defaults to the input file's folder with the base name and a '_alpaca.jsonl' postfix."
    )
    args = parser.parse_args()

    try:
        dataset = FunctionDataSet.load_from_jsonl(args.input_jsonl_file)
    except FileNotFoundError as e:
        logger.error(e)
        return

    total_functions = len(dataset.data)
    functions_with_summary = sum(1 for func in dataset.data if func.summary and len(func.summary.strip())>0)

    print(f"Total functions loaded: {total_functions}")
    print(f"Functions with summaries: {functions_with_summary}")

    if args.write_alpaca:
        # Determine the Alpaca output file path
        input_path = Path(args.input_jsonl_file)
        if args.alpaca_output:
            alpaca_output_path = Path(args.alpaca_output)
        else:
            alpaca_output_path = input_path.parent / f"{input_path.stem}_alpaca.json"

        # Write the Alpaca formatted data to disk
        dataset.write_alpaca(str(alpaca_output_path))
        print(f"Alpaca formatted data written to: {alpaca_output_path}")


if __name__ == "__main__":
    main()
