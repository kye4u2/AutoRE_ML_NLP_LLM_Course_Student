import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Any, Optional

from lab_common.common import ROOT_PROJECT_FOLDER_PATH

LAB_11_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab11")

# DEFAULT_API_CALL_TRACE_JSON_PATH = os.path.join(LAB_10_DATASET,
#                                     "benign",
#                                     "2b364c5052c0c8f12f68907551655616d74f2e89f94ad791a93e58c9fd1c8f6c",
#                                     "api_call_trace.json")

DEFAULT_API_CALL_TRACE_JSON_PATH = os.path.join(LAB_11_DATASET,
                                    "suspicious",
                                    "5b246c5d90be3bfcbfcc1eb625a064dcdcb18bd0fe86662a7ed949155e884fae",
                                    "api_call_trace.json")

cuckoo_report_file_path = os.path.join(LAB_11_DATASET,
                                       "suspicious",
                                       "5b246c5d90be3bfcbfcc1eb625a064dcdcb18bd0fe86662a7ed949155e884fae",
                                       "cuckoo.json")


@dataclass
class Argument:
    name: str
    value: Any

    def __str__(self):
        return f"{self.name}: {self.value}"


@dataclass
class APICall:
    category: str
    status: str
    return_value: str
    timestamp: str
    repeated: int
    api: str
    arguments: List[Argument]

    def __str__(self):
        arguments_str = ', '.join(str(arg) for arg in self.arguments)
        return f"Category: {self.category}, API: {self.api}, Timestamp: {self.timestamp}, Status: {self.status}, Return: {self.return_value}, Arguments: [{arguments_str}]"


@dataclass
class APICallTrace:
    api_calls: List[APICall] = field(default_factory=list)

    def __str__(self):
        api_calls_str = '\n'.join(str(api_call) for api_call in self.api_calls)
        return f"APICallsCollection:\n{api_calls_str}"

    def to_json(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self, file, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(file_path: str) -> 'APICallTrace':
        with open(file_path, 'r') as file:
            data = json.load(file)
        api_calls = [APICall(**call) for call in data.get('api_calls', [])]
        return APICallTrace(api_calls)


def load_api_calls_from_file(file_path: str) -> Optional[APICallTrace]:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"An error occurred: {e}")
        return None

    if 'behavior' not in data or 'processes' not in data['behavior'] or not data['behavior']['processes']:
        print("The data does not contain 'behavior' or 'processes' information.")
        return None

    api_calls = []

    for process in data['behavior']['processes']:
        for call in process.get('calls', []):
            arguments = [Argument(name=arg['name'], value=arg['value']) for arg in call.get('arguments', [])]
            api_call = APICall(
                category=call.get('category', ''),
                status=call.get('status', ''),
                return_value=call.get('return', ''),
                timestamp=call.get('timestamp', ''),
                repeated=call.get('repeated', 0),
                api=call.get('api', ''),
                arguments=arguments
            )
            api_calls.append(api_call)

    return APICallTrace(api_calls)


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Load and display api call trace from a JSON file.")
    parser.add_argument("-f", "--file_path",
                        type=str,
                        default=DEFAULT_API_CALL_TRACE_JSON_PATH,
                        help="The file path to the api_call_trace.json")

    # Parse the arguments
    args = parser.parse_args()

    api_calls_collection = APICallTrace.from_json(args.file_path)
    print(api_calls_collection)


if __name__ == "__main__":
    # api_call_trace = load_api_calls_from_file(cuckoo_report_file_path)
    #
    # api_call_trace_file_path = os.path.join(os.path.dirname(cuckoo_report_file_path),
    #                                         "api_call_trace.json")
    #
    # api_call_trace.to_json(api_call_trace_file_path)
    main()

