import argparse
import json
import os
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict

from logzero import logger

from lab_common.common import ROOT_PROJECT_FOLDER_PATH

LAB_11_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab11")

DEFAULT_FILE_BEHAVIOR_JSON_PATH = os.path.join(LAB_11_DATASET,
                                               "benign",
                                               "2b364c5052c0c8f12f68907551655616d74f2e89f94ad791a93e58c9fd1c8f6c",
                                               "file_behavior.json")


@dataclass
class DNSLookup:
    hostname: Optional[str] = None
    resolved_ips: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Process:
    children: List['Process'] = field(default_factory=list)
    name: Optional[str] = None
    process_id: Optional[str] = None
    time_offset: Optional[int] = None

    def __str__(self):
        # Convert the dataclass to a dictionary
        data = asdict(self)
        # Return a pretty-printed JSON string
        return json.dumps(data, indent=4)

    def to_dict(self):
        return {
            "children": [child.to_dict() for child in self.children],
            "name": self.name,
            "process_id": self.process_id,
            "time_offset": self.time_offset
        }

    @classmethod
    def from_json(cls, process_json_str):
        data = json.loads(process_json_str)
        return cls(**data)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


@dataclass
class FileBehaviour:
    analysis_date: Optional[int] = None
    behash: Optional[str] = None
    calls_highlighted: Optional[List[str]] = field(default_factory=list)
    command_executions: Optional[List[str]] = field(default_factory=list)
    dns_lookups: List[DNSLookup] = field(default_factory=list)
    files_opened: Optional[List[str]] = field(default_factory=list)
    files_written: Optional[List[str]] = field(default_factory=list)
    files_deleted: Optional[List[str]] = field(default_factory=list)
    files_attribute_changed: Optional[List[str]] = field(default_factory=list)
    has_html_report: Optional[bool] = None
    has_evtx: Optional[bool] = None
    has_memdump: Optional[bool] = None
    has_pcap: Optional[bool] = None
    hosts_file: Optional[str] = None
    ids_alerts: Optional[List[Dict[str, any]]] = field(default_factory=list)
    ip_traffic: Optional[List[Dict[str, any]]] = field(default_factory=list)
    last_modification_date: Optional[datetime] = None
    processes_tree: List[Process] = field(default_factory=list)
    processes_terminated: Optional[List[str]] = field(default_factory=list)
    processes_killed: Optional[List[str]] = field(default_factory=list)
    processes_injected: Optional[List[str]] = field(default_factory=list)
    sandbox_name: Optional[str] = None
    services_opened: Optional[List[str]] = field(default_factory=list)
    services_created: Optional[List[str]] = field(default_factory=list)
    services_started: Optional[List[str]] = field(default_factory=list)
    services_stopped: Optional[List[str]] = field(default_factory=list)
    services_deleted: Optional[List[str]] = field(default_factory=list)
    services_bound: Optional[List[str]] = field(default_factory=list)
    windows_searched: Optional[List[str]] = field(default_factory=list)
    windows_hidden: Optional[List[str]] = field(default_factory=list)
    mutexes_opened: Optional[List[str]] = field(default_factory=list)
    mutexes_created: Optional[List[str]] = field(default_factory=list)
    signals_observed: Optional[List[str]] = field(default_factory=list)
    invokes: Optional[List[str]] = field(default_factory=list)
    crypto_algorithms_observed: Optional[List[str]] = field(default_factory=list)
    crypto_keys: Optional[List[str]] = field(default_factory=list)
    crypto_plain_text: Optional[List[str]] = field(default_factory=list)
    text_decoded: Optional[List[str]] = field(default_factory=list)
    text_highlighted: Optional[List[str]] = field(default_factory=list)
    verdict_confidence: Optional[int] = None
    ja3_digests: Optional[List[str]] = field(default_factory=list)
    tls: Optional[List[Dict[str, any]]] = field(default_factory=list)
    sigma_analysis_results: Optional[List[Dict[str, any]]] = field(default_factory=list)
    signature_matches: Optional[List[Dict[str, any]]] = field(default_factory=list)
    mitre_attack_techniques: Optional[List[Dict[str, any]]] = field(default_factory=list)
    modules_loaded: Optional[List[str]] = field(default_factory=list)
    registry_keys_opened: Optional[List[str]] = field(default_factory=list)
    registry_keys_set: Optional[List[Dict[str, str]]] = field(default_factory=list)
    registry_keys_deleted: Optional[List[str]] = field(default_factory=list)
    verdicts: Optional[List[str]] = field(default_factory=list)

    def __init__(self, **kwargs):
        valid_attribute_names = {field.name for field in fields(self)}
        for field_name, value in kwargs.items():
            # logger.debug(f"Processing attribute '{field_name}' with value '{value}'")
            if field_name in valid_attribute_names:
                setattr(self, field_name, value)
            else:
                # logger.debug(f"Unexpected attribute '{field_name}' with value '{value}'")
                pass

    def __str__(self):
        return self.to_json()

    def to_dict(self):
        # Create a dictionary representation of this dataclass instance
        data = {}
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name, None)  # Safely get the field value, default to None
            if field_value is not None and hasattr(field_value, 'to_dict'):
                data[field_name] = field_value.to_dict()  # Use to_dict for complex objects
            else:
                data[field_name] = field_value  # Use the value directly for simple fields or None

        # Return the dictionary representation
        return data

    def to_json(self):
        # Note that the __str__ method returns a JSON string
        # Create a dictionary representation of this dataclass instance
        data = {}
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name, None)  # Safely get the field value, default to None
            if field_value is not None and hasattr(field_value, 'to_dict'):
                data[field_name] = field_value.to_dict()  # Use to_dict for complex objects
            else:
                data[field_name] = field_value  # Use the value directly for simple fields or None

        # Convert the dictionary to a JSON string with indentation for readability
        return json.dumps(data, indent=4, cls=CustomJSONEncoder)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)

        # Handle special deserialization logic for complex fields if necessary
        # For example, if DNSLookup or Process have their own from_json methods:
        if 'dns_lookups' in data and data['dns_lookups']:
            data['dns_lookups'] = [DNSLookup(**d) for d in data['dns_lookups']]
        if 'processes_tree' in data and data['processes_tree']:
            data['processes_tree'] = [Process(**p) for p in data['processes_tree']]

        # # Convert datetime fields back to datetime objects
        # if 'last_modification_date' in data and data['last_modification_date']:
        #     data['last_modification_date'] = datetime.fromisoformat(data['last_modification_date'])

        return cls(**data)


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Load and display file behavior from a JSON file.")
    parser.add_argument("-f", "--file_path",
                        type=str,
                        default=DEFAULT_FILE_BEHAVIOR_JSON_PATH,
                        help="The file path to the file_behavior.json")

    # Parse the arguments
    args = parser.parse_args()

    try:
        # Load the JSON data from the specified file
        with open(args.filepath, 'r') as file:
            file_behavior_json = json.load(file)

        # Create a FileBehaviour instance from the JSON data
        file_behavior = FileBehaviour.from_json(json.dumps(file_behavior_json))

        # Log the file behavior data
        logger.info(file_behavior)

    except FileNotFoundError:
        logger.error(f"File not found: {args.file_path}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON file")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
