import argparse
import json
import os
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict


from logzero import logger

from lab_common.common import ROOT_PROJECT_FOLDER_PATH
from lab_common.labs.lab11.filebehavior import FileBehaviour

LAB_11_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab11")

DEFAULT_FILE_INFO_JSON_PATH = os.path.join(LAB_11_DATASET,
                                               "benign",
                                               "2b364c5052c0c8f12f68907551655616d74f2e89f94ad791a93e58c9fd1c8f6c",
                                               "file_info.json")

@dataclass
class AnalysisResult:
    category: Optional[str] = None
    engine_name: Optional[str] = None
    engine_version: Optional[str] = None
    result: Optional[str] = None
    method: Optional[str] = None
    engine_update: Optional[str] = None



@dataclass
class AlertContext:
    url: Optional[str] = None
    hostname: Optional[str] = None
    dest_port: Optional[int] = None
    src_port: Optional[int] = None
    dest_ip: Optional[str] = None
    src_ip: Optional[str] = None

    def __init__(self, **kwargs):
        valid_fields = {field.name for field in fields(self)}
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)



@dataclass
class CrowdsourcedIdsResult:
    rule_category: Optional[str] = None
    alert_severity: Optional[str] = None
    rule_msg: Optional[str] = None
    rule_raw: Optional[str] = None
    rule_url: Optional[str] = None
    rule_source: Optional[str] = None
    rule_id: Optional[str] = None
    alert_context: Optional[List[AlertContext]] = None
    rule_references: Optional[List[str]] = None

    def __init__(self, **kwargs):
        for field_name, value in kwargs.items():
            if field_name == 'alert_context' and isinstance(value, list):
                # Convert dictionaries to AlertContext instances
                processed_alert_context = [AlertContext(**item) if isinstance(item, dict) else item for item in value]
                setattr(self, field_name, processed_alert_context)
            else:
                setattr(self, field_name, value)

@dataclass
class FileInfo:
    authentihash: Optional[str] = None
    available_tools: Optional[List] = None
    context_attributes: Optional[Dict] = None
    creation_date: Optional[datetime] = None
    detectiteasy: Optional[Dict] = None
    downloadable: Optional[bool] = None
    error: Optional[None] = None
    exiftool: Optional[Dict] = None
    first_submission_date: Optional[datetime] = None
    id: Optional[str] = None
    last_analysis_date: Optional[datetime] = None
    last_analysis_results: Optional[Dict[str, AnalysisResult]] = field(default_factory=dict)
    last_analysis_stats: Optional[Dict] = None
    last_modification_date: Optional[datetime] = None
    last_submission_date: Optional[datetime] = None
    magic: Optional[str] = None
    main_icon: Optional[Dict] = None
    md5: Optional[str] = None
    names: Optional[List] = None
    pe_info: Optional[Dict] = None
    popular_threat_classification: Optional[Dict] = None
    relationships: Optional[Dict] = None
    reputation: Optional[int] = None
    sha1: Optional[str] = None
    sha256: Optional[str] = None
    type: Optional[str] = None

    # Additional attributes
    type_description: Optional[str] = None
    tlsh: Optional[str] = None
    vhash: Optional[str] = None
    type_tags: Optional[Dict] = None
    threat_severity: Optional[Dict] = None
    type_tag: Optional[str] = None
    times_submitted: Optional[int] = None
    total_votes: Optional[Dict] = None
    size: Optional[int] = None
    meaningful_name: Optional[str] = None
    downloadable: Optional[bool] = None
    trid: Optional[str] = None
    type_extension: Optional[str] = None
    tags: Optional[List] = None
    unique_sources: Optional[int] = None
    ssdeep: Optional[str] = None
    md5: Optional[str] = None
    pe_info: Optional[Dict] = None
    magic: Optional[str] = None
    last_analysis_stats: Optional[Dict] = None
    signature_info: Optional[Dict] = None
    sigma_analysis_results: Optional[List[Dict]] = None
    sigma_analysis_summary: Optional[Dict] = None
    sandbox_verdicts: Optional[Dict] = None
    sigma_analysis_stats: Optional[Dict] = None
    crowdsourced_ids_stats: Optional[Dict] = None
    crowdsourced_ids_results: Optional[List[CrowdsourcedIdsResult]] = field(default_factory=list)
    crowdsourced_yara_results: Optional[List[Dict]] = None
    first_seen_itw_date: Optional[int] = None



    def __str__(self):
        return self.to_json()

    def __init__(self, **kwargs):
        valid_attribute_names = {field.name for field in fields(self)}
        for field_name, value in kwargs.items():
            if field_name in ('last_analysis_results', 'crowdsourced_ids_results'):
                # Special handling for last_analysis_results
                if field_name == 'last_analysis_results':
                    processed_dict = {k: AnalysisResult(**v) if isinstance(v, dict) else v for k, v in value.items()}
                    setattr(self, field_name, processed_dict)
                elif field_name == 'crowdsourced_ids_results' and isinstance(value, list):
                    # Handle crowdsourced_ids_results specifically
                    processed_list = [CrowdsourcedIdsResult(**item) if isinstance(item, dict) else item for item in value]
                    setattr(self, field_name, processed_list)
            elif field_name in valid_attribute_names:
                setattr(self, field_name, value)


    def to_json(self):
        # Note that the __str__ method returns a JSON string
        def serialize_nonstandard_types(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, FileBehaviour):
                return str(obj)
            raise TypeError("Type not serializable")

        attributes = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                if isinstance(value, list) and all(isinstance(item, FileBehaviour) for item in value):
                    attributes[field] = [str(item) for item in value]
                else:
                    attributes[field] = value

        return json.dumps(attributes, default=serialize_nonstandard_types, indent=4)

    @classmethod
    def from_json(cls, json_str):

        data = json.loads(json_str)

        virus_total_file_info = FileInfo(**data)

        # Instantiate VirusTotalFileInfo with the data
        return virus_total_file_info

    @classmethod
    def from_json_file(cls, file_path: str):
        with open(file_path, 'r') as file:
            file_info_json = json.load(file)
        return FileInfo.from_json(json.dumps(file_info_json))


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Load and display file information from a JSON file.")
    parser.add_argument("-f", "--file_path",
                        type=str,
                        default=DEFAULT_FILE_INFO_JSON_PATH,
                        help="The file path to the file_info.json")

    # Parse the arguments
    args = parser.parse_args()


    try:
        # Load the JSON data from the specified file
        with open(args.file_path, 'r') as file:
            file_info_json = json.load(file)

        # Create a FileBehaviour instance from the JSON data
        file_behavior = FileInfo.from_json(json.dumps(file_info_json))

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