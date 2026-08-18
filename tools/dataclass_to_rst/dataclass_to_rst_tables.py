# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Original source taken from https://github.com/NVIDIA/NeMo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility tool to generate an RST tables from a dataclass."""

import argparse
from dataclasses import fields, is_dataclass
from collections import deque
from tabulate import tabulate
from nvidia_tao_core.api_utils.dataclass2json_converter import import_module_from_path

def format_metadata(metadata):
    """
    Formats the metadata of a field.

    :param metadata: Metadata dictionary of a field.
    :return: Formatted metadata dictionary.
    """
    return {k: v for k, v in metadata.items()}

def extract_field_data(field):
    """
    Extracts data from a dataclass field.

    :param field: A field from a dataclass.
    :return: Dictionary containing the field's name, type, and metadata.
    """
    field_data = {
        "name": field.name,
        "type": str(field.type),
        "metadata": format_metadata(field.metadata),
    }
    return field_data

def process_dataclass(dataclass, q, all_metadata_keys):
    """
    Processes a dataclass to extract and format its fields into an RST table.

    :param dataclass: The dataclass to process.
    :param q: A deque for managing nested dataclasses.
    :param all_metadata_keys: List of metadata keys to include.
    :return: Formatted RST table string.
    """
    field_names = ["Field"] + all_metadata_keys
    table_data = []

    for field in fields(dataclass):
        field_data = extract_field_data(field)
        row_data = [f":code:`{field_data['name']}`"]
        for key in all_metadata_keys:
            row_data.append(field_data["metadata"].get(key, ""))
        table_data.append(row_data)

        if is_dataclass(field.type):
            q.append(field.type)

    table = tabulate(table_data, headers=field_names, tablefmt="grid")
    return table + "\n\n"

def iterate_dataclass_fields(dataclass, all_metadata_keys):
    """
    Iterates over the fields of a dataclass and its nested dataclasses to create a full RST table.

    :param dataclass: The root dataclass to process.
    :param all_metadata_keys: List of metadata keys to include.
    :return: Full formatted RST table string for all dataclass fields.
    """
    q = deque()
    q.append(dataclass)

    table = ""
    while len(q) > 0:
        current_dataclass = q.popleft()
        table_header = f"\n{current_dataclass.__name__} Fields"
        header_line = "=" * len(table_header)
        table += f"{table_header}\n{header_line}\n\n"
        table += process_dataclass(current_dataclass, q, all_metadata_keys)
    return table

def write_to_rst_file(dataclass, file_path, all_metadata_keys):
    """
    Writes the RST table of a dataclass's fields to a file.

    :param dataclass: The dataclass to process.
    :param file_path: Path to the output RST file.
    :param all_metadata_keys: List of metadata keys to include.
    """
    rst_content = iterate_dataclass_fields(dataclass, all_metadata_keys)
    with open(file_path, "w") as file:
        file.write(rst_content)

def main():
    """
    Main function to parse command-line arguments and generate an RST file for a dataclass.

    Command-line Arguments:
        module_path (str): Dotted path to the module (e.g., nvidia_tao_core.config.re_identification.default_config).
        output_file (str): Output file path (e.g., experiment_config.rst).
        --metadata (str): Optional metadata fields to include.

    """
    parser = argparse.ArgumentParser(description='Process a dataclass and export to an RST file.')
    parser.add_argument('module_path', type=str, help='Dotted path to the module (e.g., nvidia_tao_core.config.re_identification.default_config)')
    parser.add_argument('output_file', type=str, help='Output file path (e.g., experiment_config.rst)')
    parser.add_argument('--metadata', type=str, nargs='*', help='Optional metadata fields to include (e.g., math_cond, required, popular, etc.)')

    args = parser.parse_args()

    all_metadata_keys = [
        "value_type",
        "description",
        "default_value",
        "valid_min",
        "valid_max",
        "valid_options",
        "automl_enabled",
    ]

    if args.metadata:
        all_metadata_keys.extend(args.metadata)

    print("Importing module...")
    imported_module = import_module_from_path(args.module_path)
    print(f"Writing to {args.output_file}...")
    write_to_rst_file(imported_module.ExperimentConfig, args.output_file, all_metadata_keys)
    print("Done!")

if __name__ == "__main__":
    main()
