# Instructions to generate RST Tables from Dataclasses

## Steps
1. Run the following command to generate the RST file from the dataclass file:
    ```
    python dataclass_to_rst.py <module_path> <output_file>
    ```
2. You can use `python dataclass_to_rst.py -h` to get the list of available command line arguments.
    ```
    usage: dataclass_to_rst_tables.py [-h] [--metadata [METADATA ...]] module_path output_file

    Process a dataclass and export to an RST file.

    positional arguments:
    module_path           Dotted path to the module (e.g., nvidia_tao_core.config.re_identification.default_config)
    output_file           Output file path (e.g., experiment_config.rst)

    options:
    -h, --help            show this help message and exit
    --metadata [METADATA ...]
                            Optional metadata fields to include (e.g., math_cond, required, popular, etc.)
    ```
3. The RST file will be generated in the specified output file path
4. Optional paratemers can be added to the rst file by specifying them in the command line arguments. Use `--metadata <metadata>` to add metadata to the RST file.