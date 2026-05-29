# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare MonoDataset/StereoDataset splits in txt format."""
import os
from typing import List, Dict
import random
import omegaconf

from nvidia_tao_pytorch.config.depth_net.dataset import DNDatasetConvertConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import check_and_create


def glob_one_directory(directory_path: str,
                       match_patterns: List[str],
                       endswith_string: str = None) -> Dict[str, int]:
    """
    This function globs a directory path and returns all the files whose file path
    contains a match pattern.

    For instance, directory path could be: '/home/to/my/stereo/dataset'
    match pattern is a list could be ['/my/stereo'] or ['/my', '/stereo/']

    The depth of recursion of the directory is immaterial in this case.

    Args:
        directory_path (str): Path to root directory to be globbed.
        match_pattern (list): List of string patterns to be matched in each globbed string.
        endswith_string (str): string pattern of image extension to be matched in each globbed string.

    Returns:
        A dictionary of each matched path as key and a value of 0. The value is used as a dummy.

    """

    def check_pattern(string, patterns):
        """ Function checks whether all the patterns listed are in the path string.
        If there are, it returns true otherwise, it returns false

        Args:
            string (str): full path to be checked
            patterns (list): full list of patterns to be checked in the full path

        Returns:
            check_split_list (bool): whether pattern exists in the full path or not.
        """
        if not isinstance(patterns, omegaconf.listconfig.ListConfig):
            raise TypeError('patterns must be a ListConfig object')
        check_split_exist = True
        for pattern in patterns:
            if pattern not in string:
                check_split_exist = False
                break
        return check_split_exist

    # we use a dictionary to save the matching files to avoid duplicates.
    matching_files = dict()
    valid_match_pattern = len(match_patterns) > 0
    valid_end_string = len(endswith_string) > 1
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if valid_end_string and (not valid_match_pattern):
                # check for only endswith strng
                match_pass = full_path.endswith(endswith_string)
            elif (not valid_end_string) and valid_match_pattern:
                # check of only match strings withing the file path name
                match_pass = check_pattern(full_path, match_patterns)
            else:
                match_pass = check_pattern(full_path, match_patterns) and \
                    full_path.endswith(endswith_string)
            if match_pass:
                matching_files.update({full_path: 0})
    return matching_files


def write_files(filename: str, write_mode: str, data: dict):
    """
    This function writes data names to disk

    Args:
        filename (str): The path to write the split files.
        write_mode (str): Could be 'a', 'r' or 'w'. Same mode used in python inbuilt open() function.
        data (dict): Dictionary containing left, right or disparity files.

    Returns:
        None.

    """
    if write_mode == 'a':
        if not os.path.exists(filename):
            raise (FileExistsError(f'the file {filename} does not exist'))

    with open(filename, write_mode) as f:
        if len(data) == 1:
            # only left images in the data:
            for items in data['left']:
                f.write(items + '\n')

        elif len(data) == 2 and 'right' in data.keys():
            # write both items the data:
            for i in range(len(data['left'])):
                f.write(f'{data['left'][i]} {data['right'][i]}' + '\n')

        elif len(data) == 2 and 'disparity' in data.keys():
            for i in range(len(data['left'])):
                f.write(f'{data['left'][i]} {data['disparity'][i]}' + '\n')

        elif len(data) == 3:
            for i in range(len(data['left'])):
                f.write(f'{data['left'][i]} {data['right'][i]} {data['disparity'][i]}' + '\n')

        elif len(data) == 4:
            for i in range(len(data['left'])):
                f.write(f'{data['left'][i]} {data['right'][i]} {data['disparity'][i]} {data['mask'][i]}' + '\n')
    f.close()


def train_val_test_split(all_files, split_ratio):
    """
    This function splits all files into train and validation.
    If there is no split (split=0 or split=1), it set all the files as test.

    Args:
        all_files (list): list of all files paths
        split_ration (float): take a real number in [0, 1]

    Returns:
       output_files (dc)
    """
    output_files = {}
    random.shuffle(all_files)
    if split_ratio == 1.0 or split_ratio == 0.0:
        output_files['test'] = {'left': all_files}
    else:
        length_train = int(float(len(all_files)) * split_ratio)
        output_files['train'] = {'left': all_files[0: length_train]}
        output_files['val'] = {'left': all_files[length_train:]}
    return output_files


def create_split(experiment_config, results_dir):
    """
    This functions creates a split for train, val or test datasets.

    To avoid data mix-up, we would only glob the left files and then use replacement patterns for other paths
    Hence, we assume all training files for a certain dataset have the same root. For instance:
    for a sample stereo dataset, called stereo_data.
    the left files coult be stored at stereo_data/sub_dir/left/images.png
    the right files could be stored at stereo_data/sub_dir/right/images.png
    the disparity files could be stored at stereo_data/sub_dir/disparity/images.pfm

    For datasets with multiple sub directories, we would attempt the walk the root directory
    for one level of subdirectories.

    Args:
        experiment_config (OmegaConf.Dict): cmdline args
        results_dir (str): folder path to save split file

    Returns:
        None

    """
    # Extract the yaml configs.
    data_root = experiment_config["data_root"]
    split_ratio = experiment_config["split_ratio"]
    image_dir_pattern = experiment_config["image_dir_pattern"]
    depth_dir_pattern = experiment_config["depth_dir_pattern"]
    right_dir_pattern = experiment_config["right_dir_pattern"]
    nocc_dir_pattern = experiment_config["nocc_dir_pattern"]

    depth_extension = experiment_config["depth_extension"]
    image_extension = experiment_config["image_extension"]
    nocc_mask_extension = experiment_config["nocc_extension"]

    all_files = []

    data_root_sub_dirs = os.listdir(data_root)

    for sub_dir in data_root_sub_dirs:
        files = glob_one_directory(os.path.join(data_root, sub_dir),
                                   image_dir_pattern, endswith_string=image_extension)
        all_files.extend(list(files.keys()))

    output_file = train_val_test_split(all_files, split_ratio)

    for item in output_file:
        if len(right_dir_pattern) > 0:
            output_file[item]['right'] = []
            for x in output_file[item]['left']:
                new = x
                for i in range(len(image_dir_pattern)):
                    new = new.replace(image_dir_pattern[i], right_dir_pattern[i])
                output_file[item]['right'].append(new)

        if len(nocc_dir_pattern) > 0:
            output_file[item]['mask'] = []
            for x in output_file[item]['left']:
                new = x
                for i in range(len(image_dir_pattern)):
                    new = new.replace(image_dir_pattern[i], nocc_dir_pattern[i])
                output_file[item]['mask'].append(new)
            output_file[item]['mask'] = [x.replace(image_extension, nocc_mask_extension)
                                         for x in output_file[item]['mask']]

        if len(depth_dir_pattern) > 0:
            output_file[item]['disparity'] = []
            for x in output_file[item]['left']:
                new = x
                for i in range(len(image_dir_pattern)):
                    new = new.replace(image_dir_pattern[i], depth_dir_pattern[i])
                output_file[item]['disparity'].append(new)
            output_file[item]['disparity'] = [x.replace(image_extension, depth_extension)
                                              for x in output_file[item]['disparity']]

        # create the split file if it doesn't exist [train, val or test.txt]
        write_path = os.path.join(results_dir, item + '.txt')
        if not os.path.isfile(write_path):
            with open(write_path, "w"):
                pass

        # write all the files to the split (left | right | disparity)
        write_files(write_path, 'a', output_file[item])


def run_experiment(experiment_config, results_dir):
    """Start the Data Converter."""
    check_and_create(results_dir)
    create_split(experiment_config, results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting DepthNet dataset convert"
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="convert", schema=DNDatasetConvertConfig
)
def main(cfg: DNDatasetConvertConfig) -> None:
    """Run the convert dataset process."""
    try:
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully"
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    main()
