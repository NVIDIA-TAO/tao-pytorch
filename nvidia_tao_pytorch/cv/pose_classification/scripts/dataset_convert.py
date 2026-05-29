# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert pose data from deepstream-bodypose-3d to skeleton arrays."""
import os
import numpy as np

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import check_and_create, update_results_dir
from nvidia_tao_pytorch.config.pose_classification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import load_json_from_file, write_np_to_file


def create_data_numpy(data_numpy, pose_sequence, frame_start, frame_end, pose_type, num_joints, sequence_length_max):
    """
    Create a NumPy array for output.

    This function takes a pose sequence and converts it into a NumPy array for output. The output array
    has the shape (1, joint_dim, sequence_length_max, num_joints, 1), where joint_dim is 2 or 3 depending
    on the pose type (2D or 3D), sequence_length_max is the maximum sequence length, and num_joints is the
    number of joints in the pose.

    Args:
        data_numpy (numpy.ndarray or None): The existing NumPy array to concatenate the sequence with. If None,
            a new array will be created.
        pose_sequence (list): The pose sequence to convert.
        frame_start (int): The starting frame index.
        frame_end (int): The ending frame index.
        pose_type (str): The type of pose data ("2dbp", "3dbp", "25dbp").
        num_joints (int): The number of joints in the pose.
        sequence_length_max (int): The maximum sequence length.

    Returns:
        numpy.ndarray: The NumPy array containing the converted pose sequence.
    """
    joint_dim = 3
    if pose_type == "2dbp":
        joint_dim = 2
    sequence = np.zeros((1, joint_dim, sequence_length_max, num_joints, 1), dtype="float32")
    f = 0
    for frame in range(frame_start, frame_end):
        for j in range(num_joints):
            for d in range(joint_dim):
                sequence[0, d, f, j, 0] = pose_sequence[frame][j][d]
        f += 1
    if data_numpy is None:
        data_numpy = sequence
    else:
        data_numpy = np.concatenate((data_numpy, sequence), axis=0)
    return data_numpy


def run_experiment(experiment_config, key, data_path, results_dir):
    """
    Start the dataset conversion.

    This function is responsible for the main dataset conversion process. It loads the pose data from
    the deepstream-bodypose-3d JSON file, extracts the pose sequences, applies normalization and preprocessing,
    and saves the resulting skeleton arrays as NumPy files.

    Args:
        experiment_config (dict): The experiment configuration.
        key (str): The encryption key for data encryption.
        data_path (str): The path to the deepstream-bodypose-3d JSON file.
        results_dir (str): The directory to save the converted dataset.

    Raises:
        AssertionError: If the number of frames in a batch does not match the length of the batches.
        KeyError: If the required pose data ("pose3d" or "pose25d") is not found in the input data.
        NotImplementedError: If the pose type specified in the experiment configuration is not supported.
        Exception: If any error occurs during the conversion process.
    """
    # create output directory
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting Pose classification dataset convert")

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    # load pose data from deepstream-bodypose-3d
    pose_data = load_json_from_file(data_path)

    # extract sequences from pose data and apply normalization
    pose_type = experiment_config["dataset_convert"]["pose_type"]
    num_joints = experiment_config["dataset_convert"]["num_joints"]
    input_width = float(experiment_config["dataset_convert"]["input_width"])
    input_height = float(experiment_config["dataset_convert"]["input_height"])
    focal_length = experiment_config["dataset_convert"]["focal_length"]
    pose_sequences = {}
    for batch in pose_data:
        assert batch["num_frames_in_batch"] == len(batch["batches"]), f"batch[\"num_frames_in_batch\"] "\
            f"{batch['num_frames_in_batch']} does not match len(batch[\"batches\"]) {len(batch['batches'])}."
        for frame in batch["batches"]:
            for person in frame["objects"]:
                object_id = person["object_id"]
                if object_id not in pose_sequences.keys():
                    pose_sequences[object_id] = []
                poses = []
                if pose_type == "3dbp":
                    if "pose3d" not in list(person.keys()):
                        raise KeyError("\"pose3d\" not found in input data. "
                                       "Please run deepstream-bodypose-3d with \"--publish-pose pose3d\".")
                    assert num_joints == len(person["pose3d"]) // 4, f"The num_joints should be "\
                        f"{len(person['pose3d']) // 4}. Got {num_joints}."
                    for j in range(num_joints):
                        if person["pose3d"][j * 4 + 3] == 0.0:
                            poses.append([0.0, 0.0, 0.0])
                            continue
                        x = (person["pose3d"][j * 4 + 0] - person["pose3d"][0]) / focal_length
                        y = (person["pose3d"][j * 4 + 1] - person["pose3d"][1]) / focal_length
                        z = (person["pose3d"][j * 4 + 2] - person["pose3d"][2]) / focal_length
                        poses.append([x, y, z])
                elif pose_type in ("25dbp", "2dbp"):
                    if "pose25d" not in list(person.keys()):
                        raise KeyError("\"pose25d\" not found in input data. "
                                       "Please run deepstream-bodypose-3d with \"--publish-pose pose25d\".")
                    assert num_joints == len(person["pose25d"]) // 4, f"The num_joints should be "\
                        f"{len(person['pose25d']) // 4}. Got {num_joints}."
                    for j in range(num_joints):
                        if person["pose25d"][j * 4 + 3] == 0.0:
                            if pose_type == "25dbp":
                                poses.append([0.0, 0.0, 0.0])
                            else:
                                poses.append([0.0, 0.0])
                            continue
                        x = person["pose25d"][j * 4 + 0] / input_width - 0.5
                        y = person["pose25d"][j * 4 + 1] / input_height - 0.5
                        z = person["pose25d"][j * 4 + 2]
                        if pose_type == "25dbp":
                            poses.append([x, y, z])
                        else:
                            poses.append([x, y])
                else:
                    raise NotImplementedError(f"Pose type {pose_type} is not supported.")
                pose_sequences[object_id].append(poses)
    print(f"Number of objects: {len(pose_sequences.keys())}")
    status_logging.get_status_logger().kpi = {"Number of objects": len(pose_sequences.keys())}
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.RUNNING,
    )

    # create output of pose arrays
    sequence_length_max = experiment_config["dataset_convert"]["sequence_length_max"]
    sequence_length_min = experiment_config["dataset_convert"]["sequence_length_min"]
    sequence_length = experiment_config["dataset_convert"]["sequence_length"]
    sequence_overlap = experiment_config["dataset_convert"]["sequence_overlap"]
    step = int(sequence_length * sequence_overlap)
    for object_id in pose_sequences.keys():
        data_numpy = None
        frame_start = 0
        sequence_count = 0
        while len(pose_sequences[object_id]) - frame_start >= sequence_length_min:
            frame_end = frame_start + sequence_length
            if len(pose_sequences[object_id]) - frame_start < sequence_length:
                frame_end = len(pose_sequences[object_id])
            data_numpy = create_data_numpy(data_numpy, pose_sequences[object_id], frame_start, frame_end,
                                           pose_type, num_joints, sequence_length_max)
            frame_start += step
            sequence_count += 1
        if sequence_count > 0:
            results_path = os.path.join(results_dir, "object_" + str(object_id) + ".npy")
            write_np_to_file(results_path, data_numpy)
            print(f"Saved data {data_numpy.shape} for object {object_id} at {results_path}")
            status_logging.get_status_logger().write(
                message=f"Saved data {data_numpy.shape} for object {object_id} at {results_path}",
                status_level=status_logging.Status.RUNNING
            )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """
    Run the dataset conversion process.

    This function serves as the entry point for the dataset conversion script.
    It loads the experiment specification, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    try:
        cfg = update_results_dir(cfg, task="dataset_convert")
        run_experiment(experiment_config=cfg,
                       key=cfg.encryption_key,
                       results_dir=cfg.results_dir,
                       data_path=cfg.dataset_convert.data)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully."
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
