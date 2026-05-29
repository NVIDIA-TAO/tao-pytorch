# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action recognition dataset."""
import os
from PIL import Image
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.action_recognition.dataloader.frame_sampler import test_interval_sample


def load_image(img_path):
    """Load an image from the specified file path and handle corrupted images.

    This function attempts to load an image from the given file path. If the image
    is corrupted or cannot be read, an appropriate error message will be displayed,
    and the function will return None.

    Args:
        img_path (str): The path to the image file.

    Returns:
        Image object or None: The loaded image if successful, or None if the image
                              is corrupted or cannot be read.

    Raises:
        OSError: If the specified file path does not exist.
    """
    img = Image.open(img_path)
    try:
        img.load()
    except OSError:
        raise OSError("Corrupted image: {}".format(img_path))
    return img


class MotionDataset(Dataset):
    """A custom dataset class for loading and processing optical flow vector data.

    This dataset class is designed to handle optical flow vectors only, and it provides
    functionality for loading, preprocessing, and sampling the data. It can be used with
    PyTorch's DataLoader for efficient data loading and batching.
    """

    def __init__(self, sample_dict, output_shape,
                 seq_length, label_map, train_sampler,
                 test_sampler, input_type="2d",
                 mode="train", full_clip=False, transform=None,
                 clips_per_video=1, sample_clips_path=None,
                 sample_clips_ids=None):
        """Initialize the custom dataset object with the given parameters.

        Args:
            sample_dict (dict): A dictionary containing video patch paths as keys and action labels as values.
            output_shape (tuple): The desired output shape of the samples.
            seq_length (int): The sequence length of the samples.
            label_map (dict): A dictionary mapping labels to their corresponding indices.
            train_sampler (Sampler): A sampler object for the training set.
            test_sampler (Sampler): A sampler object for the test set.
            input_type (str, optional): The type of input data, either "2d" or "3d". Defaults to "2d".
            mode (str, optional): The mode of the dataset, either "train" or "val". Defaults to "train".
            full_clip (bool, optional): Whether to use the full clip or not. Defaults to False.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            clips_per_video (int, optional): The number of clips to be extracted from each video. Defaults to 1.
            sample_clips_path (str, optional): The path to the sample clips when use full_clip. Defaults to None.
            sample_clips_ids (list, optional): A list of sample clip IDs when use full_clip. Defaults to None.

        Raises:
            ValueError: If the input_type is not "2d" or "3d".
        """
        self.sample_name_list = list(sample_dict.keys())
        self.sample_dict = sample_dict
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.label_map = label_map
        self.seq_length = seq_length
        self.mode = mode
        self.input_type = input_type
        self.clips_per_video = clips_per_video
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        self.full_clip = full_clip

        if self.full_clip:
            assert (sample_clips_path is not None) and (sample_clips_ids is not None)
            self.sample_clips_path = sample_clips_path
            self.sample_clips_ids = sample_clips_ids

        self.transform = transform

    def __len__(self):
        """Return the length of the dataset"""
        if self.full_clip:
            return len(self.sample_clips_ids)

        return len(self.sample_name_list) * self.clips_per_video

    def stack_of(self, sample_path, img_ids):
        """Stack the u and v images of optical flow data.

        This method takes the file path of the sample and a list of image IDs as input.
        It loads the corresponding u and v images (horizontal and vertical components of
        the optical flow) and stacks them together to create a single stacked image.
        This stacked image can be used for further processing or analysis.

        Args:
            sample_path (str): The file path of the sample containing the u and v images.
            img_ids (list): A list of image IDs corresponding to the u and v images to be stacked.

        Returns:
            list: A list containing the u and v components of the optical flow data.

        Raises:
            FileNotFoundError: If the specified file path does not exist or the images cannot be found.
        """
        # video base path will contain u, v and rgb
        u = os.path.join(sample_path, "u")
        raw_u_list = sorted(os.listdir(u))
        v = os.path.join(sample_path, "v")
        raw_v_list = sorted(os.listdir(v))

        flow = []
        for _, idx in enumerate(img_ids):
            frame_name = raw_u_list[idx]
            assert raw_u_list[idx] == raw_v_list[idx]
            h_image = os.path.join(u, frame_name)
            v_image = os.path.join(v, frame_name)
            imgH = load_image(h_image)
            imgV = load_image(v_image)
            flow.append(imgH)
            flow.append(imgV)

        return flow

    def get_raw_frames(self, sample_path, img_ids):
        """Get raw frames of optical flow data for joint training.

        This method takes the file path of the sample and a list of image IDs as input.
        It loads the corresponding raw frames of optical flow data, which can be used
        for joint training with other modalities, such as RGB images.
        The raw frames are not preprocessed or transformed, allowing for flexibility
        in the subsequent processing pipeline.

        Args:
            sample_path (str): The file path of the sample containing the raw optical flow frames.
            img_ids (list): A list of image IDs corresponding to the raw optical flow frames to be loaded.

        Returns:
            list: A list of raw optical flow frames, where each frame is a Image object.

        Raises:
            FileNotFoundError: If the specified file path does not exist or the images cannot be found.
        """
        if self.mode in ["train", "val"]:
            action_label = self.label_map[self.sample_dict[sample_path]]

        data = self.stack_of(sample_path, img_ids)

        if self.mode == "train":
            return data, action_label
        if self.mode == "val":
            return sample_path, data, action_label

        return sample_path, data

    def get_frames(self, sample_path, img_ids):
        """Get transformed frames of optical flow data.

        This method takes the file path of the sample and a list of image IDs as input.
        It loads the corresponding frames of optical flow data and applies any specified
        transformations to them.

        Args:
            sample_path (str): The file path of the sample containing the optical flow frames.
            img_ids (list): A list of image IDs corresponding to the optical flow frames to be loaded and transformed.

        Returns:
            Torch.tensor: A tensor of transformed optical flow frames

        Raises:
            FileNotFoundError: If the specified file path does not exist or the images cannot be found.
        """
        if self.mode in ["train", "val"]:
            action_label = self.label_map[self.sample_dict[sample_path]]

        data = self.stack_of(sample_path, img_ids)

        data = self.transform(data)

        if self.input_type == "2d":
            # data = data.permute(1, 0, 2, 3)  # CTHW -> TCHW
            data = data.reshape([2 * self.seq_length, self.output_height, self.output_width])
        elif self.input_type == "3d":
            pass
        else:
            raise ValueError("Only 2d/3d input types are supported.")

        if self.mode == "train":
            return data, action_label
        if self.mode == "val":
            return sample_path, data, action_label

        return sample_path, data

    def __getitem__(self, idx):
        """__getitem__"""
        if self.full_clip:
            sample_path = self.sample_clips_path[idx]
            img_ids = self.sample_clips_ids[idx]
        else:
            idx = idx % len(self.sample_name_list)
            sample_path = self.sample_name_list[idx]
            max_sample_cnt = len(os.listdir(os.path.join(sample_path, "v")))

            if self.mode == "train":
                img_ids = self.train_sampler(max_sample_cnt, self.seq_length)
            else:
                img_ids = self.test_sampler(max_sample_cnt, self.seq_length)

        return self.get_frames(sample_path, img_ids)


class SpatialDataset(Dataset):
    """Dataset for RGB frames only"""

    def __init__(self, sample_dict, output_shape,
                 seq_length, label_map, transform,
                 train_sampler, test_sampler, input_type="2d",
                 mode="train", full_clip=False,
                 clips_per_video=1, sample_clips_path=None,
                 sample_clips_ids=None):
        """Initialize the SpatialDataset with the given parameters.

        Args:
            sample_dict (dict): A dictionary containing video patch paths as keys and action labels as values.
            output_shape (tuple): A tuple containing the output height and width of the frames.
            seq_length (int): The sequence length of the frames.
            label_map (dict): A dictionary mapping action labels to their corresponding integer values.
            transform (callable): A callable object for transforming the frames.
            train_sampler (function): A function for sampling frames during training.
            test_sampler (function): A function for sampling frames during testing.
            input_type (str, optional): The input type of the frames, either "2d" or "3d". Defaults to "2d".
            mode (str, optional): The mode of the dataset, either "train", "val". Defaults to "train".
            full_clip (bool, optional): Whether to use full clips or not. Defaults to False.
            clips_per_video (int, optional): The number of clips to be extracted per video. Defaults to 1.
            sample_clips_path (list, optional): A list of sample clip paths when using full clips. Defaults to None.
            sample_clips_ids (list, optional): A list of sample clip IDs when using full clips. Defaults to None.
        """
        self.sample_name_list = list(sample_dict.keys())
        self.sample_dict = sample_dict
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.seq_length = seq_length
        self.label_map = label_map
        self.mode = mode
        self.input_type = input_type
        self.full_clip = full_clip
        self.clips_per_video = clips_per_video
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        if self.full_clip:
            assert (sample_clips_path is not None) and (sample_clips_ids is not None)
            self.sample_clips_path = sample_clips_path
            self.sample_clips_ids = sample_clips_ids

        self.transform = transform

    def __len__(self):
        """Return the length of the dataset"""
        if self.full_clip:
            return len(self.sample_clips_ids)

        return len(self.sample_name_list) * self.clips_per_video

    def get_raw_frames(self, sample_path, img_ids):
        """Get raw frames for joint training with other modalities.

        This method takes the file path of the sample and a list of image IDs as input.
        It loads the corresponding raw frames of RGB data and returns them as a list of Image object.
        The raw frames can be used for joint training with other modalities, such as optical flow.

        Args:
            sample_path (str): The file path of the sample containing the RGB frames.
            img_ids (list): A list of image IDs corresponding to the RGB frames to be loaded.

        Returns:
            list: A list of raw RGB frames, where each frame is a Image object.

        Raises:
            FileNotFoundError: If the specified file path does not exist or the images cannot be found.
        """
        if self.mode in ["train", "val"]:
            action_label = self.label_map[self.sample_dict[sample_path]]

        data = []
        raw_imgs_list = sorted(os.listdir(os.path.join(sample_path, "rgb")))
        for img_idx in img_ids:
            img_name = raw_imgs_list[img_idx]
            img_path = os.path.join(sample_path, "rgb",
                                    img_name)
            data.append(load_image(img_path))

        if self.mode == "train":
            return data, action_label
        if self.mode == "val":
            return sample_path, data, action_label

        return sample_path, data

    def get_frames(self, sample_path, img_ids):
        """Get transformed frames of RGB data.

        This method takes the file path of the sample and a list of image IDs as input.
        It loads the corresponding frames of RGB data and applies any specified
        transformations to them.

        Args:
            sample_path (str): The file path of the sample containing the RGB frames.
            img_ids (list): A list of image IDs corresponding to the RGB frames to be loaded and transformed.

        Returns:
            Torch.tensor: A tensor of transformed RGB frames

        Raises:
            FileNotFoundError: If the specified file path does not exist or the images cannot be found.
        """
        if self.mode in ["train", "val"]:
            action_label = self.label_map[self.sample_dict[sample_path]]

        data = []
        raw_imgs_list = sorted(os.listdir(os.path.join(sample_path, "rgb")))
        for img_idx in img_ids:
            img_name = raw_imgs_list[img_idx]
            img_path = os.path.join(sample_path, "rgb",
                                    img_name)
            data.append(load_image(img_path))

        data = self.transform(data)

        if self.input_type == "2d":
            data = data.reshape([3 * self.seq_length, self.output_height, self.output_width])
        elif self.input_type == "3d":
            pass
        else:
            raise ValueError("Only 2d/3d input types are supported.")

        if self.mode == "train":
            return data, action_label
        if self.mode == "val":
            return sample_path, data, action_label

        return sample_path, data

    def __getitem__(self, idx):
        """getitem."""
        if self.full_clip:
            sample_path = self.sample_clips_path[idx]
            img_ids = self.sample_clips_ids[idx]
        else:
            idx = idx % len(self.sample_name_list)
            sample_path = self.sample_name_list[idx]

            max_sample_cnt = len(os.listdir(os.path.join(sample_path, "rgb")))

            if self.mode == "train":
                img_ids = self.train_sampler(max_sample_cnt,
                                             self.seq_length)
            else:
                img_ids = self.test_sampler(max_sample_cnt,
                                            self.seq_length)

        return self.get_frames(sample_path, img_ids)


class FuseDataset(Dataset):
    """Dataset for RGB frames + Optical Flow"""

    def __init__(self, sample_dict, output_shape,
                 of_seq_length, rgb_seq_length, label_map,
                 transform, train_sampler, test_sampler,
                 input_type="2d", mode="train", full_clip=False,
                 clips_per_video=1, sample_clips_path=None,
                 sample_clips_ids=None):
        """Initialize the FuseDataset with the given parameters.

        Args:
            sample_dict (dict): A dictionary containing video patch paths as keys and action labels as values.
            output_shape (tuple): A tuple containing the output height and width of the frames.
            of_seq_length (int): The sequence length of the optical flow data.
            rgb_seq_length (int): The sequence length of the RGB frames.
            label_map (dict): A dictionary mapping action labels to their corresponding integer values.
            transform (callable): A callable object for transforming the frames.
            train_sampler (function): A function for sampling frames during training.
            test_sampler (function): A function for sampling frames during testing.
            input_type (str, optional): The input type of the frames, either "2d" or "3d". Defaults to "2d".
            mode (str, optional): The mode of the dataset, either "train", "val". Defaults to "train".
            full_clip (bool, optional): Whether to use full clips or not. Defaults to False.
            clips_per_video (int, optional): The number of clips to be extracted per video. Defaults to 1.
            sample_clips_path (list, optional): A list of sample clip paths when using full clips. Defaults to None.
            sample_clips_ids (list, optional): A list of sample clip IDs when using full clips. Defaults to None.
        """
        self.sample_name_list = list(sample_dict.keys())
        self.sample_dict = sample_dict
        self.output_height = output_shape[0]
        self.output_width = output_shape[1]
        self.label_map = label_map
        self.mode = mode
        self.of_seq_length = of_seq_length
        self.rgb_seq_length = rgb_seq_length
        self.full_clip = full_clip
        self.clips_per_video = clips_per_video
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.input_type = input_type

        if self.full_clip:
            assert (sample_clips_path is not None) and (sample_clips_ids is not None)
            self.sample_clips_path = sample_clips_path
            self.sample_clips_ids = sample_clips_ids

        self.motion_dataset = MotionDataset(sample_dict=sample_dict,
                                            output_shape=output_shape,
                                            seq_length=of_seq_length,
                                            label_map=label_map,
                                            input_type=input_type,
                                            mode=mode,
                                            transform=None,
                                            train_sampler=None,
                                            test_sampler=None)
        self.spatial_dataset = SpatialDataset(sample_dict=sample_dict,
                                              output_shape=output_shape,
                                              seq_length=rgb_seq_length,
                                              label_map=label_map,
                                              input_type=input_type,
                                              mode=mode,
                                              transform=None,
                                              train_sampler=None,
                                              test_sampler=None)
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset"""
        if self.full_clip:
            return len(self.sample_clips_ids)

        return len(self.sample_name_list) * self.clips_per_video

    def __getitem__(self, idx):
        """getitem"""
        if self.full_clip:
            sample_path = self.sample_clips_path[idx]
            img_ids = self.sample_clips_ids[idx]
            if self.of_seq_length > self.rgb_seq_length:
                of_ids = img_ids
                rgb_ids = []
                can_idx = test_interval_sample(len(of_ids), self.rgb_seq_length)
                for idx_ in can_idx:
                    rgb_ids.append(of_ids[idx_])
            elif self.of_seq_length < self.rgb_seq_length:
                rgb_ids = img_ids
                of_ids = []
                can_idx = test_interval_sample(len(rgb_ids), self.of_seq_length)
                for idx_ in can_idx:
                    of_ids.append(rgb_ids[idx_])
            else:
                rgb_ids = img_ids
                of_ids = img_ids
        else:
            idx_ = idx % len(self.sample_name_list)
            sample_path = self.sample_name_list[idx_]

            # max_sample_cnt is from u and v
            max_sample_cnt = len(os.listdir(os.path.join(sample_path, "u")))

            if self.mode == "train":
                rgb_ids, of_ids = self.train_sampler(max_sample_cnt,
                                                     self.rgb_seq_length,
                                                     self.of_seq_length)
            else:
                rgb_ids, of_ids = self.test_sampler(max_sample_cnt,
                                                    self.rgb_seq_length,
                                                    self.of_seq_length)

        if self.mode in ["train", "val"]:
            action_label = self.label_map[self.sample_dict[sample_path]]
        # generate RGB frames patch + OF vectors
        if self.mode == "train":
            of_data, of_action_label = self.motion_dataset.get_raw_frames(sample_path, of_ids)
            rgb_data, rgb_action_label = self.spatial_dataset.get_raw_frames(sample_path, rgb_ids)

            rgb_data, of_data = self.transform([rgb_data, of_data])

            if self.input_type == "2d":
                rgb_data = rgb_data.reshape([3 * self.rgb_seq_length,
                                             self.output_height,
                                             self.output_width])
                of_data = of_data.reshape([2 * self.of_seq_length,
                                           self.output_height,
                                           self.output_width])
            elif self.input_type == "3d":
                pass
            else:
                raise ValueError("Only 2d/3d input types are supported.")

            assert of_action_label == action_label
            assert rgb_action_label == action_label

            return [rgb_data, of_data], action_label
        elif self.mode == "val":
            of_sample_path, of_data, of_action_label = \
                self.motion_dataset.get_raw_frames(sample_path, of_ids)
            rgb_sample_path, rgb_data, rgb_action_label = \
                self.spatial_dataset.get_raw_frames(sample_path, rgb_ids)

            rgb_data, of_data = self.transform([rgb_data, of_data])

            if self.input_type == "2d":
                rgb_data = rgb_data.reshape([3 * self.rgb_seq_length,
                                             self.output_height,
                                             self.output_width])
                of_data = of_data.reshape([2 * self.of_seq_length,
                                           self.output_height,
                                           self.output_width])
            elif self.input_type == "3d":
                pass
            else:
                raise ValueError("Only 2d/3d input types are supported.")

            assert of_action_label == action_label
            assert rgb_action_label == action_label
            assert of_sample_path == sample_path
            assert rgb_sample_path == sample_path

            return sample_path, [rgb_data, of_data], action_label
        elif self.mode == "inf":
            of_sample_path, of_data = self.motion_dataset.get_raw_frames(sample_path, of_ids)
            rgb_sample_path, rgb_data = self.spatial_dataset.get_raw_frames(sample_path, rgb_ids)

            rgb_data, of_data = self.transform([rgb_data, of_data])

            if self.input_type == "2d":
                rgb_data = rgb_data.reshape([3 * self.rgb_seq_length,
                                             self.output_height,
                                             self.output_width])
                of_data = of_data.reshape([2 * self.of_seq_length,
                                           self.output_height,
                                           self.output_width])
            elif self.input_type == "3d":
                pass
            else:
                raise ValueError("Only 2d/3d input types are supported.")

            return sample_path, [rgb_data, of_data]
        else:
            raise ValueError('There are only train, val, inf mode')
