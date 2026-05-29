# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data feeder for loading skeleton sequences."""
import numpy as np
import pickle
import random
from torch.utils.data import Dataset


def auto_pad(data_numpy, size, random_pad=False):
    """
    Apply padding to the numpy data.

    This function checks if the temporal dimension of the input data (second dimension) is smaller than the target size.
    If it is, the function pads the data up to the target size. The padding could be applied from the beginning (left padding)
    or randomly within the range, depending on the 'random_pad' flag.

    Args:
        data_numpy (np.ndarray): The input data of shape (C, T, V, M), where C is the number of channels, T is
                                 the temporal dimension, V is the number of vertices, and M is another dimension (e.g., batch size).
        size (int): The target size for the temporal dimension.
        random_pad (bool, optional): If True, padding is applied at a random position within the range.
                                      If False, padding is applied from the beginning. Defaults to False.

    Returns:
        np.ndarray: The padded data. If the temporal dimension of the input data is equal to or larger than the target size,
                    the input data is returned as is.
    """
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_padded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        data_numpy_padded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_padded
    return data_numpy


def random_choose(data_numpy, size, enable_auto_pad=True):
    """
    Randomly select a clip from the input data.

    This function checks the temporal dimension of the input data. If it's equal to the target size, the function
    returns the original data. If it's smaller than the target size, the function either pads the data to the
    target size or returns the original data based on the 'enable_auto_pad' flag. If the temporal dimension of
    the input data is greater than the target size, the function will randomly select a portion of the data
    of size 'size' and return it.

    Args:
        data_numpy (np.ndarray): The input data of shape (C, T, V, M), where C is the number of channels, T is
                                 the temporal dimension, V is the number of vertices, and M is another dimension (e.g., batch size).
        size (int): The target size for the temporal dimension.
        enable_auto_pad (bool, optional): If True, and if T < size, padding is applied using the 'auto_pad' function.
                                          If False, the original data is returned as is. Defaults to True.

    Returns:
        np.ndarray: The data clip with the temporal dimension of size 'size'. If the temporal dimension of the
                    input data is smaller than 'size' and 'enable_auto_pad' is False, the original data is returned.
    """
    # input: C,T,V,M
    data_shape = data_numpy.shape
    T = data_shape[1]
    if T == size:
        return data_numpy
    if T < size:
        if enable_auto_pad:
            return auto_pad(data_numpy, size, random_pad=True)
        return data_numpy
    begin = random.randint(0, T - size)
    return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    """
    Randomly manipulate the coordinates of the input data.

    This function randomly manipulates the coordinates of the input data by applying rotations, scaling, and transformations.
    The angle of rotation, scale factor, and the magnitude of the transformation are randomly chosen from the
    corresponding candidate lists. The manipulation is applied to the spatial dimension of the data.

    Args:
        data_numpy (np.ndarray): The input data of shape (C, T, V, M), where C is the number of channels, T is
                                 the temporal dimension, V is the number of vertices, and M is another dimension (e.g., batch size).
        angle_candidate (list, optional): List of possible rotation angles in degrees. Defaults to [-10., -5., 0., 5., 10.].
        scale_candidate (list, optional): List of possible scaling factors. Defaults to [0.9, 1.0, 1.1].
        transform_candidate (list, optional): List of possible translation magnitudes. Defaults to [-0.2, -0.1, 0.0, 0.1, 0.2].
        move_time_candidate (list, optional): List of possible 'move times' determining the granularity of the transformation over time. Defaults to [1].

    Returns:
        np.ndarray: The manipulated data with the same shape as the input data.
    """
    # input: C,T,V,M
    data_shape = data_numpy.shape
    T = data_shape[1]
    V = data_shape[2]
    M = data_shape[3]
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


class SkeletonFeeder(Dataset):
    """
    Feeder for skeleton-based action recognition.

    This feeder loads skeleton sequences and their corresponding labels from given paths and applies specified
    data processing methods such as random choosing, moving, and padding. It inherits from the PyTorch Dataset class.

    Args:
        data_path (str): The path to the '.npy' data. The data should have the shape (N, C, T, V, M).
        label_path (str): The path to the label data.
        label_map (dict): A dictionary mapping labels to their corresponding indices.
        random_choose (bool, optional): If True, a portion of the input sequence is randomly chosen for each sample. Defaults to False.
        random_move (bool, optional): If True, the input sequence is randomly moved for each sample. Defaults to False.
        window_size (int, optional): The length of the output sequence. If it is negative, the whole sequence is used. Defaults to -1.
        debug (bool, optional): If True, only the first 100 samples are used. Defaults to False.
        mmap (bool, optional): If True, memory-map the loaded data. Useful when the data is too large to fit into memory. Defaults to True.

    Attributes:
        data (np.ndarray): The loaded skeleton sequences of shape (N, C, T, V, M).
        label (list): The labels corresponding to the skeleton sequences.
        sample_name (list): The names of the skeleton sequence samples.
        N, C, T, V, M (int): The dimensions of the skeleton sequence data.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 label_map,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        """
        Initialize a skeleton feeder.
        """
        self.label = None
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.label_map = label_map
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        """
        Load skeleton sequences and their corresponding labels.

        The data is loaded from the paths specified in the constructor. The sequences are loaded either normally or
        as memory-mapped based on the 'mmap' argument. If 'debug' is True, only the first 100 samples are loaded.

        Args:
            mmap (bool): If True, memory-map the loaded data.
        """
        # data: N C T V M

        # load label
        if self.label_path:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        """
        Return the number of sequences.

        Returns:
            int: The number of skeleton sequences.
        """
        return self.N

    def __getitem__(self, index):
        """
        Get data and label at an index.

        This method retrieves the skeleton sequence and its corresponding label at the specified index. It applies
        the data processing methods specified in the constructor (random choosing, moving, and padding).

        Args:
            index (int): The index of the sequence and label to retrieve.

        Returns:
            tuple: A tuple containing the skeleton sequence and its corresponding label.
        """
        # get data
        data_numpy = np.array(self.data[index])
        label = -1
        if self.label:
            label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pad(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        return data_numpy, label
