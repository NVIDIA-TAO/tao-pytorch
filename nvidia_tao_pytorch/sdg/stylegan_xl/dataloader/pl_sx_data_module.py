# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StyleGAN Data Module"""

from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.gan_data_module import GANDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.sx_dataset import ImageFolderDataset
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.seed_dataset import SeedDataset


def init_dataset_kwargs(data):
    """
    Initializes dataset configuration parameters and validates the dataset.

    Args:
        data (str): Path to the dataset.

    Returns:
        tuple: A dictionary of dataset keyword arguments and the dataset name.

    Raises:
        IOError: If there is an error in accessing the dataset.
    """
    try:
        dataset_kwargs = dnnlib.EasyDict(
            path=data,
            use_labels=True,
            max_size=None,
            xflip=False
        )

        dataset_obj = ImageFolderDataset(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.

        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise IOError(f'--data: {err}')


class SXDataModule(GANDataModule):
    """Lightning DataModule for StyleGAN."""

    def __init__(self, dataset_config):
        """ Lightning DataModule Initialization.

        Args:
            dataset_config: dataset configuration.

        """
        super(SXDataModule, self).__init__(dataset_config)
        self.evaluate_data_dir = dataset_config['stylegan']['test_dataset']['images_dir']
        self.validate_data_dir = dataset_config['stylegan']['validation_dataset']['images_dir']
        self.train_data_dir = dataset_config['stylegan']['train_dataset']['images_dir']

    def create_dataset_kwargs(self, images_dir):
        """
        Creates and configures dataset keyword arguments based on the specified directory and dataset configuration.

        Args:
            images_dir (str): Directory path containing the dataset images.

        Returns:
            dict: A dictionary containing dataset keyword arguments for initializing the dataset.

        Raises:
            Exception: If conditional training is enabled (cond=True) but the dataset lacks labels.
        """
        data_set_kwargs, _ = init_dataset_kwargs(data=images_dir)
        if self.dataset_config['common']['cond'] and not data_set_kwargs.use_labels:
            raise Exception('--cond=True requires labels specified in dataset.json')
        data_set_kwargs.use_labels = self.dataset_config['common']['cond']
        data_set_kwargs.xflip = self.dataset_config['stylegan']['mirror']

        return data_set_kwargs

    def assertion_check(self, dataset):
        """
        Performs assertion checks to ensure the created dataset instance matches the expected configuration parameters.

        Args:
            dataset (object): The dataset object to be checked.

        Asserts:
            - The number of classes in the dataset matches the expected number of classes.
            - The number of image channels in the dataset matches the expected number of channels.
            - The image resolution in the dataset matches the expected resolution.
        """
        assert self.dataset_config['common']['num_classes'] == dataset.label_dim, (
            "The claimed number of class:", self.dataset_config['common']['num_classes'], " does not match the actual label number:", dataset.label_dim
        )
        assert self.dataset_config['common']['img_channels'] == dataset.num_channels, (
            "The claimed image channels:", self.dataset_config['common']['img_channels'], " does not match the actual image channels:", dataset.num_channels
        )
        assert self.dataset_config['common']['img_resolution'] == dataset.resolution, (
            "The claimed image resolution:", self.dataset_config['common']['img_resolution'], " does not match the actual image resolution:", dataset.resolution
        )

    def setup(self, stage=None):
        """Prepares for each dataloader.

        Args:
            stage (str): stage options from fit, validate, test, predict or None.

        """
        if stage in ('fit', None):
            self.training_set_kwargs = self.create_dataset_kwargs(images_dir=self.train_data_dir)
            self.training_set = ImageFolderDataset(**self.training_set_kwargs)
            self.assertion_check(dataset=self.training_set)

            self.validation_set_kwargs = self.create_dataset_kwargs(images_dir=self.validate_data_dir)
            self.validation_set_kwargs.update(xflip=False)
            self.validation_set = ImageFolderDataset(**self.validation_set_kwargs)
            self.assertion_check(dataset=self.validation_set)
            self.validation_sampler = self._set_unrepeated_samplers(self.validation_set)

        elif stage == 'test':
            self.evaluation_set_kwargs = self.create_dataset_kwargs(images_dir=self.evaluate_data_dir)
            self.evaluation_set_kwargs.update(xflip=False)
            self.evaluation_set = ImageFolderDataset(**self.evaluation_set_kwargs)
            self.assertion_check(dataset=self.evaluation_set)
            self.evaluation_sampler = self._set_unrepeated_samplers(self.evaluation_set)

        elif stage == 'predict':
            seeds = list(range(self.dataset_config['stylegan']['infer_dataset']['start_seed'], self.dataset_config['stylegan']['infer_dataset']['end_seed']))
            self.seed_dataset = SeedDataset(seeds)
            # The default sampler is already UnrepeatedDistributedSampler when "predict"
