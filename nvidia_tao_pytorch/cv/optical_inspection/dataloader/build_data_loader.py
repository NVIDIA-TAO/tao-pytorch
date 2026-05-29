# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build torch data loader."""

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.oi_dataset import MultiGoldenDataset, SiameseNetworkTRIDataset, get_sampler


class CalibrationDataset(Dataset):
    """Simple dataset for quantization calibration that loads image pairs from a directory."""

    def __init__(self, images_dir, transform, image_ext=".jpg"):
        """Initialize calibration dataset.

        Args:
            images_dir (str): Directory containing calibration images.
            transform: Torchvision transforms to apply.
            image_ext (str): Image file extension.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.image_ext = image_ext
        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self):
        """Return dataset length."""
        return len(self.image_files)

    def __getitem__(self, index):
        """Get image pair for calibration (uses same image as both inputs)."""
        img_path = os.path.join(self.images_dir, self.image_files[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img, 0

# START SIAMESE DATALOADER


def build_dataloader(df, weightedsampling, split, data_config):
    """Build torch dataloader.

    Args:
        df (pd.DataFrame): The input data frame.
        weightedsampling (bool): Flag indicating whether to use weighted sampling.
        split (str): The split type ('train', 'valid', 'test', 'infer').
        data_config (OmegaConf.DictConf): Configuration spec for data loading.

    Returns:
        DataLoader: The built torch DataLoader object.
    """
    workers = data_config["workers"]
    batch_size = data_config["batch_size"]
    image_width = data_config["image_width"]
    image_height = data_config["image_height"]
    rgb_mean = data_config["augmentation_config"]["rgb_input_mean"]
    rgb_std = data_config["augmentation_config"]["rgb_input_std"]
    dataset_class = MultiGoldenDataset if "num_golden" in data_config and data_config["num_golden"] > 1 else SiameseNetworkTRIDataset

    train_transforms = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]
    )

    dataloader_kwargs = {
        "pin_memory": True,
        "batch_size": batch_size,
        "num_workers": workers
    }

    if split == 'train':
        input_data_path = data_config["train_dataset"]["images_dir"]
        dataset = dataset_class(data_frame=df,
                                train=True,
                                input_data_path=input_data_path,
                                transform=train_transforms,
                                data_config=data_config)

        if weightedsampling:
            fpratio_sampling = data_config['fpratio_sampling']
            wt_sampler = get_sampler(dataset, fpratio_sampling)
            dataloader_kwargs["sampler"] = wt_sampler
        else:
            dataloader_kwargs["shuffle"] = True
        assert batch_size > 1, "Training batch size must be greater than 1."
        dataloader_kwargs["drop_last"] = True

    elif split == 'valid':
        input_data_path = data_config["validation_dataset"]["images_dir"]
        dataset = dataset_class(data_frame=df,
                                train=False,
                                input_data_path=input_data_path,
                                transform=test_transforms,
                                data_config=data_config)

        dataloader_kwargs["shuffle"] = False

    elif split == 'test':
        input_data_path = data_config["test_dataset"]["images_dir"]
        dataset = dataset_class(data_frame=df,
                                train=False,
                                input_data_path=input_data_path,
                                transform=test_transforms,
                                data_config=data_config)
        dataloader_kwargs["shuffle"] = False

    elif split == 'infer':
        input_data_path = data_config["infer_dataset"]["images_dir"]
        dataset = dataset_class(data_frame=df,
                                train=False,
                                input_data_path=input_data_path,
                                transform=test_transforms,
                                data_config=data_config)

        dataloader_kwargs["shuffle"] = False

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs
    )
    return dataloader


def build_calib_dataloader(images_dir, data_config):
    """Build dataloader for quantization calibration.

    Args:
        images_dir (str): Directory containing calibration images.
        data_config (OmegaConf.DictConf): Configuration spec for data loading.

    Returns:
        DataLoader: The built torch DataLoader object for calibration.
    """
    workers = data_config.get("workers", 8)
    batch_size = data_config.get("batch_size", 8)
    image_width = data_config.get("image_width", 224)
    image_height = data_config.get("image_height", 224)
    aug_config = data_config.get("augmentation_config", {})
    rgb_mean = aug_config.get("rgb_input_mean", [0.485, 0.456, 0.406])
    rgb_std = aug_config.get("rgb_input_std", [0.229, 0.224, 0.225])

    calib_transforms = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)
    ])

    dataset = CalibrationDataset(
        images_dir=images_dir,
        transform=calib_transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader
