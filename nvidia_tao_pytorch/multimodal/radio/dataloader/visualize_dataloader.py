# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize and save augmented images from dataloader."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List


class DataLoaderVisualizer:
    """Visualize and save images from dataloader."""

    def __init__(
        self,
        output_dir: str,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualized images
            mean: Mean used for normalization (default: ImageNet [0.485, 0.456, 0.406])
            std: Std used for normalization (default: ImageNet [0.229, 0.224, 0.225])
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default to ImageNet normalization
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

        # Convert to tensors for denormalization
        self.mean_tensor = torch.tensor(self.mean).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std).view(3, 1, 1)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor image.

        Args:
            tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Denormalized tensor
        """
        if tensor.dim() == 4:
            # Batch of images
            mean = self.mean_tensor.unsqueeze(0).to(tensor.device)
            std = self.std_tensor.unsqueeze(0).to(tensor.device)
        else:
            # Single image
            mean = self.mean_tensor.to(tensor.device)
            std = self.std_tensor.to(tensor.device)

        # Denormalize: img = img * std + mean
        denorm = tensor * std + mean

        # Clip to [0, 1]
        return torch.clamp(denorm, 0, 1)

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL Image.

        Args:
            tensor: Tensor of shape (C, H, W) with values in [0, 1]

        Returns:
            PIL Image
        """
        # Denormalize if needed
        tensor = self.denormalize(tensor)

        # Convert to numpy array
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convert from (C, H, W) to (H, W, C)
        np_img = tensor.permute(1, 2, 0).numpy()

        # Convert to uint8
        np_img = (np_img * 255).astype(np.uint8)

        return Image.fromarray(np_img)

    def save_batch(
        self,
        batch: dict,
        batch_idx: int = 0,
        num_images: Optional[int] = None,
        prefix: str = "batch",
    ):
        """
        Save a batch of images from dataloader.

        Args:
            batch: Batch dict from dataloader with 'img' and optionally 'class' keys
            batch_idx: Batch index for naming
            num_images: Number of images to save (default: all in batch)
            prefix: Prefix for saved filenames
        """
        # Extract images and labels
        images = batch.get('img', batch.get('image', None))
        labels = batch.get('class', batch.get('label', None))

        if images is None:
            raise ValueError("Batch must contain 'img' or 'image' key")

        # Determine number of images to save
        batch_size = images.shape[0]
        if num_images is None:
            num_images = batch_size
        else:
            num_images = min(num_images, batch_size)

        # Save each image
        for i in range(num_images):
            img_tensor = images[i]

            # Convert to PIL Image
            pil_img = self.tensor_to_image(img_tensor)

            # Create filename
            label_str = ""
            if labels is not None:
                if labels.dim() == 1:
                    # Single label per image
                    label = labels[i].item()
                    label_str = f"_class{label}"
                else:
                    # Multi-label or one-hot
                    label = labels[i].argmax().item()
                    label_str = f"_class{label}"

            filename = f"{prefix}_batch{batch_idx:04d}_img{i:03d}{label_str}.jpg"
            filepath = self.output_dir / filename

            # Save image
            pil_img.save(filepath, quality=95)

        print(f"Saved {num_images} images from batch {batch_idx} to {self.output_dir}")

    def visualize_dataloader(
        self,
        dataloader,
        num_batches: int = 5,
        num_images_per_batch: Optional[int] = None,
        prefix: str = "train",
    ):
        """
        Visualize multiple batches from dataloader.

        Args:
            dataloader: PyTorch DataLoader to visualize
            num_batches: Number of batches to visualize
            num_images_per_batch: Number of images to save per batch (default: all)
            prefix: Prefix for saved filenames
        """
        print(f"Visualizing {num_batches} batches from dataloader...")
        print(f"Output directory: {self.output_dir}")

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                # If batch is a tuple/list, assume first element is images
                batch = {'img': batch[0], 'class': batch[1] if len(batch) > 1 else None}

            self.save_batch(
                batch,
                batch_idx=batch_idx,
                num_images=num_images_per_batch,
                prefix=prefix,
            )

        print(f"Visualization complete! Saved to {self.output_dir}")

    def create_grid(
        self,
        batch: dict,
        nrow: int = 8,
        output_name: str = "grid.jpg",
    ) -> Image.Image:
        """
        Create a grid of images from a batch.

        Args:
            batch: Batch dict from dataloader
            nrow: Number of images per row
            output_name: Filename for the grid image

        Returns:
            PIL Image of the grid
        """
        import torchvision.utils as vutils

        images = batch.get('img', batch.get('image'))

        # Denormalize images
        images = self.denormalize(images)

        # Create grid
        grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=False)

        # Convert to PIL Image
        grid_img = self.tensor_to_image(grid)

        # Save grid
        filepath = self.output_dir / output_name
        grid_img.save(filepath, quality=95)

        print(f"Saved grid to {filepath}")
        return grid_img


def visualize_classification_dataloader(
    data_module,
    output_dir: str = "./dataloader_visualization",
    num_batches: int = 5,
    num_images_per_batch: Optional[int] = 8,
    stage: str = "fit",
):
    """
    Convenience function to visualize a classification data module.

    Args:
        data_module: PyTorch Lightning DataModule
        output_dir: Directory to save visualized images
        num_batches: Number of batches to visualize
        num_images_per_batch: Number of images to save per batch
        stage: Which dataloader to use ('fit', 'test', 'predict')
    """
    # Setup data module
    data_module.setup(stage=stage)

    # Get the appropriate dataloader
    if stage == "fit":
        dataloader = data_module.train_dataloader()
        prefix = "train"
    elif stage == "test":
        dataloader = data_module.test_dataloader()
        prefix = "test"
    elif stage == "predict":
        dataloader = data_module.predict_dataloader()
        prefix = "predict"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Get mean and std from augmentation config
    aug_config = data_module.dataset_config.get("augmentation", {})
    mean = aug_config.get("mean", [0.485, 0.456, 0.406])
    std = aug_config.get("std", [0.229, 0.224, 0.225])

    # Create visualizer
    visualizer = DataLoaderVisualizer(output_dir, mean=mean, std=std)

    # Visualize
    visualizer.visualize_dataloader(
        dataloader,
        num_batches=num_batches,
        num_images_per_batch=num_images_per_batch,
        prefix=prefix,
    )


if __name__ == "__main__":
    """
    Example usage for testing.
    """
    import argparse
    from omegaconf import OmegaConf
    from nvidia_tao_pytorch.multimodal.radio.dataloader.radio_data_module import RadioDataModule

    parser = argparse.ArgumentParser(description="Visualize dataloader output")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--output-dir", type=str, default="./dataloader_viz", help="Output directory")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of batches to visualize")
    parser.add_argument("--images-per-batch", type=int, default=8, help="Images per batch")
    parser.add_argument("--stage", type=str, default="fit", choices=["fit", "test", "predict"], help="Stage")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Create data module
    data_module = RadioDataModule(config.dataset)

    # Visualize
    visualize_classification_dataloader(
        data_module,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        num_images_per_batch=args.images_per_batch,
        stage=args.stage,
    )
