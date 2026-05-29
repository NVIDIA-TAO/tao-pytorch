# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StyleGAN Model PyTorch Lightning Module"""

import os
import random
import copy
import torch
import PIL.Image
import torch.optim as optim
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_lightning.callbacks import Callback

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.sdg.stylegan_xl.model.stylegan import build_generator, build_discriminator, build_projectedloss, build_inception, retrieve_generator_checkpoint_from_stylegan_pl_model
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import misc
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import gen_utils
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.startup import download_and_convert_pretrained_modules


# This callback is essential for loading InceptionNet for FID and class embeddings for discriminator and generator
class SubmodulesCheckpointLoader(Callback):
    """ Callback to forcefully load the stem checkpoint for a higher resolution model, even if a pretrained or resumed checkpoint is already loaded."""

    def on_train_start(self, trainer, pl_module):  # Remind that this call will be procecced even after resuming checkpoint is already loaded
        """Pytorch Lightning built-in function at the start of training."""
        # Download pretrained modules from public first
        download_and_convert_pretrained_modules()
        # Since the class embedding is trainable, the loading should be ignored when resumed or pretrained checkpoints are provided
        has_pretrained_model = (
            trainer.ckpt_path is not None or
            pl_module.experiment_spec['train']['pretrained_model_path'] is not None
        )
        if has_pretrained_model:
            pass
        else:
            if pl_module.model_config['stylegan']['metrics']['inception_fid_path'] is not None:
                pl_module.fid.inception.load_pretrained_model(pl_module.model_config['stylegan']['metrics']['inception_fid_path'])
                logging.warning("The pretrained InceptionNet is loaded.")
            else:
                logging.warning("The pretrained InceptionNet checkpoint is not provided. FID metrics cannot be correctly calculated.")
            if pl_module.model_config['input_embeddings_path'] is not None:
                pl_module.D.load_pretrained_embedding(pl_module.model_config['input_embeddings_path'])
                pl_module.G.load_pretrained_embedding(pl_module.model_config['input_embeddings_path'])
                pl_module.G_ema.load_pretrained_embedding(pl_module.model_config['input_embeddings_path'])
                logging.warning("The pretrained embedding checkpoint is loaded for discriminator/generator.")
            else:
                logging.warning("The pretrained embedding checkpoint is not provided. Initialized input embedding for discriminator/generator with random weights.")


# This callback is essential for training a super-resolution StyleGAN-XL when freezing the low-resolution backbone (stem).
class StemCheckpointLoader(Callback):
    """ Callback to forcefully load the stem checkpoint for a higher resolution model, even if a pretrained or resumed checkpoint is already loaded."""

    def on_train_start(self, trainer, pl_module):  # Remind that this call will be procecced even after resuming checkpoint is already loaded
        """Pytorch Lightning built-in function at the start of training."""
        # Load the stem checkpoint only when training higher resolution model
        if pl_module.model_config['generator']['superres']:
            # Define the condition for checking the checkpoint path and pretrained model path
            has_pretrained_model = (
                trainer.ckpt_path is not None or
                pl_module.experiment_spec['train']['pretrained_model_path'] is not None
            )
            # Check if the condition is met along with the reinitialization flag
            if has_pretrained_model and not pl_module.model_config['generator']['added_head_superres']['reinit_stem_anyway']:
                pass
            # Forcefully load the stem checkpoing
            else:
                assert pl_module.model_config['generator']['added_head_superres']['pretrained_stem_path'] \
                    is not None, "When training superres head, provide path to stem"
                stylegan_generator_ema_checkpoint = retrieve_generator_checkpoint_from_stylegan_pl_model(
                    pl_module.model_config['generator']['added_head_superres']['pretrained_stem_path']
                )

                # build the stem model for loading the stem checkpoint
                G_stem = build_generator(experiment_config=pl_module.experiment_spec, return_stem=True).train().requires_grad_(False)  # TODO .train().requires_grad_(False)?
                G_stem.load_state_dict(stylegan_generator_ema_checkpoint)

                # This is relevant when you continue training a lower-res model
                # ie. train 16 model, start training 32 model but continue training 16 model
                # then restart 32 model to reload the improved 16 model
                # Note that when training higher res, we will freeze the G_stem (grad=0), But the mapping.w_avg is still being updated.
                # So 1. resuming training w/o loading low res again and 2. resuming training w/ loading the low res stem afterwards
                # will result in different mapping.w_avg in G
                pl_module.G.reinit_stem(copy.deepcopy(G_stem))
                pl_module.G_ema.reinit_stem(copy.deepcopy(G_stem))


# The callback means non-essential logic and can be disabled without affect the training
# See the details from configure_callbacks of StyleganPlModel
class SampleImagesExporter(Callback):
    """ Callback for exporting sample images generated from StyleGAN-XL in training."""

    def __init__(self):
        """Init Callback."""
        super().__init__()
        # The folder for saving generated images. The path will be given at setup by loading the pl module configuration
        self.run_dir = None  # The path will be given at setup by loading the pl module configuration

    def setup(self, trainer, pl_module, stage):
        """Pytorch Lightning built-in function for setup. Will be called BEFORE the setup of pl module."""
        if stage in ('fit', None):
            self.run_dir = os.path.join(pl_module.experiment_spec['results_dir'], "outputs")  # TODO double train?
            if trainer.global_rank == 0:
                if not os.path.exists(self.run_dir):
                    os.makedirs(self.run_dir)

    def on_train_start(self, trainer, pl_module):
        """Pytorch Lightning built-in function at the start of training."""
        with torch.no_grad():
            # Print network summary tables.
            if trainer.global_rank == 0:
                print("Print network summary tables...")
                z = torch.empty([pl_module.batch_gpu, pl_module.G.z_dim]).to(pl_module.device)
                c = torch.empty([pl_module.batch_gpu, pl_module.G.c_dim]).to(pl_module.device)
                img = misc.print_module_summary(pl_module.G, [z, c])
                # The following printing operation will run the forward of discriminator, which will affect
                # 1. the order of the random seed switching since discriminator has random augement
                # 2. batchnorm's runnig mean and variance since the discriminator is not at eval state
                outputs = misc.print_module_summary(pl_module.D, [img, c])  # noqa: F841
                # free up cuda memory
                del img, outputs, c, z
                torch.cuda.empty_cache()

            # Export sample images.
            self.grid_size = None
            self.grid_z = None
            self.grid_c = None
            if pl_module.trainer.global_rank == 0:
                # Export image snapshot just before train start.
                print('Exporting sample images...')
                self.grid_size, images, labels = self.setup_snapshot_image_grid(training_set=pl_module.dm.training_set)
                self.save_image_grid(images, os.path.join(self.run_dir, 'reals.png'), drange=[0, 255], grid_size=self.grid_size)
                self.grid_z = torch.randn([labels.shape[0], pl_module.G.z_dim]).to(pl_module.device).split(pl_module.batch_gpu)
                self.grid_c = torch.from_numpy(labels).to(pl_module.device).split(pl_module.batch_gpu)
                images = torch.cat([pl_module.G_ema(z=z, c=c, noise_mode='const').cpu()
                                    for z, c in zip(self.grid_z, self.grid_c)]).numpy()
                self.save_image_grid(images, os.path.join(self.run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=self.grid_size)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Pytorch Lightning built-in function at the end of each validation epoch."""
        with torch.no_grad():
            if trainer.global_rank == 0:
                # Export image snapshot after validation epoch.
                images = torch.cat([pl_module.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)]).numpy()
                self.save_image_grid(images, os.path.join(self.run_dir, f'fakes{pl_module.cur_nimg // 1000:06d}.png'), drange=[-1, 1], grid_size=self.grid_size)

    def save_image_grid(self, img, fname, drange, grid_size):
        """Custom function to save image grid.
        Args:
            img: numpy image array
            fname: the name of the saved image
            drange: lower and upper bounds of the range
            grid_size: total grid number for the saved image

        Returns:
            None

        """
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

        gw, gh = grid_size
        _N, C, H, W = img.shape
        img = img.reshape([gh, gw, C, H, W])
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape([gh * H, gw * W, C])

        assert C in [1, 3]
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)

    def setup_snapshot_image_grid(self, training_set, random_seed=0, gw=None, gh=None):
        """Custom function to setup image grid.
        Args:
            training_set: training dataset
            random_seed: a specific random seed for generating the snapshot
            gw: grid width
            gh: grid height

        Returns:
            grid_size: grid size
            stacked_images: an array of stacked images
            stacked_labels: an array of stacked labels

        """
        # Use a consistent random state for exporting image grid during the whole training
        rnd = np.random.RandomState(random_seed)
        if gw is None:
            gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
        if gh is None:
            gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

        # No labels => show random subset of training samples.
        if not training_set.has_labels:
            all_indices = list(range(len(training_set)))
            rnd.shuffle(all_indices)
            grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

        else:
            # Group training samples by label.
            label_groups = dict()  # label => [idx, ...]
            for idx in range(len(training_set)):
                label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(idx)

            # Reorder.
            label_order = sorted(label_groups.keys())
            for label in label_order:
                rnd.shuffle(label_groups[label])

            # Organize into grid.
            grid_indices = []
            for y in range(gh):
                label = label_order[y % len(label_order)]
                indices = label_groups[label]
                grid_indices += [indices[x % len(indices)] for x in range(gw)]
                label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

        # Load data.
        images, labels = zip(*[training_set[i] for i in grid_indices])

        grid_size = (gw, gh)
        stacked_images = np.stack(images)
        stacked_labels = np.stack(labels)
        return grid_size, stacked_images, stacked_labels


class StyleganPlModel(TAOLightningModule):
    """ PTL module for StyleGAN-XL Model."""

    def __init__(self, experiment_spec, dm, export=False):
        """Init training for StyleGAN-XL Model."""
        super().__init__(experiment_spec)

        self.checkpoint_filename = 'styleganxl_model'
        self.dm = dm
        self.cudnn_benchmark = True

        # Number of samples processed at a time by one GPU.
        self.batch_gpu = self.dataset_config['stylegan']['batch_gpu_size']
        # Current best (smallest) fid score
        self.best_fid = 9999

        # Hyperparameters for updating generator EMA
        self.ema_rampup = 0.05
        self.ema_kimg = self.dataset_config['batch_size'] * 10 / 32
        # Current seen image number is used for updating generator EMA, determining projected loss status, and logging
        self.register_buffer('cur_nimg', torch.tensor(0, dtype=torch.long))  # will move to cuda unlike the above one #TODO other way to retreive the info?

        # Build up generator, generator EMA, discriminator, fid models for complete stylegan training
        self._build_model(export)
        # Build projected loss used in stylegan training
        self._build_criterion()

        # The phases tell the training step when should update generator and when should update discriminator
        self.phases = []
        self.G_reg_interval = 4     # How often to perform regularization for G? None = disable lazy regularization.
        self.D_reg_interval = 16    # How often to perform regularization for D? None = disable lazy regularization.

        # Training with multiple optimizers is only supported with manual optimization.
        # Set `self.automatic_optimization = False`, then access your optimizers in `training_step` with `opt1, opt2, ... = self.optimizers()`.
        self.automatic_optimization = False

    def setup(self, stage=None):
        """Pytorch Lightning built-in function for model setup before launching of training, evaluation, and inference."""
        assert self.G.training
        assert self.D.training
        assert not self.G_ema.training
        assert not self.fid.training
        # If not using resuming training such as only pretrained checkpoint for fintuning. The cur_nimg should be set to 0 to prevent saturated ema_nimg for calculating G_ema
        if stage == 'fit':
            if not self.trainer.ckpt_path:
                self.cur_nimg = self.cur_nimg * 0
        if stage in ('fit', 'predict', 'test', None):
            # IMPORTANT for making different ranks using different random seeds to prevent from generating the same latent vectors cross gpus in training
            random.seed((0 + self.experiment_spec['train']['stylegan']['gan_seed_offset']) * self.trainer.num_nodes + self.trainer.global_rank)
            np.random.seed((0 + self.experiment_spec['train']['stylegan']['gan_seed_offset']) * self.trainer.num_nodes + self.trainer.global_rank)
            torch.manual_seed((0 + self.experiment_spec['train']['stylegan']['gan_seed_offset']) * self.trainer.num_nodes + self.trainer.global_rank)

            # Set cuda configurations
            torch.backends.cudnn.benchmark = self.cudnn_benchmark   # Improves training speed.
            torch.backends.cuda.matmul.allow_tf32 = False           # Improves numerical accuracy.
            torch.backends.cudnn.allow_tf32 = False                 # Improves numerical accuracy.

            # To get deterministic results at every launching
            if self.experiment_spec['train']['deterministic_all']:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Set the CUBLAS_WORKSPACE_CONFIG
                torch.use_deterministic_algorithms(True)

        # Though generator and discriminator does not always require grad for every optimizer step,
        # enable gradients first to prevent no gradient error when creating ddp model.
        self.G.requires_grad_(True)
        self.D.requires_grad_(True)
        # Make sure the G_ema is in evaluation mode and does not requires grad in the training
        self.G_ema.requires_grad_(False).eval()
        # Make sure the fid module is in evaluation mode in the training
        self.fid.eval()

    def _build_model(self, export):
        """Internal function to build the models."""
        self.G = build_generator(experiment_config=self.experiment_spec, return_stem=False, export=export).train().requires_grad_(False)
        self.G_ema = copy.deepcopy(self.G).eval()
        self.D = build_discriminator(experiment_config=self.experiment_spec).train().requires_grad_(False)
        # Use custom inception net as backend of fid model
        inception = build_inception(experiment_config=self.experiment_spec).eval()
        self.fid = FrechetInceptionDistance(inception).eval()  # Change feature dimension to 2048
        # self.fid.inception.load_pretrained_model(self.model_config['stylegan']['metrics']['inception_fid_path'])
        # Uncomment the following line to use the default inception model of fid w/o using custom inception model
        # self.fid = FrechetInceptionDistance(2048)

    def _build_criterion(self):
        """Internal function to build the loss function."""
        self.loss = build_projectedloss(experiment_config=self.experiment_spec)

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for training step."""
        # Disable generator's and discriminator's gradients. And each module will be enabled if the phase is for updating it
        self.G.requires_grad_(False)
        self.D.requires_grad_(False)

        # Make sure using LightningOptimizer object wrapping the own optimizer. # Gmain, Greg, Dmain, Dreg
        phases_optimizers = self.optimizers()

        # Extract images and labels of class
        phase_real_img, phase_real_c = batch  # [16, ]
        # The batch size here is the global batch size
        batch_size = phase_real_img.size(0) * self.trainer.world_size

        # Normalize and split a batch-dim image/label tensor into tuple of batch_gpu-dim image/label tensor"s"
        phase_real_img = (phase_real_img.to(torch.float32) / 127.5 - 1).split(self.batch_gpu)  # normalize to -1 ~ 1 # [1, 16]
        phase_real_c = phase_real_c.split(self.batch_gpu)

        # Generate random z and c for the generator input.
        all_gen_z = torch.randn([len(self.phases) * batch_size, self.G.z_dim], device=self.device)  # e.g., 4 (phases) * 128 (batch_size)
        all_gen_z = [phase_gen_z.split(self.batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]  # [4, 128] -> [4, 8 (-1), 16 (batch_gpu)]
        all_gen_c = [self.dm.training_set.get_label(np.random.randint(len(self.dm.training_set))) for _ in range(len(self.phases) * batch_size)]  # 4 * 128
        all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(self.device)  # If running on CPU: all_gen_c = torch.from_numpy(np.stack(all_gen_c)).to(self.device)
        all_gen_c = [phase_gen_c.split(self.batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]  # [4, 128] -> [4, 8, 16]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c, opt in zip(self.phases, all_gen_z, all_gen_c, phases_optimizers):
            if batch_idx % phase.interval != 0:
                # main phase interval == 1
                # D_reg_interval == 16
                # G_reg_interval == 4
                continue

            # Accumulate gradients.
            opt.zero_grad(set_to_none=True)
            # Enable the specific module's gradient if the phase is for updating it
            phase.module.requires_grad_(True)

            # PROJECTED GAN ADDITIONS ###
            if phase.name in ['Dmain', 'Dboth', 'Dreg'] and hasattr(phase.module, 'feature_networks'):
                # Do not optimize feature extractor part of discriminator
                phase.module.feature_networks.requires_grad_(False)  # TODO what if feature network loaded failed?

            # Since the original batch size is divided by numbers of batch-gpu size batches,
            # accumulate them to simulate the original batch size in one optimizer step
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                self.loss.accumulate_gradients(self.G, self.D, self, phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=self.cur_nimg.cpu())
            # Disable the current phase module's gradient for the next phase
            phase.module.requires_grad_(False)

            # Update weights.
            params = [param for param in phase.module.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                # Prevent nan and inf gradients
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)
            opt.step()

        # Update G_ema.
        ema_nimg = self.ema_kimg * 1000
        if self.ema_rampup is not None:
            ema_nimg = min(ema_nimg, self.cur_nimg * self.ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.copy_(b)

        # Update state.
        self.cur_nimg += batch_size

        # Default logging with x-axis equal to step
        self.log("Seen images", self.cur_nimg,  prog_bar=True)
        # Custom logging with x-axis equal to self.cur_nimg
        self.logger.experiment.add_scalar("Seen images by samples", self.cur_nimg, self.cur_nimg // 1000)

    def calculate_fid_real_parts(self, fid_torchmetrics, imgs):
        """Custom function to update the infomation of real images for calculating fid.
        Args:
            fid_torchmetrics: torchmetric of fid (FrechetInceptionDistance).
            imgs: real images

        Returns:
            None.
        """
        fid_torchmetrics.update(imgs, real=True)

    def calculate_fid_fake_parts(self, fid_torchmetrics, fid_num=None):
        """Custom function to update the infomation of fake images for calculating fid.
        Args:
            fid_torchmetrics: torchmetric of fid (FrechetInceptionDistance).
            fid_num: Number of fake images generated for calculating fid.

        Returns:
            None.
        """
        fid_num = self.model_config['stylegan']['metrics']['num_fake_imgs']
        # Divide 50K into 8 (gpus) * multi_gpus_iter_num + 1 (gpus) * extra_single_gpu_iter_num
        multi_gpus_iter_num = fid_num // (4 * self.trainer.world_size)
        extra_single_gpu_iter_num = fid_num - multi_gpus_iter_num * (4 * self.trainer.world_size)

        for i in range(multi_gpus_iter_num):
            # w = gen_utils.get_w_from_seed(self.G_ema, 4, self.device, deterministic=True)
            # Use the above line to replace the following line if you want to calculate fid based on random seeds
            w = gen_utils.get_w_from_seed(self.G_ema, 4, self.device, seed=(i * self.trainer.world_size) + self.trainer.global_rank)
            fake_images = self.G_ema.synthesis(w)
            fake_images = (fake_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            fake_images = torch.nan_to_num(fake_images, nan=0)
            fid_torchmetrics.update(fake_images, real=False)

        if (self.trainer.global_rank == 0):
            for i in range(extra_single_gpu_iter_num):
                # w = gen_utils.get_w_from_seed(self.G_ema, 1, self.device, deterministic=True)
                # Use the above line to replace the following line if you want to calculate fid based on random seeds
                w = gen_utils.get_w_from_seed(self.G_ema, 1, self.device, seed=multi_gpus_iter_num * self.trainer.world_size + i)
                fake_images = self.G_ema.synthesis(w)
                fake_images = (fake_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                fake_images = torch.nan_to_num(fake_images, nan=0)
                fid_torchmetrics.update(fake_images, real=False)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for test step."""
        imgs, _ = batch
        # Collect information of real images from the testloader for calculating fid
        self.calculate_fid_real_parts(fid_torchmetrics=self.fid, imgs=imgs)  # TODO avoid next epoch re-calculation

    def on_test_epoch_end(self):
        """Pytorch Lightning built-in function for the end of an test epoch."""
        # Collect information of fake images generated from generator for calculating fid
        self.calculate_fid_fake_parts(fid_torchmetrics=self.fid)
        fid_score = self.fid.compute()
        # Reset FID metric for the next epoch
        self.fid.reset()
        if self.trainer.global_rank == 0:
            logging.info(f"Evaluation epoch end: {self.current_epoch}, rank: {self.trainer.global_rank}, fid_score: {fid_score}")

        return fid_score

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for validation step."""
        self.test_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        """Pytorch Lightning built-in function for the end of each validation epoch."""
        fid_score = self.on_test_epoch_end()

        # Log to screen's progress bar. Default logging with x-axis equal to step
        self.log('fid_score', fid_score, prog_bar=True)
        # Custom logging with x-axis equal to self.cur_nimg
        self.logger.experiment.add_scalar("fid_score by samples", fid_score, self.cur_nimg // 1000)
        # Log to status.json
        self.status_logging_dict = {}
        self.status_logging_dict["fid50k_full"] = float(fid_score)
        self.status_logging_dict["seen_images"] = int(self.cur_nimg)
        self.status_logging_dict["epoch"] = int(self.trainer.current_epoch)
        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Validation metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

        if fid_score <= self.best_fid and fid_score != float('-inf'):
            self.best_fid = fid_score
        else:
            logging.info(f"Validation epoch {self.current_epoch} Not improved FID score.")

    def predict_step(self, batch, batch_idx):
        """Pytorch Lightning built-in function for prediction step."""
        cpu_tensor_seeds = batch.cpu()  # cuda int tensors -> cpu int tensors

        outdir = self.experiment_spec['inference']['results_dir']
        truncation_psi = self.experiment_spec['inference']['truncation_psi']
        translate = self.experiment_spec['inference']['translate']
        rotate = self.experiment_spec['inference']['rotate']
        centroids_path = self.experiment_spec['inference']['centroids_path']
        class_idx = self.experiment_spec['inference']['class_idx']
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        for cpu_tensor_seed in cpu_tensor_seeds:
            if hasattr(self.G_ema.synthesis, 'input'):
                m = gen_utils.make_transform(translate, rotate)
                m = np.linalg.inv(m)
                self.G_ema.synthesis.input.transform.copy_(torch.from_numpy(m))
            # Generate images.
            w = gen_utils.get_w_from_seed(self.G_ema, 1, self.device, truncation_psi, seed=cpu_tensor_seed,
                                          centroids_path=centroids_path, class_idx=class_idx)
            img = gen_utils.w_to_img(self.G_ema, w, to_np=True)

            seed = int(cpu_tensor_seed)  # cpu int tensor -> int
            print("Saving", f'{outdir}/seed{seed:d}.png')
            PIL.Image.fromarray(gen_utils.create_image_grid(img), 'RGB').save(f'{outdir}/seed{seed:d}.png')

    def configure_optimizers(self):
        """Pytorch Lightning built-in function for optimizers initialization and configuration."""
        # Setup total phases first for extracting total involved optimizers
        self._setup_phases()
        # Extract optimizers from phases
        optimizers = [phase.opt for phase in self.phases if phase.opt is not None]
        return optimizers

    def _setup_phases(self):
        """Internal function to setup training phases."""
        logging.info('Setting up training phases...')
        iter_dict = {'G': 1, 'D': 1}  # change here if you want to do several G/D iterations at once
        G_opt_kwargs = dnnlib.EasyDict(self.experiment_spec['train']['stylegan']['optim_generator'])
        D_opt_kwargs = dnnlib.EasyDict(self.experiment_spec['train']['stylegan']['optim_discriminator'])
        for name, module, opt_kwargs, reg_interval in [('G', self.G, G_opt_kwargs, self.G_reg_interval), ('D', self.D, D_opt_kwargs, self.D_reg_interval)]:
            if reg_interval is None:
                logging.info("NO lazy regularization.")
                opt = self.create_optimizer(module, opt_kwargs)
                for _ in range(iter_dict[name]):
                    self.phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
            else:  # Lazy regularization.
                logging.info(f"{name} Lazy regularization.")
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = self.create_optimizer(module, opt_kwargs)
                for _ in range(iter_dict[name]):
                    self.phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
                self.phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    def create_optimizer(self, module, opt_kwargs):
        """Custom function for creating optimizer instances.
        Args:
            module: the module for the optimizer to update the parameters
            opt_kwargs: number of fake images generated for calculating fid.

        Returns:
            optimizer_instance: optimizer instance
        """
        if opt_kwargs.optim == "Adam":
            optimizer_instance = optim.Adam(module.parameters(), lr=opt_kwargs.lr, eps=opt_kwargs.eps, betas=opt_kwargs.betas)
            return optimizer_instance
        else:
            raise NotImplementedError("Optimizer {} is not implemented".format(opt_kwargs.optim))

    def configure_callbacks(self):
        """Pytorch Lightning built-in function for setting up callbacks."""
        callbacks = super().configure_callbacks()
        # Load pretrained submodeuls such as InceptionNet for FID and class embeddings for discriminator and generator
        submodules_checkpoint_loader = SubmodulesCheckpointLoader()
        # Enable stem checkpoint loading for training super-resolution
        stem_checkpoint_loader = StemCheckpointLoader()
        # Enable exporting sample images during training for monitoring
        sample_images_exporter = SampleImagesExporter()
        callbacks.append(submodules_checkpoint_loader)
        callbacks.append(stem_checkpoint_loader)
        callbacks.append(sample_images_exporter)
        return callbacks
