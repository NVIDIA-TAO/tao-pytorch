# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.transforms import Normalize

from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import conv2d_gradfix
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import upfirdn2d
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.blocks import Interpolate
from nvidia_tao_pytorch.sdg.stylegan_xl.model.discriminator.projector import get_backbone_normstats


class Loss(nn.Module):
    """Abstract base class for loss functions used in GAN training."""

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):  # to be overridden by subclass
        """Accumulates gradients for the loss function during training.

        This method is intended to be overridden by subclasses to implement the specific
        gradient accumulation logic for different GAN loss functions.

        Args:
            phase (str): The current training phase (e.g., 'Gmain', 'Dmain').
            real_img (torch.Tensor): The real images.
            real_c (torch.Tensor): The real image conditions.
            gen_z (torch.Tensor): The generated latent vectors.
            gen_c (torch.Tensor): The generated image conditions.
            gain (float): The learning rate gain.
            cur_nimg (int): The current number of images processed.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    """Loss function for Projected GAN, implementing various regularizations."""

    def __init__(self, blur_init_sigma=0, blur_fade_kimg=0,
                 train_head_only=False, style_mixing_prob=0.0, pl_weight=0.0,
                 cls_model='efficientnet_b1', cls_weight=0.0, **kwargs):
        """Initializes the ProjectedGANLoss.

        Args:
            blur_init_sigma (float, optional): Initial sigma for image blurring. Default is 0.
            blur_fade_kimg (int, optional): Number of thousands of images for blurring to fade. Default is 0.
            train_head_only (bool, optional): Whether to train only the head layers. Default is False.
            style_mixing_prob (float, optional): Probability of applying style mixing regularization. Default is 0.0.
            pl_weight (float, optional): Weight for path length regularization. Default is 0.0.
            cls_model (str, optional): The model name for classifier guidance. Default is 'efficientnet_b1'.
            cls_weight (float, optional): Weight for classifier guidance loss. Default is 0.0.
        """
        super().__init__()
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_head_only = train_head_only

        # SG2 techniques
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_no_weight_grad = True
        self.register_buffer('pl_mean', torch.zeros([]))
        # # classifier guidance
        cls = timm.create_model(cls_model, pretrained=True).eval()
        self.classifier = nn.Sequential(Interpolate(224), cls)
        normstats = get_backbone_normstats(cls_model)
        self.norm = Normalize(normstats['mean'], normstats['std'])
        self.cls_weight = cls_weight
        self.cls_guidance_loss = torch.nn.CrossEntropyLoss()

    def run_G(self, G, z, c, update_emas=False):
        """Runs the generator to produce images.

        Args:
            G (nn.Module): The generator model.
            z (torch.Tensor): The latent vectors.
            c (torch.Tensor): The conditioning vectors.
            update_emas (bool, optional): Whether to update exponential moving averages. Default is False.

        Returns:
            tuple: Generated images and style vectors (ws).
        """
        ws = G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = G.synthesis(ws, update_emas=False)  # enabling emas leads to collapse with PG
        return img, ws

    def run_D(self, D, img, c, blur_sigma=0, update_emas=False):
        """Runs the discriminator on the provided images.

        Args:
            D (nn.Module): The discriminator model.
            img (torch.Tensor): The input images.
            c (torch.Tensor): The conditioning vectors.
            blur_sigma (float, optional): Sigma for Gaussian blur. Default is 0.
            update_emas (bool, optional): Whether to update exponential moving averages. Default is False.

        Returns:
            torch.Tensor: The discriminator's output logits.
        """
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        return D(img, c)

    def accumulate_gradients(self, G, D, pl_module, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        """Accumulates gradients for the generator and discriminator losses.

        Args:
            G (nn.Module): The generator model.
            D (nn.Module): The discriminator model.
            pl_module (nn.Module): The model to log and report training stats.
            phase (str): The current training phase (e.g., 'Gmain', 'Dmain').
            real_img (torch.Tensor): The real images.
            real_c (torch.Tensor): The real image conditions.
            gen_z (torch.Tensor): The generated latent vectors.
            gen_c (torch.Tensor): The generated image conditions.
            gain (float): The learning rate gain.
            cur_nimg (int): The current number of images processed.
        """
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg']:
            return  # no regularization

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        # Gmain: Maximize logits for generated images.
        if do_Gmain:

            # disable gradients for superres
            if self.train_head_only:
                # Only optimize newly added head for ProGAN training
                G.mapping.requires_grad_(False)
                for name in G.synthesis.layer_names:
                    getattr(G.synthesis, name).requires_grad_(name in G.head_layer_names)

            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(G=G, z=gen_z, c=gen_c)  # pylint: disable=unused-variable
                gen_logits = self.run_D(D, gen_img, gen_c, blur_sigma=blur_sigma)

                loss_Gmain = sum([(-l).mean() for l in gen_logits])  # noqa: E741
                gen_logits = torch.cat(gen_logits)

                if self.cls_weight:
                    gen_img = self.norm(gen_img.add(1).div(2))
                    guidance_loss = self.cls_guidance_loss(self.classifier(gen_img), gen_c.argmax(1))
                    loss_Gmain += self.cls_weight * guidance_loss
                    pl_module.log('Loss/G/guidance_loss', guidance_loss)
                    pl_module.logger.experiment.add_scalar("Loss/G/guidance_loss by samples", guidance_loss, cur_nimg // 1000)

                pl_module.log('Loss/scores/fake', gen_logits.mean())
                pl_module.log('Loss/signs/fake', gen_logits.sign().mean())
                pl_module.log('Loss/G/loss', loss_Gmain)
                pl_module.logger.experiment.add_scalar('Loss/scores/fake by samples', gen_logits.mean(), cur_nimg // 1000)
                pl_module.logger.experiment.add_scalar('Loss/signs/fake by samples', gen_logits.sign().mean(), cur_nimg // 1000)
                pl_module.logger.experiment.add_scalar('Loss/G/loss by samples', loss_Gmain, cur_nimg // 1000)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                # loss_Gmain.backward()
                pl_module.manual_backward(loss_Gmain)

        # Gpl: Apply path length regularization.
        start_plreg = (cur_nimg >= 1e6)
        if start_plreg and self.pl_weight and phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(G, gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                pl_module.log('Loss/pl_penalty', pl_penalty.mean())
                pl_module.logger.experiment.add_scalar('Loss/pl_penalty by samples', pl_penalty.mean(), cur_nimg // 1000)

                loss_Gpl = pl_penalty * self.pl_weight
                pl_module.log('Loss/G/reg', loss_Gpl.mean())
                pl_module.logger.experiment.add_scalar('Loss/G/reg by samples', loss_Gpl.mean(), cur_nimg // 1000)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                # loss_Gpl.mean().backward()
                pl_module.manual_backward(loss_Gpl.mean())

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(G, gen_z, gen_c, update_emas=True)  # pylint: disable=unused-variable
                gen_logits = self.run_D(D, gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])  # noqa: E741
                gen_logits = torch.cat(gen_logits)  # noqa: E741

                pl_module.log('Loss/scores/fake', gen_logits.mean())
                pl_module.log('Loss/signs/fake', gen_logits.sign().mean())
                pl_module.logger.experiment.add_scalar('Loss/scores/fake by samples', gen_logits.mean(), cur_nimg // 1000)
                pl_module.logger.experiment.add_scalar('Loss/signs/fake by samples', gen_logits.sign().mean(), cur_nimg // 1000)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                # loss_Dgen.backward()
                pl_module.manual_backward(loss_Dgen)

            # Dmain: Maximize logits for real images.
            name = 'Dreal'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(D, real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])  # noqa: E741
                real_logits = torch.cat(real_logits)

                pl_module.log('Loss/scores/real', real_logits.mean())
                pl_module.log('Loss/signs/real', real_logits.sign().mean())
                pl_module.log('Loss/D/loss', loss_Dgen + loss_Dreal)
                pl_module.logger.experiment.add_scalar('Loss/scores/real by samples', real_logits.mean(), cur_nimg // 1000)
                pl_module.logger.experiment.add_scalar('Loss/signs/real by samples', real_logits.sign().mean(), cur_nimg // 1000)
                pl_module.logger.experiment.add_scalar('Loss/D/loss by samples', loss_Dgen + loss_Dreal, cur_nimg // 1000)

            with torch.autograd.profiler.record_function(name + '_backward'):
                # loss_Dreal.backward()
                pl_module.manual_backward(loss_Dreal)
