# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets"
#

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import dnnlib
import legacy

from pg_modules.blocks import Interpolate
import timm
from pg_modules.projector import get_backbone_normstats

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0,
                 train_head_only=False, style_mixing_prob=0.0, pl_weight=0.0,
                 cls_model='efficientnet_b1', cls_weight=0.0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_head_only = train_head_only

        # SG2 techniques
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_no_weight_grad = True
        self.pl_mean = torch.zeros([], device=device)

        self.neural_rendering_resolution_initial = 128
        self.neural_rendering_resolution_final = 128
        self.neural_rendering_resolution_fade_kimg = 1000

        # classifier guidance
        cls = timm.create_model(cls_model, pretrained=True).eval()
        self.classifier = nn.Sequential(Interpolate(224), cls).to(device)
        normstats = get_backbone_normstats(cls_model)
        self.norm = Normalize(normstats['mean'], normstats['std'])
        self.cls_weight = cls_weight
        self.cls_guidance_loss = torch.nn.CrossEntropyLoss()

    def run_G(self, z, c, neural_rendering_resolution, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)['image']  # enabling emas leads to collapse with PG
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        return self.D(img, c)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if phase in ['Dreg']: return  # no regularization
        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:

            # disable gradients for superres
            if self.train_head_only:
                self.G.backbone.mapping.requires_grad_(False)
                for name in self.G.backbone.synthesis.layer_names:
                    getattr(self.G.backbone.synthesis, name).requires_grad_(name in self.G.backbone.head_layer_names)

            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)

                loss_Gmain = sum([(-l).mean() for l in gen_logits])
                gen_logits = torch.cat(gen_logits)

                if self.cls_weight:
                    gen_img = self.norm(gen_img.add(1).div(2)) 
                    guidance_loss = self.cls_guidance_loss(self.classifier(gen_img), gen_c.argmax(1))
                    loss_Gmain += self.cls_weight * guidance_loss
                    training_stats.report('Loss/G/guidance_loss', guidance_loss)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':

            ws = self.G.mapping(gen_z, gen_c, update_emas=False)
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, neural_rendering_resolution,update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])
                gen_logits = torch.cat(gen_logits)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            name = 'Dreal'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])
                real_logits = torch.cat(real_logits)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
            with torch.autograd.profiler.record_function(name + '_backward'):
                loss_Dreal.backward()
