r""" Hypercorrelation Squeeze Network """
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.diffusion.ddim import DDIMSampler
from models.feature_backbones.autoencoder import VQModelInterface
from models.generator import ShallowAdaptiveFeatureGenerator, ShallowAdaptiveFeatureGeneratorForLabel
from models.ldm_util import get_model

# fordebug
from util.util import fbcheck, feature_normalize, get_flow, img_denorm, soft_warp

from .base.correlation import Correlation
from .base.feature import extract_feat_clip, extract_feat_res, extract_feat_vgg

"""
Latent Diffusion Autoencoder Configs
"""
vq_f4_config = yaml.safe_load(
    """
params:
    embed_dim: 3
    n_embed: 8192
    ckpt_path: models/vq-f4/model.ckpt
    ddconfig:
        double_z: false
        z_channels: 3
        resolution: 256
        in_channels: 3
        out_ch: 3
        ch: 128
        ch_mult:
        - 1
        - 2
        - 4
        num_res_blocks: 2
        attn_resolutions: []
        dropout: 0.0
    lossconfig:
        target: torch.nn.Identity
"""
)


def denoised_fn(x_start, inpaint_mask, trg_feat):
    """
    x_start : q_sampled hijack_feat (warped ref feat)
    trg_feat : q_sampled trg feat
    """
    return x_start * inpaint_mask + trg_feat * (1 - inpaint_mask)


class MIDMs(nn.Module):
    def __init__(
        self,
        backbone="clip_resnet101",
        use_original_imgsize=False,
        train=True,
        phase="corr",
        label_nc=None,
        diffusion_iteratiom_num=2,
        diffusion_config_path=None,
        diffusion_model_path=None,
        pos_embed="learnable",
        confidence_masking=True,
        maskmix=True,
        is_classification=False,
        device=None,
    ):
        super().__init__()
        self.diffusion_config_path = diffusion_config_path
        self.diffusion_model_path = diffusion_model_path
        self.diff_model = get_model(
            train, config_path=diffusion_config_path, model_path=diffusion_model_path, device=device
        )
        self.diffusion_sampler = DDIMSampler(self.diff_model)
        self.diff_model.first_stage_model.eval()
        self.diff_model.model.eval()
        self.diff_model.eval()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        N_CH = 256

        if label_nc == 150:
            nc = label_nc + 1 if maskmix else 0
        else:
            nc = label_nc if maskmix else 0
        if self.diff_model.first_stage_model.__class__.__name__ == "AutoencoderKL":
            self.real_enc = ShallowAdaptiveFeatureGenerator(ch=N_CH, inch=4, ic=4)
            self.iter_enc = ShallowAdaptiveFeatureGenerator(ch=N_CH, ic=N_CH, inch=4)
            self.seg_enc = ShallowAdaptiveFeatureGeneratorForLabel(
                label_nch=nc if label_nc == 150 else label_nc, ch=N_CH
            )
        else:
            self.real_enc = ShallowAdaptiveFeatureGenerator(ch=N_CH)
            self.iter_enc = ShallowAdaptiveFeatureGenerator(ch=N_CH, ic=N_CH)
            self.seg_enc = ShallowAdaptiveFeatureGeneratorForLabel(
                label_nch=nc if label_nc == 150 else label_nc, ch=N_CH
            )
        self.phi = nn.Conv2d(in_channels=N_CH + nc, out_channels=N_CH, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=N_CH + nc, out_channels=N_CH, kernel_size=1, stride=1, padding=0)
        # self.backbone_src = VQModelInterface(**vq_f4_config['params']).eval()

        self.layer = nn.Sequential(
            ResidualBlock(N_CH + nc, N_CH + nc, kernel_size=3, padding=1, stride=1),
            ResidualBlock(N_CH + nc, N_CH + nc, kernel_size=3, padding=1, stride=1),
            ResidualBlock(N_CH + nc, N_CH + nc, kernel_size=3, padding=1, stride=1),
            ResidualBlock(N_CH + nc, N_CH + nc, kernel_size=3, padding=1, stride=1),
        )

        for p_s in self.diff_model.first_stage_model.parameters():
            p_s.requires_grad_(False)

        for p_s in self.diff_model.model.parameters():
            p_s.requires_grad_(False)

        for p_s in self.diff_model.parameters():
            p_s.requires_grad_(False)

        self.phase = phase
        self.fb_stack = []
        self.diffusion_iteratiom_num = diffusion_iteratiom_num
        self.pe = pos_embed
        self.confidence_masking = confidence_masking
        self.device = device
        self.maskmix = maskmix

    def set_copy(self):
        self.copied_diff_model = get_model(
            False, config_path=self.diffusion_config_path, model_path=self.diffusion_model_path, device=self.device
        )
        self.copied_diffusion_sampler = DDIMSampler(self.diff_model)
        self.copied_diff_model.first_stage_model.eval()
        self.copied_diff_model.model.eval()
        self.copied_diff_model.eval()

    def forward(
        self,
        trg_img=None,
        src_img=None,
        ref_img=None,
        input_semantics=None,
        ref_semantics=None,
        mode="default",
        iter=None,
        warmup_iter=None,
        warp_bilinear=None,
    ):
        ### For debug
        debug_stack_warped_feats = {}

        # first matching
        trg_feats = trg_img
        if self.diff_model.first_stage_model.__class__.__name__ == "AutoencoderKL":
            src_feats = self.diff_model.first_stage_model.encode(src_img).sample().detach()
            ref_feats = self.diff_model.first_stage_model.encode(ref_img).sample().detach()
        else:
            src_feats = self.diff_model.first_stage_model.encode(src_img).detach()
            ref_feats = self.diff_model.first_stage_model.encode(ref_img).detach()

        input_semantics = F.interpolate(input_semantics, size=ref_feats.shape[2:4], mode="nearest")
        trg_matching_feats = self.seg_enc(input_semantics)
        src_matching_feats = self.real_enc(src_feats)
        ref_matching_feats = self.real_enc(ref_feats)

        ref_semantics = F.interpolate(ref_semantics, size=ref_feats.shape[2:], mode="nearest")
        trg_out = self.theta(self.layer(torch.cat([trg_matching_feats, input_semantics], dim=1)))
        src_out = self.phi(self.layer(torch.cat([src_matching_feats, input_semantics], dim=1)))
        ref_out = self.phi(self.layer(torch.cat([ref_matching_feats, ref_semantics], dim=1)))

        trg_out = trg_out - trg_out.mean(dim=1, keepdim=True)
        src_out = src_out - src_out.mean(dim=1, keepdim=True)
        ref_out = ref_out - ref_out.mean(dim=1, keepdim=True)

        trg_out = torch.div(trg_out, torch.norm(trg_out, 2, 1, keepdim=True))
        src_out = torch.div(src_out, torch.norm(src_out, 2, 1, keepdim=True))
        ref_out = torch.div(ref_out, torch.norm(ref_out, 2, 1, keepdim=True))

        corr_src = Correlation.correlation(trg_out, src_out)
        corr_ref = Correlation.correlation(trg_out, ref_out)

        warped_src_feat = soft_warp(src_feats, corr_src, factor=1)
        warped_ref_feat = soft_warp(ref_feats, corr_ref, factor=1)
        warped_ref_seg = soft_warp(ref_semantics, corr_ref, factor=1)

        shape = [
            self.diff_model.model.diffusion_model.in_channels,
            self.diff_model.model.diffusion_model.image_size,
            self.diff_model.model.diffusion_model.image_size,
        ]

        # diffusion process w/matching
        with torch.no_grad() if mode == "inference" else contextlib.nullcontext():  # training# training

            original_loss = F.mse_loss(warped_src_feat, src_feats.detach()) if iter < warmup_iter else 0
            if self.phase == "corr":
                warped_src = self.diff_model.first_stage_model.decode(warped_src_feat)
                warped_ref = self.diff_model.first_stage_model.decode(warped_ref_feat)
            else:
                if self.phase == "diff" or self.phase == "e2e":
                    t_list = np.sort(np.random.choice(50, self.diffusion_iteratiom_num, replace=False)) + 150
                    ddim_num_steps = 200
                # default setting: e2e_recurrent
                elif self.phase == "e2e_recurrent":
                    # number of sampling step (inference)
                    ddim_num_steps = 16 if mode == "inference" else self.diffusion_iteratiom_num * 4
                    t_list = np.sort(np.random.choice(ddim_num_steps // 4, ddim_num_steps // 4, replace=False)) + (
                        ddim_num_steps // 4 * 3
                    )
                    if iter < warmup_iter:
                        t_list = []
                        iter_warped_ref_seg = warped_ref_seg
                else:
                    raise Exception("self.phase should be diff or e2e or e2e_recurrent")

                rand_idx = np.random.randint(0, len(t_list)) if len(t_list) != 0 else 0
                pred_ref_x0 = warped_ref_feat.clone()
                pred_src_x0 = warped_src_feat.clone()

                # For debug
                if mode == "inference":
                    debug_stack_warped_feats["pred_ref_x0"] = pred_ref_x0

                ddim_param_ref = (None, None)
                ddim_param_src = (None, None)
                loss_fb = (
                    F.l1_loss(
                        soft_warp(warped_ref_feat, corr_ref.permute(0, 3, 4, 1, 2), factor=1), ref_feats.detach()
                    )
                    if iter < warmup_iter
                    else 0
                )

                for idx, t in enumerate(t_list):
                    t = torch.tensor(t).to(warped_ref_feat.device)

                    trigger = True  # (mode == 'inference') or (iter > 10000 and iter % 2 == 0)
                    # inter-domain warping
                    # import pdb; pdb.set_trace()
                    if idx in range(1, 3) and self.phase == "e2e_recurrent" and trigger:

                        if self.backbone_type == "VQGAN":
                            inter_domain_matching_feat = self.iter_enc(pred_ref_x0, trg_matching_feats)
                            src_inter_domain_matching_feat = self.iter_enc(pred_src_x0, trg_matching_feats)

                            inter_domain_matching_feat = inter_domain_matching_feat - inter_domain_matching_feat.mean(
                                dim=1, keepdim=True
                            )
                            inter_domain_matching_feat = torch.div(
                                inter_domain_matching_feat, torch.norm(inter_domain_matching_feat, 2, 1, keepdim=True)
                            )
                            src_inter_domain_matching_feat = (
                                src_inter_domain_matching_feat
                                - src_inter_domain_matching_feat.mean(dim=1, keepdim=True)
                            )
                            src_inter_domain_matching_feat = torch.div(
                                src_inter_domain_matching_feat,
                                torch.norm(src_inter_domain_matching_feat, 2, 1, keepdim=True),
                            )

                            inter_domian_ref_feats = ref_matching_feats - ref_matching_feats.mean(dim=1, keepdim=True)
                            inter_domian_ref_feats = torch.div(
                                inter_domian_ref_feats, torch.norm(inter_domian_ref_feats, 2, 1, keepdim=True)
                            )
                            src_inter_domian_ref_feats = src_matching_feats - src_matching_feats.mean(
                                dim=1, keepdim=True
                            )
                            src_inter_domian_ref_feats = torch.div(
                                src_inter_domian_ref_feats, torch.norm(src_inter_domian_ref_feats, 2, 1, keepdim=True)
                            )

                            corr_inter = Correlation.correlation(inter_domain_matching_feat, inter_domian_ref_feats)
                            src_corr_inter = Correlation.correlation(
                                src_inter_domain_matching_feat, src_inter_domian_ref_feats
                            )

                            pred_ref_x0_new = soft_warp(ref_feats, corr_inter, factor=1)
                            iter_warped_ref_seg = soft_warp(ref_semantics, corr_inter, factor=1)
                            src_pred_ref_x0_new = soft_warp(src_feats, src_corr_inter, factor=1)
                            fb_ref = soft_warp(pred_ref_x0_new, corr_inter.permute(0, 3, 4, 1, 2), factor=1)
                            src_fb_ref = soft_warp(
                                src_pred_ref_x0_new, src_corr_inter.permute(0, 3, 4, 1, 2), factor=1
                            )
                            loss_fb += F.l1_loss(fb_ref, ref_feats.detach())
                            loss_fb += F.l1_loss(src_fb_ref, src_feats.detach())
                        else:
                            raise NotImplementedError()

                        fb_out = fbcheck(*get_flow(corr_inter))[0].unsqueeze(1)
                        fb_mask = (fb_out < 0.5).float()
                        upsampler = nn.Upsample(scale_factor=2, mode="nearest")
                        fb_mask = (upsampler(F.avg_pool2d(fb_mask, 2)) > 0.5).float().detach()
                        pred_ref_x0 = (
                            pred_ref_x0_new * fb_mask + pred_ref_x0 * (1 - fb_mask)
                            if self.confidence_masking
                            else pred_ref_x0_new
                        )

                        if mode == "inference":
                            debug_stack_warped_feats[f"{idx}_fb_mask"] = fb_mask
                            debug_stack_warped_feats[f"{idx}_pred_ref_x0_new"] = pred_ref_x0_new
                            debug_stack_warped_feats[f"{idx}_pred_ref_x0"] = pred_ref_x0

                    _, pred_ref_x0, ddim_param_ref = self.diffusion_sampler.one_step_sample(
                        ddim_num_steps,
                        batch_size=trg_img.shape[0],
                        shape=shape,
                        eta=0,
                        verbose=False,
                        hijack_feat=pred_ref_x0,
                        hijack_step=t,
                        ddim_param=ddim_param_ref,
                        denoised_fn=None,
                        inpaint_mask=None,
                        trg_feat=None,
                    )
                    _, pred_src_x0, ddim_param_src = self.diffusion_sampler.one_step_sample(
                        ddim_num_steps,
                        batch_size=trg_img.shape[0],
                        shape=shape,
                        eta=0,
                        verbose=False,
                        hijack_feat=pred_src_x0,
                        hijack_step=t,
                        ddim_param=ddim_param_src,
                        denoised_fn=None,
                        inpaint_mask=None,
                        trg_feat=None,
                    )

                    # For original Diffusion Loss
                    if idx == rand_idx:
                        _, pred_src_x0_original, _ = self.diffusion_sampler.one_step_sample(
                            ddim_num_steps,
                            batch_size=trg_img.shape[0],
                            shape=shape,
                            eta=0,
                            verbose=False,
                            hijack_feat=src_feats,
                            hijack_step=t,
                            ddim_param=(None, None),
                            denoised_fn=None,
                            inpaint_mask=None,
                            trg_feat=None,
                        )
                        original_loss += F.mse_loss(pred_src_x0_original, src_feats.detach())

                if mode == "inference":
                    pred_ref_x0, _ = self.copied_diffusion_sampler.sample(
                        200,
                        batch_size=trg_img.shape[0],
                        shape=shape,
                        eta=1,
                        verbose=False,
                        hijack_feat=pred_ref_x0,
                        hijack_step=180,
                        denoised_fn=None,
                        inpaint_mask=None,
                    )

                warped_ref = self.diff_model.first_stage_model.decode(pred_ref_x0)
                warped_src = self.diff_model.first_stage_model.decode(pred_src_x0)

                if mode == "inference":
                    debug_stack = [
                        self.diff_model.first_stage_model.decode(v)
                        for k, v in debug_stack_warped_feats.items()
                        if not k.endswith("fb_mask")
                    ]
                    debug_stack += [
                        F.interpolate(v, size=warped_ref.shape[2:4]).repeat(1, 3, 1, 1)
                        for k, v in debug_stack_warped_feats.items()
                        if k.endswith("fb_mask")
                    ]

                    return warped_ref, torch.vstack(debug_stack), fb_mask

                else:
                    debug_stack = None

            return (
                warped_src,
                warped_ref,
                debug_stack,
                corr_src,
                corr_ref,
                trg_matching_feats,
                src_matching_feats,
                ref_matching_feats,
                trg_feats,
                src_feats,
                ref_feats,
                original_loss,
                loss_fb,
                iter_warped_ref_seg,
            )

    def train_mode(self):
        self.train()
        self.diff_model.first_stage_model.eval()

        if self.diff_freeze:
            self.diff_model.model.eval()
            self.diff_model.eval()

    def switch_freeze(self, freeze):

        if (not self.phase.startswith("e2e")) and (freeze == False):
            print("can't switch to unfreeze!!")
            return

        self.diff_freeze = freeze

        if freeze:
            for p_s in self.diff_model.model.parameters():
                p_s.requires_grad_(False)
            for p_s in self.diff_model.parameters():
                p_s.requires_grad_(False)
            self.train_mode()
        else:
            for p_s in self.diff_model.model.parameters():
                p_s.requires_grad_(True)
            self.train_mode()

    def reload_ddpm(self):
        self.diff_model = get_model()
        self.diffusion_sampler = DDIMSampler(self.diff_model)
        self.diff_model.first_stage_model.eval()
        self.diff_model.eval()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out
