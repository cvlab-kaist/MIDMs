import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from util.util import feature_normalize, img_denorm, masktorgb, soft_warp


def train_epoch(
    net,
    optimizer,
    train_loader,
    device,
    epoch,
    train_writer,
    vgg=None,
    contextureLoss=None,
    is_amp=False,
    hijack_step=100,
    warmup_iter=5000,
    save_path=None,
):
    n_iter = epoch * len(train_loader)
    net.module.switch_freeze(True)
    net.module.train_mode()
    running_total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    net.module.fb_stack = []
    for i, mini_batch in pbar:

        optimizer.zero_grad()

        # diffusion freeze warmup
        if n_iter >= warmup_iter and net.module.diff_freeze and net.module.phase.startswith("e2e"):
            print("Diffusion Model Freeze : False")
            net.module.switch_freeze(False)

        src = mini_batch["image"].to(device)
        ref = mini_batch["ref"].to(device)

        if train_loader.dataset.__class__.__name__ == "DeepFashionDataset":
            input_semantics = mini_batch["label"].to(device)

            input_semantics = mini_batch["label"].clone().to(device).float()
            mini_batch["label"] = mini_batch["label"][:, :3, :, :]
            ref_semantics = mini_batch["label_ref"].clone().to(device).float()
            mini_batch["label_ref"] = mini_batch["label_ref"][:, :3, :, :]

            trg = mini_batch["label"][:, 0:3].to(device)
            ref_trg = mini_batch["label_ref"][:, 0:3].to(device)
            weight_mask = 0.0
            which_perceptual = -2
            weight_perceptual = 0.002
            warp_bilinear = True

        elif train_loader.dataset.__class__.__name__ == "CelebAHQEdgeDataset":

            input_semantics = mini_batch["label"].clone().to(device).float()
            mini_batch["label"] = mini_batch["label"][:, :1, :, :]
            ref_semantics = mini_batch["label_ref"].clone().to(device).float()
            mini_batch["label_ref"] = mini_batch["label_ref"][:, :1, :, :]
            mini_batch["label"] = mini_batch["label"].long()
            mini_batch["label_ref"] = mini_batch["label_ref"].long()

            trg = mini_batch["label"][:, :1].repeat(1, 3, 1, 1).to(device)
            ref_trg = mini_batch["label_ref"][:, :1].repeat(1, 3, 1, 1).to(device)
            weight_mask = 0.0
            which_perceptual = -2
            weight_perceptual = 0.002
            warp_bilinear = False

        elif train_loader.dataset.__class__.__name__ == "LSUNChurchDataset":
            label_map = mini_batch["label"].long()
            bs, _, h, w = label_map.size()
            nc = 151
            input_label = torch.zeros((bs, nc, h, w)).float()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

            label_map = mini_batch["label_ref"].long()
            label_ref = torch.zeros((bs, nc, h, w)).float()
            ref_semantics = label_ref.scatter_(1, label_map, 1.0)
            trg = mini_batch["label"].to(device)
            ref_trg = mini_batch["label_ref"].to(device)
            weight_mask = 100.0
            which_perceptual = -1
            weight_perceptual = 0.05
            warp_bilinear = False

        else:
            raise ValueError("Unknown dataset")

        with torch.cuda.amp.autocast(enabled=is_amp):
            (
                warped_src,
                warped_ref,
                debug_stack,
                pred_corr_src,
                pred_corr_ref,
                trg_matching_feats,
                src_matching_feats,
                ref_matching_feats,
                trg_feats,
                src_feats,
                ref_feats,
                original_loss,
                loss_fb_iterative,
                iter_warped_ref_seg,
            ) = net(
                trg,
                src,
                ref,
                input_semantics=input_semantics,
                ref_semantics=ref_semantics,
                iter=n_iter,
                warmup_iter=warmup_iter,
                warp_bilinear=warp_bilinear,
            )

            original_loss = original_loss.sum()
            loss_fb_iterative = loss_fb_iterative.sum()

            fb_tmp = soft_warp(ref_feats, pred_corr_ref, factor=1)
            fb_ref = soft_warp(fb_tmp, pred_corr_ref.permute(0, 3, 4, 1, 2), factor=1)
            loss_fb = F.l1_loss(fb_ref, ref_feats.detach())

            normalied_ref_feats = feature_normalize(ref_matching_feats).detach()
            loss_domain = F.l1_loss(
                feature_normalize(trg_matching_feats), feature_normalize(src_matching_feats)
            ) + F.l1_loss(
                normalied_ref_feats * feature_normalize(trg_matching_feats),
                normalied_ref_feats * feature_normalize(src_matching_feats),
            )

            warped_src_feat = vgg(warped_src, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            warped_ref_feat = vgg(warped_ref, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            src_feat = vgg(src, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            ref_feat = vgg(ref, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            loss_feat = 0
            weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

            for i in range(len(warped_src_feat)):
                loss_feat += weights[i] * F.l1_loss(warped_src_feat[i], src_feat[i].detach())

            if train_loader.dataset.__class__.__name__ == "LSUNChurchDataset":
                scale_factor = 0.125 if train_loader.dataset.__class__.__name__ == "LSUNChurchDataset" else 0.25
                ref_trg = F.interpolate(ref_trg.float(), scale_factor=scale_factor, mode="nearest").long().squeeze(1)
                gt_label = F.interpolate(trg.float(), scale_factor=scale_factor, mode="nearest").long().squeeze(1)
                weights = []
                for i in range(ref_trg.shape[0]):
                    ref_trg_uniq = torch.unique(ref_trg[i])
                    gt_label_uniq = torch.unique(gt_label[i])
                    zero_label = [it for it in gt_label_uniq if it not in ref_trg_uniq]
                    weight = torch.ones_like(gt_label[i]).float()
                    for j in zero_label:
                        weight[gt_label[i] == j] = 0
                    weight[gt_label[i] == 0] = 0  # no loss from unknown class
                    weights.append(weight.unsqueeze(0))
                weights = torch.cat(weights, dim=0)
                warped_segmentation_loss = (
                    (
                        F.nll_loss(torch.log(iter_warped_ref_seg.contiguous() + 1e-10), gt_label, reduce=False)
                        * weights
                    ).sum()
                    / (weights.sum() + 1e-5)
                    * weight_mask
                )

            loss_perc = F.l1_loss(warped_ref_feat[which_perceptual], src_feat[which_perceptual].detach())

            contextual_style5_1 = torch.mean(contextureLoss(warped_ref_feat[-1], ref_feat[-1].detach())) * 8
            contextual_style4_1 = torch.mean(contextureLoss(warped_ref_feat[-2], ref_feat[-2].detach())) * 4
            contextual_style3_1 = (
                torch.mean(
                    contextureLoss(F.avg_pool2d(warped_ref_feat[-3], 2), F.avg_pool2d(ref_feat[-3].detach(), 2))
                )
                * 2
            )
            contextual_style2_1 = (
                torch.mean(
                    contextureLoss(F.avg_pool2d(warped_ref_feat[-4], 4), F.avg_pool2d(ref_feat[-4].detach(), 4))
                )
                * 1
            )
            loss_context = contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1

            Loss = (
                loss_domain * 10
                + loss_fb
                + loss_perc * weight_perceptual
                + loss_feat
                + original_loss
                + loss_fb_iterative
            )

            if train_loader.dataset.__class__.__name__ == "LSUNChurchDataset" and n_iter < warmup_iter:
                Loss += warped_segmentation_loss

            if n_iter > warmup_iter:
                Loss += loss_context

        if is_amp:
            scaler.scale(Loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            Loss.backward()
            optimizer.step()

        if device == 0:
            train_writer.add_scalar("train_iter/loss_total", Loss.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_domain", loss_domain.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_perc", loss_perc.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_context", loss_context.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_fb", loss_fb.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_feat", loss_feat.item(), n_iter)
            train_writer.add_scalar("train_iter/original_loss", original_loss.item(), n_iter)
            train_writer.add_scalar("train_iter/loss_fb_iterative", loss_fb_iterative.item(), n_iter)
            if train_loader.dataset.__class__.__name__ == "LSUNChurchDataset":
                train_writer.add_scalar("train_iter/warped_segmentation_loss", warped_segmentation_loss.item(), n_iter)

        # For visualize
        if n_iter % 500 == 0:
            if train_loader.dataset.__class__.__name__ == "LSUNChurchDataset":
                label = masktorgb(mini_batch["label"].long().cpu().numpy())
                label = torch.from_numpy(label).float() / 128 - 1
                save_image(
                    torch.cat(
                        [
                            img_denorm(src).clone().detach().cpu(),
                            label.clone().detach().cpu(),
                            img_denorm(ref).clone().detach().cpu(),
                            img_denorm(warped_ref).clone().detach().cpu(),
                        ],
                        dim=0,
                    ),
                    f"{save_path}/{str(n_iter)}_R{device}_result.jpg",
                    nrow=label.size(0),
                    normalize=False,
                )
            else:
                save_image(
                    torch.cat(
                        [
                            img_denorm(src).clone().detach().cpu(),
                            img_denorm(trg).clone().detach().cpu(),
                            img_denorm(ref).clone().detach().cpu(),
                            img_denorm(warped_ref).clone().detach().cpu(),
                        ],
                        dim=0,
                    ),
                    f"{save_path}/{str(n_iter)}_R{device}_result.jpg",
                    nrow=src.size(0),
                    normalize=False,
                )
            if debug_stack is not None:
                save_image(img_denorm(debug_stack), f"{save_path}/{str(n_iter)}_warped_bf_refine.jpg")
        running_total_loss += Loss.item()
        n_iter += 1
        pbar.set_description("training: R_total_loss: %.3f/%.3f" % (running_total_loss / (i + 1), Loss.item()))
    running_total_loss /= len(train_loader)
    return running_total_loss
