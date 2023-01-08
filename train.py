r"""
    modified training script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
"""
import argparse
import os
import pickle
import random
import sys
import time
from os import path as osp

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import data
import utils_training.optimize as optimize
from models.ContextualLoss import ContextualLoss_forward
from models.correspondence import VGG19_feature_color_torchversion
from models.midms import MIDMs
from utils_training.utils import boolean_string, load_checkpoint, parse_list, save_checkpoint, set_distributed


def main_worker(args, parser):
    # Setup dataset
    dataset_option_setter = data.get_option_setter(args.benchmark)
    parser = dataset_option_setter(parser, args.is_train)
    args, unknown = parser.parse_known_args()

    # Setup seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setup distributed training
    args = set_distributed(args)
    device = args.gpu

    train_dataloader = data.create_dataloader(args)

    print(f"CONFIDENCE MASKING : {args.confidence_masking}")
    print(f"WARMUP ITER : {args.warmup_iter}")

    # Model
    model = MIDMs(
        backbone=args.backbone,
        use_original_imgsize=False,
        phase=args.phase,
        label_nc=args.label_nc,
        diffusion_iteratiom_num=args.diffusion_iteratiom_num,
        diffusion_config_path=args.diffusion_config_path,
        diffusion_model_path=args.diffusion_model_path,
        pos_embed=args.pos_embed,
        confidence_masking=args.confidence_masking,
        maskmix=args.maskmix,
        device=device,
    )
    # perceptual loss

    vggnet_fix = VGG19_feature_color_torchversion(vgg_normal_correct=True)
    vggnet_fix.load_state_dict(torch.load("models/vgg19_conv.pth", map_location="cpu"))
    vggnet_fix.eval()

    param_model = [
        param
        for name, param in model.named_parameters()
        if ("diff_model.first_stage_model" not in name and "diff_model.model" not in name)
    ]
    diff_backbone = [param for name, param in model.named_parameters() if "diff_model.model" in name]

    def count_parameters(model):
        return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "backbone" not in name)

    print(f"The number of parameters: {count_parameters(model)})")
    # Optimizer
    optimizer = optim.AdamW(
        [{"params": param_model, "lr": args.lr}, {"params": diff_backbone, "lr": args.lr_diff_backbone},],
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = (
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, verbose=True)
        if args.scheduler == "cosine"
        else lr_scheduler.MultiStepLR(
            optimizer, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True,
        )
    )

    if args.pretrained:

        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(
            model, optimizer, scheduler, filename=args.pretrained
        )
        print(f"PRETRAINED LOAD. starts from {start_epoch}")
        ERR = model.diff_model.load_state_dict(
            torch.load(args.diffusion_model_path, map_location="cpu")["state_dict"], strict=False,
        )
        print(ERR)
        print(f"=> loaded checkpoint from checkpoint {args.diffusion_model_path}")

        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        best_val = 0
        start_epoch = 0
    torch.cuda.set_device(device)
    model.cuda(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    for param in vggnet_fix.parameters():
        param.requires_grad = False
    vggnet_fix.cuda(device)

    if not os.path.isdir(args.snapshots):
        os.makedirs(args.snapshots, exist_ok=True)

    cur_snapshot = args.name_exp
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, "args.pkl"), "wb") as f:
        pickle.dump(args, f)

    with open(os.path.join(args.snapshots, cur_snapshot, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f, allow_unicode=True, default_flow_style=False)

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    test_writer = SummaryWriter(os.path.join(save_path, "test"))

    contextureLoss = ContextualLoss_forward().cuda(device)

    if args.amp:
        print("Mixed Precision Mode")

    for epoch in range(start_epoch, args.epochs):

        scheduler.step(epoch)

        train_loss = optimize.train_epoch(
            model,
            optimizer,
            train_dataloader,
            device,
            epoch,
            train_writer,
            vggnet_fix,
            contextureLoss,
            is_amp=args.amp,
            hijack_step=args.hijack_step,
            warmup_iter=args.warmup_iter,
            save_path=save_path,
        )
        if device == 0:
            train_writer.add_scalar("train_epoch/loss", train_loss, epoch)
            train_writer.add_scalar("train_epoch/learning_rate", scheduler.get_lr()[0], epoch)
            # train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
            print(colored("==> ", "green") + "Train average loss:", train_loss)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": 0,
                },
                save_path,
                "epoch_{}.pth".format(epoch + 1),
            )

    print("traning finished.")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training Script")
    # Paths
    parser.add_argument(
        "--name_exp", type=str, default=time.strftime("%Y_%m_%d_%H_%M"), help="name of the experiment to save",
    )

    parser.add_argument(
        "--benchmark", type=str, default="celebahqedge", choices=["deepfashion", "celebahqedge", "lsun_church"],
    )
    parser.add_argument("--is-train", type=boolean_string, nargs="?", const=True, default=True)

    # args
    parser.add_argument("--snapshots", type=str, default="./snapshots", help="path to save training results")
    parser.add_argument(
        "--pretrained", dest="pretrained", default=None, help="path to pre-trained model",
    )
    parser.add_argument("--start_epoch", type=int, default=-1, help="start epoch")
    parser.add_argument("--epochs", type=int, default=4, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="training batch size")
    parser.add_argument(
        "--n_threads", type=int, default=2, help="number of parallel threads for dataloaders",
    )
    parser.add_argument("--seed", type=int, default=2021, help="Pseudo-RNG seed")
    parser.add_argument("--backbone", type=str, default="VQGAN")

    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument(
        "--lr", type=float, default=3e-6, metavar="LR", help="learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--lr_diff_backbone", type=float, default=3e-6, metavar="LR", help="learning rate (default: 3e-6)",
    )
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--step", type=str, default="[10, 12, 14, 24]")
    parser.add_argument("--step_gamma", type=float, default=0.3)

    parser.add_argument(
        "--label_nc",
        type=int,
        default=182,
        help="# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.",
    )

    parser.add_argument(
        "--dataroot", type=str, default="/media/dataset1/CelebAMask-HQ",
    )

    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=sys.maxsize,
        help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
    )
    parser.add_argument(
        "--real_reference_probability", type=float, default=0.7, help="self-supervised training probability",
    )
    parser.add_argument(
        "--hard_reference_probability", type=float, default=0.2, help="hard reference training probability",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="scale_width_and_crop",
        help="scaling and cropping of images at load time.",
        choices=(
            "resize_and_crop",
            "crop",
            "scale_width",
            "scale_width_and_crop",
            "scale_shortside",
            "scale_shortside_and_crop",
            "fixed",
            "none",
        ),
    )
    parser.add_argument(
        "--load_size",
        type=int,
        default=256,
        help="Scale images to this size. The final image will be cropped to --crop_size.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Crop to the width of crop_size (after initially scaling the images to load_size.)",
    )
    parser.add_argument(
        "--no_flip", action="store_true", help="if specified, do not flip the images for data argumentation",
    )

    parser.add_argument("--amp", type=boolean_string, nargs="?", const=True, default=False)
    parser.add_argument(
        "--phase", type=str, default="e2e_recurrent", choices=["corr", "diff", "e2e", "e2e_recurrent"],
    )
    parser.add_argument("--hijack_step", type=int, default=100, help="hijack step in inference")
    parser.add_argument(
        "--warmup_iter", type=int, default=10000, help="matching warmup (note: this is used when phase is e2e)",
    )
    parser.add_argument(
        "--diffusion_iteratiom_num",
        type=int,
        default=4,
        help="# of iterations of diffusion sampling process in training",
    )
    parser.add_argument(
        "--diffusion_config_path", type=str, default="../midms_weight/celeba/pretrained/config.yaml",
    )
    parser.add_argument(
        "--diffusion_model_path", type=str, default="../midms_weight/celeba/pretrained/model.ckpt",
    )
    parser.add_argument("--pos_embed", type=str, default="conv", choices=["learnable", "loftr", "conv"])
    parser.add_argument("--confidence_masking", type=boolean_string, nargs="?", const=True, default=True)
    parser.add_argument("--maskmix", type=boolean_string, nargs="?", const=True, default=True)
    parser.add_argument("--video_like", type=boolean_string, nargs="?", const=True, default=False)
    parser.add_argument("--comment", type=str, default="blablabla...")
    # Seed
    args = parser.parse_args()

    main_worker(args, parser)
