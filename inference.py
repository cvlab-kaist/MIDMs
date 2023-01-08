import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import data
from models.midms import MIDMs
from util.util import img_denorm

# random.seed(2021)
# np.random.seed(2021)
# torch.manual_seed(2021)
# torch.cuda.manual_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Inference Script")
dataset_option_setter = data.get_option_setter("celebahqedge")

parser.add_argument("--inference_mode", type=str, default="target_fixed", choices=["target_fixed", "evaluation"])
parser.add_argument("--dataroot", type=str, default="/mnt/hdd/dataset/CelebAMask-HQ/CelebAMask-HQ")
parser.add_argument("--finetuned_weight", type=str, default="pretrained/celeba/midms_celebA_finetuned.pth")
parser.add_argument("--pretrained_weight", type=str, default="pretrained/celeba/model.ckpt")
parser.add_argument("--ldm_config", type=str, default="pretrained/celeba/config.yaml")
parser.add_argument("--benchmark", type=str, default="celebahqedge")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--pick", type=int, default=11)


parser.set_defaults(is_train=False)
parser.set_defaults(max_dataset_size=sys.maxsize)
parser.set_defaults(distributed=False)
parser = dataset_option_setter(parser, is_train=False)

args, unknown = parser.parse_known_args()
val_dataloader = data.create_dataloader(args)


model = MIDMs(
    backbone="VQGAN",
    use_original_imgsize=False,
    phase="e2e_recurrent",
    label_nc=args.label_nc,
    diffusion_iteratiom_num=4,
    diffusion_config_path=args.ldm_config,
    diffusion_model_path=args.pretrained_weight,
    pos_embed="conv",
    confidence_masking=True,
)

ERR = model.load_state_dict(torch.load(args.finetuned_weight, map_location="cpu")["state_dict"], strict=False)
print(ERR)
model.set_copy()
model.eval()
model.to("cuda")
os.path.basename(val_dataloader.dataset.__getitem__(0)["path"])
cnt = 0

curr_time = time.strftime("%Y_%m_%d_%H_%M")
os.makedirs(f"result_{curr_time}")
os.makedirs(f"result_{curr_time}/real_images")
os.makedirs(f"result_{curr_time}/fake_images")

pick = args.pick
with torch.no_grad():
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for i, mini_batch in pbar:
        src = mini_batch["image"].to(device)
        ref = mini_batch["ref"].to(device)

        if val_dataloader.dataset.__class__.__name__ == "DeepFashionDataset":
            input_semantics = mini_batch["label"].to(device)

            input_semantics = mini_batch["label"].clone().to(device).float()
            mini_batch["label"] = mini_batch["label"][:, :3, :, :]
            ref_semantics = mini_batch["label_ref"].clone().to(device).float()
            mini_batch["label_ref"] = mini_batch["label_ref"][:, :3, :, :]

            trg = mini_batch["label"][:, 0:3].to(device)
            ref_trg = mini_batch["label_ref"][:, 0:3].to(device)
            weight_mask = 0.0

        elif val_dataloader.dataset.__class__.__name__ == "CelebAHQEdgeDataset":

            input_semantics = mini_batch["label"].clone().to(device).float()
            mini_batch["label"] = mini_batch["label"][:, :1, :, :]
            ref_semantics = mini_batch["label_ref"].clone().to(device).float()
            mini_batch["label_ref"] = mini_batch["label_ref"][:, :1, :, :]
            mini_batch["label"] = mini_batch["label"].long()
            mini_batch["label_ref"] = mini_batch["label_ref"].long()

            if args.inference_mode == "target_fixed":
                input_semantics = (
                    val_dataloader.dataset.__getitem__(pick)["label"]
                    .clone()
                    .float()
                    .unsqueeze(0)
                    .repeat(10, 1, 1, 1)
                    .to(device)
                )
                mini_batch["label"] = (
                    val_dataloader.dataset.__getitem__(pick)["label"].long().unsqueeze(0).repeat(10, 1, 1, 1)
                )

            trg = mini_batch["label"][:, :1].repeat(1, 3, 1, 1).to(device)
            ref_trg = mini_batch["label_ref"][:, :1].repeat(1, 3, 1, 1).to(device)
            weight_mask = 0.0
        else:
            trg = None
            input_semantics = None

        warped_ref, debug_stack, fb_mask = model(
            trg,
            src,
            ref,
            input_semantics=input_semantics,
            ref_semantics=ref_semantics,
            mode="inference",
            iter=1,
            warmup_iter=0,
        )

        save_image(
            torch.cat([img_denorm(warped_ref), img_denorm(trg), img_denorm(ref)]),
            f"result_{curr_time}/{str(i)}_inference.jpg",
            nrow=args.batch_size,
        )
        for in_batch_num, (warped_ref_items, src_items) in enumerate(zip(warped_ref, src)):
            img_name = os.path.basename(mini_batch["path"][in_batch_num])
            save_image(
                img_denorm(warped_ref_items),
                f"result_{curr_time}/fake_images/{img_name}",
                nrow=1,
                padding=0,
                normalize=False,
            )
            save_image(
                img_denorm(src_items), f"result_{curr_time}/real_images/{img_name}", nrow=1, padding=0, normalize=False
            )
