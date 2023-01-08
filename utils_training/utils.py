import os
import re

import torch
import torch.nn.functional as F

r"""
    source code from DeiT
    https://github.com/facebookresearch/deit
"""


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # set distributed training
        print("=> set distributed training")
        args.distributed = True
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])

        # device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    else:
        args.distributed = False
        print("=> set single node training")
        args.world_size = 1
        args.rank = 0
        args.gpu = "cuda" if torch.cuda.is_available() else "cpu"

    torch.distributed.barrier()
    print(f"=> set cuda device = {args.gpu}")
    return args


r"""
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
"""


def save_checkpoint(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))
    # if is_best:
    #     shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))


def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth.tar"):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val = -1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        ERR = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(ERR)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val = checkpoint["best_loss"]
        except:
            best_val = -1
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def log_args(args):
    r"""Log program arguments"""
    print("\n+================================================+")
    for arg_key in args.__dict__:
        print("| %20s: %-24s |" % (arg_key, str(args.__dict__[arg_key])))
    print("+================================================+\n")


def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r"\d+", list_str)))


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def flow2kps(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode="bilinear") * (upsample_size[0] / h)

    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)

        kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)
