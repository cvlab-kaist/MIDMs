# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from data.image_folder import make_dataset
from data.pix2pix_dataset import Pix2pixDataset


class LSUNChurchDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode="resize_and_crop")
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = "val" if not opt.is_train else "train"
        subfolder = "validation" if not opt.is_train else "training"
        cache = False if not opt.is_train else True
        all_images = sorted(make_dataset(root + "/" + subfolder, recursive=True, read_cache=cache, write_cache=False))
        image_paths = []
        label_paths = []
        for p in all_images:
            # if '_%s_' % phase not in p:
            #     continue
            if p.endswith(".jpg"):
                image_paths.append(p)
            elif p.endswith(".png"):
                label_paths.append(p)

        return label_paths, image_paths

    def get_ref(self, opt):
        extra = "_test" if not opt.is_train else ""
        with open("./data/lsun_church_ref{}.txt".format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(",")
            key = items[0]
            if not opt.is_train:
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ("validation", "validation")
        return ref_dict, train_test_folder
