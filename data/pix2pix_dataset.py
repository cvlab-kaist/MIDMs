# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import random
import re

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform

# from scipy.ndimage.filters import gaussian_filter


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--no_pairing_check",
            action="store_true",
            help="If specified, skip sanity check of correct label-image file pairing",
        )
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)

        if opt.benchmark != "celebahq" and opt.benchmark != "deepfashion":
            natural_sort(label_paths)
            natural_sort(image_paths)

        label_paths = label_paths[: opt.max_dataset_size]
        image_paths = image_paths[: opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), (
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this."
                    % (path1, path2)
                )

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        self.real_reference_probability = 1 if not opt.is_train else opt.real_reference_probability
        self.hard_reference_probability = 0 if not opt.is_train else opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)

        # label_paths, image_paths = self.get_paths(opt)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(
            self.opt, params1, method=transforms.InterpolationMode.NEAREST, normalize=False
        )
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label_tensor, params1 = self.get_label_tensor(label_path)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), "The label_path %s and image_path %s don't match." % (
                label_path,
                image_path,
            )
        image = Image.open(image_path)
        image = image.convert("RGB")

        transform_image, cats_additional_transform = get_transform(self.opt, params1, CATs_aug=True, normalize=False)

        image_tensor = transform_image(Image.fromarray(cats_additional_transform(image=np.array(image))["image"]))
        image_tensor = image_tensor * 2 - 1
        original_image_tensor = transform_image(image)
        original_image_tensor = original_image_tensor * 2 - 1
        ref_tensor = 0
        original_ref_tensor = 0
        label_ref_tensor = 0

        random_p = random.random()
        if random_p < self.real_reference_probability or self.opt.is_train == False:
            key = (
                image_path.replace("\\", "/").split("DeepFashion/")[-1]
                if self.opt.benchmark == "deepfashion"
                else os.path.basename(image_path)
            )
            val = self.ref_dict[key]
            if random_p < self.hard_reference_probability:
                path_ref = val[1]  # hard reference
            else:
                path_ref = val[0]  # easy reference
            if self.opt.benchmark == "deepfashion":
                path_ref = os.path.join(self.opt.dataroot, path_ref)
            else:
                path_ref = (
                    os.path.dirname(image_path).replace(self.train_test_folder[1], self.train_test_folder[0])
                    + "/"
                    + path_ref
                )

            image_ref = Image.open(path_ref).convert("RGB")
            if self.opt.benchmark != "deepfashion":
                path_ref_label = path_ref.replace(".jpg", ".png")
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else:
                path_ref_label = self.imgpath_to_labelpath(path_ref)

            label_ref_tensor, params = self.get_label_tensor(path_ref_label)
            transform_image, cats_additional_transform = get_transform(
                self.opt, params, normalize=False, CATs_aug=True
            )
            ref_tmp = Image.fromarray(cats_additional_transform(image=np.array(image_ref))["image"])

            ref_tensor = transform_image(ref_tmp)
            original_ref_tensor = transform_image(image_ref)
            # Normalize
            ref_tensor = ref_tensor * 2 - 1
            original_ref_tensor = original_ref_tensor * 2 - 1
            # ref_tensor = self.reference_transform(image_ref)
            self_ref_flag = torch.zeros_like(ref_tensor)
        else:
            pair = False
            if self.opt.benchmark == "deepfashion" and self.opt.video_like:
                # if self.opt.hdfs:
                #     key = image_path.split('DeepFashion.zip@/')[-1]
                # else:
                #     key = image_path.split('DeepFashion/')[-1]
                key = image_path.replace("\\", "/").split("DeepFashion/")[-1]
                val = self.ref_dict[key]
                ref_name = val[0]
                key_name = key
                if (
                    os.path.dirname(ref_name) == os.path.dirname(key_name)
                    and os.path.basename(ref_name).split("_")[0] == os.path.basename(key_name).split("_")[0]
                ):
                    path_ref = os.path.join(self.opt.dataroot, ref_name)
                    image_ref = Image.open(path_ref).convert("RGB")
                    label_ref_path = self.imgpath_to_labelpath(path_ref)
                    label_ref_tensor, params = self.get_label_tensor(label_ref_path)
                    transform_image, cats_additional_transform = get_transform(
                        self.opt, params1, CATs_aug=True, normalize=False
                    )
                    ref_tensor = transform_image(
                        Image.fromarray(cats_additional_transform(image=np.array(image_ref))["image"])
                    )
                    original_ref_tensor = transform_image(image_ref)

                    ref_tensor = ref_tensor * 2 - 1
                    original_ref_tensor = original_ref_tensor * 2 - 1
                    pair = True
            if not pair:
                label_ref_tensor, params = self.get_label_tensor(label_path)
                transform_image, cats_additional_transform = get_transform(
                    self.opt, params1, CATs_aug=True, normalize=False
                )
                ref_tensor = transform_image(
                    Image.fromarray(cats_additional_transform(image=np.array(image))["image"])
                )
                original_ref_tensor = transform_image(image)
                ref_tensor = ref_tensor * 2 - 1
                original_ref_tensor = original_ref_tensor * 2 - 1
            # ref_tensor = self.reference_transform(image)
            self_ref_flag = torch.ones_like(ref_tensor)

        input_dict = {
            "label": label_tensor,
            "image": image_tensor,
            "original_image": original_image_tensor,
            "path": image_path,
            "self_ref": self_ref_flag,
            "ref": ref_tensor,
            "original_ref": original_ref_tensor,
            "label_ref": label_ref_tensor,
            "theta": compute_syn_theta(),
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split("(\d+)", text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def atoi(text):
    return int(text) if text.isdigit() else text


def compute_syn_theta():
    rot_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 12
    # between -np.pi/12 and np.pi/12
    sh_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 6
    # between -np.pi/6 and np.pi/6
    lambda_1 = 1 + (2 * np.random.rand(1) - 1) * 0.25
    # between 0.75 and 1.25
    lambda_2 = 1 + (2 * np.random.rand(1) - 1) * 0.25
    # between 0.75 and 1.25
    tx = (2 * np.random.rand(1) - 1) * 0.25
    # between -0.25 and 0.25
    ty = (2 * np.random.rand(1) - 1) * 0.25

    R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])], [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
    R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])], [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

    D = np.diag([lambda_1[0], lambda_2[0]])

    A = R_alpha @ R_sh.transpose() @ D @ R_sh
    theta_aff = np.array([A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]])

    return torch.tensor(theta_aff).type(torch.FloatTensor)
