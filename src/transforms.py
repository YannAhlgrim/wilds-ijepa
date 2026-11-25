# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()

NORMALIZATION_STATS = {
    "imagenet": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "clip": (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    "inception": (
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
}


def resolve_normalization(name="imagenet"):
    """Resolve an explicit preprocessing preset without altering legacy defaults."""
    if isinstance(name, str):
        key = name.lower()
        if key not in NORMALIZATION_STATS:
            choices = ", ".join(sorted(NORMALIZATION_STATS))
            raise ValueError(f"Unknown normalization {name!r}; choose from {choices}")
        return NORMALIZATION_STATS[key]
    # Keep accepting a (mean, std) pair for direct callers of make_transforms.
    return name


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    use_random_resized_crop=True,
    normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):
    logger.info("making imagenet data transforms")

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    transform_list = []
    if use_random_resized_crop:
        transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    else:
        resize_size = int(crop_size * 256 / 224)
        transform_list += [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
        ]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


def make_transform_eval(
    crop_size=224, normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
):
    logger.info("making imagenet evaluation transforms")

    resize_size = int(crop_size * 256 / 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ]
    )
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
