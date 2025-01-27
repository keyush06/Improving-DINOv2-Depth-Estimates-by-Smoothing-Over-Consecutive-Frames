import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import math
import itertools
from functools import partial
import mmcv
from mmcv.runner import load_checkpoint
from collections import defaultdict
import cv2
import requests
from torchvision import transforms
from dinov2.eval.depth.models import build_depther
import urllib
import json
import random


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def load_backbone(backbone_size = "small"):

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    return backbone_name, backbone_model


def load_dino_model(backbone_name, backbone_model, backbone_size = "small", head_dataset = "nyu", head_type = "dpt"):

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = create_depther(
        cfg,
        backbone_model=backbone_model,
        backbone_size=backbone_size,
        head_type=head_type,
    )

    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.eval()
    model.cuda()

    return model


def enumerate_datasets(data_directory):

    datasets = []
    num_images_dataset = []

    for dataset_dir, _, files in os.walk(data_directory):
        if dataset_dir != data_directory:
            dataset_name = os.path.basename(dataset_dir)
            datasets.append(dataset_name)

            image_count = sum(1 for file in files if file.endswith('.jpg'))
            num_images_dataset.append(image_count)

    num_images = sum(num_images_dataset)

    return datasets, num_images_dataset, num_images


def load_training_logs(num_epochs, num_val_checkpoints, model_string):

    original_train_file_path = './training_logs/original_mse_train_' + model_string + '.pt'
    if os.path.exists(original_train_file_path):
        original_mse_train = torch.load(original_train_file_path, weights_only=True)
    else:
        original_mse_train = torch.zeros(num_epochs)

    updated_train_file_path = './training_logs/updated_mse_train_' + model_string + '.pt'
    if os.path.exists(updated_train_file_path):
        updated_mse_train = torch.load(updated_train_file_path, weights_only=True)
    else:
        updated_mse_train = torch.zeros(num_epochs)

    original_val_file_path = './training_logs/original_mse_val_' + model_string + '.pt'
    if os.path.exists(original_val_file_path):
        original_mse_val = torch.load(original_val_file_path, weights_only=True)
    else:
        original_mse_val = torch.zeros(num_epochs, num_val_checkpoints + 1)

    updated_val_file_path = './training_logs/updated_mse_val_' + model_string + '.pt'
    if os.path.exists(updated_val_file_path):
        updated_mse_val = torch.load(updated_val_file_path, weights_only=True)
    else:
        updated_mse_val = torch.zeros(num_epochs, num_val_checkpoints + 1)

    return original_mse_train, updated_mse_train, original_mse_val, updated_mse_val

