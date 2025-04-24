import torch
import cv2
import numpy as np
import random
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
import argparse
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--weights_path', default='/workspace/src/best.pth')
    parser.add_argument('--task', default="va", type=str, choices=['exp', 'va'])
    parser.add_argument('--feature_embedding', type=int, default=2048)
    parser.add_argument('--w', type=int, default=7, help='width of the attention map')
    parser.add_argument('--h', type=int, default=7, help='height of the attention map')
    parser.add_argument('--dataset_name', type=str, default='AffectNet')
    return parser.parse_args()


def get_transform():
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    return data_transforms


def pre_trained_wegiths_load(model, cp):
    new_state_dict = OrderedDict()
    for k, v in cp.items():
        new_state_dict[k[7:]] = v
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)
    return model


def resize_image(image, target_width):
    # 원본 이미지의 가로와 세로 크기 가져오기
    height, width = image.shape[:2]

    # 새로운 크기 계산
    ratio = target_width / width
    new_height = int(height * ratio)
    new_dimensions = (target_width, new_height)
    center_h, center_w, length = 212, 312, 362
    new_center_h, new_center_w, new_length = int(center_h * ratio), int(center_w * ratio), int(length * ratio)
    # 크기 조절
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_image, (new_center_h, new_center_w, new_length)