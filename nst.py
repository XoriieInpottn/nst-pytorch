#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2022-02-16
"""

import argparse
import os
import shutil
from typing import List

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchcommon.optim import AdamX
from torchvision.models import vgg19
from tqdm import tqdm


def load_image(path, height, width):
    image = cv.imread(path, cv.IMREAD_COLOR)
    original_height, original_width, _ = image.shape
    image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    cv.cvtColor(image, cv.COLOR_BGR2RGB, image)
    return image, original_height, original_width


def save_image(path, image, height, width):
    cv.cvtColor(image, cv.COLOR_RGB2BGR, image)
    image = cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
    cv.imwrite(path, image)


class ImageNet(object):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
    STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

    @staticmethod
    def encode_image(image: np.ndarray) -> np.ndarray:
        image = np.array(image, dtype=np.float32)
        image -= ImageNet.MEAN
        image /= ImageNet.STD
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def decode_image(tensor: np.ndarray) -> np.ndarray:
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor *= ImageNet.STD
        tensor += ImageNet.MEAN
        np.clip(tensor, 0, 255, out=tensor)
        return np.array(tensor, dtype=np.uint8)


class VGGAdapter(nn.Module):

    def __init__(self, model: nn.Module):
        super(VGGAdapter, self).__init__()
        self._model = model

        self._y_list = []
        for layer in self._model.features:
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(self._forward_hook)

        mode = self.training
        self.train(False)
        dummy = torch.rand((1, 3, 128, 128), dtype=torch.float32)
        self.sizes = [y.shape[1] for y in self(dummy)]
        self.train(mode)

    def _forward_hook(self, _module, _x, y):
        self._y_list.append(y)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._model(x)
        y_list = self._y_list
        self._y_list = []
        return y_list


class NST(nn.Module):

    def __init__(
            self,
            backbone: VGGAdapter,
            content_image: torch.Tensor,
            style_image: torch.Tensor,
            content_weight: float,
            style_weight: float,
    ):
        super(NST, self).__init__()

        self.backbone = backbone
        self.content_image = content_image
        self.style_image = style_image
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.backbone.train(False)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.opt_image = nn.Parameter(content_image.clone())

        self.content_layers = [9]
        self.style_layers = [0, 2, 4, 8, 12]

        self.content_targets = self.backbone(content_image)
        self.content_targets = [self.content_targets[i].detach() for i in self.content_layers]
        self.style_target = self.backbone(style_image)
        self.style_target = [NST.gram_matrix(self.style_target[i]).detach() for i in self.style_layers]

        self.content_weights = [1.0]
        self.style_weights = np.array([1.0 / self.backbone.sizes[i] ** 2 for i in self.style_layers])
        self.style_weights /= np.sqrt(np.square(self.style_weights).sum())

    def forward(self):
        outs = self.backbone(self.opt_image)
        content_outs = [outs[i] for i in self.content_layers]
        style_outs = [outs[i] for i in self.style_layers]
        content_loss = sum([
            F.mse_loss(o, t) * w
            for o, t, w in zip(content_outs, self.content_targets, self.content_weights)
        ])
        style_loss = sum([
            F.mse_loss(NST.gram_matrix(o), t) * w
            for o, t, w in zip(style_outs, self.style_target, self.style_weights)
        ])
        loss = content_loss * self.content_weight + style_loss * self.style_weight
        return loss

    @staticmethod
    def gram_matrix(x):
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        g.div_(h * w)
        return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--content-image', '-c', required=True)
    parser.add_argument('--style-image', '-s', required=True)
    parser.add_argument('--output-dir', '-o', required=True)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--content-weight', type=float, default=1e-1)
    parser.add_argument('--style-weight', type=float, default=1e4)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=1.2e-2)
    parser.add_argument('--momentum', type=float, default=0.93)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-loops', type=int, default=10000)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    backbone = vgg19(pretrained=True)
    print('vgg19 initialized')

    backbone = VGGAdapter(backbone).to(device)
    content_image, oh, ow = load_image(args.content_image, args.image_size, args.image_size)
    content_image = torch.tensor(ImageNet.encode_image(content_image)[None, ...]).to(device)
    style_image, _, _ = load_image(args.style_image, args.image_size, args.image_size)
    style_image = torch.tensor(ImageNet.encode_image(style_image)[None, ...]).to(device)
    model = NST(
        backbone=backbone,
        content_image=content_image,
        style_image=style_image,
        content_weight=args.content_weight,
        style_weight=args.style_weight
    ).to(device)
    print('neural style transfer model initialized')

    optimizer = AdamX(
        list(model.parameters()),
        lr=args.lr,
        betas=(args.momentum, 0.999),
        weight_decay=args.weight_decay
    )
    print('optimizer initialized')

    loop = tqdm(range(args.num_loops), leave=False, ncols=96)
    for i in loop:
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = float(loss)
        loop.set_description(f'[{i + 1}/{args.num_loops}] L={loss:.06f}', False)
        if (i + 1) % args.save_interval == 0:
            # loop.write(f'[{i + 1}/{args.num_loops}] L={loss:.06f}')
            image = ImageNet.decode_image(model.opt_image.detach().to('cpu').numpy()[0])
            path = os.path.join(args.output_dir, f'{i + 1:08d}.jpg')
            save_image(path, image, oh, ow)
    print('complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
