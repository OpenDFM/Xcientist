#!/usr/bin/env python3
"""
DeepEraser - 图像去水印/文字工具

将 demo.py 封装成一个易用的函数
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deeperaser.pth")


# =============================================================================
# 模型定义 (来自 model.py, update.py, extractor.py)
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        self.conv1 = nn.Conv2d(4, 16, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 16
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=1)
        self.layer3 = self._make_layer(32, stride=1)
        self.layer4 = self._make_layer(64, stride=1)
        self.layer5 = self._make_layer(64, stride=1)
        self.layer6 = self._make_layer(128, stride=1)

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.conv2(x)
        return x


class PredHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(PredHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.outact = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.outact(self.conv2(self.relu(self.conv1(x))))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        self.convf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64-3, 3, padding=1)

    def forward(self, flow):
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        return torch.cat((flo, flow), dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=32):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=64, input_dim=64+64)
        self.pre_head = PredHead(64, hidden_dim=128)

    def forward(self, net, inp, rec_image):
        rec_image_features = self.encoder(rec_image)
        inp = torch.cat([inp, rec_image_features], dim=1)
        net = self.gru(net, inp)
        drec_image = self.pre_head(net)
        return net, drec_image, rec_image_features, inp


class DeepEraser(nn.Module):
    def __init__(self):
        super(DeepEraser, self).__init__()
        self.hidden_dim = hdim = 64
        self.context_dim = 64

        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.update_block = BasicUpdateBlock(hidden_dim=64)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image1, mask, iters=8, test_mode=False):
        image1 = image1.contiguous()

        image2 = torch.cat([image1, mask], dim=1)

        hdim = self.hidden_dim
        cdim = self.context_dim
        fmap1 = self.fnet(image2)

        net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        rec_image0 = image1
        rec_image = image1
        image_list = []

        for itr in range(iters):
            net, d_rec_image, rec_image_features, inpf = self.update_block(net, inp, rec_image)
            rec_image = rec_image0 + d_rec_image
            image_list.append(rec_image)

        if test_mode:
            return rec_image

        return image_list


# =============================================================================
# 工具函数
# =============================================================================

def remove_text(
    input_image_path: str,
    mask_image_path: str,
    output_image_path: str = None,
    model_path: str = _DEFAULT_MODEL_PATH,
    use_cuda: bool = True,
    return_array: bool = False,
) -> Optional[np.ndarray]:
    """
    使用 DeepEraser 去除图片中的水印/文字

    Args:
        input_image_path: 输入图片路径 (原图)
        mask_image_path: 掩码图片路径 (白色区域为要去除的部分)
        output_image_path: 输出图片路径 (可选，默认在同目录下生成 _cleaned.png)
        model_path: 模型权重路径 (默认 ./deeperaser.pth)
        use_cuda: 是否使用 GPU (默认 True，如果不可用会自动回退到 CPU)
        return_array: 是否返回 numpy 数组 (默认 False，返回文件路径)

    Returns:
        如果 return_array=True，返回修复后的 numpy 数组 (H x W x 3)
        否则返回输出文件路径
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图片不存在: {input_image_path}")
    if not os.path.exists(mask_image_path):
        raise FileNotFoundError(f"掩码图片不存在: {mask_image_path}")

    if output_image_path is None:
        base, ext = os.path.splitext(input_image_path)
        output_image_path = f"{base}_cleaned{ext}"

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"[DeepEraser] Using device: {device}")

    net = DeepEraser()
    if device == 'cuda':
        net = net.cuda()

    model_dict = net.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()

    img = np.array(Image.open(input_image_path))[:, :, :3]
    mask = np.array(Image.open(mask_image_path))[:, :]

    im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    mask_tensor = torch.from_numpy(mask / 255.0).unsqueeze(0).float()

    with torch.no_grad():
        if device == 'cuda':
            pred_img = net(im.unsqueeze(0).cuda(), mask_tensor.unsqueeze(0).cuda())
        else:
            pred_img = net(im.unsqueeze(0), mask_tensor.unsqueeze(0))

        pred_img[-1] = torch.clamp(pred_img[-1], 0, 1)
        out = (pred_img[-1][0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    cv2.imwrite(output_image_path, out[:, :, ::-1])
    print(f"[DeepEraser] 已保存到: {output_image_path}")

    if return_array:
        return out

    return output_image_path


def remove_text_batch(
    input_image_paths: List[str],
    mask_image_paths: List[str],
    output_image_paths: List[str] = None,
    model_path: str = _DEFAULT_MODEL_PATH,
    use_cuda: bool = True,
) -> List[str]:
    """批量处理多张图片"""
    if len(input_image_paths) != len(mask_image_paths):
        raise ValueError("输入图片和掩码图片数量必须一致")

    if output_image_paths is None:
        output_image_paths = [None] * len(input_image_paths)

    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"[DeepEraser] Using device: {device}")

    net = DeepEraser()
    if device == 'cuda':
        net = net.cuda()

    model_dict = net.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()

    results = []

    for i, (img_path, mask_path, out_path) in enumerate(zip(input_image_paths, mask_image_paths, output_image_paths)):
        print(f"[DeepEraser] 处理 {i+1}/{len(input_image_paths)}: {img_path}")

        img = np.array(Image.open(img_path))[:, :, :3]
        mask = np.array(Image.open(mask_path))[:, :]

        im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask / 255.0).unsqueeze(0).float()

        with torch.no_grad():
            if device == 'cuda':
                pred_img = net(im.unsqueeze(0).cuda(), mask_tensor.unsqueeze(0).cuda())
            else:
                pred_img = net(im.unsqueeze(0), mask_tensor.unsqueeze(0))

            pred_img[-1] = torch.clamp(pred_img[-1], 0, 1)
            out = (pred_img[-1][0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if out_path is None:
            base, ext = os.path.splitext(img_path)
            out_path = f"{base}_cleaned{ext}"

        cv2.imwrite(out_path, out[:, :, ::-1])
        results.append(out_path)
        print(f"[DeepEraser] 已保存: {out_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepEraser - 图像去水印/文字工具")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("mask", help="掩码图片路径")
    parser.add_argument("-o", "--output", help="输出图片路径 (默认: input_cleaned.png)")
    parser.add_argument("-m", "--model", default="./deeperaser.pth", help="模型权重路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")

    args = parser.parse_args()

    remove_text(
        input_image_path=args.input,
        mask_image_path=args.mask,
        output_image_path=args.output,
        model_path=args.model,
        use_cuda=not args.cpu,
    )


if __name__ == "__main__":
    main()
