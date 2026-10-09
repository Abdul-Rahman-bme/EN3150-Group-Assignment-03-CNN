"""Notebook architectures; module names retain saved checkpoint compatibility."""

import torch
from torch import nn


class StandardCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers = []
        for inputs, outputs in [(3, 32), (32, 64), (64, 128)]:
            layers.extend([nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), start_dim=1))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.block(x)


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(DepthwiseSeparableBlock(3, 32),
                                      DepthwiseSeparableBlock(32, 64),
                                      DepthwiseSeparableBlock(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), start_dim=1))


class WideLightweightCNN(nn.Module):
    """Model C: Model B's separable blocks with twice the output channels."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(DepthwiseSeparableBlock(3, 64),
                                      DepthwiseSeparableBlock(64, 128),
                                      DepthwiseSeparableBlock(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), start_dim=1))


def build_model(name, num_classes=10):
    return {"model_a": StandardCNN, "model_b": LightweightCNN,
            "model_c": WideLightweightCNN}[name](num_classes)


def model_cost(model, weights_path=None):
    """Count convolution/linear MACs per 64 x 64 image, excluding other operations."""
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = 0
    handles = []

    def count(layer, inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            macs += output.numel() * layer.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    result = {"trainable_parameters": parameters, "estimated_fp32_parameter_bytes": parameters * 4,
              "conv_linear_macs_per_image": macs}
    if weights_path is not None:
        from pathlib import Path
        result["actual_weights_file_bytes"] = Path(weights_path).stat().st_size
    return result
