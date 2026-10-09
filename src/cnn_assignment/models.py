"""Notebook architectures; module names retain saved checkpoint compatibility."""

import torch
from torch import nn
from .model_specs import CUSTOM_MODELS, PRETRAINED_MODELS, PRETRAINED_SPECS


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


class PretrainedWeightsError(RuntimeError):
    """A fresh pretrained run must stop rather than use a random backbone."""


def load_saved_state(model, state):
    """Our saved checkpoints must include every parameter and BatchNorm buffer.

    Torch's compatibility loader can fill missing legacy BN counters; that is
    appropriate for the official old ImageNet file, but not for our own saves.
    """
    expected, actual = set(model.state_dict()), set(state)
    if actual != expected:
        raise RuntimeError(f"Saved state mismatch. Missing key(s): {sorted(expected - actual)}; "
                           f"Unexpected key(s): {sorted(actual - expected)}")
    return model.load_state_dict(state, strict=True)


def _load_pretrained_state(weights, weights_file, progress):
    from pathlib import Path
    import re
    from urllib.parse import urlparse
    from .utils import sha256

    filename = Path(urlparse(weights.url).path).name
    prefix = re.search(r"-([0-9a-f]+)\.pth$", filename).group(1)
    path = Path(weights_file) if weights_file is not None else Path(torch.hub.get_dir()) / "checkpoints" / filename
    # Torch's download verifies new files. Verify cached/manual files as well.
    if path.is_file() and not sha256(path).startswith(prefix):
        raise ValueError(f"Official checkpoint SHA-256 prefix mismatch: {path}; expected {prefix}")
    if weights_file is not None:
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        state = weights.get_state_dict(progress=progress, check_hash=True, weights_only=True)
    digest = sha256(path)
    if not digest.startswith(prefix):
        raise ValueError(f"Official checkpoint SHA-256 prefix mismatch: {path}; expected {prefix}")
    return state, digest


def build_model(name, num_classes=10, *, pretrained=False, pretrained_weights_file=None, progress=True):
    """Load ImageNet only for a fresh run; restore saved states using pretrained=False."""
    if name in CUSTOM_MODELS:
        if pretrained or pretrained_weights_file is not None:
            raise ValueError("ImageNet initialization only applies to the two pretrained candidates")
        return {"model_a": StandardCNN, "model_b": LightweightCNN,
                "model_c": WideLightweightCNN}[name](num_classes)
    if name not in PRETRAINED_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    if pretrained_weights_file is not None and not pretrained:
        raise ValueError("A source weights file is only used for fresh pretrained initialization")
    from torchvision import models
    spec = PRETRAINED_SPECS[name]
    enum_name, member = spec["pretrained_weights"].split(".")
    weights = getattr(getattr(models, enum_name), member)
    model = getattr(models, name)(weights=None)
    if pretrained:
        try:
            if weights.url != spec["pretrained_source"]:
                raise ValueError("Installed Torchvision weight URL differs from the recorded source")
            state, digest = _load_pretrained_state(weights, pretrained_weights_file, progress)
            # Restore the entire original 1,000-class model before replacing its head.
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise PretrainedWeightsError(
                f"Cannot load {spec['pretrained_weights']}: {error}. No random fallback was used. "
                f"Download the official file from {spec['pretrained_source']} and supply "
                "--pretrained-weights-file PATH for a fresh train command. "
                "The file's official SHA-256 prefix and full state dictionary will be checked."
            ) from error
        model.pretrained_initialization = {**spec, "pretrained_source_sha256": digest,
                                           "pretrained_backbone_loaded": True}
    if name == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def model_cost(model, weights_path=None):
    """Count convolution/linear MACs per 64 x 64 image, excluding other operations."""
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
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
    result = {"total_parameters": total_parameters, "trainable_parameters": parameters,
              "estimated_fp32_parameter_bytes": parameters * 4,
              "estimated_fp32_total_parameter_bytes": total_parameters * 4,
              "conv_linear_macs_per_image": macs,
              "mac_input_shape": [1, 3, 64, 64],
              "mac_scope": "Convolution and linear multiply-accumulates only; not measured inference speed",
              "size_units": {"byte": "8 bits", "MB": "1,000,000 bytes", "MiB": "1,048,576 bytes"}}
    if weights_path is not None:
        from pathlib import Path
        result["actual_weights_file_bytes"] = Path(weights_path).stat().st_size
        result["actual_weights_file_MB"] = result["actual_weights_file_bytes"] / 1_000_000
        result["actual_weights_file_MiB"] = result["actual_weights_file_bytes"] / 1_048_576
    return result
