"""Model metadata without torch imports, so saved-history plotting stays light."""

from copy import deepcopy

CUSTOM_MODELS = ("model_a", "model_b", "model_c", "model_d")
PRETRAINED_MODELS = ("mobilenet_v2", "shufflenet_v2_x0_5")
MODEL_NAMES = CUSTOM_MODELS + PRETRAINED_MODELS
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PRETRAINED_SPECS = {
    "mobilenet_v2": {
        "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V2",
        "pretrained_source": "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        "pretrained_documentation": "https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.mobilenet_v2.html",
    },
    "shufflenet_v2_x0_5": {
        "pretrained_weights": "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1",
        "pretrained_source": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
        "pretrained_documentation": "https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.shufflenet_v2_x0_5.html",
    },
}


def pretrained_settings(name):
    settings = deepcopy(PRETRAINED_SPECS[name])
    settings.update(initialization="imagenet_pretrained_then_new_classifier",
                    fine_tune_all_layers=True, normalization_source="ImageNet",
                    normalization_mean=list(IMAGENET_MEAN), normalization_std=list(IMAGENET_STD))
    return settings
