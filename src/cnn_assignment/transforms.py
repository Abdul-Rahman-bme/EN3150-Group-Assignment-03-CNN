"""Original orientation augmentation and saved RGB normalization."""

import random
from PIL import Image
from torchvision import transforms


class RandomQuarterTurn:
    def __call__(self, image):
        rotation = random.choice([None, Image.Transpose.ROTATE_90,
                                  Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
        return image.copy() if rotation is None else image.transpose(rotation)


def make_augmentation():
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5), RandomQuarterTurn()])


def make_transform(config, training=False):
    steps = [make_augmentation()] if training else []
    steps.extend([transforms.ToTensor(), transforms.Normalize(
        mean=config["normalization_mean"], std=config["normalization_std"])])
    return transforms.Compose(steps)
