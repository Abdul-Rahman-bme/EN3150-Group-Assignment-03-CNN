"""Read saved splits without rebuilding them or estimating normalization."""

import hashlib
from pathlib import Path, PurePosixPath
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .utils import sha256, seed_worker

SPLIT_SIZES = {"train": 18900, "validation": 4050, "test": 4050}


def load_split(split_dir, name, class_names):
    if name not in SPLIT_SIZES:
        raise ValueError(f"Unknown split: {name}")
    frame = pd.read_csv(Path(split_dir) / f"{name}.csv")
    if not {"relative_path", "label", "class_name", "pixel_hash"}.issubset(frame.columns):
        raise ValueError("Split CSV is missing required columns")
    if len(frame) != SPLIT_SIZES[name] or frame.isna().any().any():
        raise ValueError(f"Invalid {name} split size or missing values")
    if not frame["label"].isin(range(len(class_names))).all():
        raise ValueError("Invalid labels")
    expected = frame["label"].map(dict(enumerate(class_names)))
    if not expected.equals(frame["class_name"]):
        raise ValueError("Split label mapping differs from saved class order")
    if not frame["pixel_hash"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid saved pixel hashes")
    for relative, class_name in zip(frame["relative_path"], frame["class_name"]):
        path = PurePosixPath(relative)
        if "\\" in relative or ":" in relative or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe relative image path: {relative}")
        if len(path.parts) != 2 or path.parts[0] != class_name:
            raise ValueError(f"Image path does not match class: {relative}")
    if frame["relative_path"].duplicated().any() or frame["pixel_hash"].duplicated().any():
        raise ValueError("Repeated paths or exact duplicate pixels in split")
    return frame


def split_fingerprints(split_dir):
    return {name: sha256(Path(split_dir) / f"{name}.csv") for name in SPLIT_SIZES}


def check_splits(split_dir, class_names):
    frames = {name: load_split(split_dir, name, class_names) for name in SPLIT_SIZES}
    merged = pd.concat(frames.values(), ignore_index=True)
    if merged["relative_path"].duplicated().any() or merged["pixel_hash"].duplicated().any():
        raise ValueError("Overlapping paths or pixel hashes across splits")
    total = merged.groupby("label").size()
    for name, proportion in [("train", .7), ("validation", .15), ("test", .15)]:
        counts = frames[name].groupby("label").size()
        if not counts.equals((total * proportion).astype("int64")):
            raise ValueError(f"Incorrect stratification in {name}")
    return {"sizes": {name: len(frame) for name, frame in frames.items()},
            "class_names": class_names, "split_sha256": split_fingerprints(split_dir),
            "unique_pixel_hashes": int(merged["pixel_hash"].nunique())}


def check_images(data_root, split_dir, class_names):
    """Optional quality audit, including test images; this does no model evaluation."""
    from tqdm import tqdm
    checked = 0
    for name in SPLIT_SIZES:
        frame = load_split(split_dir, name, class_names)
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Checking {name}"):
            with Image.open(Path(data_root) / row.relative_path) as image:
                image.load()
                if image.size != (64, 64) or image.mode != "RGB":
                    raise ValueError(f"Unexpected image format: {row.relative_path}")
                if hashlib.sha256(image.tobytes()).hexdigest() != row.pixel_hash:
                    raise ValueError(f"Pixels differ from saved split: {row.relative_path}")
            checked += 1
    return checked


def check_archive(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != "c8fa014336c82ac7804f0398fcb19387":
        raise ValueError("EuroSAT archive checksum differs from the notebook")
    return digest.hexdigest()


class EuroSATSplit(Dataset):
    def __init__(self, dataframe, transform, data_root=None):
        # The path-column form keeps the original notebook interface usable.
        self.paths = (dataframe["path"].tolist() if data_root is None else
                      [Path(data_root) / relative for relative in dataframe["relative_path"]])
        self.labels = dataframe["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError(f"Expected 64 x 64 RGB image: {self.paths[index]}")
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


def make_loader(frame, data_root, transform, batch_size, device, *, training=False,
                generator=None, num_workers=0):
    return DataLoader(EuroSATSplit(frame, transform, data_root), batch_size=batch_size,
                      shuffle=training, generator=generator, num_workers=num_workers,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
