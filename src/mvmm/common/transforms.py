"""Standard image transforms.

Keep these centralized so anomaly/pose/metrology pipelines share the
same normalization conventions (ImageNet mean/std unless stated otherwise).
"""

from __future__ import annotations

from typing import Any

# torchvision is imported lazily so the common package stays importable
# in CPU-only / no-torch environments (e.g. doc builds).


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(image_size: int = 256, crop_size: int | None = 224) -> Any:
    """Resize → CenterCrop → ToTensor → Normalize (ImageNet).

    Returns a torchvision.transforms.Compose. Lazy-import keeps base deps clean.
    """
    from torchvision import transforms as T  # local import

    crop = crop_size or image_size
    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(
    image_size: int = 256,
    crop_size: int | None = 224,
    color_jitter: float = 0.1,
    flip: bool = True,
) -> Any:
    """Light augmentation for anomaly-detection-style fine-tuning.

    For unsupervised AD (PatchCore, EfficientAD) we deliberately avoid heavy
    augmentation — only what defects-free samples can tolerate.
    """
    from torchvision import transforms as T

    crop = crop_size or image_size
    ops: list[Any] = [
        T.Resize(image_size),
        T.CenterCrop(crop),
    ]
    if flip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    if color_jitter > 0:
        ops.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def denormalize_imagenet(tensor: Any) -> Any:
    """Inverse of ImageNet normalize for visualization. Accepts CxHxW tensor."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor)
    return (tensor * std + mean).clamp(0, 1)
