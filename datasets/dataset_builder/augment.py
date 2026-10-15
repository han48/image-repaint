"""
Augmentation pipelines for sketch training.
"""
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms():
    """
    Training augmentation pipeline.
    Applies same geometric transforms to both photo and sketch.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required. Install: pip install albumentations")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomCrop(224, 224),
        A.Rotate(limit=15, p=0.3),
        A.Perspective(p=0.2),
    ], additional_targets={'sketch': 'image'})


def get_val_transforms():
    """
    Validation/test augmentation pipeline (center crop only).
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required. Install: pip install albumentations")
    return A.Compose([
        A.CenterCrop(224, 224),
    ], additional_targets={'sketch': 'image'})


if __name__ == '__main__':
    print("Train transforms:", get_train_transforms())
    print("Val transforms:", get_val_transforms())
