from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd, ToTensord,
    EnsureTyped, Resized, RandAffined, Rand3DElasticd, RandZoomd, DivisiblePadd, CenterSpatialCropd
)
import os
from glob import glob

def get_data_dicts(data_dir):
    data_dicts = []

    for patient_id in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, patient_id, "preRT")
        if not os.path.isdir(patient_dir):
            continue

        image_list = glob(os.path.join(patient_dir, "*_T2.nii*"))
        label_list = glob(os.path.join(patient_dir, "*_mask.nii*"))

        if not image_list or not label_list:
            print(f"[Warning] File mancanti per {patient_dir}")
            continue

        data_dicts.append({
            "image": image_list[0],
            "label": label_list[0]
        })

    return data_dicts

def get_loaders(train_dir, train_dir2, val_dir, batch_size, patch_size):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=16),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest")
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            prob=0.2,
            mode=("bilinear", "nearest")
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.2,
            mode=("trilinear", "nearest")
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=patch_size, mode=("trilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=16),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_data = get_data_dicts(train_dir) + get_data_dicts(train_dir2)

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = CacheDataset(data=get_data_dicts(val_dir), transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader
