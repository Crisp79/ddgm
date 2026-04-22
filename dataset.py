import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt


# =========================
# 1. TRANSFORMS
# =========================
def get_transforms(image_size=64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
    ])


# =========================
# 2. LOAD FULL DATASET
# =========================
def load_full_dataset(data_path, image_size=64):
    transform = get_transforms(image_size)

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )

    return dataset


# =========================
# 3. CREATE SUBSET
# =========================
def get_subset(dataset, num_samples=20000):
    assert num_samples <= len(dataset), "Subset size larger than dataset"

    indices = torch.randperm(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)

    return subset


# =========================
# 4. CREATE DATALOADER
# =========================
def get_dataloader(dataset, batch_size=32, num_workers=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


# =========================
# 5. MAIN FUNCTION
# =========================
def get_celeba_loader(
    data_path,
    image_size=64,
    batch_size=32,
    subset_size=20000,
    num_workers=4
):
    dataset = load_full_dataset(data_path, image_size)

    if subset_size is not None:
        dataset = get_subset(dataset, subset_size)

    loader = get_dataloader(dataset, batch_size, num_workers)

    return loader


# =========================
# 6. SANITY CHECK
# =========================
if __name__ == "__main__":
    data_path = "data/celeba"  # adjust if needed

    loader = get_celeba_loader(data_path,subset_size=2)

    images, _ = next(iter(loader))

    img = images[0].permute(1, 2, 0)   # C,H,W → H,W,C
    img = (img + 1) / 2                # [-1,1] → [0,1]

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("Shape:", images.shape)  # should be (B, 3, 64, 64)