import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

# Resize images, then convert them to tensors and apply augmentation
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True), 
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomPerspective(0.5, 0.5),
    v2.Normalize(mean=[0.594, 0.478, 0.396], std=[0.207, 0.204, 0.210])
])
test_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

train_data = ImageFolder(root='train', transform=train_transforms)
val_data = ImageFolder(root='val', transform=test_transforms)
test_data = ImageFolder(root='test', transform=test_transforms)

# Batch images to process faster
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

for X_batch, y_batch in train_loader:
    for i in range(len(X_batch)):
        print(f"train input: {X_batch[i]}")
        print(f"train output: {y_batch[i]}")

for X_batch, y_batch in val_loader:
    for i in range(len(X_batch)):
        print(f"validation input: {X_batch[i]}")
        print(f"validation output: {y_batch[i]}")

for X_batch, y_batch in test_loader:
    for i in range(len(X_batch)):
        print(f"test input: {X_batch[i]}")
        print(f"test output: {y_batch[i]}")