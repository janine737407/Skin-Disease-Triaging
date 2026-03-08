import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim

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
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.594, 0.478, 0.396], std=[0.207, 0.204, 0.210])
])

train_data = ImageFolder(root='train', transform=train_transforms)
val_data = ImageFolder(root='val', transform=test_transforms)
test_data = ImageFolder(root='test', transform=test_transforms)

# Batch images to process faster
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# for X_batch, y_batch in train_loader:
    # for i in range(len(X_batch)):
        # print(f"train input: {X_batch[i]}")
        # print(f"train output: {y_batch[i]}")

# for X_batch, y_batch in val_loader:
    # for i in range(len(X_batch)):
        # print(f"validation input: {X_batch[i]}")
        # print(f"validation output: {y_batch[i]}")

# for X_batch, y_batch in test_loader:
    # for i in range(len(X_batch)):
        # print(f"test input: {X_batch[i]}")
        # print(f"test output: {y_batch[i]}")

class ConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 48, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28 * 28 * 48, 400)
        self.fc2 = nn.Linear(400, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = self.relu(self.conv3(X))
        X = self.pool(X)
        X = X.flatten(start_dim=1)
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output
    
model = ConvNet()
model.train()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    for X_batch, y_batch in train_loader:
        train_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(train_preds, y_batch.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_val_correct = 0

    for X_batch, y_batch in val_loader:
        val_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(val_preds, y_batch.unsqueeze(1))
        class_preds = val_preds > 0
        total_val_correct += (class_preds == y_batch.unsqueeze(1)).sum()

    val_accuracy = total_val_correct / len(val_data)
    print(f"Epoch {epoch} | Loss: {loss.item()} | Accuracy: {val_accuracy}")

print("\n------------------------Testing Phase-----------------------------\n")

model.eval()

with torch.no_grad():

    total_correct = 0

    for X_batch, y_batch in test_loader:
        test_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(test_preds, y_batch.unsqueeze(1))
        class_preds = test_preds > 0
        total_correct += (class_preds == y_batch.unsqueeze(1)).sum()

    test_accuracy = total_correct / len(test_data)
    print(f"Loss: {loss.item()} | Accuracy: {test_accuracy}")