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

class ConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1) # We decided to keep kernel size, stride, and padding the same for all three convolution layers.
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 48, 3, 1, 1) # We added another convolution layer to increase the amount of features the model could pick up on.
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28 * 28 * 48, 400)
        self.fc2 = nn.Linear(400, 1) # We made two linear layers instead of one because the jump from 37632 to 1 seemed like a lot.
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

loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
optimizer = optim.Adam(model.parameters(), lr=0.01)
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    total_train_loss = 0
    total_train_correct = 0

    for X_batch, y_batch in train_loader:
        train_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(train_preds, y_batch.unsqueeze(1))
        total_train_loss += loss
        class_preds = train_preds > 0
        total_train_correct += (class_preds == y_batch.unsqueeze(1)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_data)
    print(f"Epoch {epoch} | Training Loss: {train_loss.item()} | Train Accuracy: {train_accuracy}")

    total_val_loss = 0
    total_val_correct = 0

    for X_batch, y_batch in val_loader:
        val_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(val_preds, y_batch.unsqueeze(1))
        total_val_loss += loss
        class_preds = val_preds > 0
        total_val_correct += (class_preds == y_batch.unsqueeze(1)).sum()

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / len(val_data)
    print(f"Epoch {epoch} | Validation Loss: {val_loss.item()} | Validation Accuracy: {val_accuracy}")

print("\n------------------------Testing Phase-----------------------------\n")

model.eval()

with torch.no_grad():

    total_correct = 0
    total_test_loss = 0

    for X_batch, y_batch in test_loader:
        test_preds = model(X_batch)
        y_batch = y_batch.float()
        loss = loss_fn(test_preds, y_batch.unsqueeze(1))
        total_test_loss += loss
        class_preds = test_preds > 0
        total_correct += (class_preds == y_batch.unsqueeze(1)).sum()

    test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / len(test_data)
    print(f"Loss: {test_loss.item()} | Accuracy: {test_accuracy}")