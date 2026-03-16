import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
# import os

# print(len(os.listdir('train/malignant')))
# exit()
# Resize images, then convert them to tensors and apply augmentation
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True), 
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomPerspective(0.2, 0.5), # Lowered distortion scale
    v2.Normalize(mean=[0.594, 0.478, 0.396], std=[0.207, 0.204, 0.210]),
    # Added new augmentations
    v2.RandomVerticalFlip(0.5), # New
    v2.RandomRotation(15), # New
    v2.ColorJitter(0.2, 0.2), # New 
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
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1) # We decided to keep kernel size, stride, and padding the same for all three convolution layers.
        self.batchNorm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.batchNorm2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 48, 3, 1, 1) # We added another convolution layer to increase the amount of features the model could pick up on.
        self.batchNorm3 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.3) # 30% probability dropout
        self.fc1 = nn.Linear(28 * 28 * 48, 400)
        self.fc2 = nn.Linear(400, 1) # We made two linear layers instead of one because the jump from 37632 to 1 seemed like a lot.
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.batchNorm1(X)
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.batchNorm2(X)
        X = self.pool(X)
        X = self.relu(self.conv3(X))
        X = self.batchNorm3(X)
        X = self.pool(X)
        X = X.flatten(start_dim=1)
        X = self.dropout(X) # Dropout before fc1
        X = self.relu(self.fc1(X))
        X = self.dropout(X) # Dropout before fc2
        X = self.fc2(X)
        output = self.sig(X)
        return output
    
model = ConvNet()
model.train()

# loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # lowered learning rate 0.01 -> 0.001

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Added lr scheduler
NUM_EPOCHS = 10
THRESHOLD = 0.2

for epoch in range(NUM_EPOCHS):

    total_train_loss = 0
    total_train_correct = 0
    train_tp = 0
    train_fp = 0
    train_fn = 0

    model.train() # train mode for dropout

    for X_batch, y_batch in train_loader:
        train_preds = model(X_batch)
        y_batch = y_batch.float().unsqueeze(1)
        loss = loss_fn(train_preds, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() 
        class_preds = train_preds > THRESHOLD
        total_train_correct += (class_preds == y_batch).sum().item()

        # Calculate true positives, false positives, false negatives
        train_tp += ((class_preds == 1) & (y_batch == 1)).sum().item()
        train_fp += ((class_preds == 1) & (y_batch == 0)).sum().item()
        train_fn += ((class_preds == 0) & (y_batch == 1)).sum().item()

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_data)


    # Calculate training precision and recall
    train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0.0
    train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")

    total_val_loss = 0
    total_val_correct = 0
    val_tp = 0
    val_fp = 0
    val_fn = 0

    total_malignant = 0

    model.eval() # No dropout for validation
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_preds = model(X_batch)
            y_batch = y_batch.float().unsqueeze(1)
            loss = loss_fn(val_preds, y_batch)

            total_val_loss += loss
            class_preds = val_preds > THRESHOLD
            total_val_correct += (class_preds == y_batch).sum().item()

            val_tp += ((class_preds == 1) & (y_batch == 1)).sum().item()
            val_fp += ((class_preds == 1) & (y_batch == 0)).sum().item()
            val_fn += ((class_preds == 0) & (y_batch == 1)).sum().item()
            total_malignant += (class_preds == 1).sum().item()


    val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / len(val_data)
    val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
    val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0

    print(f"Malignant preds: {total_malignant}")
    
    print(f"Epoch {epoch+1} | Val Loss:   {val_loss:.4f} | Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} \n")
    scheduler.step() # Step learning rate
print("\n--------------------------------Testing Phase-------------------------------------\n")

model.eval()

with torch.no_grad():

    total_correct = 0
    total_test_loss = 0
    
    all_class_preds = [] # for confusion matrix
    test_outputs = []

    for X_batch, y_batch in test_loader:
        test_preds = model(X_batch)
        y_batch = y_batch.float().unsqueeze(1)
        loss = loss_fn(test_preds, y_batch)

        total_test_loss += loss
        class_preds = test_preds > THRESHOLD
        all_class_preds.append(class_preds.squeeze().tolist())
        test_outputs.append(y_batch.squeeze().tolist()) #JSDGDSIUFGNDIUFNGIUNIFDUNGIDFgFIU
        total_correct += (class_preds == y_batch).sum().item() # Changed from .unsqueeze to .item

    all_class_preds = [pred for batch in all_class_preds for pred in batch]
    all_class_preds = torch.tensor(all_class_preds)
    test_outputs = [output for batch in test_outputs for output in batch]
    test_outputs = torch.tensor(test_outputs)

    test_precision = precision_score(test_outputs, all_class_preds, average=None)
    test_recall = recall_score(test_outputs, all_class_preds, average=None)

    test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / len(test_data)
    

    print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f} | Precision: {test_precision} | Recall: {test_recall}")

    

    
    cm = confusion_matrix(test_outputs, all_class_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['Benign', 'Malignant'])
    disp.plot()
    plt.show()


