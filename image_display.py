import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import numpy as np
from torchvision.transforms import v2
import splitfolders
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch

DATA_DIR = './ddidiversedermatologyimages'
CSV_FILENAME = 'ddi_metadata.csv' 
IMAGE_COL = 'DDI_file'            
LABEL_COL = 'malignant'         
NUM_IMAGES = 100
"""
def visualize_dataset(data_dir, csv_file, img_col, label_col, num_images):
    csv_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"NO CSV AT {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path)
    
    # Samples dataframe
    if len(df) > num_images:
        sample_df = df.sample(n=num_images, random_state=42)
    else:
        sample_df = df
        num_images = len(df)

    # Grid Setup
    grid_size = math.ceil(math.sqrt(num_images))
    plt.figure(figsize=(10, 10)) # Adjusted size for better visibility on laptop
    
    success_count = 0
    
    print(f"Loading {len(sample_df)} images from: {os.path.abspath(data_dir)}")

    for i, (index, row) in enumerate(sample_df.iterrows()):
        img_name = row[img_col]
        label = row[label_col]
        img_path = os.path.join(data_dir, img_name)
        
        # Make subplot slot
        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')

        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"{label}\n{img.size}", fontsize=8)
            success_count += 1
        except Exception as e:
            plt.text(0.5, 0.5, "MISSING\nFILE", horizontalalignment='center', 
                     verticalalignment='center', color='red', fontsize=10)
            plt.title(f"{label}\n(Not Found)", fontsize=8, color='red')
            # Prints the first few errors to help debug
            if i < 5: 
                print(f"COULDNT LOAD: {img_name} -> {e}")

    plt.tight_layout()
    plt.show()

    
    print(f"Displayed {success_count} out of {num_images}")
    
# Run the function
visualize_dataset(DATA_DIR, CSV_FILENAME, IMAGE_COL, LABEL_COL, NUM_IMAGES)
"""
# Data augmentation list
# , v2.RandomHorizontalFlip(0.5), v2.RandomPerspective(0.5, 0.5)
# ^ I commented this out because I am just trying to figure out what the rgb values should be for the mean for v2.Normalize()
# basically make the code below work and then whatever is printed out for the mean/std input it for the v2.Normalize tranform
transforms = v2.Compose([v2.ToTensor(), v2.Resize((950, 550))])

# Data preparation for training loop
root = "DDI"
splitfolders.ratio(root, output="DDI", ratio=(0.6, 0.2, 0.2))

train_data = ImageFolder(os.path.join(root, "train"), transform=transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

channels_sum = torch.zeros(3)
channels_squared_sum = torch.zeros(3)
num_batches = 0

for data, _ in train_loader:
    channels_sum += torch.mean(data, dim=[0, 2, 3])
    channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
    num_batches += 1

mean = channels_sum / num_batches
std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

print(f"Mean: {mean}")
print(f"Std: {std}")