import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

DATA_DIR = './ddidiversedermatologyimages'
CSV_FILENAME = 'ddi_metadata.csv' 
IMAGE_COL = 'DDI_file'            
LABEL_COL = 'malignant'         
ID_COL = 'DDI_ID'  # Added reference for the ID column
NUM_IMAGES = 100

def visualize_dataset(data_dir, csv_file, img_col, label_col, id_col, num_images):
    csv_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"NO CSV AT {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path)
    
    if len(df) > num_images:
        sample_df = df.sample(n=num_images, random_state=42)
    else:
        sample_df = df
        num_images = len(df)

    grid_size = math.ceil(math.sqrt(num_images))
    # Increased height slightly to accommodate the extra line of text in titles
    plt.figure(figsize=(12, 14)) 
    
    success_count = 0
    print(f"Loading {len(sample_df)} images from: {os.path.abspath(data_dir)}")

    for i, (index, row) in enumerate(sample_df.iterrows()):
        img_name = row[img_col]
        label = row[label_col]
        ddi_id = row[id_col] # Extract the ID value
        img_path = os.path.join(data_dir, img_name)
        
        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')

        try:
            img = Image.open(img_path)
            plt.imshow(img)
            # Updated title to include ID, Label, and Size
            plt.title(f"ID: {ddi_id}\n{label}\n{img.size}", fontsize=7)
            success_count += 1
        except Exception as e:
            plt.text(0.5, 0.5, "MISSING\nFILE", horizontalalignment='center', 
                     verticalalignment='center', color='red', fontsize=8)
            plt.title(f"ID: {ddi_id}\n(Not Found)", fontsize=7, color='red')
            if i < 5: 
                print(f"COULDNT LOAD: {img_name} -> {e}")

    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {success_count} out of {num_images}")
    
# Run the function with the new ID_COL argument
visualize_dataset(DATA_DIR, CSV_FILENAME, IMAGE_COL, LABEL_COL, ID_COL, NUM_IMAGES)
