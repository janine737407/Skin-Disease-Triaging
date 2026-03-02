import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

#Convert to tensors scaled between 0.0 and 1.0 + resize them
calc_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True), 
    v2.ToDtype(torch.float32, scale=True) 
])


train_dataset = ImageFolder(root='train', transform=calc_transforms)

# Batch images to process faster
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

def calculate_mean_std(loader):
    print(f"Calculating stats across {len(loader.dataset)} training images...")
    
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        #images shape: (batch_size, channels, height, width)
        batch_size, _, height, width = images.shape
        num_pixels += batch_size * height * width
        
        # Sum pf batch, height, and width
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        channel_sq_sum += torch.sum(images ** 2, dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    variance = (channel_sq_sum / num_pixels) - (mean ** 2)
    std = torch.sqrt(variance)
    
    return mean.tolist(), std.tolist()

# Calculate and print the stats
if __name__ == '__main__':
    train_mean, train_std = calculate_mean_std(train_loader)

    
    print(f"mean = {train_mean}")
    print(f"std  = {train_std}")