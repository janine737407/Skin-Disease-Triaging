import torch
from PIL import Image
from torchvision.transforms import v2
from data_augmentation import ConvNet

model = ConvNet()
model.load_state_dict(torch.load("final+weights.pt", weights_only=True))
model.eval()

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.594, 0.478, 0.396], std=[0.207, 0.204, 0.210])
])

img = Image.open("melanoma.png").convert('RGB')
img = transforms(img)

# print(img.shape) # check image shape is correct, if it isn't, unsqueeze
img = torch.unsqueeze(img, 0)

pred = model(img)
print(pred.item())
# If you are doing classification, use Softmax to turn the output into percentages (see Week 4 Day 2 activity document).
# Also, try to convert the raw number output into understandable classes.