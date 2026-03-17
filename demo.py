import torch
from PIL import Image
from torchvision.transforms import v2

from data_augmentation import ConvNet

model = ConvNet()
model.load_state_dict(torch.load("weights.pt", weights_only=True))
model.eval()

transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.594, 0.478, 0.396], std=[0.207, 0.204, 0.210])
])

# Kaedon can you find a picture that the model predicts correctly and add it to the repo, then change "image.png" to the name
img = Image.open("melanoma.png").convert('RGB')
img = transforms(img)

img = torch.unsqueeze(img, 0)


THRESHOLD = 0.6 

with torch.no_grad():
    raw_logit = model(img)
    
    #Sigmoid for binary classification
    probability = torch.sigmoid(raw_logit).item()
    
    #Convert the raw number output into understandable classes
    if probability > THRESHOLD:
        prediction_class = "Malignant"
    else:
        prediction_class = "Benign"

    print("------- Model Results -------")
    print(f"Raw Logit Output: {raw_logit.item():.4f}")
    print(f"Malignant Probability: {probability * 100:.2f}%")
    print(f"Final Prediction: {prediction_class}")