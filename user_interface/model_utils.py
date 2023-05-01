import torch
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVATION = "softmax2d"
ENCODER_NAME = "timm-gernet_l"
ENCODER_WEIGHTS= "imagenet"

IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 832 

MODEL_DIR = os.path.expanduser('~') + "/AVS1/saved_models/best_model_3.pth.tar"

# Loads the model with the "MODEL_DIR" path
def load_model():
    print("Loading model from: ", MODEL_DIR)
    try:
        # UNet Model
        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=6,
            activation=ACTIVATION,
        ).to(DEVICE)

        m = torch.load(MODEL_DIR)
        model.load_state_dict(m['model_state_dict'])
        model.eval()

        print("Model succesfully loaded")
        return model
    except:
        raise Exception('Could not load model')

# Predicts the image on "path" using the "model"
def predict(path, model):
    print("Predicting image using model")
    image = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize([512, 832]),
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0],[1.0, 1.0, 1.0])
    ])
    image = trans(image).to(DEVICE).unsqueeze(0)

    model.eval()
    preds = model(image)

    #Convert the prediction from a tensor of 6 channels (one for each class) to tensor containing rgb channels
    #Classes | 0: Background | 1: Fairway | 2: Green | 3: Tees | 4: Bunkers | 5: Water |
    class_to_color = [torch.tensor([0.0, 0.0, 0.0], device='cuda'), torch.tensor([0.0, 140.0/255, 0.0],  device='cuda'), torch.tensor([0.0, 1.0, 0.0],  device='cuda'), torch.tensor([1.0, 0.0, 0.0],  device='cuda'), torch.tensor([217.0/255, 230.0/255, 122.0/255],  device='cuda'), torch.tensor([7.0/255, 15.0/255, 247.0/255],  device='cuda')]
    
    output = torch.zeros(3, preds.size(-2), preds.size(-1), dtype=torch.float,  device=DEVICE) #Output size is set to preds.shape[0] as the size automatically changes to fit the remaining batch_size.
    for class_idx, color in enumerate(class_to_color):
        mask = preds[:,class_idx,:,:] == torch.max(preds, dim=1)[0]
        #mask = mask.unsqueeze(1)
        curr_color = color.reshape(3, 1, 1)
        segment = mask*curr_color 
        output += segment
    transform = transforms.ToPILImage()
    image = transform(output)
    image = np.array(image)
    print("Predicting succesful")
    return image


