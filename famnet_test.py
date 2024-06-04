import torch
import torchvision.models as models

# Define the model architecture (this should match the architecture of the model when it was saved)
model = models.resnet50()

# Specify the path to your .pth file
model_path = 'FamNet_Save1.pth'

# Load the state dictionary
state_dict = torch.load(model_path)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model in evaluation mode
model.eval()
