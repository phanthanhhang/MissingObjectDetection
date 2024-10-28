import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2

class base_model(nn.Module):
    def __init__(self, num_classes):
        super(base_model, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3))
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(32,64, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3))
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(p=0.2)
        
#         self.conv5 = nn.Conv2d(64,256, kernel_size=(72,72))
#         self.dropout3 = nn.Dropout(p=0.2)
#         self.conv6 = nn.Conv2d(256,2, kernel_size=(1,1))

        self.conv5 = nn.Conv2d(64,256, kernel_size=(53,53))
        self.dropout3 = nn.Dropout(p=0.2)
        self.conv6 = nn.Conv2d(256,2, kernel_size=(1,1))
#         self.fc1 = nn.Linear(64*53*53,256)
#         self.dropout3 = nn.Dropout(p=0.2)
#         self.fc2 = nn.Linear(256,num_classes)
        
        self.sequential1 = nn.Sequential(self.conv1, self.conv2, self.maxpool1, self.dropout1)
        self.sequential2 = nn.Sequential(self.conv3, self.conv4, self.maxpool2, self.dropout2)
        self.sequential3 = nn.Sequential(self.conv5, self.dropout3, self.conv6)
#         self.sequential3 = nn.Sequential(self.fc1,self.dropout3,self.fc2)
        
    def forward(self, x):
        x = self.sequential1(x)
#         print('1: ',x.shape)
        x = self.sequential2(x)
#         print('2: ',x.shape)
#         x = torch.flatten(x, start_dim=1)
        x = self.sequential3(x)
#         print('3: ',x.shape)
#         y = self.cnn(x)
#         return x
        return x.squeeze(-1).squeeze(-1)
    

from torchvision import models, transforms

# Load the model
# model = models.resnet50(pretrained=True)
model = base_model(2)
ckpt = torch.load('context_model_last.pt',map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

# Prepare the input image
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch
    return input_batch, input_image

# Get Grad-CAM
def generate_grad_cam(model, input_batch, target_layer, original_image):
    # Hook to get gradients
    gradients = []
    def save_gradient(grad):
        gradients.append(grad)
        
    feature_maps = []
    def get_feature_map(module, input, output):
        feature_maps.append(output)
    
    # Register the hook
    target_layer.register_backward_hook(lambda m, grad_input, grad_output: save_gradient(grad_output[0]))
    target_layer.register_forward_hook(get_feature_map)

    # Forward pass
    output = model(input_batch)
    pred_class = output.argmax(dim=1)
    print('output:', output)
    print('pred_class:',pred_class)

    # Zero gradients
    model.zero_grad()

    # Backward pass
    output[0, pred_class].backward()

    # Get the gradients and feature maps
    grad = gradients[0].cpu().data.numpy()
#     feature_map = target_layer.output.cpu().data.numpy()
    feature_map = feature_maps[0].cpu().data.numpy() 
    print('feature_map:',feature_map.shape) # Shape: (1, 2048, 7, 7)
    print('grad:', grad.shape) # Shape: (1, 2048, 7, 7)

    # Compute weights
    weights = np.mean(grad, axis=(2, 3))[0, :] # Shape: (2048,)
    print('weights:', weights.shape, len(weights))
    
    # Compute Grad-CAM
    cam = np.zeros(feature_map.shape[2:], dtype=np.float32)
    print('cam:',cam.shape) # Shape: (7, 7)
    for i in range(len(weights)):
        # print(weights[i])
        cam += weights[i] * feature_map[0, i, :, :]
    
    cam = np.maximum(cam, 0)  # Apply ReLU
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))  # Resize to input image size
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)  # Normalize

    return cam

# Overlay the heatmap on the original image
def overlay_heatmap(original_image, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.5, np.array(original_image), 0.5, 0)
    return overlay


# Example usage

# i = Image.open('/data2/thang/SIA/ScaledYOLO/unfold_missing_wheel_data/SINSQ78473_SQ114036_SINSQ78473_09122023_MotionsCloud_SQ114036_back_647VPPSQ114036back.jpeg')
# i = np.asarray(i)
# crop = i[2400:,:800,:]
# crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

image_path = 'test_pos_3.jpg'
# image_path = 'test_neg_4.jpg'

# image_path = 'missing_wheel_luggage.jpg'

input_batch, original_image  = preprocess_image(image_path)
# target_layer = model.layer4[-1]  # Last convolutional layer for ResNet
target_layer = model.sequential2[-3] # Last convolutional layer for ResNet

cam = generate_grad_cam(model, input_batch, target_layer, original_image)
# original_image = cv2.imread(image_path)
overlay = overlay_heatmap(original_image, cam)

# Display the result
# plt.imshow('Grad-CAM', overlay)
# plt.show()

img = cv2.imwrite(f'grad-cam-{image_path}', overlay)