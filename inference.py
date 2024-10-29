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
    
transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                    #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    # preprocess = transforms.Compose([
    #     transforms.Resize(224),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch
    return input_batch, input_image

# Load the model
# model = models.resnet50(pretrained=True)
model = base_model(2)
ckpt = torch.load('context_model_last.pt',map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()


image_path = 'test_pos_6.jpg'
# image_path = 'test_neg_4.jpg'

# image_path = 'missing_wheel_luggage.jpg'

input_batch, original_image  = preprocess_image(image_path)
# Forward pass
output = model(input_batch)
pred_class = output.argmax(dim=1)
print('output:', output, torch.softmax(output, dim=1))
print('pred_class:',pred_class)