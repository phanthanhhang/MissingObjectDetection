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
        
        self.conv5 = nn.Conv2d(64,256, kernel_size=(53,53))
        self.dropout3 = nn.Dropout(p=0.2)
        self.conv6 = nn.Conv2d(256,2, kernel_size=(1,1))
        # self.fc1 = nn.Linear(64*53*53,256)
        # self.dropout3 = nn.Dropout(p=0.2)
        # self.fc2 = nn.Linear(256,2)
        
        self.sequential1 = nn.Sequential(self.conv1, self.conv2, self.maxpool1, self.dropout1)
        self.sequential2 = nn.Sequential(self.conv3, self.conv4, self.maxpool2, self.dropout2)
        self.sequential3 = nn.Sequential(self.conv5, self.dropout3, self.conv6)
        # self.sequential3 = nn.Sequential(self.fc1,self.dropout3,self.fc2)
        
    def forward(self, x):
        x = self.sequential1(x)
#         print('1: ',x.shape)
        x = self.sequential2(x)
#         print('2: ',x.shape)
        # x = torch.flatten(x, start_dim=1)
        x = self.sequential3(x)
        # print('3: ',x.shape)
#         y = self.cnn(x)
        return x.squeeze(-1).squeeze(-1)

class Class_loss(nn.Module):
    def __init__(self):
        super(Class_loss, self).__init__()
    
    def forward(self, output, target):
        return F.cross_entropy(output, target)
#         return (-output[range(target.shape[0]), target] + torch.log(torch.sum(torch.exp(output),1))).mean()

class Context_loss(nn.Module):
    def __init__(self):
        super(Context_loss, self).__init__()
        
    def forward(self, output1, output2):
        return torch.norm(output1 - output2, p=2, dim=1).mean()

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion,self).__init__()
        self.class_loss = Class_loss()
        self.context_loss = Context_loss()
    
    def forward(self,output1,output2,target):
        return self.class_loss(output1,target) + 0.5*self.context_loss(output1,output2)

class SiameseDataset(Dataset):
    def __init__(self, csv_file=None, transform=None):
        # used to prepare the labels and images path
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, index):
        raw_file_name = self.data.iloc[index]['file_name']
        
        label = self.data.iloc[index]['label']
        
        raw_image = Image.open(raw_file_name).convert('RGB')
        mask_image = raw_image.copy()
        
        if label :
#             print('read mask : !!',)
            mask_image = Image.open(raw_file_name.replace('_raw','_masked')).convert('RGB')
        
#         print('check sum : ',np.asarray(raw_image).sum() == np.asarray(mask_image).sum() )

        # Apply image transformations
        if self.transform is not None:
            raw_image = self.transform(raw_image)
            mask_image = self.transform(mask_image)

        return raw_image, mask_image, torch.tensor(label)

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    # Separate the pairs of images and the classes
    image1_list = []
    image2_list = []
    labels = []

    for item in batch:
        image1, image2, label = item
        image1_list.append(image1)
        image2_list.append(image2)
        labels.append(label)

    # Stack images into tensors
    image1_tensor = torch.stack(image1_list)
    image2_tensor = torch.stack(image2_list)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    return (image1_tensor, image2_tensor), labels_tensor

class SiameseValidDataset(Dataset):
    def __init__(self, csv_file=None, transform=None):
        # used to prepare the labels and images path
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['file_name'].str.contains('raw')].reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, index):
        raw_file_name = self.data.iloc[index]['file_name']
        
        label = self.data.iloc[index]['label']
        
        raw_image = Image.open(raw_file_name).convert('RGB')
        # Apply image transformations
        if self.transform is not None:
            raw_image = self.transform(raw_image)
#             mask_image = self.transform(mask_image)

        return raw_image, torch.tensor(label)

    def __len__(self):
        return len(self.data)

num_classes = 2
epochs = 100
device = 'cuda:1'
batch_size = 64

transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = SiameseDataset('train.csv',transform)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=custom_collate_fn,num_workers=10)

valid_dataset = SiameseValidDataset('valid.csv',transform)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=True, num_workers=10)

model = base_model(num_classes=num_classes).to(device)
loss_func = Criterion()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)


def train_1_epoch(model,loss_func,optimizer,dataloader,epoch, epochs,device):
    model.train()
    
    print(('\n' + '%10s   ' * 3) % ('Epoch', 'total_loss', 'targets'))
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=len(dataloader))
    
    train_loss = 0
    for batch_id,((raw_images,mask_images),labels) in pbar:
        optimizer.zero_grad()
        
        raw_images = raw_images.to(device)
        mask_images = mask_images.to(device)
        labels = labels.to(device)
        
        out_raw  =  model(raw_images)
        out_mask = model(mask_images)
        
        loss = loss_func(out_raw,out_mask,labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        s = f'       {epoch}/{epochs}  {train_loss/(batch_id+1):4.4f}             {labels.shape[0]}'
        pbar.set_description(s)
        ckpt = {'model':model.state_dict()}
        torch.save(ckpt,'context_model_last_2810.pt') 

def test(model,dataloader,device):
    model.eval()
    total= 0
    correct = 0
    with torch.no_grad():
        print(('\n' + '%10s   ' * 3) % ('acc', 'correct', 'total'))
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar,total=len(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            s = f'      {correct/total: 4.4f}  (    {correct}  /      {total}) '
            pbar.set_description(s)

    return correct / total

best_acc = 0
for epoch in range(epochs):
    train_1_epoch(model,loss_func,optimizer,train_dataloader,epoch,epochs,device)
    acc = test(model,valid_dataloader,device)
    if acc > best_acc:
        ckpt = {'model':model.state_dict()}
        torch.save(ckpt,'context_model_best_2810.pt')