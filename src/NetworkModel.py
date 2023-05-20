import torch as t
import torch.nn as nn
import torchvision.transforms as tTrans


class CNNModel(nn.Module):
    
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.subModuleOne = nn.Sequential(nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0), 
                                      nn.ReLU(), 
                                      nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.subModuleTwo = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape))    
    pass

    def forward(self, x: t.Tensor):
        print("pass1")
        x = self.subModuleOne(x)
        print("pass2")
        x = self.subModuleTwo(x)
        print("pass3")        
        x = self.classifier(x)
        print("pass4")
        return x
    pass

model = CNNModel(input_shape=3, 
    hidden_units=10, 
    output_shape=1).to('cuda')
model

loss_fn = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(params=model.parameters(), #might be adam or smth
                             lr=0.1)

###############################################################################################
import torch as t
import torch.nn as nn
import torchvision.transforms as tTrans
import torchvision.models as models
import DatasetManagerMini as dm
import random
import PIL.Image as Image
import cv2
import numpy as np

seed = 78
split = 0.9
numberOfClasses = len(dm.ClassGesturesNames)
batchSize = 50 # 1-100 for now
epochIterations = 5#..15

lr = 0.005 # 0.0001-0.1
momentum = 0.9
weight_decay = 5e-4

def Collate(batch):
    batch_targets = list()
    images = list()

    for b in batch:
        images.append(b[0])
        batch_targets.append({"labels": b[1]["labels"]})
    return images, batch_targets

    #setting up main variables 
t.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = t.device('cuda') 
transform = tTrans.ToTensor()

    #Load train and test sets (not in memory yet)
trainSet = dm.DatasetManagerMini( "D:\\Politechnika\\BIAI\\subsample", True, transform, seed, split)
testSet = dm.DatasetManagerMini( "D:\\Politechnika\\BIAI\\subsample", False, transform, seed, split)

dataLoaderTrain = t.utils.data.DataLoader(trainSet, batch_size=batchSize,collate_fn=Collate, shuffle=True)#, num_workers=4) #wrapper
dataLoaderTest = t.utils.data.DataLoader(testSet, batch_size=batchSize,collate_fn=Collate, shuffle=True)#, num_workers=4)   #wrapper
    ####################################
    
model = CNNModel(input_shape=100, 
    hidden_units=10, 
    output_shape=len(dm.ClassGesturesNames)).to(device)

optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ####################################
warmup_factor = 1.0 / 1000
warmup_iters = min(1000, len(trainSet) - 1)

lr_scheduler = t.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    ####################################
   
for epoch in range(epochIterations):
       
    if __name__ == '__main__':
        model.train()
      #  total = 0
      #  sum_loss = 0

#        for (images, targets) in enumerate(dataLoaderTrain):
    for images, targets in dataLoaderTrain:

        batch = len(images)
        imagesTensor = []
        for image in images:
            imagesTensor.append(transform(image).to(device))

        for target in targets:
            for key, value in target.items():
                target[key] = value.to(device)
        print(transform(images[0]).shape)
#        print(type(imagesTensor),"\n",type(imagesTensor[0]),"\n",type(targets),"\n",type(images[0]),"\n",type[images])
        loss_dict = model(imagesTensor)
        input("Press Enter to continue...")
       # loss_dict = model(imagesTensor, targets)
        losses = sum(loss for loss in loss_dict.values())
     #       loss = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()