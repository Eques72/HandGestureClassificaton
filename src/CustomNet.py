import torch
import torch.nn as nn
import random
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import pathlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



################################################################################
##################################### DATA #####################################
################################################################################

def grepClasses(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory))
    if not classes:
        raise FileNotFoundError(f"No classes in {directory}.")
        
    classToIndex = {className: i for i, className in enumerate(classes)}
    return classes, classToIndex

class DatasetFromFolderCustom(Dataset):
    
    def __init__(self, targetDir: str, transform=None, seed=1, split=0.8, trainSet = True) -> None:
        #super().__init__() #Is needed?
        self.paths = list(pathlib.Path(targetDir).glob("*/*.jpg"))
        
        random.Random(seed).shuffle(self.paths)
        self.paths = self.paths.copy() 
        if(trainSet):
            self.paths = self.paths[:int(len(self.paths) * split)]
        else:
            self.paths = self.paths[int(len(self.paths) * split):]

        self.transform = transform
        self.classes, self.classToIndex = grepClasses(targetDir)

    def getPaths(self, numberOfPaths = 10, getAllPaths = False):
        if getAllPaths:
            return self.paths
        tmpPaths = self.paths.copy()
        random.Random().shuffle(tmpPaths)
        return tmpPaths[:numberOfPaths]

    def load_image(self, index: int) -> Image.Image:
        imagePath = self.paths[index]
        return Image.open(imagePath) 
    
    def __len__(self) -> int:
        return len(self.paths)
    
    #Override
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        className  = self.paths[index].parent.name 
        classIndex = self.classToIndex[className]

        if self.transform:
            return self.transform(img), classIndex 
        else:
            return img, classIndex

################################################################################
##################################### MODEL ####################################
################################################################################

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.convBlock_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=6, 
                      stride=1, 
                      padding=2),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=5,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.convBlock_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=30*30*hidden_units, out_features=output_shape)
        )
    
    def forward(self, mod: torch.Tensor):
        mod = self.convBlock_1(mod)
        mod = self.convBlock_2(mod)
        mod = self.classifier(mod)
        return mod

################################################################################
################################ TEST AND TRAIN ################################
################################################################################

def trainStep(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               lossFun: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    model.train()
    
    trainLoss, trainAcc = 0, 0
    
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        prediction = model(X)

        loss = lossFun(prediction, Y)
        trainLoss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictedClass = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
        trainAcc += (predictedClass == Y).sum().item()/len(prediction)

    trainLoss = trainLoss / len(dataloader)
    trainAcc = trainAcc / len(dataloader)
    return trainLoss, trainAcc

def testStep(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):

    model.eval() 
    testLoss, testAcc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
    
            prediction = model(X)

            loss = loss_fn(prediction, Y)
            testLoss += loss.item()
            
            predictionLabels = prediction.argmax(dim=1)
            testAcc += ((predictionLabels == Y).sum().item()/len(predictionLabels))
            
    testLoss = testLoss / len(dataloader)
    testAcc = testAcc / len(dataloader)
    return testLoss, testAcc

def trainAndStat(model: torch.nn.Module, 
          trainDataloader: torch.utils.data.DataLoader, 
          testDataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          lossFun: torch.nn.Module,
          epochs: int,
          device: torch.device):
    
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):
        trainLoss, trainAcc = trainStep(
                                            model=model,
                                            dataloader=trainDataloader,
                                            lossFun=lossFun,
                                            optimizer=optimizer, 
                                            device=device)
        testLoss, testAcc = testStep(
            model=model,
            dataloader=testDataloader,
            loss_fn=lossFun,
            device=device)
        
        print(f"Epoch: {epoch+1} | "f"train_loss: {trainLoss:.4f} | "f"train_acc: {trainAcc:.4f} | "f"test_loss: {testLoss:.4f} | "f"test_acc: {testAcc:.4f}")

        results["train_loss"].append(trainLoss)
        results["train_acc"].append(trainAcc)
        results["test_loss"].append(testLoss)
        results["test_acc"].append(testAcc)
    return results

################################################################################
################################### IO MODEL ###################################
################################################################################

def LoadModel(path:str, filename:str) -> nn.Module():
    model = torch.load(os.path.join(os.getcwd(),"{path}/{filename}.pth".format(path=path, filename=filename)))
    return model    

def SaveModel(model, path:str, filename:str): 
    torch.save(model, os.path.join(os.getcwd(),f"{path}\\{filename}.pth".format(path=path, filename=filename)))
    pass

################################################################################
################################### ANALYSIS ###################################
################################################################################

def plotLoss(results: Dict[str, List[float]], epochs: int):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='train_loss')
    plt.plot(epochs,  results['test_loss'], label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='train_accuracy')
    plt.plot(epochs, results['test_acc'], label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.show()
    pass





