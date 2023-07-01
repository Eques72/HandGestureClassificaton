import torch
import torch.nn as nn
import random
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Type
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
    
    def countClassesInstances(self):
        print("class to index contains: ", self.classToIndex)
        classesInstances = [0] * len(self.classes)
        for img in self.paths:
            className  = img.parent.name
            classesInstances[self.classToIndex[className]] += 1
        print("class numbers: ", classesInstances)
        print("img sum: ", sum(classesInstances))
        pass

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

class CustomTinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        # self.block = nn.Sequential(
        #     nn.Conv2d(in_channels=input_shape, 
        #               out_channels=hidden_units, 
        #               kernel_size=3, 
        #               stride=1, 
        #               padding=1),
        # nn.BatchNorm2d(hidden_units),
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm2d(hidden_units),
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm2d(hidden_units),
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        # nn.Flatten(),
        # nn.Linear(2560, hidden_units*2),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(20, output_shape)
        # )

        self.convBlock_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=6, 
                      stride=1, 
                      padding=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                       out_channels=hidden_units,
                       kernel_size=6,
                       stride=1,
                       padding=1),
            nn.Dropout(0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.convBlock_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=6, padding=1),
            nn.Dropout(0.9),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=pow(32,2)*hidden_units, out_features=output_shape)
        )
    
    def forward(self, mod: torch.Tensor):
        # mod = self.block(mod)
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




# datasetDirPath = "D:\\Politechnika\\BIAI\\cropped"
# import testCustomNet as tcn

# testTransforms = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor()
# ])
# SEED = 8502
# BATCH_SIZE = 32
# NUM_EPOCHS = 30
# SPLIT = 0.8

# testData = DatasetFromFolderCustom(targetDir=datasetDirPath, 
#                                         transform=testTransforms, seed=SEED, split=SPLIT, trainSet=False)
# model = LoadModel( 
#     "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
#     "model_seed8502_epoch10_batch32_spec-largeKernel_lowLR_cus")
# tcn.testModelVisually(testData, model, 5)


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out


class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 18
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x