import CustomNet
from CustomNet import TinyVGG
import numpy as np
import torch
import torch.nn as nn
import os
import PIL.Image as Image

print("IM HERE")
def LoadModel(path:str, filename:str) -> nn.Module():
    model = torch.load(os.path.join(os.getcwd(),"{path}/{filename}.pth".format(path=path, filename=filename)))
    return model  

def see_resoult(model: nn.Module, test_dataloader: torch.utils.data.DataLoader, num_of_img = 5) -> None:
    
    num_of_img_shown = 0;

    with torch.no_grad():
        model.eval()    

        for batch, (X, y) in enumerate(test_dataloader):
            if num_of_img <= num_of_img_shown:
                break
            num_of_img_shown += 1
            # Send data to target device
            X_cpy = X
            X = X.to(torch.device('cuda'))    
            # 1. Forward pass
            predictions = model(X)
            print(predictions)
            image = np.array(X_cpy)
            Image.fromarray(image).show()  

import numpy as np
import cv2
def showClassifiedImage(image, label, score=0.0):
    image = np.array(image)
    image = cv2.putText(image, f'{label}, score: {score}', (1,12), cv2.FONT_HERSHEY_PLAIN,
                 0.7, (255, 0, 0), 1)
    Image.fromarray(image).show()    
    pass 

test_data_custom = CustomNet.ImageFolderCustom(targ_dir=CustomNet.dir_path, 
                                      transform=CustomNet.train_transforms, seed=CustomNet.SEED, split=0.8, trainSet=False)
# test_dataloader_simple = CustomNet.DataLoader(test_data_custom, 
#                                      batch_size=CustomNet.BATCH_SIZE, 
#                                      shuffle=True) 

model = LoadModel( 
    "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
    "model_seed8502_epoch10_batch32_spec-largeKernel_lowLR_cus")
# see_resoult(model, test_dataloader_simple, 5)


import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

tmpNumOfImgs = 5
testImgPaths = test_data_custom.getPaths(tmpNumOfImgs)
imagePredictions = []#list of tuples, guessed label and probability

for i in range(tmpNumOfImgs):

    image4Model = torchvision.io.read_image(str(testImgPaths[i]))
    image4Model = image4Model / 255.0

    image4ModelTransformPipeline = transforms.Compose([
        transforms.Resize((128,128)),
    ])
    image4ModelTransformed = image4ModelTransformPipeline(image4Model)

    model.eval()
    with torch.inference_mode():
        
        imagePrediction = model(image4ModelTransformed.unsqueeze(dim=0).to(torch.device('cuda')))
        imagePrediction_probabilities = torch.softmax(imagePrediction, dim=1)
        imagePrediction_class_pred_val = torch.argmax(imagePrediction_probabilities, dim=1)
        imagePrediction_class_pred = test_data_custom.classes[imagePrediction_class_pred_val.cpu()] # put pred label to CPU, otherwise will error
        imagePredictions.append((imagePrediction_class_pred, imagePrediction_probabilities.max().cpu()))

plt.figure(figsize=(32, 8))

for j, imgPath in enumerate(testImgPaths):
    #read image and adjust for plt
    image = torchvision.io.read_image(str(imgPath))
    image = image / 255.0
        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]

        #target image == custom image
    plt.subplot(1, tmpNumOfImgs, j+1)
    plt.imshow(image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    title = f"Pred: {imagePredictions[j][0]} | Prob: {imagePredictions[j][1]:.3f}"
    plt.title(title)
    plt.axis(False);
plt.tight_layout(pad=12)
plt.show()    
