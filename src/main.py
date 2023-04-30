import numpy as np
import random
import os

import DatasetManager as dm

import torch as t
import torch.nn as nn
import torchvision.transforms as tTrans
import torchvision.models as models

import PIL.Image as Image
import cv2
 
# Learn
# Save
# Load
# Test (predict)
# Data
# Create / Use model

PATH_IMAGES = "D:\\Politechnika\\BIAI\\subsample"
PATH_ANNOTATIONS = "D:\\Politechnika\\BIAI\\ann_subsample\\ann_subsample"



def LoadModel(path:str, filename:str) -> nn.Module():
   # model = nn.Module()
    model = t.load(os.path.join(os.getcwd(),"{path}/{filename}.pth".format(path=path, filename=filename)))
   # model.load_state_dict(t.load(os.path.join(os.getcwd(),"{path}/{filename}.pth".format(path=path, filename=filename))))
    model.eval()
    #model.load_state_dict(t.load(os.path.join(os.getcwd(),"{path}/{filename}.pth".format(path=path, filename=filename))))
    return model    

def SaveModel(model, path:str, filename:str): 
    t.save(model, os.path.join(os.getcwd(),f"{path}\\{filename}.pth".format(path=path, filename=filename)))
    
    #t.save(model.state_dict(),os.path.join(os.getcwd(),f"{path}\\{filename}.pth".format(path=path, filename=filename)))
    pass

# def TeachModel():
#     pass

# def TeachPreTrainedModel():
#     pass

#use haGRID database
#Custom: seed, split, batch, epoch

# Params declarations:
seed = 78
split = 0.9
numberOfClasses = len(dm.ClassGesturesNames)
batchSize = 20 # 1-100 for now
epochIterations = 3#..15

def main():
    #TEST
  #  D = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, True, None, 42, 0.8)
  #  D.GetSample(666)



    #Optimizer parameters
    lr = 0.005
    momentum = 0.9
    weight_decay = 5e-4


    #setting up main variables 
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = t.device('cuda') 
    transform = tTrans.ToTensor()

    #Load train and test sets (not in memory yet)
    trainSet = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, True, transform, seed, split)
    testSet = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, False, transform, seed, split)

    dataLoaderTrain = t.utils.data.DataLoader(trainSet, batch_size=batchSize,collate_fn=Collate, shuffle=True)#, num_workers=4) #wrapper
    dataLoaderTest = t.utils.data.DataLoader(testSet, batch_size=batchSize,collate_fn=Collate, shuffle=True)#, num_workers=4)   #wrapper
    ####################################
    
    model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=numberOfClasses + 1, pretrained_backbone=True) #MODEL, for now is pre trained 

    model.to(device)

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
            imagesTensor = list(transform(image).to(device) for image in images)

            for target in targets:
                for key, value in target.items():
                    target[key] = value.to(device)
            loss_dict = model(imagesTensor, targets)
            losses = sum(loss for loss in loss_dict.values())
     #       loss = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr_scheduler.step()

          #  total = total + batch
         #   sum_loss = sum_loss + loss

      #  metrics = eval(model, dataLoaderTest, epoch)
       # print(f"epoch : {epoch}  |||  loss : {sum_loss / total} ||| MAP : {metrics['map']}")
        SaveModel(model, "checkpoints", str(epoch))

    SaveModel(model, "trainedFinals", "model_{a}_{b}_{c}_{d}_{e}".format(a=seed,b=split,c=numberOfClasses,d=batchSize,e=epochIterations)+"_pre")
    # @interact
    # def show_images(file=os.listdir(out_dir)):
    #     display(DImage(out_dir+file, width=600, height=300))
    # pass


#To rework function
def Collate(batch):
    batch_targets = list()
    images = list()

    for b in batch:
        images.append(b[0])
        batch_targets.append({"boxes": b[1]["boxes"],
                              "labels": b[1]["labels"]})
    return images, batch_targets


def Test(modelName:str, testSet = None):
    device = t.device('cuda') 
    transform = tTrans.ToTensor()
    model = LoadModel("trainedFinals", modelName)

    TestModel(transform, model, device, testSet)
    pass

def TestModel(transform, model, device, testSet = None):
            ####################TESTING ################ 
    images_tensors_input = []
    images = []
    if testSet is None:
        for gesture in dm.ClassGesturesNames[:-1]:
        # image_path = dm.DatasetManager.__GetAbsoluteFilePathsFromDir(os.path.join(PATH_IMAGES, gesture))[0]
        #   image_path = glob(PATH_IMAGES + '\\{gesture}/*.jpg')[0] #get all files in folder, takes first one
            image_path = os.path.join(PATH_IMAGES, "{}".format(gesture) + "\\") + dm.DatasetManager.GetFileNamesFromDir(os.path.join(PATH_IMAGES, "{}".format(gesture)))[0]
            images.append(Image.open(image_path))
            images_tensors = images.copy()

        images_tensors_input = list(transform(image).to(device) for image in images_tensors)
    else:
        imagesTen = []
        for images, targets in testSet:
            imagesTen = list(transform(image)
            .to(device) for image in images)
            images = list(image for image in images)

        images_tensors_input = imagesTen

    with t.no_grad():
        model.eval()
        predictions = model(images_tensors_input)

        ####################################
        bboxes = []

    scores = []
    labels = []
    for singlePrediction in predictions:
        #.cpu() - memeory from gpu to cpu (tensor data)
        #.numpy() - tensor to numpy array
        #.astype(np.int) - convert to int
        #[:2] - take first 2 elements
        #[ids] - take only elements with ids (ids are bool array)
        #["{x}"] - take only {x}
        ids = singlePrediction['scores'] >= 0.2
        print(singlePrediction['scores'][ids].cpu().numpy())
        bboxes.append(singlePrediction['boxes'][ids][:2].cpu().numpy().astype(np.int))
        scores.append(singlePrediction['scores'][ids][:2].cpu().numpy())
        labels.append(singlePrediction['labels'][ids][:2].cpu().numpy())
    #    print(singlePrediction['labels'][ids][:2].cpu().numpy())
    #    print("Sens: " + str(singlePrediction['labels']))


    final_images = []


    for bbox, score, label, image in zip(bboxes, scores, labels, images):
        image = np.array(image)
        for i, box in enumerate(bbox):
            #_,width,_  = image.shape
        
            # if len(label) > 0:
            #     print(dm.ClassGesturesNames[label[i]])
            # else :
            #     print("No gesture")
            print(box[0]," AND ", box[1])
            image = cv2.rectangle(image, box[:2], box[2:], thickness=3, color=[255, 0, 255])
            cv2.putText(image, f'{dm.ClassGesturesNames[label[i]]}: {score[i]:0.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN,
                 2.5, (255, 0, 0), 3)
                            # width / 780, (0, 0, 255), 2)
        final_images.append(Image.fromarray(image))

    out_images = []
    for i, image in enumerate(final_images):
      #  out_name = f"out_images/{i}.png"
      #  out_images.append(out_name)
        print(i)
        image.show()
        wait = input("PRESS ENTER TO CONTINUE.")
    pass

# def eval(model, test_dataloader, epoch, device):
#     model.eval()
#     with t.no_grad():
#         mapmetric = MeanAveragePrecision()
        
#         for images, targets in test_dataloader:
#             images = list(image.to(device) for image in images)
#             output = model(images)
            
#             for pred in output:
#                 for key, value in pred.items():
#                     pred[key] = value.cpu()
                    
#             mapmetric.update(output, targets)

#     metrics = mapmetric.compute()
#     return metrics

#main()

#Test("model_42_0.8_19_20_2_pre")
#Test("model_78_0.9_19_20_15_pre")

testSet = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, False, tTrans.ToTensor(), 72, split)
dataLoaderTest = t.utils.data.DataLoader(testSet, batch_size=100
,collate_fn=Collate, shuffle=True)#, num_workers=4)   #wrapper
Test("model_78_0.9_19_20_15_pre", dataLoaderTest)


def PrintCudaStatus():

    print("Is CUDA on: ",t.cuda.is_available())
    print(t.cuda.current_device())
    dev = t.cuda.current_device()
    print(t.cuda.device_count())
    print(t.cuda.device(dev))
    print(t.cuda.get_device_name(dev))
    print('Memory Usage:')
    print('Allocated:', round(t.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(t.cuda.memory_reserved(0)/1024**3,1), 'GB')