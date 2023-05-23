#OBSOLETE

import torch as t
import os
import pandas as pd
import random
import PIL.Image as Image

ClassGesturesNames = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace_inverted',
   'peace',
   'rock',
   'stop_inverted',
   'stop',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

class DatasetManagerMini(t.utils.data.Dataset):

    __dataFileExtension = ".jpg"

    def __init__(self, path_images, is_train, transform=None, _seed=42, _split=0.8):
        self.seed = _seed    
        self.split = _split
    
        self.is_train = is_train
        self.transform = transform
        self.path_images = path_images
        self.labels = {label: num for (label, num) in
                       zip(ClassGesturesNames, range(len(ClassGesturesNames)))}
        self.annotations = self.__CreateAnnotations()
        pass 

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def GetFileNamesFromDir(pth: str) -> list:
        if not os.path.exists(pth):
            print(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(DatasetManagerMini.__dataFileExtension)]
        return files

    @staticmethod
    def __GetAbsoluteFilePathsFromDir(pth: str) -> list:
        if not os.path.exists(pth):
            print(f"Dataset directory doesn't exist {pth}")
            return []
        files = [os.path.join(pth,f) for f in os.listdir(pth) if f.endswith(DatasetManagerMini.__dataFileExtension)]
        return files

    def __CreateAnnotations(self) -> pd.DataFrame:
        #Dataframe holds only gesture and path to image
        
        annotationsAll = []
        
        for gesture in ClassGesturesNames:
            if(gesture == "no_gesture"):
                continue
            allFileNames = self.GetFileNamesFromDir(os.path.join(self.path_images, gesture))
            for fileName in allFileNames:
                sample = {}
                sample['target'] = gesture
                sample['name'] = fileName
                annotationsAll.append(sample)

        random.Random(self.seed).shuffle(annotationsAll)

        annotationsAll = annotationsAll.copy() 

        if self.is_train:
            annotationsAll = annotationsAll[:int(len(annotationsAll) * self.split)] 
        else:
            annotationsAll = annotationsAll[int(len(annotationsAll) * self.split):]

        return annotationsAll

    def GetOneSample(self, index: int):
        row = self.annotations[index] #Gets row (a dictionary) from data frame

        imagePath = os.path.join(self.path_images, row["target"], row["name"])
        image = Image.open(imagePath).convert("RGB")
        image = image.resize((128,128))

        labels = t.LongTensor([self.labels[row["target"]]])

        target = {}
        target["labels"] = labels #label index
        return image, target

    def __getitem__(self, index: int):
        image, target = self.GetOneSample(index)
       # if self.transform:
         #   image = self.transform(image)
        return image, target


   
# trainSet = DatasetManagerMini("D:\\Politechnika\\BIAI\\cropped", True,None, 777, 0.8)
# print(trainSet.GetOneSample(50))
# print(trainSet.GetOneSample(7))
# print(trainSet.GetOneSample(115))
# print(trainSet.GetOneSample(1))
# print(trainSet.GetOneSample(2))
# print(trainSet.GetOneSample(3))
# a = trainSet.GetOneSample(3)

#One sample has: {'labels': tensor([0]), 'boxes': tensor([[253.0019, 248.6007, 400.8069, 398.1805]]), 'orig_size': tensor([562, 843])}
# So tensor for type as int, tensor for bounding box as float (four absolute coordinates (pixels of points (2?))), tensor for original image size
# And a tensor for image looking like this:
# device='cuda:0'), tensor([[[1.0000, 1.0000, 1.0000,  ..., 0.3922, 0.3529, 0.3490],
#          [1.0000, 1.0000, 1.0000,  ..., 0.4118, 0.4039, 0.4039],
#          [1.0000, 1.0000, 1.0000,  ..., 0.4706, 0.4627, 0.4549],
#          ...,