import torch as t
import os
import pandas as pd
import json
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

class DatasetManager(t.utils.data.Dataset):

    __dataFileExtension = ".jpg"

    def __init__(self, path_annotation, path_images, is_train, transform=None, _seed=42, _split=0.8):
        self.seed = _seed    
        self.split = _split
    
        self.is_train = is_train
        self.transform = transform
        self.path_annotation = path_annotation
        self.path_images = path_images
        self.transform = transform
        self.labels = {label: num for (label, num) in
                       zip(ClassGesturesNames, range(len(ClassGesturesNames)))}
        self.annotations = self.__ReadAnnotations(self.path_annotation)
        pass 

    def __len__(self):
        return self.annotations.shape[0]

    @staticmethod
    def __getFileNamesFromDir(pth: str) -> list:
        if not os.path.exists(pth):
            print(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(DatasetManager.__dataFileExtension)]
      #  print(files)
        return files

    def __ReadAnnotations(self, path:str) -> pd.DataFrame:
        annotationsAll = None
        existsImages = []
        
        for gesture in ClassGesturesNames:
            pathToJson = os.path.join(path, f"{gesture}.json")
            if os.path.exists(pathToJson):
                json_annotation = json.load(open( os.path.join(path, f"{gesture}.json") ))
                #List of dictionaries, each dictionary is a data of one image from current gesture class
                #Dictionary keys are: bboxes labels landmarks leading_conf leading_hand user_id name
                json_annotation = [dict(annotation, **{"name": f"{fileName}{DatasetManager.__dataFileExtension}"}) for fileName, annotation in zip(json_annotation, json_annotation.values())]
                
                # for key, val in list(json_annotation[0].items()):
                #     print(key, val)
                annotation = pd.DataFrame(json_annotation)

                annotation["target"] = gesture
                annotationsAll = pd.concat([annotationsAll, annotation], ignore_index=True)

                existsImages.extend(self.__getFileNamesFromDir(os.path.join(self.path_images, gesture)))
            else:
                if gesture != 'no_gesture':
                    print(f"Database for {gesture} not found")

        annotationsAll["exists"] = annotationsAll["name"].isin(existsImages)

        annotationsAll = annotationsAll[annotationsAll["exists"]] #Removes images that are not in the database

        users = annotationsAll["user_id"].unique()
        users = sorted(users)
        random.Random(self.seed).shuffle(users)
        train_users = users[:int(len(users) * self.split)] #Splits data into train and validation
        val_users = users[int(len(users) * self.split):]

        annotationsAll = annotationsAll.copy() 

        if self.is_train:
            annotationsAll = annotationsAll[annotationsAll["user_id"].isin(train_users)]
        else:
            annotationsAll = annotationsAll[annotationsAll["user_id"].isin(val_users)]

        return annotationsAll

    def getSample(self, index: int):
        row = self.annotations.iloc[[index]].to_dict('records')[0] #Gets row (a dictionary) from data frame

        imagePath = os.path.join(self.path_images, row["target"], row["name"])
        image = Image.open(imagePath).convert("RGB")

        image.show()
        labels = t.LongTensor([self.labels[label] for label in row["labels"]])

        target = {}
        width, height = image.size
        bBoxes = [] #Absolute coordinates

        for bbox in row["bboxes"]:
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
            bBoxes.append(bbox_abs)

        target["labels"] = labels
        target["boxes"] = t.as_tensor(bBoxes, dtype=t.float32)
        target["orig_size"] = t.as_tensor([int(height), int(width)])

        return image, target

    def __getItem__(self, index: int):
        image, target = self.getSample(index)
        if self.transform:
            image = self.transform(image)
        return image, target