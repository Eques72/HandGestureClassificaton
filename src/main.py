import numpy as np
import random
import os

import DatasetManager as dm

import torch as t
import torchvision.transforms as tTrans
import torchvision.models as models

import PIL.Image as Image
import cv2
 
PATH_IMAGES = "D:\\Politechnika\\BIAI\\subsample"
PATH_ANNOTATIONS = "D:\\Politechnika\\BIAI\\ann_subsample\\ann_subsample"

#use haGRID database
def main():
    #TEST
  #  D = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, True, None, 42, 0.8)
  #  D.GetSample(666)

    random_seed = 42
    split = 0.8
    num_classes = len(dm.ClassGesturesNames)
    batch_size = 16
    num_epoch = 15
    t.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    device = t.device('cuda')
    transform = tTrans.ToTensor()

    trainSet = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, True, transform, random_seed, split)
    testSet = dm.DatasetManager(PATH_ANNOTATIONS, PATH_IMAGES, True, transform, random_seed, split)
    ####################################
    dataLoaderTrain = t.utils.data.DataLoader(trainSet, batch_size=batch_size,collate_fn=Collate, shuffle=True, num_workers=4)
    dataLoaderTest = t.utils.data.DataLoader(testSet, batch_size=batch_size,collate_fn=Collate, shuffle=True, num_workers=4)
    ####################################
    lr = 0.005
    momentum = 0.9
    weight_decay = 5e-4

    model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=len(num_classes) + 1, pretrained_backbone=True)
    model.to(device)

    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ####################################
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(trainSet) - 1)

    lr_scheduler_warmup = t.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    ####################################
#    !mkdir checkpoints
    for epoch in range(num_epoch):
        model.train()
        total = 0
        sum_loss = 0
        for images, targets in dataLoaderTrain:
            batch = len(images)
            images = list(image.to(device) for image in images)
            for target in targets:
                for key, value in target.items():
                    target[key] = value.to(device)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr_scheduler_warmup.step()

            total = total + batch
            sum_loss = sum_loss + loss

      #  metrics = eval(model, dataLoaderTest, epoch)
       # print(f"epoch : {epoch}  |||  loss : {sum_loss / total} ||| MAP : {metrics['map']}")
        t.save(model.state_dict(),f"checkpoints/{epoch}.pth")
        

  
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

def TestModel(transform, model, device):
            ####################TESTING AKA OPTIONAL################ 
    images = []
    for gesture in dm.ClassGesturesNames[:-1]:
       # image_path = dm.DatasetManager.__GetAbsoluteFilePathsFromDir(os.path.join(PATH_IMAGES, gesture))[0]
     #   image_path = glob(PATH_IMAGES + '\\{gesture}/*.jpg')[0] #get all files in folder, takes first one
        image_path = os.path.join(PATH_IMAGES, "{}".format(gesture) + "\\") + dm.DatasetManager.GetFileNamesFromDir(os.path.join(PATH_IMAGES, "{}".format(gesture)))[0]
        images.append(Image.open(image_path))
        images_tensors = images.copy()

    images_tensors_input = list(transform(image).to(device) for image in images_tensors)

    with t.no_grad():
        model.eval()
        out = model(images_tensors_input)
        ####################################
        bboxes = []

    scores = []
    labels = []
    for pred in out:
        ids = pred['scores'] >= 0.2
        bboxes.append(pred['boxes'][ids][:2].cpu().numpy().astype(np.int))
        scores.append(pred['scores'][ids][:2].cpu().numpy())
        labels.append(pred['labels'][ids][:2].cpu().numpy())
        short_class_names = []

    for name in dm.ClassGesturesNames:
        if name == 'stop_inverted':
            short_class_names.append('stop inv.')
        elif name == 'peace_inverted':
            short_class_names.append('peace inv.')
        elif name == 'two_up':
            short_class_names.append('two up')
        elif name == 'two_up_inverted':
            short_class_names.append('two up inv.')
        elif name == 'no_gesture':
            short_class_names.append('no gesture')
        else:
            short_class_names.append(name)
    final_images = []
    for bbox, score, label, image in zip(bboxes, scores, labels, images):
        image = np.array(image)
        for i, box in enumerate(bbox):
            _,width,_  = image.shape
            image = cv2.rectangle(image, box[:2], box[2:], thickness=3, color=[255, 0, 255])
            cv2.putText(image, f'{short_class_names[label[i]]}: {score[i]:0.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            width / 780, (0, 0, 255), 2)
        final_images.append(Image.fromarray(image))
 #   !mkdir out_images
    out_images = []
    for i, image in enumerate(final_images):
        out_name = f"out_images/{i}.png"
        out_images.append(out_name)
        image.save(out_name)
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
