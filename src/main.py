import CustomNet as cn
from CustomNet import CustomTinyVGG
from CustomNet import BasicBlock
from CustomNet import ResNet
import testCustomNet as tcn
from ConfMatrix import ConfusionMatrixGenerator
from timeit import default_timer as timer 
import torchvision.transforms.functional as TF
import argparse

parser = argparse.ArgumentParser(prog="CNN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-tr", "--istrain", help="Set true if train", required=False)
parser.add_argument("-bch", "--batch", help="Provide an size of batch",required=False)
parser.add_argument("-ep", "--epoch", help="Provide an number of epochs", required=False)
parser.add_argument("-inf", "--info", help="Provide an info to saved model name",required=False)
parser.add_argument("-ld", "--load", help="Provide an name of the model",required=False)

args = parser.parse_args()
training = False
if args.istrain is not None:
    training = bool(args.istrain)


def displayBasicInfo():
    print(cn.torch.__version__)
    print(cn.device)
    iterateDir(cn.datasetDirPath)
    pass

def iterateDir(dirPath):
  for dirpath, dirnames, filenames in cn.os.walk(dirPath):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def main():
    trainData = cn.DatasetFromFolderCustom(targetDir=datasetDirPath, 
                                        transform=trainTransforms, seed=SEED, split=SPLIT, trainSet=True)
    testData = cn.DatasetFromFolderCustom(targetDir=datasetDirPath, 
                                        transform=trainTransforms, seed=SEED, split=SPLIT, trainSet=False)

    trainData.countClassesInstances()
    testData.countClassesInstances()

    trainDataloader = cn.DataLoader(trainData, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True) 
    testDataloader = cn.DataLoader(testData, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True) 

    cn.torch.manual_seed(SEED) 
    cn.torch.cuda.manual_seed(SEED)


    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=18).to(device)

    # model = cn.CustomTinyVGG(input_shape=3, #rgb
    #                 hidden_units=10, #10 4 now
    #                 output_shape=len(trainData.classes)).to(device)

    lossFun = cn.nn.CrossEntropyLoss()
    optimizer = cn.torch.optim.Adam(params=model.parameters(), lr=0.00008)


    startTime = timer()

    modelResults = cn.trainAndStat(model=model, 
                            trainDataloader=trainDataloader,
                            testDataloader=testDataloader,
                            optimizer=optimizer,
                            lossFun=lossFun, 
                            epochs=NUM_EPOCHS,
                            device=device)

    endTime = timer()
    print(f"Total training time: {endTime-startTime:.3f} seconds")
    specsModelInfo = "rasnet_best_ContrastPostProc"
    if args.info is not None:
        specsModelInfo = str(args.info)
    cn.SaveModel(
        model, 
        "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
        "model_seed{a}_epoch{b}_batch{c}_res{d}_spec-{e}".format(a=SEED,b=NUM_EPOCHS,c=BATCH_SIZE,d=IMAGE_SIZE,e=specsModelInfo)+"_cus")

    cn.plotLoss(modelResults, range(NUM_EPOCHS))


################################################################################
################################### VARIABLES ##################################
################################################################################

SEED = 9512
BATCH_SIZE = 64
if args.batch is not None:
    BATCH_SIZE = int(args.batch)
NUM_EPOCHS = 25
if args.epoch is not None:
    NUM_EPOCHS = int(args.epoch)
SPLIT = 0.9
IMAGE_SIZE = 224 #256 #128
device = "cuda" if cn.torch.cuda.is_available() else "cpu"
datasetDirPath = "D:\\Politechnika\\BIAI\\cropped"
#Train pipeline

class OwnColorTransform:

    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, x):
        return TF.adjust_contrast(x, self.contrast)

trainTransforms = cn.transforms.Compose([
    cn.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    # cn.transforms.ColorJitter( contrast=0.9),
    # cn.transforms.Grayscale(num_output_channels=1),
    # OwnColorTransform(contrast=0.5),
    cn.transforms.ToTensor()
])
#Test pipeline
testTransforms = cn.transforms.Compose([
    cn.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    # cn.transforms.ColorJitter(contrast=0.9),
    #  OwnColorTransform(contrast=0.5),
    # cn.transforms.Grayscale(num_output_channels=1),
    cn.transforms.ToTensor()
])

 

if training :
    main()
else:
    testData = cn.DatasetFromFolderCustom(targetDir=datasetDirPath, 
                                        transform=testTransforms, seed=SEED, split=SPLIT, trainSet=False)
    
    modelName = "model_seed9512_epoch25_batch64_spec-rasnet_best_noPostProc_cus"
    if args.load is not None:
        modelName = str(args.load)
    model = cn.LoadModel( 
        "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
        modelName)
    tcn.testModelVisually(testData, model, IMAGE_SIZE,OwnColorTransform,0.5, 5)
    testDataloader = cn.DataLoader(testData, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True) 
    ConfusionMatrixGenerator.generateConfMatrix(model, testDataloader, "outSimpleras")










