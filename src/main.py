import CustomNet as cn
from CustomNet import CustomTinyVGG
import testCustomNet as tcn

training = True;




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

    model = cn.CustomTinyVGG(input_shape=3, #rgb
                    hidden_units=10, #10 4 now
                    output_shape=len(trainData.classes)).to(device)

    lossFun = cn.nn.CrossEntropyLoss()
#    optimizer = cn.torch.optim.SGD(params=model.parameters(), lr=0.001)
    optimizer = cn.torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.01)

    from timeit import default_timer as timer 
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
    specsModelInfo = "100epoch_dropout_wegDec_cusKernel_lowLR_Adam_reworked"
    # cn.SaveModel(
    #     model, 
    #     "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
    #     "model_seed{a}_epoch{b}_batch{c}_spec-{d}".format(a=SEED,b=NUM_EPOCHS,c=BATCH_SIZE,d=specsModelInfo)+"_cus")

    cn.plotLoss(modelResults, range(NUM_EPOCHS))


################################################################################
################################### VARIABLES ##################################
################################################################################

SEED = 9512
BATCH_SIZE = 64
NUM_EPOCHS = 15
SPLIT = 0.9
IMAGE_SIZE = 128 #256 #128
device = "cuda" if cn.torch.cuda.is_available() else "cpu"
datasetDirPath = "D:\\Politechnika\\BIAI\\cropped"
#Train pipeline
trainTransforms = cn.transforms.Compose([
    cn.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    cn.transforms.ToTensor()
])
#Test pipeline
testTransforms = cn.transforms.Compose([
    cn.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    cn.transforms.ToTensor()
])

 

if training :
    main()
else:
    testData = cn.DatasetFromFolderCustom(targetDir=datasetDirPath, 
                                        transform=testTransforms, seed=SEED, split=SPLIT, trainSet=False)
    model = cn.LoadModel( 
        "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
        "model_seed9512_epoch15_batch32_spec-cusKernel_lowLR_Adam_reworked_cus")
    tcn.testModelVisually(testData, model, 5)










