import CustomNet as cn
import testCustomNet as tcn

training = False;




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

    trainDataloader = cn.DataLoader(trainData, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True) 
    testDataloader = cn.DataLoader(testData, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True) 

    cn.torch.manual_seed(SEED) 
    cn.torch.cuda.manual_seed(SEED)

    model = cn.TinyVGG(input_shape=3, #rgb
                    hidden_units=10, #10 4 now
                    output_shape=len(trainData.classes)).to(device)

    lossFun = cn.nn.CrossEntropyLoss()
    optimizer = cn.torch.optim.SGD(params=model.parameters(), lr=0.0001)
#    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.0001, weight_decay=0.005)

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

    specsModelInfo = "Anyrelativename1"
    cn.SaveModel(
        model, 
        "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
        "model_seed{a}_epoch{b}_batch{c}_spec-{d}".format(a=SEED,b=NUM_EPOCHS,c=BATCH_SIZE,d=specsModelInfo)+"_cus")

    cn.plotLoss(modelResults, range(NUM_EPOCHS))


################################################################################
################################### VARIABLES ##################################
################################################################################

SEED = 8502
BATCH_SIZE = 32
NUM_EPOCHS = 30
SPLIT = 0.8
device = "cuda" if cn.torch.cuda.is_available() else "cpu"
datasetDirPath = "D:\\Politechnika\\BIAI\\cropped"
#Train pipeline
trainTransforms = cn.transforms.Compose([
    cn.transforms.Resize((128,128)),
    cn.transforms.ToTensor()
])
#Test pipeline
testTransforms = cn.transforms.Compose([
    cn.transforms.Resize((128,128)),
    cn.transforms.ToTensor()
])



if training :
    main()
else:
    testData = cn.DatasetFromFolderCustom(targetDir=datasetDirPath, 
                                        transform=trainTransforms, seed=SEED, split=SPLIT, trainSet=False)
    model = cn.LoadModel( 
        "D:/Dane/Moje projekty/Python/HandGestureClassification/HandGestureClassificaton/trainedFinals", 
        "model_seed8502_epoch10_batch32_spec-largeKernel_lowLR_cus")
    tcn.testModelVisually(testData, model, 5)