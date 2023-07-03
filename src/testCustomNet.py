import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def testModelVisually(testDataset, model, imageSize, contrFun = None, contrast:float=0, numOfImgs: int = 5, ):

    testImgPaths = testDataset.getPaths(numOfImgs)
    imagePredictions = []#list of tuples, guessed label and probability

    for i in range(numOfImgs):

        image4Model = torchvision.io.read_image(str(testImgPaths[i]))
        image4Model = image4Model / 255.0


        image4ModelTransformPipeline = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
        ])
        if contrFun is not None:
            image4ModelTransformPipeline = transforms.Compose([
                transforms.Resize((imageSize,imageSize)),
                contrFun(contrast=contrast),
            ])


        image4ModelTransformed = image4ModelTransformPipeline(image4Model)

        model.eval()
        with torch.inference_mode():
            
            imagePrediction = model(image4ModelTransformed.unsqueeze(dim=0).to(torch.device('cuda')))
            imagePrediction_probabilities = torch.softmax(imagePrediction, dim=1)
            imagePredictionClassPredVal = torch.argmax(imagePrediction_probabilities, dim=1)
            imagePredictionClassPred = testDataset.classes[imagePredictionClassPredVal.cpu()] # put pred label to CPU, otherwise will error
            imagePredictions.append((imagePredictionClassPred, imagePrediction_probabilities.max().cpu()))

    plt.figure(figsize=(32, 8))

    for j, imgPath in enumerate(testImgPaths):

        image = torchvision.io.read_image(str(imgPath))
        image = image / 255.0

        plt.subplot(1, numOfImgs, j+1)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        title = f"Pred: {imagePredictions[j][0]} | Prob: {imagePredictions[j][1]:.3f}"
        plt.title(title)
        plt.axis(False);
    plt.tight_layout(pad=12)
    plt.show()    
