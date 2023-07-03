from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch



class ConfusionMatrixGenerator:
    
    @staticmethod
    def generateConfMatrix(net, testSet, filename:str):
        yPred = []
        yTrue = []
        net.eval()
        for batch, (inputs, labels) in enumerate(testSet):
                inputs, labels = inputs.cuda(), labels.cuda()
                output = net(inputs)

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                yPred.extend(output)
                
                labels = labels.data.cpu().numpy()
                yTrue.extend(labels)

        classes = ('call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted')

        matrix = confusion_matrix(yTrue, yPred)
        dfMatrix = pd.DataFrame(matrix / np.sum(matrix, axis=1)[:, None], index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(dfMatrix, annot=True)
        plt.savefig(filename+'_output.png')