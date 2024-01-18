import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def conf_matrix(output, target, device, num_classes=32):
    metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    pred = torch.argmax(output, dim=1)
    metric = metric(pred, target)
    return metric

def save_conf_matrix(conf_matrix, classes, saving_path="."):
    output_file='/conf_matrix.png'
    column_sums = conf_matrix.sum(dim=0)
    normalized_tensor = conf_matrix / column_sums
    #print(normalized_tensor)
    numpy = normalized_tensor.to("cpu").numpy()
    df_cm = pd.DataFrame(numpy / np.sum(numpy, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (20,20))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(saving_path+output_file)
