from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os, re, copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, precision_recall_fscore_support
import wandb
import matplotlib.pyplot as plt
from spacecutter.models import OrdinalLogisticModel
import spacecutter
import pandas as pd
import pickle
import math
from torch import nn

# From: https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/wing_loss.py
# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))



def weights_init_xavier(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv3d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def weights_init_cnn(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)


    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

#https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def weighted_mse_loss(input, target, weights):
    error_per_sample = (input - target) ** 2
    numerator = 0
    
    for key in weights:
        numerator += weights[key]

    weights_list = [numerator / weights[int(i.data.tolist()[0])]  for i in target]
    weight_tensor = torch.FloatTensor(weights_list)
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss

#https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547
def log_weighted_mse_loss(input, target, weights):
    error_per_sample = (input - target) ** 2
    numerator = 0
    
    for key in weights:
        numerator += weights[key]

    inv_weights_list = [numerator / weights[int(i.data.tolist()[0])]  for i in target]

    weight_tensor = torch.FloatTensor(inv_weights_list)
    weight_tensor = torch.log(weight_tensor) # Take log of the tensor
    weight_tensor = weight_tensor.unsqueeze(1).cuda()

    loss = torch.mul(weight_tensor, error_per_sample).mean()
    return loss



def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy




def plot_confusion_matrix( y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[1] is not len(classes):
        # print("our CM is not the right size!!")

        all_labels = y_true + y_pred
        y_all_unique = list(set(all_labels))
        y_all_unique.sort()


        try:
            max_cm_size = len(classes)
            print('max_cm_size: ', max_cm_size)
            cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
            for i in range(len(y_all_unique)):
                for j in range(len(y_all_unique)):
                    i_global = y_all_unique[i]
                    j_global = y_all_unique[j]
                    
                    cm_new[i_global, j_global] = cm[i,j]
        except:
            print('CM failed++++++++++++++++++++++++++++++++++++++')
            print('cm_new', cm_new)
            print('cm', cm)
            print('classes', classes)
            print('y_all_unique', y_all_unique)
            print('y_true', list(set(y_true)))
            print('y_pred', list(set(y_pred)))
            print('max_cm_size: ', max_cm_size)
            max_cm_size = max([len(classes), y_all_unique[-1]])

            cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
            for i in range(len(y_all_unique)):
                for j in range(len(y_all_unique)):
                    i_global = y_all_unique[i]
                    j_global = y_all_unique[j]
                    try:
                        cm_new[i_global, j_global] = cm[i,j]
                    except:
                        print('CM failed second time++++++++++++++++++++++++++++++++++++++')
                        print('cm_new', cm_new)
                        print('cm', cm)
                        print('classes', classes)
                        print('y_all_unique', y_all_unique)
                        print('y_true', list(set(y_true)))
                        print('y_pred', list(set(y_pred)))
                        print('max_cm_size: ', max_cm_size)



        cm = cm_new

        classes = [i for i in range(max_cm_size)]

    # print(cm)
    # classes = classes[unique_labels(y_true, y_pred).astype(int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')
# 
    #print(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange( cm.shape[1]),
        yticks=np.arange( cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    
    ax.set_xlim(-0.5, cm.shape[1]-0.5)
    ax.set_ylim(cm.shape[0]-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    # print(ax.get_xticklabels())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def regressionPlot(labels, raw_preds, classes, fig_title):
    labels = np.asarray(labels)
    true_labels_jitter = labels + np.random.random_sample(labels.shape)/6

    fig = plt.figure()
    plt.plot(true_labels_jitter, raw_preds, 'bo', markersize=6)
    plt.title(fig_title)

    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)

    plt.xlabel("True Label")
    plt.ylabel("Regression Value")
    return fig