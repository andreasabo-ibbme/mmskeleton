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
import wandb
import shutil


def setup_eval_pipeline(pipeline):
    # eval_pipeline = copy.deepcopy(pipeline)
    eval_pipeline = []
    for item in pipeline:
        if item['type'] != "datasets.skeleton.scale_walk" and item['type'] != "datasets.skeleton.shear_walk":
            eval_pipeline.append(item)
    return eval_pipeline

# Processing a batch of data for label prediction
# process a batch of data
def batch_processor(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        flip_loss_mult = kwargs['flip_loss_mult']
    except:
        flip_loss_mult = 0

    try:
        balance_classes = kwargs['balance_classes']
    except:
        balance_classes = False

    try:
        class_weights_dict = kwargs['class_weights_dict']
    except:
        class_weights_dict = {}

    #torch.cuda.empty_cache()
    #print('have cuda: ', torch.cuda.is_available())
    #print('using device: ', torch.cuda.get_device_name())
    
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    model_2 = copy.deepcopy(model)
    have_flips = 0
    try:
        data, label, name, num_ts = datas
    except:
        data, data_flipped, label, name, num_ts = datas
        have_flips = 1

    data_all = data.cuda()
    label = label.cuda()

    # Remove the -1 labels
    y_true = label.data.reshape(-1, 1).float()
    condition = y_true >= 0.
    row_cond = condition.all(1)
    y_true = y_true[row_cond, :]
    data = data_all.data[row_cond, :]
    num_valid_samples = data.shape[0]

    if have_flips:
        data_all = data_all.data
        data_all_flipped = data_flipped.cuda()
        data_all_flipped = data_all_flipped.data 
        output_all_flipped = model_2(data_all_flipped)
        torch.clamp(output_all_flipped, min=-1, max=num_class)


    # Get predictions from the model
    output_all = model(data_all)
    # print(output_all)
    if torch.sum(output_all) == 0:        
        raise ValueError("=============================== got all zero output...")
    output = output_all[row_cond]
    loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 

    # Clip the predictions to avoid large loss that would otherwise be dealt with using the raw predictions
    torch.clamp(output_all, min=-1, max=num_class)

    if have_flips:
        loss_flip_tensor = mse_loss(output_all_flipped, output_all)
        if loss_flip_tensor.data > 10:
            pass
            # print('output_all_flipped', output_all_flipped, 'output_all', output_all)

    if not flip_loss_mult:
        loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 
        loss_flip_tensor = loss_flip_tensor.cuda()
    else:
        loss_flip_tensor = loss_flip_tensor * flip_loss_mult

    # if we don't have any valid labels for this batch...
    if num_valid_samples < 1:
        labels = []
        preds = []
        raw_preds = []
        # loss_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 
        # loss_tensor = loss_tensor.cuda()

        if type(num_ts) is not list:
            num_ts = [num_ts]

        log_vars = dict(loss_label=0, loss_flip = loss_flip_tensor.item(), loss_all=loss_flip_tensor.item())
        log_vars['mae_raw'] = 0
        log_vars['mae_rounded'] = 0
        output_labels = dict(true=labels, pred=preds, raw_preds=raw_preds, name=name, num_ts=num_ts)
        outputs = dict(loss=loss_flip_tensor, log_vars=log_vars, num_samples=0)

        return outputs, output_labels, loss_flip_tensor.item()
    


    y_true_orig_shape = y_true.reshape(1,-1).squeeze()
    losses = loss(output, y_true)


    if balance_classes:
        if type(loss) == type(mse_loss):
            losses = weighted_mse_loss(output, y_true, class_weights_dict)
        if type(loss) == type(mae_loss):
            losses = weighted_mae_loss(output, y_true, class_weights_dict)
    # Convert the output to classes and clip from 0 to number of classes
    y_pred_rounded = output.detach().cpu().numpy()
    output = y_pred_rounded
    output_list = output.squeeze().tolist()
    y_pred_rounded = y_pred_rounded.reshape(1, -1).squeeze()
    y_pred_rounded = np.round(y_pred_rounded, 0)
    y_pred_rounded = np.clip(y_pred_rounded, 0, num_class-1)
    preds = y_pred_rounded.squeeze().tolist()


    labels = y_true_orig_shape.data.tolist()
    num_ts = num_ts.data.tolist()
    # Case when we have a single output
    if type(labels) is not list:
        labels = [labels]
    if type(preds) is not list:
        preds = [preds]
    if type(output_list) is not list:
        output_list = [output_list]    
    if type(num_ts) is not list:
        num_ts = [num_ts]

    try:
        labels = [int(cl) for cl in labels]
        preds = [int(cl) for cl in preds]
    except Exception as e:
        print(labels)
        print(preds)
        print("got an error: ", e)


    overall_loss = losses + loss_flip_tensor
    log_vars = dict(loss_label=losses.item(), loss_flip = loss_flip_tensor.item(), loss_all=overall_loss.item())
    # print('l1', losses, 'l2', loss_flip_tensor)

    try:
        log_vars['mae_raw'] = mean_absolute_error(labels, output)
    except:
        print("labels: ", labels, "output", output)
        print('input', torch.sum(torch.isnan(data_all)))
        print('output_all', output_all, 'output_all_flipped', output_all_flipped)
        raise ValueError('stop')
    log_vars['mae_rounded'] = mean_absolute_error(labels, preds)
    output_labels = dict(true=labels, pred=preds, raw_preds=output_list, name=name, num_ts=num_ts)
    outputs = dict(loss=overall_loss, log_vars=log_vars, num_samples=len(labels))
    # print(type(labels), type(preds))
    # print('this is what we return: ', output_labels)
    return outputs, output_labels, overall_loss


def set_up_results_table(workflow, num_class):
    col_names = []
    for i, flow in enumerate(workflow):
        mode, _ = flow
        class_names_int = [int(i) for i in range(num_class)]

        # Calculate the mean metrics across classes
        average_types = ['macro', 'micro', 'weighted']
        average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
        prefix_name = mode + '/'
        for av in average_types:
            for m in average_metrics_to_log:
                col_names.append(prefix_name + m +'_average_' + av)


        # Calculate metrics per class
        for c in range(len(average_metrics_to_log)):
            for s in range(len(class_names_int)):
                col_names.append(prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c])

        col_names.append(prefix_name + 'mae_rounded')
        col_names.append(prefix_name + 'mae_raw')
        col_names.append(prefix_name + 'accuracy')


    df = pd.DataFrame(columns=col_names)
    return df

def final_stats_per_trial(final_results_local_path, wandb_group, wandb_project, num_class, workflow, num_epochs, results_table):
    try:
        # Compute summary statistics (accuracy and confusion matrices)
        print("getting final results from: ", final_results_local_path)
        log_vars = {'num_epochs': num_epochs}

        # final results +++++++++++++++++++++++++++++++++++++++++
        for i, flow in enumerate(workflow):
            mode, _ = flow

            class_names = [str(i) for i in range(num_class)]
            class_names_int = [int(i) for i in range(num_class)]
            results_file = os.path.join(final_results_local_path, mode+".csv")

            print("loading from: ", results_file)
            df = pd.read_csv(results_file)
            true_labels = df['true_score']
            preds = df['pred_round']
            preds_raw = df['pred_raw']

            # Calculate the mean metrics across classes
            average_types = ['macro', 'micro', 'weighted']
            average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
            prefix_name = mode + '/'
            for av in average_types:
                results_tuple = precision_recall_fscore_support(true_labels, preds, average=av)
                for m in range(len(average_metrics_to_log)):      
                    log_vars[prefix_name +  average_metrics_to_log[m] +'_average_' + av] = results_tuple[m]


            # Calculate metrics per class
            results_tuple = precision_recall_fscore_support(true_labels, preds, average=None, labels=class_names_int)

            for c in range(len(average_metrics_to_log)):
                cur_metrics = results_tuple[c]
                print(cur_metrics)
                for s in range(len(class_names_int)):
                    log_vars[prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c]] = cur_metrics[s]


            # Keep the original metrics for backwards compatibility
            log_vars[prefix_name + 'mae_rounded'] = mean_absolute_error(true_labels, preds)
            log_vars[prefix_name + 'mae_raw'] = mean_absolute_error(true_labels, preds_raw)
            log_vars[prefix_name + 'accuracy'] = accuracy_score(true_labels, preds)

        # print(log_vars)
        # print(results_table.columns)
        df = pd.DataFrame(log_vars, index=[0])
        results_table = results_table.append(df)

        return results_table
    except:
        logging.exception("in batch stats after trials: \n")


def final_stats_variance(results_df, wandb_group, wandb_project, total_epochs, num_class, workflow):
    wandb.init(name="ALL", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
    stdev = results_df.std().to_dict()
    means = results_df.mean().to_dict()
    all_stats = dict()
    for k,v in stdev.items():
        all_stats[k + "_stdev"] = stdev[k]
        all_stats[k + "_mean"] = means[k]


    wandb.log(all_stats)


def final_stats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow):
    try:
        max_label = num_class
        # Compute summary statistics (accuracy and confusion matrices)
        final_results_dir = os.path.join(work_dir, 'all_final_eval', wandb_group)
        final_results_dir2 = os.path.join(work_dir, 'all_test', wandb_group)

        wandb.init(name="ALL", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
        print("getting final results from: ", final_results_dir)
        for e in range(0, total_epochs):
            log_vars = {}
            results_file = os.path.join(final_results_dir, "test_" + str(e + 1) + ".csv")
            try:
                df = pd.read_csv(results_file)
            except:
                break
            true_labels = df['true_score']
            preds = df['pred_round']
            preds_raw = df['pred_raw']

            log_vars['eval/mae_rounded'] = mean_absolute_error(true_labels, preds)
            log_vars['eval/mae_raw'] = mean_absolute_error(true_labels, preds_raw)
            log_vars['eval/accuracy'] = accuracy_score(true_labels, preds)
            wandb.log(log_vars, step=e+1)

            if e % 5 == 0:
                class_names = [str(i) for i in range(num_class)]

                fig = plot_confusion_matrix( true_labels,preds, class_names, max_label)
                wandb.log({"confusion_matrix/eval_"+ str(e)+".png": fig}, step=e+1)
                fig_title = "Regression for ALL unseen participants"
                reg_fig = regressionPlot(true_labels,preds_raw, class_names, fig_title)
                try:
                    wandb.log({"regression/eval_"+ str(e)+".png": [wandb.Image(reg_fig)]}, step=e+1)
                except:
                    pass

        # final results +++++++++++++++++++++++++++++++++++++++++

        for i, flow in enumerate(workflow):
            mode, _ = flow

            class_names = [str(i) for i in range(num_class)]
            class_names_int = [int(i) for i in range(num_class)]

            log_vars = {}
            results_file = os.path.join(final_results_dir, mode+".csv")
            print("loading from: ", results_file)
            df = pd.read_csv(results_file)
            true_labels = df['true_score']
            preds = df['pred_round']
            preds_raw = df['pred_raw']

            # Calculate the mean metrics across classes
            average_types = ['macro', 'micro', 'weighted']
            average_metrics_to_log = ['precision', 'recall', 'f1score', 'support']
            average_dict = {}
            prefix_name = 'final/'+ mode + '/'
            for av in average_types:
                results_tuple = precision_recall_fscore_support(true_labels, preds, average=av)
                for m in range(len(average_metrics_to_log)):      
                    average_dict[prefix_name + '_'+ average_metrics_to_log[m] +'_average_' + av] = results_tuple[m]

            wandb.log(average_dict)

            # Calculate metrics per class
            results_tuple = precision_recall_fscore_support(true_labels, preds, average=None, labels=class_names_int)

            per_class_stats = {}
            for c in range(len(average_metrics_to_log)):
                cur_metrics = results_tuple[c]
                print(cur_metrics)
                for s in range(len(class_names_int)):
                    per_class_stats[prefix_name + str(class_names_int[s]) + '_'+ average_metrics_to_log[c]] = cur_metrics[s]

            wandb.log(per_class_stats)


            # Keep the original metrics for backwards compatibility
            log_vars['early_stop_eval/'+mode+ '/mae_rounded'] = mean_absolute_error(true_labels, preds)
            log_vars['early_stop_eval/'+mode+ '/mae_raw'] = mean_absolute_error(true_labels, preds_raw)
            log_vars['early_stop_eval/'+mode+ '/accuracy'] = accuracy_score(true_labels, preds)
            wandb.log(log_vars)

            
            fig = plot_confusion_matrix( true_labels,preds, class_names, max_label)
            
            wandb.log({"early_stop_eval/" + mode + "_final_confusion_matrix.png": fig})
            fig_title = "Regression for ALL unseen participants"
            reg_fig = regressionPlot(true_labels, preds_raw, class_names, fig_title)
            try:
                wandb.log({"early_stop_eval/" + mode + "_final_regression_plot.png": [wandb.Image(reg_fig)]})
            except:
                try:
                    wandb.log({"early_stop_eval/" + mode + "_final_regression_plot.png": reg_fig})
                except:
                    print("failed to log regression plot")

            # Log the final dataframe to wandb for future analysis
            header = ['amb', 'walk_name', 'num_ts', 'true_score', 'pred_round', 'pred_raw']
            try:
                wandb.log({"final_results_csv/"+mode: wandb.Table(data=df.values.tolist(), columns=header)})
            except: 
                logging.exception("Could not save final table =================================================\n")
        
        # Remove the files generated so we don't take up this space
        shutil.rmtree(final_results_dir)
        shutil.rmtree(final_results_dir2)


    except:
        print('something when wrong in the summary stats')
        logging.exception("Error message =================================================")    



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

def weighted_mae_loss(input, target, weights):
    error_per_sample = abs(input - target)
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


def plot_confusion_matrix( y_true, y_pred, classes, max_label, normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # How many classes are there? Go from 0 -> max in preds or labels
    # max_label = max(max(y_true), max(y_pred))
    class_names_int = [int(i) for i in range(max_label)]
    classes = [str(i) for i in range(max_label)]
    cm = confusion_matrix(y_true, y_pred, labels=class_names_int)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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



# def plot_confusion_matrix( y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'



#     cm = confusion_matrix(y_true, y_pred)
#     if cm.shape[1] is not len(classes):
#         # print("our CM is not the right size!!")

#         all_labels = y_true + y_pred
#         y_all_unique = list(set(all_labels))
#         y_all_unique.sort()


#         try:
#             max_cm_size = len(classes)
#             print('max_cm_size: ', max_cm_size)
#             cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
#             for i in range(len(y_all_unique)):
#                 for j in range(len(y_all_unique)):
#                     i_global = y_all_unique[i]
#                     j_global = y_all_unique[j]
                    
#                     cm_new[i_global, j_global] = cm[i,j]
#         except:
#             print('CM failed++++++++++++++++++++++++++++++++++++++')
#             print('cm_new', cm_new)
#             print('cm', cm)
#             print('classes', classes)
#             print('y_all_unique', y_all_unique)
#             print('y_true', list(set(y_true)))
#             print('y_pred', list(set(y_pred)))
#             print('max_cm_size: ', max_cm_size)
#             max_cm_size = max([len(classes), y_all_unique[-1]])

#             cm_new = np.zeros((max_cm_size, max_cm_size), dtype=np.int64)
#             for i in range(len(y_all_unique)):
#                 for j in range(len(y_all_unique)):
#                     i_global = y_all_unique[i]
#                     j_global = y_all_unique[j]
#                     try:
#                         cm_new[i_global, j_global] = cm[i,j]
#                     except:
#                         print('CM failed second time++++++++++++++++++++++++++++++++++++++')
#                         print('cm_new', cm_new)
#                         print('cm', cm)
#                         print('classes', classes)
#                         print('y_all_unique', y_all_unique)
#                         print('y_true', list(set(y_true)))
#                         print('y_pred', list(set(y_pred)))
#                         print('max_cm_size: ', max_cm_size)



#         cm = cm_new

#         classes = [i for i in range(max_cm_size)]

#     # print(cm)
#     # classes = classes[unique_labels(y_true, y_pred).astype(int)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#         # print("Normalized confusion matrix")
#     # else:
#         # print('Confusion matrix, without normalization')
# # 
#     #print(cm)

#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange( cm.shape[1]),
#         yticks=np.arange( cm.shape[0]),
#         # ... and label them with the respective list entries
#         xticklabels=classes, yticklabels=classes,
#         title=title,
#         ylabel='True label',
#         xlabel='Predicted label')
    
#     ax.set_xlim(-0.5, cm.shape[1]-0.5)
#     ax.set_ylim(cm.shape[0]-0.5, -0.5)

#     # Rotate the tick labels and set their alignment.
#     # print(ax.get_xticklabels())
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = '.3f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return fig

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