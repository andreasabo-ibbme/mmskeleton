from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os, re, copy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import wandb
import matplotlib.pyplot as plt
from spacecutter.models import OrdinalLogisticModel
import spacecutter
import pandas as pd
import pickle
from mmskeleton.processor.utils_recognition import *
from mmskeleton.processor.supcon_loss import *



os.environ['WANDB_MODE'] = 'dryrun'

# Global variables
num_class = 3
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False


def train(
        work_dir,
        model_cfg,
        loss_cfg,
        dataset_cfg,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        test_ids=None,
        cv=5,
        exclude_cv=False,
        notes=None,
        flip_loss=0,
        weight_classes=False,
        group_notes='',
        launch_from_windows=False,
        wandb_project="mmskel",
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience_1=5,
        es_start_up_1=5,
        es_patience_2=10,
        es_start_up_2=50,
        head='stgcn',
        freeze_encoder=True,
):
    # Set up for logging 
    outcome_label = dataset_cfg[0]['data_source']['outcome_label']

    global flip_loss_mult
    flip_loss_mult = flip_loss

    global balance_classes
    balance_classes = weight_classes

    global num_class
    num_class = model_cfg['num_class']
    wandb_group = wandb.util.generate_id() + "_" + outcome_label + "_" + group_notes
    print("ANDREA - TRI-recognition: ", wandb_group)

    id_mapping = {27:25, 33:31, 34:32, 37:35, 39:37,
                  46:44, 47:45, 48:46, 50:48, 52:50, 
                  55:53, 57:55, 59:57, 66:63}


    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())

    # All data dir (use this for finetuning with the flip loss)
    data_dir_all_data = dataset_cfg[0]['data_source']['data_dir']
    all_files = [os.path.join(data_dir_all_data, f) for f in os.listdir(data_dir_all_data)]
    print("all files: ", len(all_files))

    all_file_names_only = os.listdir(data_dir_all_data)

    # PD lablled dir (only use this data for supervised contrastive)
    data_dir_pd_data = dataset_cfg[1]['data_source']['data_dir']
    pd_all_files = [os.path.join(data_dir_pd_data, f) for f in os.listdir(data_dir_pd_data)]
    pd_all_file_names_only = os.listdir(data_dir_pd_data)
    print("pd_all_files: ", len(pd_all_files))


    original_wandb_group = wandb_group
    workflow_orig = copy.deepcopy(workflow)
    for test_id in test_ids:
        plt.close('all')
        ambid = id_mapping[test_id]

        # These are all of the walks (both labelled and not) of the test participant and cannot be included in training data at any point (for LOSOCV)
        test_subj_walks_name_only_all = [i for i in all_file_names_only if re.search('ID_'+str(test_id), i) ]
        test_subj_walks_name_only_pd_only = [i for i in pd_all_file_names_only if re.search('ID_'+str(test_id), i) ]
        
        print(f"test_subj_walks_name_only_all: {len(test_subj_walks_name_only_all)}")
        print(f"test_subj_walks_name_only_pd_only: {len(test_subj_walks_name_only_pd_only)}")

        # These are the walks that can potentially be included in the train/val sets at some stage
        non_test_subj_walks_name_only_all = list(set(all_file_names_only).difference(set(test_subj_walks_name_only_all)))
        non_test_subj_walks_name_only_pd_only = list(set(pd_all_file_names_only).difference(set(test_subj_walks_name_only_pd_only)))
        non_test_subj_walks_name_only_non_pd_only = list(set(non_test_subj_walks_name_only_all).difference(set(non_test_subj_walks_name_only_pd_only)))


        print(f"non_test_subj_walks_name_only_all: {len(non_test_subj_walks_name_only_all)}")
        print(f"non_test_subj_walks_name_only_pd_only: {len(non_test_subj_walks_name_only_pd_only)}")



        # These are all of the labelled walks from the current participant that we want to evaluate our eventual model on
        test_walks_pd_labelled = [os.path.join(data_dir_pd_data, f) for f in test_subj_walks_name_only_pd_only]
        non_test_walks_pd_labelled = [os.path.join(data_dir_pd_data, f) for f in non_test_subj_walks_name_only_pd_only]
        non_test_walks_all = [os.path.join(data_dir_all_data, f) for f in non_test_subj_walks_name_only_all]
        non_test_walks_all_no_pd_label = [os.path.join(data_dir_all_data, f) for f in non_test_subj_walks_name_only_non_pd_only]



        # A list of whether a walk from the non_test_walks_all list has a pd label as well
        non_test_is_lablled = [1 if i in non_test_walks_pd_labelled else 0 for i in non_test_walks_all]



        datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
        work_dir_amb = work_dir + "/" + str(ambid)
        for ds in datasets:
            ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

        # Don't bother training if we have no test data
        if len(test_walks_pd_labelled) == 0:
            continue
        
        # data exploration
        print(f"test_walks_pd_labelled: {len(test_walks_pd_labelled)}")
        print(f"non_test_walks_pd_labelled: {len(non_test_walks_pd_labelled)}")


        # Split the non_test walks into train/val
        kf = KFold(n_splits=cv, shuffle=True, random_state=1)
        kf.get_n_splits(non_test_walks_all_no_pd_label)

        # PD labelled walks only
        kf_pd = KFold(n_splits=cv, shuffle=True, random_state=1)
        kf_pd.get_n_splits(non_test_walks_pd_labelled)


        num_reps = 1
        for train_ids, val_ids in kf.split(non_test_walks_all_no_pd_label):
            if num_reps > 1:
                break
            
            plt.close('all')
            ambid = id_mapping[test_id]
            num_reps += 1

            # Divide all of the data into:
            # Stage 1 train/val
            # Stage 2 train/val
            print(f"we have {len(non_test_walks_all_no_pd_label)} non_test_walks_all")
            print(f"we have {len(train_ids)} train_ids and {len(val_ids)} val_ids. ")

            # These are from the walks without pd labels (for pretraining)
            stage_1_train = [non_test_walks_all_no_pd_label[i] for i in train_ids]
            stage_1_val = [non_test_walks_all_no_pd_label[i] for i in val_ids]



            # These are from the pd labelled set
            num_reps_pd = 1
            for train_ids_pd, val_ids_pd in kf.split(non_test_walks_pd_labelled):
                if num_reps_pd > 1:
                    break
                num_reps_pd += 1

                stage_2_train = [non_test_walks_all_no_pd_label[i] for i in train_ids_pd]
                stage_2_val = [non_test_walks_all_no_pd_label[i] for i in val_ids_pd]


            print(f"we have {len(stage_1_train)} stage_1_train and {len(stage_1_val)} stage_1_val. ")
            print(f"we have {len(stage_2_train)} stage_2_train and {len(stage_2_val)} stage_2_val. ")
           

            # ================================ STAGE 1 ====================================
            # Stage 1 training
            datasets[0]['data_source']['data_dir'] = stage_1_train[:50]
            datasets[1]['data_source']['data_dir'] = stage_1_val[:50]
            datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled
            datasets_stage_1 = copy.deepcopy(datasets)
            datasets_stage_1.pop(2)

            workflow_stage_1 = copy.deepcopy(workflow)
            workflow_stage_1.pop(2)

            work_dir_amb = work_dir + "/" + str(ambid)
            for ds in datasets:
                ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

            things_to_log = {'num_ts_predicting': model_cfg['num_ts_predicting'], 'es_start_up_1': es_start_up_1, 'es_patience_1': es_patience_1, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg, 'optimizer_cfg': optimizer_cfg, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

            # print("train walks: ", stage_1_train)

            print('stage_1_train: ', len(stage_1_train))
            print('stage_1_val: ', len(stage_1_val))
            print('test_walks_pd_labelled: ', len(test_walks_pd_labelled))

            pretrained_model = pretrain_model(
                work_dir_amb,
                model_cfg,
                loss_cfg,
                datasets_stage_1,
                optimizer_cfg,
                batch_size,
                total_epochs,
                training_hooks,
                workflow,
                gpus,
                log_level,
                workers,
                resume_from,
                load_from, 
                things_to_log,
                early_stopping,
                force_run_all_epochs,
                es_patience_1,
                es_start_up_1
                )


            # ================================ STAGE 2 ====================================

            # Stage 2 training
            datasets[0]['data_source']['data_dir'] = stage_2_train
            datasets[1]['data_source']['data_dir'] = stage_2_val
            datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled


            # Reset the head
            pretrained_model.module.set_stage_2()
            pretrained_model.module.head.apply(weights_init_xavier)

            things_to_log = {'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg, 'optimizer_cfg': optimizer_cfg, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

            # print("final model for fine_tuning is: ", pretrained_model)

            finetune_model(work_dir,
                        pretrained_model,
                        loss_cfg,
                        datasets,
                        optimizer_cfg,
                        batch_size,
                        total_epochs,
                        training_hooks,
                        workflow,
                        gpus,
                        log_level,
                        workers,
                        resume_from,
                        load_from,
                        things_to_log,
                        early_stopping,
                        force_run_all_epochs,
                        es_patience_2,
                        es_start_up_2, 
                        freeze_encoder)



    # Compute summary statistics (accuracy and confusion matrices)
    final_results_dir = os.path.join(work_dir, 'all_test', wandb_group)
    wandb.init(name="ALL", project=wandb_project, group=wandb_group, tags=['summary'], reinit=True)
    print(final_results_dir)
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

            fig = plot_confusion_matrix( true_labels,preds, class_names)
            wandb.log({"confusion_matrix/eval_"+ str(e)+".png": fig}, step=e+1)
            fig_title = "Regression for ALL unseen participants"
            reg_fig = regressionPlot(true_labels,preds_raw, class_names, fig_title)
            try:
                wandb.log({"regression/eval_"+ str(e)+".png": [wandb.Image(reg_fig)]}, step=e+1)
            except:
                pass

    # final results
    final_results_dir = os.path.join(work_dir, 'all_eval', wandb_group)

    for i, flow in enumerate(workflow):
        mode, _ = flow


        log_vars = {}
        results_file = os.path.join(final_results_dir, mode+".csv")
        print("loading from: ", results_file)
        df = pd.read_csv(results_file)
        true_labels = df['true_score']
        preds = df['pred_round']
        preds_raw = df['pred_raw']

        log_vars['early_stop_eval/'+mode+ '/mae_rounded'] = mean_absolute_error(true_labels, preds)
        log_vars['early_stop_eval/'+mode+ '/mae_raw'] = mean_absolute_error(true_labels, preds_raw)
        log_vars['early_stop_eval/'+mode+ '/accuracy'] = accuracy_score(true_labels, preds)
        wandb.log(log_vars)

        class_names = [str(i) for i in range(num_class)]

        fig = plot_confusion_matrix( true_labels,preds, class_names)
        wandb.log({"early_stop_eval/final_confusion_matrix.png": fig})
        fig_title = "Regression for ALL unseen participants"
        reg_fig = regressionPlot(true_labels, preds_raw, class_names, fig_title)
        try:
            wandb.log({"early_stop_eval/final_regression_plot.png": [wandb.Image(reg_fig)]})
        except:
            try:
                wandb.log({"early_stop_eval/final_regression_plot.png": reg_fig})
            except:
                print("failed to log regression plot")


def finetune_model(
        work_dir,
        model,
        loss_cfg,
        datasets,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
        freeze_encoder=False, 
):
    print("Starting STAGE 2: Fine-tuning...")

    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False) for d in datasets
    ]

    global balance_classes
    global class_weights_dict

    if balance_classes:
        dataset_train = call_obj(**datasets[0])
        class_weights_dict = dataset_train.data_source.class_dist

    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)


    loss = call_obj(**loss_cfg_local)

    # print('training hooks: ', training_hooks_local)
    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    final_model = runner.run(data_loaders, workflow, total_epochs, loss=loss)
    return final_model




def pretrain_model(
        work_dir,
        model_cfg,
        loss_cfg,
        datasets,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience=10,
        es_start_up=50,
):
    print("Starting STAGE 1: Pretraining...")

    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False) for d in datasets
    ]

    global balance_classes
    global class_weights_dict

    if balance_classes:
        dataset_train = call_obj(**datasets[0])
        class_weights_dict = dataset_train.data_source.class_dist

    model_cfg_local = copy.deepcopy(model_cfg)
    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)


    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg_local]
        model = torch.nn.Sequential(*model)

    else:
        model = call_obj(**model_cfg_local)
    # print("the model is: ", model)

    # print("These are the model parameters:")
    # for param in model.parameters():
    #     print(param.data)

    # Step 1: Initialize the model with random weights, 
    model.apply(weights_init_xavier)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)

    # print('training hooks: ', training_hooks_local)
    # build runner

    # loss = SupConLoss()
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor_position_pretraining, optimizer, work_dir, log_level, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    pretrained_model = runner.run(data_loaders, workflow, total_epochs, loss=loss, supcon_pretraining=True)
    return pretrained_model


# process a batch of data
def batch_processor_position_pretraining(model, datas, train_mode, loss):

    try:
        data, label = datas
    except:
        data, data_flipped, label = datas
        have_flips = 1

    # Even if we have flipped data, we only want to use the original in this stage

    data_all = data.cuda()
    label = label.cuda()
    num_valid_samples = data.shape[0]

    # Predict the future joint positions using all data
    predicted_joint_positions = model(data)

    if torch.sum(predicted_joint_positions) == 0:        
        raise ValueError("=============================== got all zero output...")


    # Calculate the supcon loss for this data
    try:

        batch_loss = loss(predicted_joint_positions, label)
        # print("the batch loss is: ", batch_loss)
    except Exception as e:
        logging.exception("loss calc message=================================================")
    # raise ValueError("the supcon batch loss is: ", batch_loss)

    preds = []
    raw_preds = []

    label_placeholders = [-1 for i in range(num_valid_samples)]

    # Case when we have a single output
    if type(label_placeholders) is not list:
        label_placeholders = [label_placeholders]

    log_vars = dict(loss_pretrain_position=batch_loss.item())
    output_labels = dict(true=label_placeholders, pred=preds, raw_preds=raw_preds)
    outputs = dict(loss=batch_loss, log_vars=log_vars, num_samples=num_valid_samples)

    return outputs, output_labels, batch_loss.item()
    







# process a batch of data
def batch_processor(model, datas, train_mode, loss):
    #torch.cuda.empty_cache()
    #print('have cuda: ', torch.cuda.is_available())
    #print('using device: ', torch.cuda.get_device_name())
    
    mse_loss = torch.nn.MSELoss()
    model_2 = copy.deepcopy(model)
    have_flips = 0
    try:
        data, label = datas
    except:
        data, data_flipped, label = datas
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
        # print('input_flipped', torch.sum(torch.isnan(data_all_flipped)))     
        # print('raw', torch.sum(torch.isnan(data_all)))   
        output_all_flipped = model_2(data_all_flipped)


    # Get predictions from the model
    output_all = model(data_all)

    if torch.sum(output_all) == 0:        
        raise ValueError("=============================== got all zero output...")
    output = output_all[row_cond]
    loss_flip_tensor = torch.tensor([0.], dtype=torch.float, requires_grad=True) 

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



        log_vars = dict(loss_label=0, loss_flip = loss_flip_tensor.item(), loss_all=loss_flip_tensor.item())
        log_vars['mae_raw'] = 0
        log_vars['mae_rounded'] = 0
        output_labels = dict(true=labels, pred=preds, raw_preds=raw_preds)
        outputs = dict(loss=loss_flip_tensor, log_vars=log_vars, num_samples=0)

        return outputs, output_labels, loss_flip_tensor.item()
    


    y_true_orig_shape = y_true.reshape(1,-1).squeeze()
    losses = loss(output, y_true)


    if type(loss) == type(mse_loss):
        if balance_classes:
            losses = log_weighted_mse_loss(output, y_true, class_weights_dict)
        # Convert the output to classes and clip from 0 to number of classes
        y_pred_rounded = output.detach().cpu().numpy()
        output = y_pred_rounded
        output_list = output.squeeze().tolist()
        y_pred_rounded = y_pred_rounded.reshape(1, -1).squeeze()
        y_pred_rounded = np.round(y_pred_rounded, 0)
        y_pred_rounded = np.clip(y_pred_rounded, 0, num_class-1)
        preds = y_pred_rounded.squeeze().tolist()
    else:    
        rank = output.argsort()
        preds = rank[:,-1].data.tolist()

    labels = y_true_orig_shape.data.tolist()

    # Case when we have a single output
    if type(labels) is not list:
        labels = [labels]
    if type(preds) is not list:
        preds = [preds]
    if type(output_list) is not list:
        output_list = [output_list]

    try:
        labels = [int(cl) for cl in labels]
        preds = [int(cl) for cl in preds]
    except TypeError as e:
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
    output_labels = dict(true=labels, pred=preds, raw_preds=output_list)
    outputs = dict(loss=overall_loss, log_vars=log_vars, num_samples=len(labels))
    # print(type(labels), type(preds))
    # print('this is what we return: ', output_labels)
    return outputs, output_labels, overall_loss

