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
import shutil
from mmskeleton.processor.utils_recognition import *
from mmskeleton.processor.supcon_loss import *

turn_off_wd = True
fast_dev = False
# os.environ['WANDB_MODE'] = 'dryrun'

# Global variables
num_class = 4
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False

local_data_base = '/home/saboa/data'
cluster_data_base = '/home/asabo/projects/def-btaati/asabo'
local_output_base = '/home/saboa/data/mmskel_out'
local_long_term_base = '/home/saboa/data/mmskel_long_term'

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
        launch_from_local=False,
        wandb_project="mmskel",
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience_1=5,
        es_start_up_1=5,
        es_patience_2=10,
        es_start_up_2=50,
        do_position_pretrain=True,
        head='stgcn',
        freeze_encoder=True,
):
    # Set up for logging 
    outcome_label = dataset_cfg[0]['data_source']['outcome_label']

    eval_pipeline = setup_eval_pipeline(dataset_cfg[1]['pipeline'])

    if turn_off_wd:
        for stage in range(len(optimizer_cfg)):
            optimizer_cfg[stage]['weight_decay'] = 0

    global flip_loss_mult
    flip_loss_mult = flip_loss

    global balance_classes
    balance_classes = weight_classes

    # Add the wandb group to work_dir to prevent conflicts if running multiple repetitions of the same configuration
    model_type = get_model_type(model_cfg)


    group_notes = model_type + '_pretrain15' + "_dropout" + str(model_cfg['dropout']) + '_tempkernel' + str(model_cfg['temporal_kernel_size']) + "_batch" + str(batch_size)
    num_class = model_cfg['num_class']
    wandb_group = wandb.util.generate_id() + "_" + outcome_label + "_" + group_notes
    work_dir = os.path.join(work_dir, wandb_group)

    
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


    # Correctly set the full data path
    if launch_from_local:
        work_dir = os.path.join(local_data_base, work_dir)
        
        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(local_data_base, dataset_cfg[i]['data_source']['data_dir'])
    else:
        global fast_dev
        fast_dev = False
        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(cluster_data_base, dataset_cfg[i]['data_source']['data_dir'])

    simple_work_dir = work_dir

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
        try:

            plt.close('all')
            ambid = id_mapping[test_id]
            work_dir_amb = work_dir + "/" + str(ambid)

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


            # This is the entire dataset (not just the PD labelled set)
            datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
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
            kf.get_n_splits(non_test_walks_all)

            # PD labelled walks only
            kf_pd = KFold(n_splits=cv, shuffle=True, random_state=1)
            kf_pd.get_n_splits(non_test_walks_pd_labelled)


            num_reps = 1
            for train_ids, val_ids in kf.split(non_test_walks_all):
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
                all_train = [non_test_walks_all[i] for i in train_ids]
                all_val = [non_test_walks_all[i] for i in val_ids]

                # These are from the pd labelled set
                num_reps_pd = 1
                for train_ids_pd, val_ids_pd in kf.split(non_test_walks_pd_labelled):
                    if num_reps_pd > 1:
                        break
                    num_reps_pd += 1

                    pd_train = [non_test_walks_pd_labelled[i] for i in train_ids_pd]
                    pd_val = [non_test_walks_pd_labelled[i] for i in val_ids_pd]



                # ================================ STAGE 1 ====================================
                # Stage 1 training
                datasets[0]['data_source']['data_dir'] = pd_train
                datasets[1]['data_source']['data_dir'] = pd_val
                datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled

                if fast_dev:
                    datasets[0]['data_source']['data_dir'] = pd_train[:100]
                    datasets[1]['data_source']['data_dir'] = pd_val[:100]
                    datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled[:100]

                # Use same train/val set for training and validation for both sets
                # stage_2_train = stage_1_train[:100]
                # stage_2_val = stage_1_val[:100]

                # Don't do testing during pretraining
                datasets_stage_1 = copy.deepcopy(datasets)
                datasets_stage_1.pop(2)

                workflow_stage_1 = copy.deepcopy(workflow)
                workflow_stage_1.pop(2)

                loss_cfg_stage_1 = copy.deepcopy(loss_cfg[0])
                optimizer_cfg_stage_1 = optimizer_cfg[0]

                # print('optimizer_cfg_stage_1 ', optimizer_cfg_stage_1)

                work_dir_amb = work_dir + "/" + str(ambid)
                simple_work_dir_amb = simple_work_dir + "/" + str(ambid)
                for ds in datasets:
                    ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

                optimizer_cfg_stage_2 = optimizer_cfg[1]
                loss_cfg_stage_2 = copy.deepcopy(loss_cfg[1])

                things_to_log_1 = {'num_ts_predicting': model_cfg['num_ts_predicting'], 'es_start_up_1': es_start_up_1, 'es_patience_1': es_patience_1, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_1, 'optimizer_cfg': optimizer_cfg_stage_1, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }
                things_to_log_2 = {'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, 'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

                # print("train walks: ", stage_1_train)

                print('stage_1_train: ', len(pd_train))
                print('stage_1_val: ', len(pd_val))
                print('test_walks_pd_labelled: ', len(test_walks_pd_labelled))

                print(f"we have {len(pd_train)} stage_1_train and {len(pd_val)} stage_1_val. ")
                print(f"we have {len(pd_train)} stage_2_train and {len(pd_val)} stage_2_val. ")

                # train_one_cycle(
                #     work_dir_amb,
                #     simple_work_dir_amb,
                #     model_cfg,
                #     loss_cfg,
                #     datasets,
                #     optimizer_cfg,
                #     batch_size,
                #     total_epochs,
                #     training_hooks,
                #     workflow,
                #     gpus,
                #     log_level,
                #     workers,
                #     resume_from,
                #     load_from, 
                #     things_to_log_1,
                #     things_to_log_2,
                #     early_stopping,
                #     force_run_all_epochs,
                #     es_patience_1,
                #     es_start_up_1)

                train_one_cycle(
                        work_dir_amb,
                        simple_work_dir_amb,
                        model_cfg,
                        loss_cfg_stage_1,
                        loss_cfg_stage_2,
                        datasets,
                        optimizer_cfg_stage_1,
                        optimizer_cfg_stage_2,
                        batch_size,
                        total_epochs,
                        training_hooks,
                        workflow,
                        gpus,
                        log_level,
                        workers,
                        resume_from,
                        load_from,
                        things_to_log_1,
                        things_to_log_2,
                        early_stopping,
                        force_run_all_epochs,
                        es_patience_1,
                        es_start_up_1,
                        pd_train,
                        pd_val,
                        test_walks_pd_labelled,
                        eval_pipeline,
                        dataset_cfg,
                        es_patience_2,
                        es_start_up_2, 
                        freeze_encoder, 
                        do_position_pretrain,
                        )


            try:
                shutil.rmtree(work_dir_amb)
            except:
                print('failed to delete the participant folder')

        except:
            logging.exception("this went wrong")
            # Done with this participant, we can delete the temp foldeer
            try:
                shutil.rmtree(work_dir_amb)
            except:
                print('failed to delete the participant folder')

    # Final stats
    final_stats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow)

    # Delete the work_dir
    try:
        shutil.rmtree(work_dir)
    except:
        logging.exception('This: ')
        print('failed to delete the work_dir folder: ', work_dir)

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
        num_class=4, 
        data_loaders=None,
):
    print("Starting STAGE 2: Fine-tuning...")

    
    if not data_loaders:
        print("RELOADING DATA FOR FINETUNING")
        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets
        ]

    if data_loaders:
        for dnum in range(len(data_loaders)):
            data_loaders[dnum].dataset.pipeline = datasets[dnum]['pipeline']

    # print(data_loaders[0])
    # print("data_loaders: ", len(data_loaders), "\nworkflow: ", len(workflow), workflow)
    # print("Pipeline: ", datasets[0]['pipeline'])
    # print("Datasets: ", datasets[0]['pipeline'])

    global balance_classes
    global class_weights_dict

    if balance_classes:
        dataset_train = call_obj(**datasets[0])
        class_weights_dict = dataset_train.data_source.class_dist

    loss_cfg_local = copy.deepcopy(loss_cfg)
    training_hooks_local = copy.deepcopy(training_hooks)
    optimizer_cfg_local = copy.deepcopy(optimizer_cfg)

    try:
        loss = call_obj(**loss_cfg_local)
    except:
        print(loss)

    # print('training hooks: ', training_hooks_local)
    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    final_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')    
    
    return final_model


def pretrain_model(
        work_dir,
        simple_work_dir_amb,
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
        do_position_pretrain=True):
    print("============= Starting STAGE 1: Pretraining...")

    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False) for d in datasets
    ]

    global balance_classes
    global class_weights_dict

    # if balance_classes:
    #     dataset_train = call_obj(**datasets[0])
    #     class_weights_dict = dataset_train.data_source.class_dist

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
    model.apply(weights_init)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)
    # loss = WingLoss()

    visualize_preds = {'visualize': False, 'epochs_to_visualize': ['first', 'last'], 'output_dir': os.path.join(local_long_term_base, simple_work_dir_amb)}

    # print('training hooks: ', training_hooks_local)
    # build runner
    # loss = SupConLoss()
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor_position_pretraining, optimizer, work_dir, log_level, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up, visualize_preds=visualize_preds)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    if do_position_pretrain:
        # [('train', 5), ('val', 1)]
        pretrained_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, supcon_pretraining=True)
    else:
        print("SKIPPING PRETRAINING")
        pretrained_model = model

    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')

    return pretrained_model, data_loaders


# process a batch of data
def batch_processor_position_pretraining(model, datas, train_mode, loss, num_class, **kwargs):

    try:
        data, data_flipped, label, name, num_ts = datas
    except:
        data, data_flipped, label, name, num_ts, true_future_ts = datas

    # Even if we have flipped data, we only want to use the original in this stage
    data_all = data.cuda()
    label = label.cuda()
    num_valid_samples = data.shape[0]

    # Predict the future joint positions using all data
    predicted_joint_positions = model(data_all)

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
    output_labels = dict(true=label_placeholders, pred=preds, raw_preds=raw_preds, name=name, num_ts=num_ts)
    outputs = dict(predicted_joint_positions=predicted_joint_positions, loss=batch_loss, log_vars=log_vars, num_samples=num_valid_samples)

    return outputs, output_labels, batch_loss.item()
    

def train_one_cycle(
        work_dir_amb,
        simple_work_dir_amb,
        model_cfg,
        loss_cfg_stage_1,
        loss_cfg_stage_2,
        datasets,
        optimizer_cfg_stage_1,
        optimizer_cfg_stage_2,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
        things_to_log_1=None,
        things_to_log_2=None,
        early_stopping=False,
        force_run_all_epochs=True,
        es_patience_1=10,
        es_start_up_1=50,
        stage_2_train=None,
        stage_2_val=None,
        test_walks_pd_labelled=None,
        eval_pipeline=None,
        dataset_cfg=None,
        es_patience_2=None,
        es_start_up_2=None, 
        freeze_encoder=None,
        do_position_pretrain=True, 
        ):



    pretrained_model, data_loaders = pretrain_model(
        work_dir_amb,
        simple_work_dir_amb,
        model_cfg,
        loss_cfg_stage_1,
        datasets,
        optimizer_cfg_stage_1,
        batch_size,
        total_epochs,
        training_hooks,
        workflow,
        gpus,
        log_level,
        workers,
        resume_from,
        load_from, 
        things_to_log_1,
        early_stopping,
        force_run_all_epochs,
        es_patience_1,
        es_start_up_1, 
        do_position_pretrain,
        )


    # ================================ STAGE 2 ====================================
    # Make sure we're using the correct dataset and workflow
    datasets = [copy.deepcopy(dataset_cfg[1]) for i in range(len(workflow))]
    print(datasets)
    for ds in datasets:
        ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

    # Stage 2 training
    datasets[0]['data_source']['data_dir'] = stage_2_train
    datasets[1]['data_source']['data_dir'] = stage_2_val
    datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled

    # Don't shear or scale the test or val data
    datasets[1]['pipeline'] = eval_pipeline
    datasets[2]['pipeline'] = eval_pipeline



    print('optimizer_cfg_stage_2 ', optimizer_cfg_stage_2)


    # Reset the head
    pretrained_model.module.set_stage_2()
    pretrained_model.module.head.apply(weights_init)




    # print("final model for fine_tuning is: ", pretrained_model)

    finetune_model(work_dir_amb,
                pretrained_model,
                loss_cfg_stage_2,
                datasets,
                optimizer_cfg_stage_2,
                batch_size,
                total_epochs,
                training_hooks,
                workflow,
                gpus,
                log_level,
                workers,
                resume_from,
                load_from,
                things_to_log_2,
                early_stopping,
                force_run_all_epochs,
                es_patience_2,
                es_start_up_2, 
                freeze_encoder, 
                num_class, 
                data_loaders)
