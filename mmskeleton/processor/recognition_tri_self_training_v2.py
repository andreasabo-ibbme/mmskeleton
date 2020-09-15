from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner, TooManyRetriesException
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
import time
from pathlib import Path

turn_off_wd = True
fast_dev = True
os.environ['WANDB_MODE'] = 'dryrun'

# Global variables
num_class = 4
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False

local_data_base = '/home/saboa/data'
cluster_data_base = '/home/asabo/projects/def-btaati/asabo'
local_output_base = '/home/saboa/data/mmskel_out'
local_long_term_base = '/home/saboa/data/mmskel_long_term'
cluster_output_wandb = '/home/asabo/projects/def-btaati/asabo/mmskeleton/wandb_dryrun'
local_output_wandb = '/home/saboa/code/mmskeleton/wandb_dryrun'


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    # print("deleting: ", directory)
    # print("with: ", os.listdir(directory))
    # for f in os.listdir(directory):
    #     to_rm = os.path.join(directory,f)
    #     print("removing this: ", to_rm)
    #     os.remove(to_rm)
    # print("with2: ", os.listdir(directory))

    try:
        directory.rmdir()
    except:
        pass
        # print("couldn't delete: ", directory)


def robust_rmtree(path, logger=None, max_retries=3):
    """Robustly tries to delete paths.
    Retries several times (with increasing delays) if an OSError
    occurs.  If the final attempt fails, the Exception is propagated
    to the caller.
    """
    print("removing robustly")
    dt = 1
    for i in range(max_retries):
        print("removing robustly: ", i)

        try:
            shutil.rmtree(path)
            return
        except Exception as e:
            # print(e)
            # print('Unable to remove path: %s' % path)
            # print('Retrying after %d seconds' % dt)
            # print('files it has: ', os.listdir(path))
            rmdir(path)
            time.sleep(dt)
            dt *= 1.5

    # Final attempt, pass any Exceptions up to caller.
    shutil.rmtree(path)


def sync_wandb(wandb_local_id):
    # Sync everything to wandb at the end
    try:
        os.system('wandb sync ' + wandb_local_id)


        # Delete the work_dir if successful sync
        try:
            robust_rmtree(wandb_local_id)
            # shutil.rmtree(work_dir)
        except:
            logging.exception('This: ')
            print('failed to delete the wandb_log_local_group folder: ', wandb_local_id)

    except:
        print('failed to sync wandb')



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
        train_extrema_for_epochs=0,
        head='stgcn',
        freeze_encoder=True,
        do_position_pretrain=True,
        model_increase_iters=None,
        model_increase_mults=None,
        num_self_train_iter=0,
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

    model_type = ''

    if model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_2_position_pretrain':
        model_type = "v2"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_10_position_pretrain':
        model_type = "v10"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_smaller_11_position_pretrain':
        model_type = "v11"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_orig_position_pretrain':
        model_type = "v0"
    elif model_cfg['type'] == 'models.backbones.ST_GCN_18_ordinal_orig_position_pretrain_dynamic_v1':
        model_type = "dynamic_v1"
        
    else: 
        model_type = model_cfg['type']



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

    wandb_local_id = wandb.util.generate_id()

    # Correctly set the full data path
    if launch_from_local:
        work_dir = os.path.join(local_data_base, work_dir)
        wandb_log_local_group = os.path.join(local_output_wandb, wandb_local_id)

        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(local_data_base, dataset_cfg[i]['data_source']['data_dir'])
    else:
        global fast_dev
        fast_dev = False
        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(cluster_data_base, dataset_cfg[i]['data_source']['data_dir'])

        wandb_log_local_group = os.path.join(cluster_output_wandb, wandb_local_id)

    simple_work_dir = work_dir
    os.makedirs(wandb_log_local_group)

    os.environ["WANDB_RUN_GROUP"] = wandb_group

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

            if num_self_train_iter >= 0:
                self_train_iteration_count = 0
            plt.close('all')
            ambid = id_mapping[test_id]
            work_dir_amb = work_dir + "/" + str(self_train_iteration_count) + "/" + str(ambid)

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

                    stage_2_train = [non_test_walks_pd_labelled[i] for i in train_ids_pd]
                    stage_2_val = [non_test_walks_pd_labelled[i] for i in val_ids_pd]


                print(f"we have {len(stage_1_train)} stage_1_train and {len(stage_1_val)} stage_1_val. ")
                print(f"we have {len(stage_2_train)} stage_2_train and {len(stage_2_val)} stage_2_val. ")
            

                # ================================ STAGE 1 ====================================
                # Stage 1 training
                datasets[0]['data_source']['data_dir'] = stage_1_train
                datasets[1]['data_source']['data_dir'] = stage_1_val
                datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled

                if fast_dev:
                    datasets[0]['data_source']['data_dir'] = stage_1_train[:100]
                    datasets[1]['data_source']['data_dir'] = stage_1_val[:100]
                    datasets[2]['data_source']['data_dir'] = test_walks_pd_labelled[:100]


                datasets_stage_1 = copy.deepcopy(datasets)
                datasets_stage_1.pop(2)

                workflow_stage_1 = copy.deepcopy(workflow)
                workflow_stage_1.pop(2)

                loss_cfg_stage_1 = copy.deepcopy(loss_cfg[0])

                optimizer_cfg_stage_1 = optimizer_cfg[0]

                print('optimizer_cfg_stage_1 ', optimizer_cfg_stage_1)

                work_dir_amb = work_dir + "/" + str(self_train_iteration_count) + "/" + str(ambid)
                # work_dir_amb = work_dir + "/" + str(ambid)
                simple_work_dir_amb = simple_work_dir + "/" + str(ambid)
                for ds in datasets:
                    ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

                things_to_log = {'dir': wandb_log_local_group, 'model_increase_mults': model_increase_mults, 'model_increase_iters': model_increase_iters, 'train_extrema_for_epochs': train_extrema_for_epochs, 'self_train_iteration_count': self_train_iteration_count, 'num_ts_predicting': model_cfg['num_ts_predicting'], 'es_start_up_1': es_start_up_1, 'es_patience_1': es_patience_1, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_1, 'optimizer_cfg': optimizer_cfg_stage_1, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

                # print("train walks: ", stage_1_train)

                print('stage_1_train: ', len(stage_1_train))
                print('stage_1_val: ', len(stage_1_val))
                print('test_walks_pd_labelled: ', len(test_walks_pd_labelled))


                pretrained_model = pretrain_model(
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
                    things_to_log,
                    early_stopping,
                    force_run_all_epochs,
                    es_patience_1,
                    es_start_up_1, do_position_pretrain,
                    )

                pretrained_model_copy = copy.deepcopy(pretrained_model)

                # ================================ STAGE 2 ====================================
                # Make sure we're using the correct dataset
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

                optimizer_cfg_stage_2 = optimizer_cfg[1]
                loss_cfg_stage_2 = copy.deepcopy(loss_cfg[1])

                print('optimizer_cfg_stage_2 ', optimizer_cfg_stage_2)


                # Reset the head
                pretrained_model.module.set_stage_2()
                pretrained_model.module.head.apply(weights_init)

                things_to_log = {'dir': wandb_log_local_group, 'model_increase_mults': model_increase_mults, 'model_increase_iters': model_increase_iters, 'self_train_iteration_count': self_train_iteration_count, 'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, 'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

                # print("final model for fine_tuning is: ", pretrained_model)

                finetuned_model = finetune_model(work_dir_amb,
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
                            things_to_log,
                            early_stopping,
                            force_run_all_epochs,
                            es_patience_2,
                            es_start_up_2, 
                            freeze_encoder, 
                            num_class, 
                            None, # dataloaders
                            train_extrema_for_epochs)


                # Now load in all of the data for pseudolabelling 
                # Only use the true labels for val and test sets for evaluation
                all_trainset = stage_2_train + stage_1_train + stage_1_val # Stage 1 are all unlabelled
                all_valset = stage_2_val # These all have labels
                all_testset = test_walks_pd_labelled


                # Create dataloaders from these new dataset (they will be used to train all subsequent models)
                datasets[0]['data_source']['data_dir'] = all_trainset
                datasets[1]['data_source']['data_dir'] = all_valset
                datasets[2]['data_source']['data_dir'] = all_testset
                if fast_dev:
                    datasets[0]['data_source']['data_dir'] = all_trainset[0:100]
                # for i in range(len(datasets)):
                #     datasets[i]['sampler'] = torch.utils.data.RandomSampler(data_source=datasets[i]['data_source']['data_dir']*2)

                # all_data_loaders = [
                #     torch.utils.data.DataLoader(dataset=call_obj(**d),
                #                                 batch_size=batch_size,
                #                                 sampler=torch.utils.data.RandomSampler(data_source=d['data_source']['data_dir']*2),
                #                                 num_workers=workers,
                #                                 drop_last=False) for d in datasets
                # ]
                all_data_loaders = [ 
                    torch.utils.data.DataLoader(dataset=call_obj(**d),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=workers,
                                                drop_last=False) for d in datasets
                ]

                # SELF_TRAINING======================================================================================
                # SELF_TRAINING======================================================================================
                # SELF_TRAINING======================================================================================
                # SELF_TRAINING======================================================================================
                
                for iter_num in range(1, num_self_train_iter):
                    self_train_iteration_count = iter_num
                    print("itertest"*20)
                    changed_model = False

                    for ds in range(len(all_data_loaders)):
                        all_data_loaders[ds].dataset.pipeline = eval_pipeline

                    # Label the originally unlabelled training data (do not alter the data with labels or the val/test sets)             
                    relabelData(finetuned_model, all_data_loaders[0])

                    # Increase model capacity if needed
                    pretrained_model = pretrained_model_copy
                    if iter_num in model_increase_iters:
                        local_ind = model_increase_iters.index(iter_num)
                        pretrained_model.module.addLayer(model_increase_mults[local_ind])
                        changed_model = True
                        print('increased model size')


                    # Reset the model and train
                    for ds in range(len(all_data_loaders)):
                        all_data_loaders[ds].dataset.pipeline = dataset_cfg[0]['pipeline']
                    datasets_stage_1 = copy.deepcopy(datasets)
                    datasets_stage_1.pop(2)

                    workflow_stage_1 = copy.deepcopy(workflow)
                    workflow_stage_1.pop(2)

                    loss_cfg_stage_1 = copy.deepcopy(loss_cfg[0])

                    optimizer_cfg_stage_1 = optimizer_cfg[0]

                    print('optimizer_cfg_stage_1 ', optimizer_cfg_stage_1)

                    # work_dir_amb = work_dir + "/" + str(ambid) + "/" + str(iter_num)
                    simple_work_dir_amb = simple_work_dir + "/" + str(self_train_iteration_count) + "/" + str(ambid)
                    work_dir_amb = work_dir + "/" + str(self_train_iteration_count) + "/" + str(ambid)

                    for ds in datasets:
                        ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

                    things_to_log = {'dir': wandb_log_local_group, 'model_increase_mults': model_increase_mults, 'model_increase_iters': model_increase_iters, 'train_extrema_for_epochs': train_extrema_for_epochs,'self_train_iteration_count': self_train_iteration_count, 'num_ts_predicting': model_cfg['num_ts_predicting'], 'es_start_up_1': es_start_up_1, 'es_patience_1': es_patience_1, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_1, 'optimizer_cfg': optimizer_cfg_stage_1, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

                    # print("train walks: ", stage_1_train)

                    print('stage_1_train: ', len(stage_1_train))
                    print('stage_1_val: ', len(stage_1_val))
                    print('test_walks_pd_labelled: ', len(test_walks_pd_labelled))

                    # Only redo pretraining if we changed the model, other wise just reuse what we had
                    if changed_model:
                        pretrained_model = pretrain_model(
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
                            things_to_log,
                            early_stopping,
                            force_run_all_epochs,
                            es_patience_1,
                            es_start_up_1, 
                            do_position_pretrain,
                            all_data_loaders
                            )
                        pretrained_model_copy = copy.deepcopy(pretrained_model)

                    # Finetune
                    for ds in range(len(all_data_loaders)):
                        all_data_loaders[ds].dataset.pipeline = dataset_cfg[1]['pipeline']
                    # print(datasets)
                    # for ds in datasets:
                    #     ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

                    # Stage 2 training


                    # Don't shear or scale the test or val data
                    datasets[1]['pipeline'] = eval_pipeline
                    datasets[2]['pipeline'] = eval_pipeline

                    optimizer_cfg_stage_2 = optimizer_cfg[1]
                    loss_cfg_stage_2 = copy.deepcopy(loss_cfg[1])

                    print('optimizer_cfg_stage_2 ', optimizer_cfg_stage_2)


                    # Reset the head
                    pretrained_model.module.set_stage_2()
                    pretrained_model.module.head.apply(weights_init)

                    things_to_log = {'dir': wandb_log_local_group, 'model_increase_mults': model_increase_mults, 'model_increase_iters': model_increase_iters, 'train_extrema_for_epochs': train_extrema_for_epochs, 'self_train_iteration_count': self_train_iteration_count, 'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': ambid, 'test_AMBID_num': len(test_walks_pd_labelled), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, 'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

                    # print("final model for fine_tuning is: ", pretrained_model)

                    finetuned_model = finetune_model(work_dir_amb,
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
                                things_to_log,
                                early_stopping,
                                force_run_all_epochs,
                                es_patience_2,
                                es_start_up_2, 
                                freeze_encoder, 
                                num_class, 
                                all_data_loaders, 
                                train_extrema_for_epochs)



            try:
                robust_rmtree(work_dir_amb)

            except:
                print('failed to delete the participant folder')

        except TooManyRetriesException:
            # print("CAUGHT TooManyRetriesException - something is very wrong. Stopping")
            # sync_wandb(wandb_log_local_group)

            break

        except:
            logging.exception("this went wrong")
            # Done with this participant, we can delete the temp foldeer
            try:
                shutil.rmtree(work_dir_amb)
            except:
                print('failed to delete the participant folder')

    # Final stats
    final_stats(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, num_self_train_iter)
    # final_stats_by_iter(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, num_self_train_iter)
    # wandb.init(name='END', project=wandb_project, group=wandb_group, reinit=True)

    sync_wandb(wandb_log_local_group)

    # Delete the work_dir
    try:
        robust_rmtree(work_dir)
        # shutil.rmtree(work_dir)
    except:
        logging.exception('This: ')
        print('failed to delete the work_dir folder: ', work_dir)

    return work_dir




def relabelData(model, data_loader):
    model.eval()
    print("relabelData now..."* 5)
    print(len(data_loader))
    for i, datas in enumerate(data_loader):
        print("+"* 50, i, "  ", len(datas[0]))
        try:
            try:
                data, label, name, num_ts, index, non_pseudo_label = datas
            except:
                data, data_flipped, label, name, num_ts, index, non_pseudo_label = datas
                have_flips = 1
        except:
            print("datas: ", len(datas))
            raise RuntimeError("SOMETHING IS UP WITH THE DATA")

        # Predict
        data_all = data.cuda()
        output_all = model(data_all)
        output_all = output_all.detach().cpu().numpy().squeeze().tolist()
        index = index.numpy()

        for i in range(len(output_all)):
            data_loader.dataset.reassignLabel(index[i], output_all[i])




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
        all_data_loaders=None,
        train_extrema_for_epochs=0,
):
    print("Starting STAGE 2: Fine-tuning...")

    if all_data_loaders is None:
        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets
        ]
    else:
        print("REUSING labelled data... finetuning")
        data_loaders = all_data_loaders

    data_loaders[0].dataset.data_source.sample_extremes = True
    workflow = [tuple(w) for w in workflow]
    global balance_classes
    global class_weights_dict
    for i in range(len(data_loaders)):
        class_weights_dict[workflow[i][0]] = data_loaders[i].dataset.data_source.get_class_dist()

    # if balance_classes:
    #     class_weights_dict = data_loaders[0].dataset.data_source.class_dist

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
    
    # [('train', 5), ('val', 1)]
    final_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict, train_extrema_for_epochs=train_extrema_for_epochs)

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
        do_position_pretrain=True, 
        all_data_loaders=None):
    print("============= Starting STAGE 1: Pretraining...")

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

    # Step 1: Initialize the model with random weights, 
    model.apply(weights_init)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)
    loss = WingLoss()

    # print(model)
    # input("Original model...")
    # model.module.addLayer(2)
    # print(model)
    # input("Added model...")

    if not do_position_pretrain:
        return model

    if all_data_loaders is None:
        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets
        ]
    else:
        print("Using existing data loaders")
        data_loaders = all_data_loaders

    global balance_classes
    global class_weights_dict
    data_loaders[0].dataset.data_source.sample_extremes = False

    # if balance_classes:
    #     dataset_train = call_obj(**datasets[0])
    #     class_weights_dict = dataset_train.data_source.class_dist


    # print("the model is: ", model)

    # print("These are the model parameters:")
    # for param in model.parameters():
    #     print(param.data)



    visualize_preds = {'visualize': False, 'epochs_to_visualize': ['first', 'last'], 'output_dir': os.path.join(local_long_term_base, simple_work_dir_amb)}

    # print('training hooks: ', training_hooks_local)
    # build runner
    # loss = SupConLoss()
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor_position_pretraining, optimizer, work_dir, log_level, things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, es_patience=es_patience, es_start_up=es_start_up, visualize_preds=visualize_preds)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    if do_position_pretrain:
        pretrained_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, supcon_pretraining=True)
    else:
        pretrained_model = model
    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')

    return pretrained_model


# process a batch of data
def batch_processor_position_pretraining(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        data, data_flipped, label, name, num_ts, index, non_pseudo_label = datas
    except:
        data, data_flipped, label, name, num_ts, true_future_ts, index, non_pseudo_label = datas

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
    
