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
# from spacecutter.models import OrdinalLogisticModel
# import spacecutter
import pandas as pd
import pickle
import shutil
from mmskeleton.processor.utils_recognition import *
from mmskeleton.processor.supcon_loss import *
import time

turn_off_wd = True
fast_dev = True
log_incrementally = True
log_code = False
log_conf_mat = False
os.environ['WANDB_MODE'] = 'dryrun'


# Global variables
num_class = 4
balance_classes = False
class_weights_dict = {}
flip_loss_mult = False

local_data_base = '/home/saboa/data/OBJECTIVE_2_ML_DATA/data'

local_output_base = '/home/saboa/data/mmskel_out'
local_long_term_base = '/home/saboa/data/mmskel_long_term'

local_output_wandb = '/home/saboa/code/mmskeleton/wandb_dryrun'

local_model_zoo_base = '/home/saboa/data/OBJECTIVE_2_ML_DATA/model_zoo'


local_dataloader_temp = '/home/saboa/data/OBJECTIVE_2_ML_DATA/dataloaders'



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
        model_save_root='None',
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
):
    # Reproductibility
    set_seed(0)
    print("==================================")
    print('have cuda: ', torch.cuda.is_available())
    print('using device: ', torch.cuda.get_device_name())
    
    global log_incrementally
    if log_incrementally:
        os.environ['WANDB_MODE'] = 'run'
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    if log_code:
        os.environ['WANDB_DISABLE_CODE'] = 'false'
    else:
        os.environ['WANDB_DISABLE_CODE'] = 'true'

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

    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    # Check if we should use gait features
    if 'use_gait_feats' in dataset_cfg[1]['data_source']:
        model_cfg['use_gait_features'] = dataset_cfg[1]['data_source']['use_gait_feats']



    wandb_local_id = wandb.util.generate_id()

    # Correctly set the full data path depending on if launched from local or cluster env
    if launch_from_local:
        work_dir = os.path.join(local_data_base, work_dir)
        wandb_log_local_group = os.path.join(local_output_wandb, wandb_local_id)

        model_zoo_root = local_model_zoo_base
        dataloader_temp = local_dataloader_temp

        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(local_data_base, dataset_cfg[i]['data_source']['data_dir'])
    else: # launching from the cluster
        global fast_dev
        fast_dev = False
        model_zoo_root = cluster_model_zoo_base
        dataloader_temp = cluster_dataloader_temp

        for i in range(len(dataset_cfg)):
            dataset_cfg[i]['data_source']['data_dir'] = os.path.join(cluster_data_base, dataset_cfg[i]['data_source']['data_dir'])

        wandb_log_local_group = os.path.join(cluster_output_wandb, wandb_local_id)
        work_dir = os.path.join(cluster_workdir_base, work_dir)

    simple_work_dir = work_dir
    os.makedirs(simple_work_dir)

    print(simple_work_dir)
    print(wandb_log_local_group)


    os.environ["WANDB_RUN_GROUP"] = wandb_group

    # All data dir (use this for finetuning with the flip loss)
    data_dir_all_data = dataset_cfg[0]['data_source']['data_dir']
    all_files = [os.path.join(data_dir_all_data, f) for f in os.listdir(data_dir_all_data) if os.path.isfile(os.path.join(data_dir_all_data, f))]
    print("all files: ", len(all_files))

    all_file_names_only = [f for f in os.listdir(data_dir_all_data) if os.path.isfile(os.path.join(data_dir_all_data, f))]

    # PD lablled dir (only use this data for supervised contrastive)
    data_dir_pd_data = dataset_cfg[1]['data_source']['data_dir']
    pd_all_files = [os.path.join(data_dir_pd_data, f) for f in os.listdir(data_dir_pd_data) if os.path.isfile(os.path.join(data_dir_pd_data, f))]
    # pd_all_file_names_only = os.listdir(data_dir_pd_data)
    pd_all_file_names_only = [f for f in os.listdir(data_dir_pd_data) if os.path.isfile(os.path.join(data_dir_pd_data, f))]
    print("pd_all_files: ", len(pd_all_files))


    # Set up test data
    data_dir_all_data_test = dataset_cfg[1]['data_source']['data_dir']
    all_files_test = [os.path.join(data_dir_all_data_test, f) for f in os.listdir(data_dir_all_data_test) if os.path.isfile(os.path.join(data_dir_all_data_test, f))]
    all_files_test_names_only = [f for f in os.listdir(data_dir_all_data_test) if os.path.isfile(os.path.join(data_dir_all_data_test, f))]

    print("all files test: ", len(all_files_test))


    # sort all of the files
    all_files.sort()
    all_file_names_only.sort()
    pd_all_files.sort()
    pd_all_file_names_only.sort()
    all_files_test.sort()
    all_files_test_names_only.sort()

    try:
        plt.close('all')
        test_walks = all_files_test
        non_test_walks_all = all_files



        # data exploration
        print(f"test_walks: {len(test_walks)}")
        print(f"non_test_walks: {len(non_test_walks_all)}")


        # Split the non_test walks into train/val
        kf = KFold(n_splits=cv, shuffle=True, random_state=1)
        kf.get_n_splits(non_test_walks_all)



        # This loop is for pretraining
        num_reps = 1
        for train_ids, val_ids in kf.split(non_test_walks_all):
            plt.close('all')
            test_id = num_reps
            num_reps += 1

            # Divide all of the data into train/val
            train_walks = [non_test_walks_all[i] for i in train_ids]
            val_walks = [non_test_walks_all[i] for i in val_ids]

            datasets = [copy.deepcopy(dataset_cfg[0]) for i in range(len(workflow))]
            for ds in datasets:
                ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

            datasets_stage_1 = datasets
            # ================================ STAGE 1 ====================================
            # Stage 1 training
            datasets_stage_1[0]['data_source']['data_dir'] = train_walks
            datasets_stage_1[1]['data_source']['data_dir'] = val_walks
            datasets_stage_1[2]['data_source']['data_dir'] = test_walks

            if fast_dev:
                datasets_stage_1[0]['data_source']['data_dir'] = train_walks[:50]
                datasets_stage_1[1]['data_source']['data_dir'] = val_walks[:50]
                datasets_stage_1[2]['data_source']['data_dir'] = test_walks[:50]



            # datasets_stage_1 = copy.deepcopy(datasets)
            # datasets_stage_1.pop(2)

            workflow_stage_1 = copy.deepcopy(workflow)
            # workflow_stage_1.pop(2)

            # print(len(datasets_stage_1))
            # print(len(workflow_stage_1))
            # input('here')
            loss_cfg_stage_1 = copy.deepcopy(loss_cfg[0])

            optimizer_cfg_stage_1 = optimizer_cfg[0]

            print('optimizer_cfg_stage_1 ', optimizer_cfg_stage_1)

            work_dir_amb = work_dir + "/" + str(test_id)
            simple_work_dir_amb = simple_work_dir + "/" + str(test_id)
            for ds in datasets:
                ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']

            things_to_log = {'num_ts_predicting': model_cfg['num_ts_predicting'], 'es_start_up_1': es_start_up_1, 'es_patience_1': es_patience_1, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': num_reps, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_1, 'optimizer_cfg': optimizer_cfg_stage_1, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }


            print('stage_1_train: ', len(train_walks))
            print('stage_1_val: ', len(val_walks))
            print('stage_1_test: ', len(test_walks))

            # path_to_pretrained_model = os.path.join(model_zoo_root, model_save_root, dataset_cfg[0]['data_source']['outcome_label'], model_type, \
            #                                         str(model_cfg['temporal_kernel_size']), str(model_cfg['dropout']), str(test_id))

            # Pretrain doesnt depend on the outcome label
            path_to_pretrained_model = os.path.join(model_zoo_root, model_save_root, model_type, \
                                                    str(model_cfg['temporal_kernel_size']), str(model_cfg['dropout']), str(test_id))


            load_all = flip_loss != 0
            path_to_saved_dataloaders = os.path.join(dataloader_temp, outcome_label, model_save_root, str(num_reps), \
                                                    "load_all" + str(load_all), "gait_feats_" + str(model_cfg['use_gait_features']), str(test_id))
            

            print('path_to_pretrained_model', path_to_pretrained_model)
            if not os.path.exists(path_to_pretrained_model):
                os.makedirs(path_to_pretrained_model)


            pretrained_model = pretrain_model(
                work_dir_amb,
                simple_work_dir_amb,
                model_cfg,
                loss_cfg_stage_1,
                datasets_stage_1,
                optimizer_cfg_stage_1,
                batch_size,
                total_epochs,
                training_hooks,
                workflow_stage_1,
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
                path_to_pretrained_model, 
                path_to_saved_dataloaders
                )



            # ================================ STAGE 2 ====================================
            # Make sure we're using the correct dataset
            datasets = [copy.deepcopy(dataset_cfg[1]) for i in range(len(workflow))]
            print(datasets)
            for ds in datasets:
                ds['data_source']['layout'] = model_cfg['graph_cfg']['layout']


            # Stage 2 training
            datasets[0]['data_source']['data_dir'] = train_walks
            datasets[1]['data_source']['data_dir'] = val_walks
            datasets[2]['data_source']['data_dir'] = test_walks


            if fast_dev:
                datasets[0]['data_source']['data_dir'] = train_walks[:50]
                datasets[1]['data_source']['data_dir'] = val_walks[:50]
                datasets[2]['data_source']['data_dir'] = test_walks[:50]


            # Don't shear or scale the test or val data (also always just take the middle 120 crop)
            datasets[1]['pipeline'] = eval_pipeline
            datasets[2]['pipeline'] = eval_pipeline

            optimizer_cfg_stage_2 = optimizer_cfg[1]
            loss_cfg_stage_2 = copy.deepcopy(loss_cfg[1])

            print('optimizer_cfg_stage_2 ', optimizer_cfg_stage_2)

                
            # if we don't want to use the pretrained head, reset to norm init
            if not pretrain_model:
                pretrained_model.module.encoder.apply(weights_init_xavier)

            # Reset the head for finetuning
            pretrained_model.module.set_stage_2()
            pretrained_model.module.head.apply(weights_init_xavier)

            things_to_log = {'do_position_pretrain': do_position_pretrain, 'num_reps_pd': test_id, 'train_extrema_for_epochs': train_extrema_for_epochs, 'supcon_head': head, 'freeze_encoder': freeze_encoder, 'es_start_up_2': es_start_up_2, 'es_patience_2': es_patience_2, 'force_run_all_epochs': force_run_all_epochs, 'early_stopping': early_stopping, 'weight_classes': weight_classes, 'keypoint_layout': model_cfg['graph_cfg']['layout'], 'outcome_label': outcome_label, 'num_class': num_class, 'wandb_project': wandb_project, 'wandb_group': wandb_group, 'test_AMBID': test_id, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg_stage_2, 'optimizer_cfg': optimizer_cfg_stage_2, 'dataset_cfg_data_source': dataset_cfg[0]['data_source'], 'notes': notes, 'batch_size': batch_size, 'total_epochs': total_epochs }

            # print("final model for fine_tuning is: ", pretrained_model)
            # input("here: " + work_dir_amb)
            print("starting finetuning", "*" *100)
            _, num_epochs = finetune_model(work_dir_amb,
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
                        train_extrema_for_epochs, 
                        path_to_saved_dataloaders, 
                        path_to_pretrained_model)



        # Done CV
        try:
            robust_rmtree(work_dir_amb)
        except:
            print('failed to delete the participant folder')

    except TooManyRetriesException:
        print("CAUGHT TooManyRetriesException - something is very wrong. Stopping")
        # sync_wandb(wandb_log_local_group)

    except:
        logging.exception("this went wrong")
        # Done with this participant, we can delete the temp foldeer
        try:
            robust_rmtree(work_dir_amb)
        except:
            print('failed to delete the participant folder')


    # Final stats
    if exclude_cv:
        final_stats_numbered(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, 1)
    else:
        final_stats_numbered(work_dir, wandb_group, wandb_project, total_epochs, num_class, workflow, cv)


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
        train_extrema_for_epochs=0,
        path_to_saved_dataloaders=None, 
        path_to_pretrained_model=None
):
    print("Starting STAGE 2: Fine-tuning...")

    load_data = True
    base_dl_path = os.path.join(path_to_saved_dataloaders, 'finetuning') 
    full_dl_path = os.path.join(base_dl_path, 'data.pt')
    os.makedirs(base_dl_path, exist_ok=True) 
    if os.path.isfile(full_dl_path):
        try:
            data_loaders = torch.load(full_dl_path)
            load_data = False
        except:
            print(f'failed to load dataloaders from file: {full_dl_path}, loading from individual files')

    if load_data:
        set_seed(0)

        train_dataloader = torch.utils.data.DataLoader(dataset=call_obj(**datasets[0]),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False)
    

        # Normalize by the train scaler
        for d in datasets[1:]:
            d['data_source']['scaler'] = train_dataloader.dataset.get_scaler()

        data_loaders = [
            torch.utils.data.DataLoader(dataset=call_obj(**d),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=workers,
                                        drop_last=False) for d in datasets[1:]
        ]

        data_loaders.insert(0, train_dataloader) # insert the train dataloader
        data_loaders[0].dataset.data_source.sample_extremes = True

        # Save for next time
        torch.save(data_loaders, full_dl_path)
        

    # return None, 0

    workflow = [tuple(w) for w in workflow]
    global balance_classes
    global class_weights_dict
    for i in range(len(data_loaders)):
        class_weights_dict[workflow[i][0]] = data_loaders[i].dataset.data_source.get_class_dist()

    # Set up model for finetuning
    set_seed(0)
    model.module.set_classification_head_size(data_loaders[i].dataset.data_source.get_num_gait_feats())
    model.module.set_stage_2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # input(model)
    set_seed(0)

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
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, num_class=num_class, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, freeze_encoder=freeze_encoder, finetuning=True,\
                    log_conf_mat=log_conf_mat)
    runner.register_training_hooks(**training_hooks_local)

    # run
    final_model, num_epoches_early_stop_finetune = runner.run(data_loaders, workflow, total_epochs, train_extrema_for_epochs=train_extrema_for_epochs, loss=loss, flip_loss_mult=flip_loss_mult, balance_classes=balance_classes, class_weights_dict=class_weights_dict)
    
    try:
        # Wait half a minute so the WANDB thread can sync
        time.sleep(30)
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')    
    
    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_final.pt')
        torch.save(final_model.module.state_dict(), checkpoint_file)


    return final_model, num_epoches_early_stop_finetune


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
        path_to_pretrained_model=None, 
        path_to_saved_dataloaders=None):
    print("============= Starting STAGE 1: Pretraining...", "="*50)
    print(path_to_pretrained_model)
    set_seed(0)

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


    print("path_to_pretrained_model===============================================================", path_to_pretrained_model)
    if not do_position_pretrain:
        print("SKIPPING PRETRAINING-------------------")
        model = MMDataParallel(model, device_ids=range(gpus)).cuda()
        return model

    if path_to_pretrained_model is not None:
        checkpoint_file = os.path.join(path_to_pretrained_model, 'checkpoint_pretrain.pt')
        if os.path.isfile(checkpoint_file):
            print(checkpoint_file)

            # Only copy over the ST-GCN layer from this model
            model_state = model.state_dict()

            pretrained_state = torch.load(checkpoint_file)
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }  


            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            # input(model.use_gait_features)

            model = MMDataParallel(model, device_ids=range(gpus)).cuda()

            return model

    # Step 1: Initialize the model with random weights, 
    set_seed(0)
    model.apply(weights_init_xavier)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()


    load_data = True
    base_dl_path = os.path.join(path_to_saved_dataloaders, 'finetuning') 
    full_dl_path = os.path.join(base_dl_path, 'data.pt')
    os.makedirs(base_dl_path, exist_ok=True)     
    if os.path.isfile(full_dl_path):
        try:
            data_loaders = torch.load(full_dl_path)
            load_data = False
        except:
            print(f'failed to load dataloaders from file: {full_dl_path}, loading from individual files')

    if load_data:
        set_seed(0)
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


    torch.cuda.set_device(0)
    loss = call_obj(**loss_cfg_local)
    loss = WingLoss()

    visualize_preds = {'visualize': False, 'epochs_to_visualize': ['first', 'last'], 'output_dir': os.path.join(local_long_term_base, simple_work_dir_amb)}

    # print('training hooks: ', training_hooks_local)
    # build runner
    # loss = SupConLoss()
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg_local)
    runner = Runner(model, batch_processor_position_pretraining, optimizer, work_dir, log_level, \
                    things_to_log=things_to_log, early_stopping=early_stopping, force_run_all_epochs=force_run_all_epochs, \
                    es_patience=es_patience, es_start_up=es_start_up, visualize_preds=visualize_preds, log_conf_mat=log_conf_mat)
    runner.register_training_hooks(**training_hooks_local)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    pretrained_model, _ = runner.run(data_loaders, workflow, total_epochs, loss=loss, supcon_pretraining=True)
    
    try:
        shutil.rmtree(wandb.run.dir)
    except:
        print('failed to delete the wandb folder')

    # print(pretrained_model)
    # input('model')
    if path_to_pretrained_model is not None:
        torch.save(pretrained_model.module.state_dict(), checkpoint_file)
        # print(checkpoint_file)

        # input('saved')
    return pretrained_model


# process a batch of data
def batch_processor_position_pretraining(model, datas, train_mode, loss, num_class, **kwargs):
    try:
        data, data_flipped, label, name, num_ts, index, non_pseudo_label = datas
    except:
        data, data_flipped, label, name, num_ts, true_future_ts, index, non_pseudo_label = datas

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  
    # Even if we have flipped data, we only want to use the original in this stage
    gait_features = np.empty([1, 9])# default value if we dont have any gait features to load in
    if isinstance(data, dict):
        gait_features = data['gait_feats'].type(dtype)
        data = data['data'].type(dtype)

    data_all = data.cuda()
    gait_features_all = gait_features.cuda()

    label = label.cuda()
    num_valid_samples = data.shape[0]

    # Predict the future joint positions using all data
    predicted_joint_positions = model(data_all, gait_features_all)

    if torch.sum(predicted_joint_positions) == 0:        
        raise ValueError("=============================== got all zero output...")


    # Calculate the supcon loss for this data
    try:
        batch_loss = loss(predicted_joint_positions, label)
        # print("the batch loss is: ", batch_loss)
    except Exception as e:
        print(predicted_joint_positions.shape)
        print(label.shape)
        input("710")
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
