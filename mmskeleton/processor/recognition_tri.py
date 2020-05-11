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


def test(model_cfg, dataset_cfg, checkpoint, batch_size=64, gpus=1, workers=4):
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    results = []
    labels = []
    prog_bar = ProgressBar(len(dataset))
    for data, label in data_loader:
        with torch.no_grad():
            output = model(data).data.cpu().numpy()
        results.append(output)
        labels.append(label)
        for i in range(len(data)):
            prog_bar.update()
    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


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
):
    print("ANDREA - TRI-recognition")
    id_mapping = {27:25, 33:31, 34:32, 37:35, 39:37,
                  46:44, 47:45, 48:46, 50:48, 52:50, 
                  55:53, 57:55, 59:57, 66:63}


    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]

    print("==================================")
    # print(dataset_cfg[0])
    assert len(dataset_cfg) == 1
    data_dir = dataset_cfg[0]['data_source']['data_dir']
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for test_id in test_ids:
        ambid = id_mapping[test_id]

        test_walks = [i for i in all_files if re.search('ID_'+str(test_id), i) ]
        non_test_walks = list(set(all_files).symmetric_difference(set(test_walks)))
    
        datasets = [copy.deepcopy(dataset_cfg[0]), copy.deepcopy(dataset_cfg[0])]
        datasets[0]['data_source']['data_dir'] = non_test_walks
        datasets[1]['data_source']['data_dir'] = test_walks
        work_dir_amb = work_dir + "/" + str(ambid)
        things_to_log = {'test_AMBID': ambid, 'test_AMBID_num': len(test_walks), 'model_cfg': model_cfg, 'loss_cfg': loss_cfg, 'optimizer_cfg': optimizer_cfg, 'batch_size': batch_size, 'total_epochs': total_epochs }

        train_model(
                work_dir_amb,
                model_cfg,
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
                things_to_log)

        continue
        if len(test_walks) == 0:
            continue
        
        # Split the non_test walks into train/val
        kf = KFold(n_splits=cv, shuffle=True)
        kf.get_n_splits(non_test_walks)

        for train_ids, val_ids in kf.split(non_test_walks):
            train_walks = [non_test_walks[i] for i in train_ids]
            val_walks = [non_test_walks[i] for i in val_ids]

            # print(len(train_walks), len(val_walks), len(test_walks))

            datasets = [copy.deepcopy(dataset_cfg[0]), copy.deepcopy(dataset_cfg[0])]
            datasets[0]['data_source']['data_dir'] = train_walks
            datasets[1]['data_source']['data_dir'] = val_walks

            train_model(
                    work_dir,
                    model_cfg,
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
                    load_from)


def train_model(
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
        
):
    # print(all_files)
    print("==================================")

    # print(datasets)

    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=False) for d in datasets
    ]

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    model.apply(weights_init)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    loss = call_obj(**loss_cfg)


    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level, things_to_log=things_to_log)
    runner.register_training_hooks(**training_hooks)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    # [('train', 5), ('val', 1)]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)


# process a batch of data
def batch_processor(model, datas, train_mode, loss):
    data, label = datas
    data = data.cuda()
    label = label.cuda()
    # forward
    output = model(data)
    losses = loss(output, label)
    rank = output.argsort()
    # print(output, rank)

    # output
    log_vars = dict(loss=losses.item())
    # if not train_mode:
    #     log_vars['top1'] = topk_accuracy(output, label)
    #     log_vars['top5'] = topk_accuracy(output, label, 5)

    labels = dict(true=label.data.tolist(), pred=rank[:,-1].data.tolist())
    outputs = dict(loss=losses, log_vars=log_vars, num_samples=len(data.data))
    return outputs, labels


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


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
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
