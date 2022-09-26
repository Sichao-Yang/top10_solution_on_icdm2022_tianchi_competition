DEBUG = True

import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import os
from datetime import datetime
import gc
import random
import json
import shutil
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch.nn.functional as F
from lib_dgl.data import PrepareData
from lib_dgl.utils import (gen_output_torch, train_rlu, gen_model, get_n_params, train_v2, val_v2, check_cv_nids,
                           final_res, train_pseudo_v1, train_pseudo_v2, get_git_revision_hash, zipDir, get_logger, FocalLoss)
from lib_dgl.plots import Expplots
from lib_dgl.argparser import ArgsParser

def env_set(seed=42):
    # for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    # Setup log
    log = get_logger(f'run_{current_time}.log',method='w2file')
    return log, current_time

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step+1 < num_warmup_steps + num_wait_steps:
            return float(current_step+1) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step+1 - num_warmup_steps - num_wait_steps) / \
                float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_dataloader(stage, train_nid, val_nid, test_nid, args, enhance_idx=None):
    if train_nid is not None:
        if stage == 0:
            train_loader = torch.utils.data.DataLoader(
                        torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        else:
            if args.use_rlu:
                train_loader = torch.utils.data.DataLoader(
                            torch.arange(len(train_nid)), 
                            batch_size=int(args.batch_size*len(train_nid)/(len(enhance_idx)+len(train_nid))), 
                            shuffle=True, drop_last=False)
            else:
                train_loader = torch.utils.data.DataLoader(
                            torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid),len(train_nid)+len(val_nid)),
                batch_size=args.batch_size, shuffle=False, drop_last=False)
        all_loader = torch.utils.data.DataLoader(
                torch.arange(len(train_nid)+len(val_nid)+len(test_nid)), 
                batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        train_loader = val_loader = None
        all_loader = torch.utils.data.DataLoader(
                torch.arange(len(test_nid)), 
                batch_size=args.batch_size, shuffle=False, drop_last=False)
        
    return train_loader, val_loader, all_loader

def training(args, stage, epoch, epochs, ema_step, model, teacher_model, feats, labels, label_emb, loss_fcn, optimizer, 
             train_loader, enhance_loader, all_loader, predict_prob, log=None, pseudo_labels=None):
    train_node_num = len(train_loader.dataset)
    if args.use_rlu:
        if stage == 0:
            train_loss, train_acc, train_ap=train_v2(model, feats, labels, label_emb, loss_fcn, optimizer, train_loader, args.use_rlu)
        else:
            train_loss, train_acc, train_ap=train_rlu(model, feats, labels, label_emb, loss_fcn, optimizer, 
                                                    train_loader, enhance_loader, predict_prob, args.gamma)
    elif args.use_pseudo:
        if stage == 0:
            train_loss, train_acc, train_ap=train_v2(model, feats, labels, label_emb, loss_fcn, optimizer, train_loader, args.use_rlu)
        else:
            enhance_node_num = len(enhance_loader.dataset)
            # make combo dataloaders
            if train_node_num != 0:
                train_loader = torch.utils.data.DataLoader(torch.arange(train_node_num), 
                                                        batch_size=int(args.batch_size*train_node_num/(enhance_node_num+train_node_num)),
                                                        shuffle=True, drop_last=False)
            if args.use_scr:
                if args.adap:
                    alpha = min(1 - 1 /(ema_step+1), args.ema_decay)
                else:
                    alpha = args.ema_decay
                for mean_param, param in zip(teacher_model.parameters(), model.parameters()):
                    mean_param.data.mul_(alpha).add_(1-alpha, param.data)
            
            if args.adap:
                args.pseudo_lam = min((ema_step+1)/args.ramp_epochs[stage]*args.pseudo_lam_max, args.pseudo_lam_max)
            train_loss, train_acc, train_ap=train_pseudo_v2(model, teacher_model, feats, labels, device, loss_fcn, optimizer, 
                                                        train_loader, enhance_loader, label_emb, args, pseudo_labels, log=log)
            ema_step += 1
    else:
        train_loss, train_acc, train_ap=train_v2(model, feats, labels, label_emb, loss_fcn, optimizer, train_loader, args.use_rlu)    
    
    return train_loss, train_acc, train_ap, ema_step


def get_models(subset_dim, feat_dim, num_classes):
    teacher_models = []
    for i in range(args.kfold):
        m = gen_model(args, subset_dim, feat_dim, num_classes)
        fp = osp.join(args.out_dir, '{}_{}_{}.pkl'.format(args.cv_id, args.method, stage-1))
        m.load_state_dict(torch.load(fp))
        if i == args.cv_id:
            model = m.to(device)
        else:
            teacher_models.append(m.to(device))
    return model, teacher_models

def get_pseudo_labels(args, stage, masked_node_num, train_node_num, 
                      subset_dim, feat_dim, num_classes, all_loader, feats, label_emb, log):
    model, teacher_models = get_models(subset_dim, feat_dim, num_classes)
    preds = []
    for m in teacher_models:
        preds.append(gen_output_torch(m, feats, all_loader, device, label_emb, args.use_rlu).unsqueeze(dim=2))
    preds = torch.mean(torch.cat(preds, dim=-1), dim=-1)
    pseudo_labels = preds.argmax(dim=1).to(device)

    prob_teacher = preds.softmax(dim=1)
    # select pseudo labels by class-specific thresholds, higher at begining
    thresholds = []
    for top, down in zip(args.tops, args.downs):
        thresholds.append(top - (top-down)*stage/len(args.stages))
    while True:
        for i, th in enumerate(thresholds):
            if i == 0:
                confident_mask = [prob_teacher[:,i] > th][0]
            else:
                tmp = [prob_teacher[:,i] > th][0]
                confident_mask = torch.logical_or(tmp, confident_mask)
        confident_nid = torch.arange(len(prob_teacher))[confident_mask]
        enhance_idx = confident_nid[confident_nid >= masked_node_num]
        if len(enhance_idx)>100:
            break
        else:
            thresholds = [x-0.02 for x in thresholds]
    
    log.info(f'Pseudo Label thresholds set to: {thresholds}')
    log.info(f"At stage{stage}, selected {enhance_idx.shape[0]} pseudo labels from {len(prob_teacher)-masked_node_num} unlabeled data")
    # make enhance_loader by fix total batch_size
    enhance_loader = torch.utils.data.DataLoader(enhance_idx, 
                                                batch_size=int(args.batch_size*len(enhance_idx)/(len(enhance_idx)+train_node_num)), 
                                                shuffle=True, drop_last=False)
    del preds, teacher_models
    torch.cuda.empty_cache()
    return pseudo_labels, model, enhance_loader

def get_lossfcn(args):
    if args.loss_fcn == 'focal':
        return FocalLoss(gamma=args.focal_gamma, alpha=None)
    elif args.loss_fcn == 'ce':
        return nn.CrossEntropyLoss(weight=args.class_weight)

def get_opt(args, model):
    if args.opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.opt == 'sgd':
        return torch.optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.l2, 
                            nesterov=args.nesterov)

def get_sched(args, optimizer, warmup_steps=10, total_epochs=1000):
    if args.sched == 'plateau':
        return ReduceLROnPlateau(optimizer, 'max', patience=args.stop_patience//2, factor=0.1)
    elif args.sched == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_epochs)


def Pipeline(device, args, stage, epochs, log, logsum, dataset=None):

    log.info(f'\n\nCV_{args.cv_id} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    log.info(f'Stage{stage} >>>>>>>>>>>>>>>>>')
    expP = Expplots(dir=args.out_dir)
    # get enhance loader for rlu
    teacher_probs = enhance_idx = predict_prob = enhance_loader = pseudo_labels = None

    log.info(f'Start dataloading and preprocessing')
    # Load datasets
    # preprocess - feature & label msg passing
    if stage==0 and args.cv_id==0:
        dataset = PrepareData(device, args, teacher_probs, log)
        feats, labels, label_emb, num_classes, train_nid, val_nid, test_nid, rev_item_map = dataset.dat
    else:
        dataset.update(args.cv_id, stage)
        feats, labels, label_emb, num_classes, train_nid, val_nid, test_nid, rev_item_map = dataset.dat
    torch.cuda.empty_cache()
    gc.collect()

    train_loader, val_loader, all_loader = get_dataloader(stage, train_nid, val_nid, test_nid, args, enhance_idx)
    # Create models
    log.info("Make model and Set-up traning env")
    _, subset_dim, feat_dim = feats[0].shape
    if stage != 0:
        if args.use_pseudo:
            log.info("use pseudo labels")
            train_node_num = len(train_loader.dataset)
            masked_node_num = train_node_num+len(val_loader.dataset)
            pseudo_labels, model, enhance_loader = get_pseudo_labels(args, stage, masked_node_num, train_node_num, subset_dim, 
                                                    feat_dim, num_classes, all_loader, feats, label_emb, log)
        else:
            model, _ = get_models(subset_dim, feat_dim, num_classes)
    else:
        model = gen_model(args, subset_dim, feat_dim, num_classes).to(device)
        pseudo_labels = None
    #     from torchinfo import summary
    #     summary(model)
    log.info("Params: {}".format(get_n_params(model)))
    teacher_model = None
    # for consistensy loss2
    if args.use_scr:
        teacher_model = gen_model(args, subset_dim, feat_dim, num_classes)
        teacher_model = teacher_model.to(device)
        for param in teacher_model.parameters():
            param.detach_()

    # Set loss fcns, optimizer, scheduler
    loss_fcn = get_lossfcn(args)
    optimizer = get_opt(args, model)
    scheduler = get_sched(args, optimizer, warmup_steps=args.warmup_epochs, total_epochs=epochs)
    
    # start training
    log.info("Start training")
    best_epoch = 0
    max_val_ap = 0
    no_improve = 0    
    ema_step = 0
    for epoch in range(1,epochs+1):
        gc.collect()
        # train - three methods: rlu, scr or plain traning
        train_loss, train_acc, train_ap, ema_step = training(args, stage, epoch, epochs, ema_step, model, teacher_model, feats, labels, label_emb, 
                                                        loss_fcn, optimizer, train_loader, enhance_loader, all_loader, predict_prob, log=log, 
                                                        pseudo_labels=pseudo_labels)
        gc.collect()
        log.info(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}')
        # evaluate
        # if epoch % args.eval_every == 0:
        val_loss, val_acc, val_ap = val_v2(model, feats, labels, label_emb, loss_fcn, val_loader, args.use_rlu)
        log.info(f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}')
        # save best model
        if args.early_stopping and no_improve>args.stop_patience:
            log.info("Early Stopping")
            break
        if val_ap > max_val_ap:
            max_val_ap = val_ap
            best_epoch = epoch
            fp = osp.join(args.out_dir, '{}_{}_{}.pkl'.format(args.cv_id, args.method, stage))
            torch.save(model.state_dict(), fp)
            log.info('Save best_model on val_AP: {}\n'.format(max_val_ap))
            no_improve=0
        else:
            no_improve+=1
        if args.sched == 'plateau':
            scheduler.step(val_ap)
        else:
            scheduler.step()
        # draw training results
        # self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        lr = optimizer.param_groups[0]['lr']
        log.info(f'learning rate: {lr}')
        losses = {'train': train_loss, 'valid': val_loss}
        aps = {'train': train_ap, 'valid': val_ap}
        expP.new_epoch_draw(aps, losses, lr, epoch, tag=str(args.cv_id)+'_'+str(stage))
    
    # save best model, training done
    tmp = "Stage {} CV {}: Best Epoch {}, Val_AP {:.4f}".format(stage, args.cv_id, best_epoch, max_val_ap)
    log.info(tmp)
    logsum.append(tmp)
    model.load_state_dict(torch.load(fp))
    
    # Infernce - save test result or not
    if stage < len(args.stages)-1:
        del feats, label_emb, labels
        gc.collect()
    else:
        preds = gen_output_torch(model, feats, all_loader, labels.device, label_emb, args.use_rlu)
        log.info(f"Saving result on CV_{args.cv_id} >>>>>>>")
        pred_test = preds[len(train_nid)+len(val_nid):len(train_nid)+len(val_nid)+len(test_nid)]
        pred_test = F.softmax(pred_test, dim=1)[:, 1].numpy()
        test_nid = test_nid.cpu().numpy()
        # get original test_id
        ori = []
        for i, key in enumerate(test_nid):
            ori.append(rev_item_map[key])
        # save to submit file & store
        with open(osp.join(args.out_dir, f"{args.cv_id}_submit.json"), 'w+') as f:
            for i in range(len(ori)):
                y_dict = {}
                y_dict["item_id"] = int(ori[i])
                y_dict["score"] = float(pred_test[i])
                json.dump(y_dict, f)
                f.write('\n')
        log.info(f"Done on CV_{args.cv_id}!")
    
    return dataset, logsum


def get_paths(args, log):
    args.code_root = osp.abspath(osp.dirname(__file__))
    args.tmp_dir = osp.abspath(osp.join(osp.dirname(__file__), '../tmp_feats'))
    args.data_root = osp.abspath(osp.join(osp.dirname(__file__), '../dataset'))
    args.out_dir = osp.abspath(osp.join(osp.dirname(__file__), '../output'))
    if osp.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
        log.info("Found existing out_dir, delete & make new one")
    os.mkdir(args.out_dir)
    assert osp.isdir(args.data_root)
    return args

def debug_args(args):
    args.num_hidden = 24
    args.num_hops = 3
    args.stages = [2,2,2,2]
    args.ramp_epochs = [1,1,1,2]
    args.relation_subset_path = 'config/subsets/icdm2022_rand_subsets_debug'
    args.data_root = osp.join(args.data_root, 'sample')
    args.use_pseudo = True
    args.tops = [0.9,0.9]
    args.downs = [0.8,0.8]
    return args


if __name__ == "__main__":

    args = ArgsParser(description="ICDM2022").parse_args()
    log, current_time = env_set(seed=args.seed)
    args = get_paths(args, log)
    
    log.info(args)
    # log.info('This run\'s git commit id: '+get_git_revision_hash())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.class_weight = torch.tensor(args.class_weight).to(device)
    ############################################### DEBUG
    if DEBUG:
        args = debug_args(args)

    dataset = None
    if args.cv_id == -1:
        check_cv_nids(device, args, log)
    
    logsum = []
    for stage, epochs in enumerate(args.stages):
        for cv in range(args.kfold):
            args.cv_id = cv
            dataset, logsum = Pipeline(device, args, stage, epochs, log, logsum, dataset=dataset)
            torch.cuda.empty_cache()
            gc.collect()
    selected_cv = final_res(args, logsum)
    log.info('\n'.join(logsum))
    log.info(f'The final result is selected from cv {selected_cv}')
    
    # for result-reproduce purpurse
    zipDir(dirpath=osp.abspath(osp.dirname(__file__)), outFullName=osp.join(args.out_dir, 'code.zip'), log=log)
    shutil.copyfile(src=log.handlers[0].baseFilename, dst=osp.join(args.out_dir, 'run.log'))
    shutil.copyfile(src=osp.join(args.code_root, 'config/dgl_base.yml'), dst=osp.join(args.code_root, f'exp_{current_time}.yml'))