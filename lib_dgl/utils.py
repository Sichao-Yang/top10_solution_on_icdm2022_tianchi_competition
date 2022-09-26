from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import json
from sklearn.metrics import average_precision_score
import sys
import subprocess
import zipfile
import os
import logging
from os import path as osp
p = osp.abspath(osp.join(osp.dirname(__file__),'.'))
sys.path.append(p)
from model import *
from data import load_icdm
###############################################################################
# small utilities
###############################################################################
def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def zipDir(dirpath, outFullName, log):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        log.info(fpath)
        log.info(path)
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

def get_logger(filename, verb_level='info', name=None, method=None):
    level_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARNING}
    formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verb_level])

    if method == 'w2file':
        fh = logging.FileHandler(filename, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def check_cv_nids(device, args, log):
    cv_dir = osp.join(args.tmp_dir, args.relation_subset_path.split('/')[-1])
    if os.path.exists(cv_dir):
        if len([x for x in os.listdir(cv_dir) if 'cv' in x]) == args.kfold:
            return logging.info('Found cv_nids, check passed')
        else:
            os.makedirs(cv_dir,exist_ok=True)
    logging.info('Found no cv_nids, start cv making')
    stat = load_icdm(device, args, session='session1')
    if stat == 0:
        log.info(f'Finished cv making, find them at {cv_dir}')


def final_res(args, logsum):
    final_stage_res = [x for x in logsum if f'Stage {len(args.stages)-1}' in x]
    vap = [float(x.split(' ')[-1]) for x in final_stage_res]
    cv = np.argmax(np.array(vap))
    os.rename(src=osp.join(args.out_dir, f"{cv}_submit.json"),dst=osp.join(args.out_dir, f"final_submit.json"))
    return cv

###############################################################################
# Training and testing for one epoch
###############################################################################
def train(model, feats, labels, train_nid, optimizer, batch_size, device, class_weight, history=None):
    model.train()
    train_loader = torch.utils.data.DataLoader(
                    train_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    
    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Training')
    total_loss = total_correct = total_examples = 0
    y_true = []
    y_pred = []
    for batch in train_loader:
        batch = batch.long()
        batch_feats = [x[batch].to(device) for x in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [x[batch].to(device) for x in history])
        y_hat = model(batch_feats)
        y = labels[batch].to(device)
        loss = F.cross_entropy(y_hat, y, weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += loss.item() * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        del y, y_hat, batch_feats
        torch.cuda.empty_cache()
        pbar.update(batch_size)
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ap_score

@torch.no_grad()
def val(model, feats, labels, val_nid, batch_size, device, class_weight, history=None):
    model.eval()
    device = labels.device
    val_loader = torch.utils.data.DataLoader(
                val_nid, batch_size=batch_size, shuffle=False, drop_last=False)
    
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Validating')
    total_loss = total_correct = total_examples = 0
    y_true = []
    y_pred = []
    for batch in val_loader:
        batch = batch.long()
        batch_feats = [feat[batch].to(device) for feat in feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [x[batch].to(device) for x in history])
        y_hat = model(batch_feats)
        y = labels[batch].to(device)
        loss = F.cross_entropy(y_hat, y, weight=class_weight)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += loss.item() * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        del y, y_hat, batch_feats
        torch.cuda.empty_cache()
        pbar.update(batch_size)
    pbar.close()

    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
    return total_loss / total_examples, total_correct / total_examples, ap_score


def train_v2(model, feats, labels, label_emb, loss_fcn, optimizer, train_loader, use_rlu):
    model.train()
    device = labels.device
    # pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    # pbar.set_description(f'Training')
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        if use_rlu:
            output_att=model(batch_feats,label_emb[batch].to(device))
        else:
            output_att=model(batch_feats)
        L1 = loss_fcn(output_att, labels[batch])
        loss_train = L1
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        y_true.append(labels[batch].cpu())
        y_pred.append(F.softmax(output_att, dim=1)[:, 1].detach().cpu())        
        total_loss += loss_train.item() * len(batch)
        total_correct += int((output_att.argmax(dim=-1) == labels[batch]).sum())
        total_examples += len(batch)
        del output_att, batch_feats
        torch.cuda.empty_cache()        
    #     pbar.update(len(batch))
    # pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss/total_examples, total_correct/total_examples, ap_score

@torch.no_grad()
def val_v2(model, feats, labels, label_emb, loss_fcn, val_loader, use_rlu):
    model.eval()
    device = labels.device

    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Validating')
    total_loss = total_correct = total_examples = 0
    y_true = []
    y_pred = []
    for batch in val_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if use_rlu:
            y_hat = model(batch_feats, label_emb[batch].to(device))
        else:
            y_hat = model(batch_feats)
        y = labels[batch].to(device)
        loss = loss_fcn(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += loss.item() * len(batch)
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += len(batch)
        del y, y_hat, batch_feats
        torch.cuda.empty_cache()
        pbar.update(len(batch))
    pbar.close()

    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
    return total_loss/total_examples, total_correct/total_examples, ap_score


def train_pseudo_v1(model, teacher_model, feats, labels, device, loss_fcn, optimizer, 
                train_loader, enhance_loader, label_emb, args, pseudo_labels, log=None):
    model.train()
    # pseudo loss are smoothed
    loss_fcn2 = LabelSmoothingCrossEntropy(epsilon=0.1)
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    if train_loader is not None:
        pbar = tqdm(total=int(len(train_loader.dataset)+len(enhance_loader.dataset)), ascii=True)
        pbar.set_description(f'Training with pseudo labels')
        for idx_1, idx_2 in zip(train_loader, enhance_loader):
            idx = torch.cat((idx_1, idx_2), dim=0)
            feat_list = [x[idx].to(device) for x in feats]
            if args.use_rlu:
                output_att=model(feat_list,label_emb[idx].to(device))
            else:
                output_att=model(feat_list)
            L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1].to(device))
            L2 = loss_fcn2(output_att[len(idx_1):], pseudo_labels[idx_2])
            loss_train =L1*args.sup_lam + L2*args.pseudo_lam
            if args.use_scr:
                feat_list_teacher = [x[idx_2].to(device) for x in feats]
                with torch.no_grad():
                    if args.use_rlu:
                        p_t = teacher_model(feat_list_teacher,label_emb[idx_2].to(device))
                    else:
                        p_t = teacher_model(feat_list_teacher)
                p_s = output_att[len(idx_1):]
                loss_consis = consis_loss(p_t, p_s, args.temp, args.unsup_losstype)
                loss_train += loss_consis*args.unsup_lam
                del feat_list_teacher, p_s, p_t
        
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            y_true.append(labels[idx_1].cpu())
            y_pred.append(F.softmax(output_att[:len(idx_1)], dim=1)[:len(idx_1), 1].detach().cpu())
            total_loss += loss_train.item() * len(idx)
            total_correct += int((output_att[:len(idx_1)].argmax(dim=-1) == labels[idx_1]).sum())
            total_examples += len(idx_1)
            del output_att, feat_list
            torch.cuda.empty_cache() 
            pbar.update(len(idx_1)+len(idx_2))
        pbar.close()
        ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
        
    return total_loss/total_examples, total_correct/total_examples, ap_score


def train_pseudo_v2(model, teacher_model, feats, labels, device, loss_fcn, optimizer, 
                train_loader, enhance_loader, label_emb, args, pseudo_labels, log=None):
    model.train()
    loss_fcn2 = LabelSmoothingCrossEntropy(epsilon=0.1)
    #teacher_model.train()
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    do_w = nn.Dropout(args.do_w)
    if train_loader is not None:
        # pbar = tqdm(total=int(len(train_loader.dataset)+len(enhance_loader.dataset)), ascii=True)
        # pbar.set_description(f'Training with pseudo labels')
        for idx_1, idx_2 in zip(train_loader, enhance_loader):     
            feat_list_s = [torch.cat([x[idx_1].to(device), do_w(x[idx_1]).to(device), x[idx_2].to(device)], dim=0) for x in feats]
            feat_list_t = [x[idx_2].to(device) for x in feats]
            output_s = model(feat_list_s)
            loss_l = loss_fcn(output_s[:len(idx_1)], labels[idx_1].to(device))
            loss_pl = loss_fcn2(output_s[2*len(idx_1):], pseudo_labels[idx_2])
            if args.use_dcr:
                loss_consis = consis_loss(output_s[:len(idx_1)], output_s[len(idx_1):2*len(idx_1)], args.temp, args.unsup_losstype)
            else:
                loss_consis = 0
            if args.use_scr:
                with torch.no_grad():
                    p_t = teacher_model(feat_list_t)
                p_s = output_s[2*len(idx_1):]
                loss_consis2 = consis_loss(p_t, p_s, args.temp, args.unsup_losstype)
                loss_consis += loss_consis2
            if args.use_dcr and args.use_scr:
                loss_consis = loss_consis/2
            loss_train = loss_l*args.sup_lam + loss_pl*args.pseudo_lam + loss_consis*args.unsup_lam
        
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            y_true.append(labels[idx_1].cpu())
            y_pred.append(F.softmax(output_s[:len(idx_1)], dim=1)[:len(idx_1), 1].detach().cpu())
            total_loss += loss_train.item() * len(idx_1)
            total_correct += int((output_s[:len(idx_1)].argmax(dim=-1) == labels[idx_1]).sum())
            total_examples += len(idx_1)
            del output_s, feat_list_s
            torch.cuda.empty_cache() 
        #     pbar.update(len(idx_1)+len(idx_2))
        # pbar.close()
        ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss/total_examples, total_correct/total_examples, ap_score


def train_pseudo_pretrain(student_model, teacher_model, feats, labels, device, loss_fcn, s_opt, t_opt, 
                    labeled_loader, unlabeled_loader, args, thresholds, log=None):

    student_model.train()
    teacher_model.train()
    loss_fcn2 = LabelSmoothingCrossEntropy(epsilon=0.1)
    #teacher_model.train()
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    do_s = nn.Dropout(args.do_s)
    do_w = nn.Dropout(args.do_w)

    pbar = tqdm(total=int(len(labeled_loader.dataset)*2), ascii=True)
    pbar.set_description(f'Training with pseudo labels')
    ul_iter = iter(unlabeled_loader)
    tot_selected = torch.tensor(0).to(device)
    for idx_1 in labeled_loader:
        try:
            idx_2 = ul_iter.next()
        except:
            ul_iter = iter(unlabeled_loader)
            idx_2 = ul_iter.next()
        batch_size = len(idx_1)
        feats_t = [torch.cat([x[idx_1].to(device), do_w(x[idx_2]).to(device), do_s(x[idx_2]).to(device)], dim=0) for x in feats]
        t_logits = teacher_model(feats_t)
        t_logits_l = t_logits[:batch_size]
        t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
        del t_logits

        t_loss_l = loss_fcn(t_logits_l, labels[idx_1].to(device))
        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temp, dim=-1)
        _, hard_pseudo_label  = torch.max(soft_pseudo_label, dim=-1)       
        while True:
            for i, th in enumerate(thresholds):
                if i == 0:
                    confident_mask = [soft_pseudo_label[:,i] > th][0]
                else:
                    tmp = [soft_pseudo_label[:,i] > th][0]
                    mask = torch.logical_or(tmp, confident_mask)
            if torch.sum(mask)>100:
                break
            else:
                thresholds = [x-0.02 for x in thresholds]
        
        tot_selected+=torch.sum(mask)
        
        a = consis_loss(t_logits_us[mask], t_logits_uw[mask], args.temp, 'mse')
        b = loss_fcn(t_logits_us[mask], hard_pseudo_label)
        
        t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask.float())
        t_loss = t_loss_l + args.pseudo_lam*t_loss_u
        
        feats_s = [do_s(x[idx_2]).to(device) for x in feats]
        s_logits_us = student_model(feats_s)
        s_loss = loss_fcn2(s_logits_us[mask], hard_pseudo_label[mask])
    
        s_opt.zero_grad()
        s_loss.backward()
        s_opt.step()
        t_opt.zero_grad()
        t_loss.backward()
        t_opt.step()
        
        y_true.append(labels[idx_1].cpu())
        y_pred.append(F.softmax(t_logits_l, dim=1)[:, 1].detach().cpu())
        total_loss += t_loss_l.item() * len(idx_1)
        total_correct += int((t_logits_l.argmax(dim=-1) == labels[idx_1]).sum())
        total_examples += len(idx_1)
        torch.cuda.empty_cache() 
        pbar.update(len(idx_1)+len(idx_2))
    pbar.close()
    log.info(f"Selected {tot_selected} pseudo labels from {len(labeled_loader.dataset)} unlabeled data")
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
    del do_s, do_w
    return total_loss/total_examples, total_correct/total_examples, ap_score

def train_pseudo_finetune(student_model, feats, labels, device, loss_fcn, s_opt, 
                    labeled_loader, unlabeled_loader, args, thresholds, log=None):

    student_model.train()
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]

    do_w = nn.Dropout(args.do_w)

    pbar = tqdm(total=int(len(labeled_loader.dataset)*2), ascii=True)
    pbar.set_description(f'FineTune with pseudo labels')
    ul_iter = iter(unlabeled_loader)
    tot_selected = torch.tensor(0).to(device)
    for idx_1 in labeled_loader:
        try:
            idx_2 = ul_iter.next()
        except:
            ul_iter = iter(unlabeled_loader)
            idx_2 = ul_iter.next()
        batch_size = len(idx_1)
        
        feats_s = [torch.cat([x[idx_1].to(device), x[idx_2].to(device), do_w(x[idx_2]).to(device)]) for x in feats]
        s_logits = student_model(feats_s)
        s_logits_l = s_logits[:batch_size]
        s_logits_u, s_logits_uw = s_logits[batch_size:].chunk(2)
        del s_logits
        s_loss_l = loss_fcn(s_logits_l, labels[idx_1].to(device))
        soft_pseudo_label = torch.softmax(s_logits_u.detach() / args.temp, dim=-1)

        for i, th in enumerate(thresholds):
            if i == 0:
                mask = [soft_pseudo_label[:,i] > th][0]
            else:
                tmp = [soft_pseudo_label[:,i] > th][0]
                mask = torch.logical_or(tmp, mask)
        tot_selected+=torch.sum(mask)
        mask = mask.float()
        s_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(s_logits_uw, dim=-1)).sum(dim=-1) * mask)
        s_loss = s_loss_l + args.pseudo_lam*s_loss_u
    
        s_opt.zero_grad()
        s_loss.backward()
        s_opt.step()
        
        y_true.append(labels[idx_1].cpu())
        y_pred.append(F.softmax(s_logits_l, dim=1)[:, 1].detach().cpu())
        total_loss += s_loss_l.item() * len(idx_1)
        total_correct += int((s_logits_l.argmax(dim=-1) == labels[idx_1]).sum())
        total_examples += len(idx_1)
        torch.cuda.empty_cache() 
        pbar.update(len(idx_1)+len(idx_2))
    pbar.close()
    log.info(f"Selected {tot_selected} pseudo labels from {len(labeled_loader.dataset)} unlabeled data")
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss/total_examples, total_correct/total_examples, ap_score


def train_scr(model, teacher_model, feats, labels, device, loss_fcn, optimizer, 
                train_loader, enhance_loader, label_emb, args):
    model.train()
    #teacher_model.train()
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    pbar = tqdm(total=int(len(train_loader.dataset)+len(enhance_loader.dataset)), ascii=True)
    pbar.set_description(f'Training SCR')
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        feat_list = [x[idx].to(device) for x in feats]
        feat_list_teacher = [x[idx_2].to(device) for x in feats]
        if args.use_rlu:
            output_att=model(feat_list,label_emb[idx].to(device))
        else:
            output_att=model(feat_list)
        L1 = loss_fcn(output_att[:len(idx_1)], labels[idx_1].to(device))
        loss_supervised = args.sup_lam*L1
        with torch.no_grad():
            if args.use_rlu:
                p_t = teacher_model(feat_list_teacher,label_emb[idx_2].to(device))
            else:
                p_t = teacher_model(feat_list_teacher)
        p_s = output_att[len(idx_1):]
        loss_consis = consis_loss(p_t, p_s, args.temp, args.unsup_losstype)
        if args.unsup_losstype == 'mse':
            loss_train = loss_supervised + loss_consis*args.lam
        elif args.unsup_losstype == 'kl':
            loss_train = loss_supervised + loss_consis*args.kl_lam  
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        y_true.append(labels[idx_1].cpu())
        y_pred.append(F.softmax(output_att[:len(idx_1)], dim=1)[:len(idx_1), 1].detach().cpu())
        total_loss += loss_train.item() * len(idx)
        total_correct += int((output_att[:len(idx_1)].argmax(dim=-1) == labels[idx_1]).sum())
        total_examples += len(idx_1)
        del output_att, feat_list_teacher, feat_list, p_s, p_t
        torch.cuda.empty_cache() 
        pbar.update(len(idx_1)+len(idx_2))
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())

    return total_loss/total_examples, total_correct/total_examples, ap_score


def train_rlu(model, feats, labels, label_emb, loss_fcn, optimizer, 
            train_loader, enhance_loader, predict_prob, gamma):
    model.train()
    device = labels.device
    pbar = tqdm(total=int(len(train_loader.dataset)+len(enhance_loader.dataset)), ascii=True)
    pbar.set_description(f'Training')
    total_loss = total_correct = total_examples = 0
    y_true=[]
    y_pred=[]
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        batch = torch.cat((idx_1, idx_2), dim=0)
        batch_feats = [x[batch].to(device) for x in feats]
        y = labels[idx_1].to(torch.long).to(device)
        optimizer.zero_grad()
        output_att= model(batch_feats, label_emb[batch].to(device))
        ratio = len(idx_1)*1.0/(len(idx_1)+len(idx_2))
        L1 = loss_fcn(output_att[:len(idx_1)],  y)*ratio
        teacher_soft = predict_prob[idx_2].to(device)
        teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
        L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)
                                          -torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*ratio
        loss_train = L1 + L3*gamma
        loss_train.backward()
        optimizer.step()
        y_true.append(labels[idx_1].cpu())
        y_pred.append(F.softmax(output_att, dim=1)[:len(idx_1), 1].detach().cpu())        
        total_loss = loss_train.item()
        total_correct += int((output_att[:len(idx_1)].argmax(dim=-1) == labels[idx_1]).sum())
        total_examples += len(batch)
        del output_att, batch_feats
        torch.cuda.empty_cache()        
        pbar.update(len(batch))
    pbar.close()
    ap_score = average_precision_score(torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy())
    return total_loss/total_examples, total_correct/total_examples, ap_score


@torch.no_grad()
def gen_output_torch(model, feats, all_loader, device, label_emb, use_rlu):
    model.eval()
    preds = []
    for batch in all_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        if use_rlu:
            preds.append(model(batch_feats,label_emb[batch].to(device)).cpu())
        else:
            preds.append(model(batch_feats).cpu())
    preds = torch.cat(preds, dim=0)
    return preds
###############################################################################
# Evaluator for different datasets
###############################################################################
def batched_acc(pred, labels):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)


def ap_score(pred, labels):
    return average_precision_score(torch.hstack(labels).numpy(), torch.hstack(pred).numpy())


def get_evaluator():
    # return batched_acc
    return ap_score


def compute_mean(metrics, nid):
    num_nodes = len(nid)
    return [m[nid].float().sum().item() / num_nodes for m in metrics]


def consis_loss(p_t, p_s, temp, losstype):
    sharp_p_t = F.softmax(p_t/temp, dim=1)
    p_s = F.softmax(p_s,dim=1)
    if losstype == 'mse':
        return F.mse_loss(sharp_p_t, p_s, reduction='mean')
        # return torch.mean(torch.pow(p_s - sharp_p_t, 2))
    elif losstype == 'kl':
        log_sharp_p_t = torch.log(sharp_p_t+1e-8)
        return F.kl_div(log_sharp_p_t, p_s, reduction = 'mean')
        # return torch.mean(p_s * (torch.log(p_s+1e-8) - log_sharp_p_t))


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def _linear_combination(self, x, y):
        return self.epsilon*x + (1-self.epsilon)*y

    def _reduce_loss(self, loss, reduction='mean'):
        if reduction == 'mean':
            return loss.mean()  
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, preds, target):
        num_class = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self._linear_combination(loss/num_class, nll)
    

from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
###############################################################################
# Model creator
###############################################################################
def gen_model(args,num_feats,feat_dim,num_classes):
    if args.method=="NARS_R_GAMLP":
        if args.use_rlu:
            return NARS_R_GAMLP_RLU(feat_dim, args.num_hidden, num_classes, args.num_hops+1,
                                    num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3,
                                    args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop,
                                    args.pre_process, args.residual, args.pre_dropout, args.bns)
        else:
            return NARS_R_GAMLP(feat_dim, args.num_hidden, num_classes, args.num_hops+1, num_feats, 
                                args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, 
                                args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, 
                                args.pre_process, args.residual, args.pre_dropout, args.bns)
    elif args.method=="NARS_JK_GAMLP":
        if args.use_rlu:
            return NARS_JK_GAMLP_RLU(feat_dim, args.num_hidden, num_classes, args.num_hops+1,
                                    num_feats, args.alpha, args.n_layers_1, args.n_layers_2,
                                    args.n_layers_3, args.act, args.dropout, args.input_drop, 
                                    args.att_drop, args.label_drop, args.pre_process,
                                    args.residual, args.pre_dropout, args.bns)
        else:
            return NARS_JK_GAMLP(feat_dim, args.num_hidden, num_classes, args.num_hops+1, 
                                 num_feats, args.alpha, args.n_layers_1, args.n_layers_2, args.n_layers_3, 
                                 args.act, args.dropout, args.input_drop, args.att_drop, args.label_drop, 
                                 args.pre_process, args.residual, args.pre_dropout, args.bns)
    elif 'SIGNV' in args.method:
        return eval(args.method)(num_feats, feat_dim, args.num_hidden, num_classes, args.num_hops+1,
                    args.n_layers_1, args.n_layers_2, args.dropout, input_drop=args.input_drop, alpha=args.alpha, bns=args.bns)
