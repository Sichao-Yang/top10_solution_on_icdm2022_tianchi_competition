
from dis import code_info
import os
import numpy as np
import random
import torch
import dgl
import dgl.function as fn
import gc
import torch.nn.functional as F
from sklearn.model_selection import KFold
import sys
from os import path as osp
p = osp.abspath(osp.join(osp.dirname(__file__),'.'))
sys.path.append(p)
from dataset import ICDM2022Dataset
#############################################################################
@torch.no_grad()
def load_icdm(device, args, session=None, enhance=False, mu=1):
    # for all returns, only g and labels sent to device
    if session == 'session1':
        load_labels = True
    else:
        load_labels = False
    dataset = ICDM2022Dataset(session=session, load_features=True, load_labels=load_labels, raw_dir=args.data_root, verbose=True)
    g = dataset[0]
    category = dataset.category
    rev_item_map = dataset.rev_item_map
    num_classes = dataset.num_classes
    if session == 'session1':
        labels = g.nodes[category].data['label'].to(device).squeeze()
        train_mask = g.nodes[category].data['train_mask'].bool()
        val_mask = g.nodes[category].data['val_mask'].bool()
    test_mask = g.nodes[category].data['test_mask'].bool()

    full_indice = np.arange(g.num_nodes(category))
    # convert to tensor format
    if session == 'session1':
        train_nid = torch.tensor(full_indice[train_mask]).to(torch.long)
        val_nid = torch.tensor(full_indice[val_mask]).to(torch.long)
        # load nid if cv_id!=-1, else use kfold to shuffle and save nid sets
        cv_dir = osp.join(args.tmp_dir, args.relation_subset_path.split('/')[-1])
        if not os.path.exists(cv_dir):
            os.mkdir(cv_dir)
        if args.cv_id == -1:
            all_idx = torch.cat((train_nid, val_nid))
            np.random.shuffle(all_idx.numpy())
            kf = KFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)
            for cv, (tr, va) in enumerate(kf.split(all_idx)):
                train_nid = all_idx[tr]
                val_nid = all_idx[va]
                with open(osp.join(cv_dir, f'cv_{cv}.npy'), 'wb') as f:
                    np.save(f, train_nid.numpy())
                    np.save(f, val_nid.numpy())
            return 0
        else:
            with open(osp.join(cv_dir, f'cv_{args.cv_id}.npy'), 'rb') as f:
                train_nid = torch.tensor(np.load(f)).to(torch.long)
                val_nid = torch.tensor(np.load(f)).to(torch.long)
    else:
        labels = train_nid = val_nid = None
        
    test_nid = torch.tensor(full_indice[test_mask]).to(torch.long)
    # include background data
    if enhance:
        enhance_nid = np.random.choice(full_indice[~test_nid], len(test_nid)*mu)
        enhance_nid = torch.tensor(enhance_nid).to(torch.long)
        test_nid = torch.cat([test_nid, enhance_nid],dim=-1)
    
    g = g.to(device)     
    return g, labels, num_classes, train_nid, val_nid, test_nid, rev_item_map, category
#############################################################################
# Make or load Relation Subsets
def read_relation_subsets(fname, log):
    log.info("Reading Relation Subsets:")
    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = line.strip().split(',')
            rel_subsets.append(relations)
            log.info(relations)
    return rel_subsets

def make_subsets(g, category='item', num_subsets=12, output="lib_dgl/icdm2022_rand_subsets"):
    # each relation has prob 0.5 to be kept
    prob = 0.5
    # # predefined edge_types
    # edge_types = [('a', 'G_1', 'f'), ('a', 'H_1', 'e'), ('b', 'A_1', 'item'), 
    #         ('c', 'D_1', 'f'), ('d', 'C_1', 'f'), ('e', 'F_1', 'f'), 
    #         ('e', 'H', 'a'), ('f', 'B', 'item'), ('f', 'C', 'd'), 
    #         ('f', 'D', 'c'), ('f', 'F', 'e'), ('f', 'G', 'a'), 
    #         ('item', 'A', 'b'), ('item', 'B_1', 'f')]
    target_ntype = category
    edge_types = g.canonical_etypes
    # make sure every target_nodes should be included in the subset
    edges = []
    must_edges = []
    for u, e, v in edge_types:
        edges.append((u, e, v))
        if u == target_ntype or v == target_ntype:
            must_edges.append(e)

    must_edges = [x[0] for x in must_edges if len(x)>1]
    # assert not os.path.exists(args.output)
    subsets = set()

    while len(subsets) < num_subsets:
        selected = []
        for (u, e, v) in edges:
            edge = e[0]
            if random.random() < prob:
                if edge not in selected:
                    selected.append(edge)

        # retry if no edge is selected
        if len(selected) == 0:
            continue
        sorted(selected)
        subsets.add(tuple(selected))

    with open(output, "w") as f:
        for edges in subsets:
            tmp = must_edges.copy()
            etypes = list(edges)
            # only save subsets that touches all target node's edges
            target_touched = False
            for e in edges:
                if e[0] in tmp:
                    tmp.remove(e[0])
                if len(tmp) == 0:
                    target_touched = True
            print(etypes, target_touched and "touched" or "not touched")
            if target_touched:
                f.write(",".join(etypes) + "\n")

#############################################################################
# Generate multi-hop neighbor-averaged feature for each relation subset
def gen_rel_subset_feature(g, rel_subset, args, device, log, train_nid, val_nid, test_nid, category, fp):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        if device == 'cpu':
            src = src.cpu().numpy()
            dst = dst.cpu().numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges).to(device)
    log.debug('generated new_g for rel_subset')
    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :].to(device)
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"].to(device)
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)
    log.debug('done node feature and calc deg for new_g')
    gc.collect()
    # compute k-hop feature
    for hop in range(1, args.num_hops + 1):
        log.debug('start on hop_{}'.format(hop))
        ntype2feat = {}
        # propogate msgs along relational edges
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        log.debug('done msg passing')
        # save target-node's feature on hop n then normalize features per node type and save for hop n+1
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is undirectional, so all nodes should be updated
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            # only store the normalized features for target_node_type
            if ntype == category:
                tmp = old_feat.cpu()
                torch.save(torch.cat((tmp[train_nid], tmp[val_nid], tmp[test_nid]), axis=0), f=fp+f'_{hop-1}.pt')
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])
        log.debug('done feature normalize')
        # release memory on cpu (old_feat, cpu) & gpu (ntype2feat)
        del old_feat, tmp, ntype2feat
        torch.cuda.empty_cache()
        gc.collect()
    tmp = new_g.nodes[category].data.pop(f"hop_{args.num_hops}").cpu()
    torch.save(torch.cat((tmp[train_nid], tmp[val_nid], tmp[test_nid]), axis=0), f=fp+f'_{args.num_hops}.pt')
    del tmp, new_g, feat_dict
    torch.cuda.empty_cache()
    gc.collect()
    return 0

@torch.no_grad()
def prepare_features(g, metapaths, args, device, train_nid, val_nid, test_nid, category, log, cache_dir):
    """NOTE: Because the biggest memory bound happens here, detailed memory control is applied:
    features are saved each metapath and each hop with necessary node_ids to reduce mermory size.
        full size in memory: khop=10;m_nodes=11933366;n_feat_dim=256;dtype=float32=8byte; 8*k*m*n//1024//1024//1024=227G
        reduce this to: 8*k=1*m=(8364+77198+36924)*n//1024//1024=239M
    """
    num_item, feat_size = g.nodes[category].data["feat"].shape
    if train_nid is None and val_nid is None:
        train_nid = val_nid = torch.tensor([]).to(torch.long)
    num_item = len(train_nid)+len(val_nid)+len(test_nid)
    
    for mpath_id, mpath in enumerate(metapaths):
        log.info('process on {}'.format(mpath))
        fp = osp.join(cache_dir, f'{mpath_id}')
        gen_rel_subset_feature(g, mpath, args, device, log, train_nid, val_nid, test_nid, category, fp)
        gc.collect()
    log.debug('Done feature aggragation and put them into one')
    # [Khop][Sub][f_dim]
    new_feats = [torch.zeros(num_item, len(metapaths), feat_size) for _ in range(args.num_hops + 1)]
    for mpath_id, mpath in enumerate(metapaths):
        for i in range(args.num_hops + 1):
            fp = osp.join(cache_dir, f'{mpath_id}_{i}.pt')
            new_feats[i][:, mpath_id, :] = torch.load(f=fp)
            os.remove(fp)
    return new_feats
#############################################################################
# Generate multi-hop neighbor-averaged label for target nodes
def neighbor_average_labels(g, feat):
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')
    return feat

def prepare_label_emb(args, g, labels, n_classes, train_idx, valid_idx, test_idx, category, log, label_teacher_emb=None):
    # below using cpu to do the label_embedding
    target_type_id = g.get_ntype_id(category)
    num_nodes = g.num_nodes()
    homo_g = dgl.to_homogeneous(g, ndata=None)
    homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
    homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
    # log.info(n_classes)
    # log.info(labels.shape[0])
    # make label matrix for the labeled nodes, default vale set to zeros
    # then assgin one-hot labels to coresponding nodes
    y = np.zeros(shape=(labels.shape[0], int(n_classes)))
    if label_teacher_emb != None:
        log.info("use teacher label")
        y[valid_idx] = label_teacher_emb[len(train_idx):len(train_idx)+len(valid_idx)]
        y[test_idx] = label_teacher_emb[len(train_idx)+len(valid_idx):len(train_idx)+len(valid_idx)+len(test_idx)]
    y[train_idx] = F.one_hot(labels[train_idx].to(torch.long), num_classes=n_classes).float().squeeze(1)
    y = torch.Tensor(y)
    # assgin labels to the whole graph matrix (default zeros)
    target_mask = homo_g.ndata["target_mask"]
    num_target = target_mask.sum().item()
    new_label_emb = torch.zeros((num_nodes,n_classes), dtype=y.dtype, device=y.device)
    new_label_emb[target_mask] = y
    y = new_label_emb
    del labels
    gc.collect()
    # propagate by msg passing
    for _ in range(args.label_num_hops):
        y = neighbor_average_labels(homo_g, y.to(torch.float))
        gc.collect()
    # get propagated labels from graph to target node
    target_mask = homo_g.ndata['target_mask']
    num_target = target_mask.sum().item()
    res = torch.zeros((num_target,n_classes), dtype=y.dtype, device=y.device)
    res = y[target_mask]
    # only save the right-ordered train/val/test labels
    return torch.cat([res[train_idx], res[valid_idx], res[test_idx]], dim=0)


def prepare_data(device, args, teacher_probs, log, session=None, enhance=False, mu=0):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    assert session is not None, "Should input session No."
    if args.cpu_preprocess:
        device_pre = 'cpu'
    else:
        device_pre = device
    g, labels, num_classes, train_nid, val_nid, test_nid, rev_item_map, category = load_icdm(device_pre, 
                                                                                             args, 
                                                                                             session=session, 
                                                                                             enhance=enhance, 
                                                                                             mu=mu)
    log.info('Done dataloading')
    if args.remake_subsets:
        make_subsets(g, category)
        log.info('Done subset making, exit program')
        exit()
    fn_sub = osp.join(args.code_root, args.relation_subset_path)
    rel_subsets = read_relation_subsets(fn_sub, log)
    # all features are stored in cpu only load to device at training
    cache_dir = osp.join(args.tmp_dir, osp.join(fn_sub.split('/')[-1], session))
    if osp.exists(cache_dir) and (len(os.listdir(cache_dir)) >= args.num_hops+1):
        log.info(f"Found cached features & loading")
        feats = []
        for i in range(args.num_hops+1):
            feat = torch.load(osp.join(cache_dir, f'{i}.pt'))
            if args.sampled_path_num != -1:
                feat = feat[:,:args.sampled_path_num,:]
            feats.append(feat)
    else:
        log.info(f"No cached features found, start making and save to {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        feats = prepare_features(g, rel_subsets, args, device_pre, train_nid, val_nid, test_nid, category, log, cache_dir)
        for i, feat in enumerate(feats):
            torch.save(feat, f=osp.join(cache_dir, f'{i}.pt'))

    log.info("Done feature preprocessing")

    label_emb = None
    if args.use_rlu:
        # label embedding is done on cpu
        labels = labels.to('cpu')
        label_emb = prepare_label_emb(args, g.to('cpu'), labels, num_classes, 
                                    train_nid, val_nid, test_nid, category, log, teacher_probs)
        log.info("Done label preprocessing")
    # release gpu memory
    del g
    torch.cuda.empty_cache()
    # move to device
    if session == 'session1':
        train_nid = train_nid.to(device)
        val_nid = val_nid.to(device)
        labels = labels.to(device)
    test_nid = test_nid.to(device)
    if label_emb is not None:
        label_emb = label_emb.to(device)
    if session == 'session1':
        # postprocess on test_labels:
        # due to format_dgl setting(np.nan->float), the values are all -9223372036854775808, 
        # now set to -1 for readability
        labels[test_nid] = -1
        labels = torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]])

    return feats, labels, label_emb, num_classes, train_nid, val_nid, test_nid, rev_item_map

#############################################################################
# Dataset loading
class PrepareData(object):
    def __init__(self, device, args, teacher_probs, logging):
        self.device = device
        self.args = args
        self.teacher_probs = teacher_probs
        self.logging = logging
        self.dat = prepare_data(device, args, teacher_probs, logging, session='session1')
        self.feats, self.labels, self.label_emb, self.num_classes, self.train_nid, self.val_nid, self.test_nid, self.rev_item_map = self.dat

    def update(self, cv_id, stage):
        all_nid = torch.cat([self.train_nid, self.val_nid]).cpu().numpy()
        cv_dir = osp.join(self.args.tmp_dir, self.args.relation_subset_path.split('/')[-1])
        # find position in the existing array by matching node_id
        with open(osp.join(cv_dir, f'cv_{cv_id}.npy'), 'rb') as f:
            train_nid = np.load(f)
            val_nid = np.load(f)
        
        tr = []
        for i in range(len(train_nid)):
            tr.append(np.where(all_nid == train_nid[i])[0][0])
        va = []
        for i in range(len(val_nid)):
            va.append(np.where(all_nid == val_nid[i])[0][0])
        # reorder exsiting data by chunk concat the coresponding node sets(train/val)
        for i in range(len(self.feats)):
            self.feats[i] = torch.cat([torch.cat([self.feats[i][tr], self.feats[i][va]],dim=0), self.feats[i][len(all_nid):]], dim=0)
        self.labels = torch.cat([torch.cat([self.labels[tr], self.labels[va]],dim=0), self.labels[len(all_nid):]], dim=0)
        # assign new node_ids to sets
        self.train_nid = torch.tensor(train_nid).to(self.device).to(torch.long)
        self.val_nid = torch.tensor(val_nid).to(self.device).to(torch.long)
        
        if not self.args.session1_only:      
            if stage==1 and cv_id==0:
                for i in range(len(self.feats)):
                    self.feats[i] = self.feats[i][:len(self.train_nid)+len(self.val_nid)]
                gc.collect()
                self.feats_s2, _, _, _, _, _, self.test_nid, self.rev_item_map = prepare_data(self.device,
                                                                                    self.args, 
                                                                                    self.teacher_probs, 
                                                                                    self.logging, 
                                                                                    session='session2',
                                                                                    enhance=self.args.enhance,
                                                                                    mu=self.args.mu)
                for i in range(len(self.feats)):
                    self.feats[i] = torch.cat([self.feats[i][:len(self.train_nid)+len(self.val_nid)], self.feats_s2[i]], dim=0)
            elif stage==len(self.args.stages)-1 and cv_id==0:
                if self.args.enhance:
                    true_test_len = self.feats_s2[0].shape[0]//(1+self.args.mu)
                    self.test_nid = self.test_nid[:true_test_len]
                else:
                    true_test_len = self.feats_s2[0].shape[0]
                for i in range(len(self.feats)):
                    self.feats[i] = torch.cat([self.feats[i][:len(self.train_nid)+len(self.val_nid)], self.feats_s2[i][:true_test_len]], dim=0)
            
        self.dat = self.feats, self.labels, self.label_emb, self.num_classes, self.train_nid, self.val_nid, self.test_nid, self.rev_item_map


if __name__ == '__main__':
    code_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    sys.path.append(code_root)
    from lib_dgl.utils import get_logger, check_cv_nids
    from lib_dgl.argparser import ArgsParser
    logging = get_logger('run.log',method='w2file',verb_level='debug')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = ArgsParser(description="Neighbor-Averaging over Relation Subgraphs").parse_args()
    
    args.code_root = code_root
    args.tmp_dir = osp.abspath(osp.join(osp.dirname(__file__), '../../tmp_feats'))
    args.data_root = osp.abspath(osp.join(osp.dirname(__file__), '../../dataset'))
    args.out_dir = osp.abspath(osp.join(osp.dirname(__file__), '../../output'))    

    feats_s2, _, _, _, _, _, test_nid, rev_item_map = prepare_data(device,
                                                                    args, 
                                                                    None, 
                                                                    logging, 
                                                                    session='session2',
                                                                    enhance=True,
                                                                    mu=1)