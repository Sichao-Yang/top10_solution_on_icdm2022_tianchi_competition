# 添加属性n_idx
# 将edge_index 转换成 torch.long
# 数据集划分信息

import argparse
from pydoc import describe
from tkinter import W
from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl

edge_size = 157814864
node_size = 13806619
root = r'D:\comp_gnn\dataset\comp_data'
nodef = root+r'\input\icdm2022_session1_nodes.csv'
edgef = root+r'\input\icdm2022_session1_edges.csv'
trainlf = root+r'\input\icdm2022_session1_train_labels.csv'
storefile = root+r'\output\icdm2022_session1.pt'
mapfile = root+r'\output\icdm2022_session1_nodemap.pkl'

f_dim = 256
edge_order = [('b', 'A_1', 'item'),
                ('f', 'B', 'item'),
                ('a', 'G_1', 'f'),
                ('f', 'G', 'a'),
                ('a', 'H_1', 'e'),
                ('f', 'C', 'd'),
                ('f', 'D', 'c'),
                ('c', 'D_1', 'f'),
                ('f', 'F', 'e'),
                ('item', 'B_1', 'f'),
                ('item', 'A', 'b'),
                ('e', 'F_1', 'f'),
                ('e', 'H', 'a'),
                ('d', 'C_1', 'f')]
labeled_class = 'item'


def read_node_atts(node_file, storefile, label_file=None):
    node_maps = {}
    node_embeds = {}
    count = 0
    node_counts = node_size
    if osp.exists(storefile + ".nodes.pyg") == False:
        process = tqdm(total=node_counts, desc="Generating " + storefile + ".nodes.pyg")
        with open(node_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()

                node_maps.setdefault(node_type, {})
                node_id_v2 = len(node_maps[node_type])
                node_maps[node_type][node_id] = node_id_v2

                node_embeds.setdefault(node_type, {})
                node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                count += 1
                if count % 100000 == 0:
                    process.update(100000)

        process.close()

        print("Num of total nodes:", count)
        print('Node_types:', node_maps.keys())
        for node_type in node_maps:
            print(node_type, len(node_maps[node_type]))

        labels = []
        if label_file is not None:
            labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
            for i in range(len(labels_info)):
                x = labels_info[i]
                item_id = node_maps[labeled_class][int(x[0])]
                label = int(x[1])
                labels.append([item_id, label])

        nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
        nodes_dict['labels'] = {}
        nodes_dict['labels'][labeled_class] = labels
        print('Start saving pkl-style node information\n')
        pkl.dump(nodes_dict, open(storefile + ".nodes.pyg", 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Complete saving pkl-style node information\n')

    else:
        nodes = pkl.load(open(storefile + ".nodes.pyg", 'rb'))
        node_embeds = nodes['embeds']
        node_maps = nodes['maps']
        labels = nodes['labels'][labeled_class]

    return node_embeds, node_maps, labels


def format_pyg_graph(edge_file, node_file, storefile, mapfile, label_file=None):
    node_embeds, node_maps, labels = read_node_atts(node_file, storefile, label_file)

    # 将结点特征储存到pyg 图数据中
    graph = HeteroData()

    print("Start converting into pyg data")
    # 1. 转换结点特征
    for node_type in tqdm(node_embeds, desc="Node features, numbers and mapping", ascii=True):
        graph[node_type].x = torch.empty((len(node_maps[node_type]), f_dim))
        for nid, embedding in tqdm(node_embeds[node_type].items()):
            graph[node_type].x[nid] = torch.from_numpy(embedding)
        graph[node_type].num_nodes = len(node_maps[node_type])
        graph[node_type].maps = node_maps[node_type]

    if label_file is not None:
        # 2. 转换标签
        graph[labeled_class].y = torch.zeros(len(node_maps[labeled_class]), dtype=torch.long) - 1
        for index, label in tqdm(labels, desc="Node labels", ascii=True):
            graph[labeled_class].y[index] = label

        # 3. 划分数据集
        indices = (graph[labeled_class].y != -1).nonzero().squeeze()
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        # 得到训练集和验证集划分
        train_val_random = torch.randperm(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        print("trian_idx:{}".format(train_idx.numpy()))
        print("test_idx:{}".format(val_idx.numpy()))
        # 添加到item类型结点的属性中
        graph[labeled_class].train_idx = train_idx
        graph[labeled_class].val_idx = val_idx

    # # 添加每个节点的索引信息n_id
    # for ntype in graph.node_types:
    #     graph[ntype].n_id = torch.arange(graph[ntype].num_nodes)

    process = tqdm(total=edge_size)
    edges = {}
    count = 0
    with open(edge_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = graph[source_type].maps[int(source_id)]
            dest_id = graph[dest_type].maps[int(dest_id)]
            edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(int(source_id))
            edges[edge_type].setdefault('dest', []).append(int(dest_id))
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
            if count % 100000 == 0:
                process.update(100000)
    process.close()
    print('Complete reading edge information\n')

    print('Start converting edge information\n')
    for edge_type in edges:
        source_type = edges[edge_type]['source_type']
        dest_type = edges[edge_type]['dest_type']
        source = torch.tensor(edges[edge_type]['source'], dtype=torch.long)
        dest = torch.tensor(edges[edge_type]['dest'], dtype=torch.long)
        graph[(source_type, edge_type, dest_type)].edge_index = torch.vstack([source, dest])

    # edge_type 重新排序,pyg处理异质图一般是将其转换为同质图再利用edge_type这个属性确定边的类型,所以最好先把所有图的edge_type按照统一的标准进行排序
    for edge_type in edge_order:
        try:
            temp = graph[edge_type].edge_index
            del graph[edge_type]
            graph[edge_type].edge_index = temp
        except:
            del graph[edge_type]
            continue
    
    for ntype in graph.node_types:
        if ntype == labeled_class:
            pkl.dump(graph[ntype].maps, open(mapfile, 'wb'), pkl.HIGHEST_PROTOCOL)
        del graph[ntype].maps
        
    print('Complete converting edge information\n')
    print('Start saving into pyg data\n')
    torch.save(graph, storefile)
    print('Complete saving into pyg data\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', type=bool, default=False, help="Whether node features should be reloaded")
    args = parser.parse_args()
    args.graph = edgef
    args.label = trainlf
    args.node = nodef
    args.storefile = storefile
    args.mapfile = mapfile
    
    if args.graph is not None and args.storefile is not None and args.node is not None:
        format_pyg_graph(args.graph, args.node, args.storefile, args.mapfile, args.label)
        # read_node_atts(args.node, args.storefile, args.label)
