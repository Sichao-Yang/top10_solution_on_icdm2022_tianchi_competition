# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser("Random relation subset generator")
parser.add_argument("--num-subsets", type=int, default=12)
parser.add_argument("--output", type=str, default=None, help="Output file name")
parser.add_argument("--dataset", type=str, default='icdm2022')
parser.add_argument(
    "--target-node-type",
    type=str,
    default="item",
    help="Node types that predictions are made for",
)
args = parser.parse_args()
args.data_root = r'D:\comp_gnn\dataset'
print(args)

random.seed(33)

# from dataset import ICDM2022Dataset
# dataset = ICDM2022Dataset(raw_dir=args.data_root)
# g = dataset[0]
# target_ntype = dataset.target_ntype

# each relation has prob 0.5 to be kept
prob = 0.5

edge_types = [('a', 'G_1', 'f'), ('a', 'H_1', 'e'), ('b', 'A_1', 'item'), 
         ('c', 'D_1', 'f'), ('d', 'C_1', 'f'), ('e', 'F_1', 'f'), 
         ('e', 'H', 'a'), ('f', 'B', 'item'), ('f', 'C', 'd'), 
         ('f', 'D', 'c'), ('f', 'F', 'e'), ('f', 'G', 'a'), 
         ('item', 'A', 'b'), ('item', 'B_1', 'f')]
target_ntype = args.target_node_type

edges = []
must_edges = []
# for u, e, v in g.canonical_etypes:
for u, e, v in edge_types:
    edges.append((u, e, v))
    if u == target_ntype or v == target_ntype:
        must_edges.append(e)

must_edges = [x[0] for x in must_edges if len(x)>1]

n_edges = len(edges)

if args.output is None:
    args.output = "lib_dgl/{}_rand_subsets".format(args.dataset)
# assert not os.path.exists(args.output)
subsets = set()

while len(subsets) < args.num_subsets:
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

with open(args.output, "w") as f:
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
