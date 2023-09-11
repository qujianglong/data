import json
import math
import os
import random

from tap import Tap
import numpy as np


class Params(Tap):
    margin: float = 0.3
    test_size: float = 0.2
    neg_test_size: float = 1.0
    fn: str = 'data/connect.csv'
    out_dir: str = 'data'


def dataprocessing(args: Params):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("Reading adj matrix...")
    my_data = np.genfromtxt(args.fn, delimiter=',')
    adj = my_data[1:, 1:]
    print(f"Before cut-off: {np.sum((adj != 0))}")
    adj[adj < args.margin] = 0.0
    print(f"After cut-off: {np.sum((adj != 0))}")
    all_nodes = list(range(adj.shape[0]))
    n = adj.shape[0]
    src, dst = np.nonzero(adj)
    src = src.tolist()
    dst = dst.tolist()
    all_edges = list(zip(src, dst))
    print("Spliting...")
    train_edges_size = math.floor(len(all_edges) * (1 - args.test_size))
    random.shuffle(all_edges)
    train_edges = all_edges[:train_edges_size]
    test_edges_pos = all_edges[train_edges_size:]
    test_labels = [1 for _ in test_edges_pos]
    test_edges_neg = list()
    neg_size = math.floor(len(test_edges_pos) * args.neg_test_size)
    ncount = 0
    print("Generating negative samples for test...")
    while ncount < neg_size:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if adj[u, v] == 0:
            test_edges_neg.append((u, v))
            ncount += 1
    test_edges = test_edges_pos + test_edges_neg
    test_labels.extend([0 for _ in test_edges_neg])
    print(f"Training Size: {len(train_edges)}")
    print(f"Test Positive Size: {len(test_edges_pos)}")
    print(f"Test Negative Size: {len(test_edges_neg)}")
    print(f"Total Test: {len(test_edges)}")
    print("Writing into files...")
    train_fp = open(os.path.join(args.out_dir, 'edge.train'), 'w')
    test_fp = open(os.path.join(args.out_dir, 'edge.test'), 'w')
    train_fp.write(f"source,target\n")
    test_fp.write(f"source,target\n")
    for (u, v) in train_edges:
        train_fp.write(f"{u},{v}\n")
    for (u, v) in test_edges:
        test_fp.write(f"{u},{v}\n")
    train_fp.close()
    test_fp.close()
    json.dump(test_labels, open(os.path.join(args.out_dir, 'test.labels'), 'w'))
    print("Done.")


if __name__ == '__main__':
    args = Params().parse_args()
    dataprocessing(args)
