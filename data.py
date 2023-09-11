import torch
import dgl
import numpy as np
import scipy.sparse as sp
import os.path as op
import json
import time
import pickle
from utils import shuffle_walks


def load_matrix(fn: str):
    return torch.load(fn)


def build_dgl_graph(fn: str = 'connect.csv',
                    margin: float = 0.5,
                    device=torch.device('cpu')):
    my_data = np.genfromtxt(fn, delimiter=',')
    adj = my_data[1:, 1:]
    adj[adj < margin] = 0.0
    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g = g.to(device)
    return g


def build_train_test(g, eid_fn='eids.json', test_size=0.2):
    """

    :param g: DGLGraph
    :param eid_fn: eids filename for train_test
    :return: train_g, (train_pos_g, train_neg_g), (test_pos_g, test_neg_g)
    """
    u, v = g.edges()

    if not op.isfile(eid_fn):
        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        eids_lt = eids.tolist()
        json.dump(eids_lt, open(eid_fn, 'w'))
    else:
        eids = json.load(open(eid_fn))
        eids = torch.tensor(eids, dtype=torch.long)
    test_size = int(len(eids) * test_size)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # 采样所有负样例并划分为训练集和测试集中。
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    return train_g, (train_pos_g, train_neg_g), (test_pos_g, test_neg_g)


def net2graph(net_sm):
    """ Transform the network to DGL graph
    Return
    ------
    G DGLGraph : graph by DGL
    """
    start = time.time()
    G = dgl.DGLGraph(net_sm)
    end = time.time()
    t = end - start
    print("Building DGLGraph in %.2fs" % t)
    return G


def make_undirected(G):
    # G.readonly(False)
    G.add_edges(G.edges()[1], G.edges()[0])
    return G


def find_connected_nodes(G):
    nodes = G.out_degrees().nonzero().squeeze(-1)
    return nodes


class DeepwalkDataset:
    def __init__(self,
                 net_file,
                 map_file,
                 walk_length,
                 window_size,
                 num_walks,
                 batch_size,
                 negative=5,
                 gpus=[0],
                 fast_neg=True,
                 ogbl_name="",
                 load_from_ogbl=False,
                 ):
        """ This class has the following functions:
        1. Transform the txt network file into DGL graph;
        2. Generate random walk sequences for the trainer;
        3. Provide the negative table if the user hopes to sample negative
        nodes according to nodes' degrees;
        Parameter
        ---------
        net_file str : path of the txt network file
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        num_walks int : number of walks for each node
        batch_size int : number of node sequences in each batch
        negative int : negative samples for each positve node pair
        fast_neg bool : whether do negative sampling inside a batch
        """
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_walks = num_walks
        self.batch_size = batch_size
        self.negative = negative
        self.num_procs = len(gpus)
        self.fast_neg = fast_neg

        if load_from_ogbl:
            assert len(gpus) == 1, "ogb.linkproppred is not compatible with multi-gpu training (CUDA error)."
            from load_dataset import load_from_ogbl_with_name
            self.G = load_from_ogbl_with_name(ogbl_name)
            self.G = make_undirected(self.G)
        else:
            self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(net_file)
            self.save_mapping(map_file)
            self.G = net2graph(self.sm)

        self.num_nodes = self.G.number_of_nodes()

        # random walk seeds
        start = time.time()
        self.valid_seeds = find_connected_nodes(self.G)
        if len(self.valid_seeds) != self.num_nodes:
            print("WARNING: The node ids are not serial. Some nodes are invalid.")

        seeds = torch.cat([torch.LongTensor(self.valid_seeds)] * num_walks)
        self.seeds = torch.split(shuffle_walks(seeds),
                                 int(np.ceil(len(self.valid_seeds) * self.num_walks / self.num_procs)),
                                 0)
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = self.G.out_degrees(self.valid_seeds).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=np.int)
            self.neg_table = []

            for idx, node in enumerate(self.valid_seeds):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.long)
            del node_degree

    def create_sampler(self, i):
        """ create random walk sampler """
        return DeepwalkSampler(self.G, self.seeds[i], self.walk_length)

    def save_mapping(self, map_file):
        """ save the mapping dict that maps node IDs to embedding indices """
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)


class DeepwalkSampler(object):
    def __init__(self, G, seeds, walk_length):
        """ random walk sampler

        Parameter
        ---------
        G dgl.Graph : the input graph
        seeds torch.LongTensor : starting nodes
        walk_length int : walk length
        """
        self.G = G
        self.seeds = seeds
        self.walk_length = walk_length

    def sample(self, seeds):
        walks = dgl.sampling.random_walk(self.G, seeds, length=self.walk_length - 1)[0]
        return walks


if __name__ == '__main__':
    build_train_test(build_dgl_graph())
