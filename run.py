import json
import os.path

import numpy as np
import torch
import encoder
from encoder import Node2Vec
import utils
from tqdm import tqdm
import random
import argparse
from tap import Tap
from typing import Optional
import pickle


class Params(Tap):
    data_dir: str = 'data'
    out_dir: str = 'results'
    walks: Optional[str] = 'walks.L25'
    rewalk: bool = False
    embed_ckpt: Optional[str] = 'no/ckpts'
    max_iter: int = 100
    embed_dim: int = 128
    gpu_id: int = -1
    nega_ratio: int = 10
    walk_num: int = 10
    walk_length: int = 25


class Trainer:
    def __init__(self, params: Params):
        self.params = params
        train_path = os.path.join(params.data_dir, 'edge.train')
        test_path = os.path.join(params.data_dir, 'edge.test')
        labels = os.path.join(params.data_dir, 'test.labels')
        self.train_edges = utils.read_graph(train_path)
        self.test_edges = np.loadtxt(test_path,
                                     dtype=np.int64,
                                     skiprows=1,
                                     delimiter=',')
        self.test_edges = torch.tensor(self.test_edges)
        self.test_labels = json.load(open(labels))

        if os.path.exists(params.walks) and not params.rewalk:
            self.train_edges.Walks = pickle.load(open(params.walks, 'rb'))
        else:
            self.random_walk()
            fn = f'walks.L{params.walk_length}'
            pickle.dump(self.train_edges.Walks, open(fn, 'wb'))
        self.encoder = Node2Vec(G=self.train_edges,
                                dim=params.embed_dim,
                                gpu_id=params.gpu_id,
                                test_edges=self.test_edges,
                                test_labels=self.test_labels)
        # self.test_labels = json.load(open(labels))
        if not os.path.exists(params.out_dir):
            os.mkdir(params.out_dir)
        print("Finish initializaton.")

    def random_walk(self):

        print("Perform random walk...")
        for i in tqdm(range(self.train_edges.n)):
            for r in range(self.params.walk_num):
                self.train_edges.perform_walk(i, p=2.0, q=1.0,
                                              length=self.params.walk_length)

    def train(self):
        print("Start training...")
        self.encoder.training(self.params.max_iter,
                              negative=self.params.walk_length * self.params.nega_ratio,
                              rewalk=False)
        embed_fn = os.path.join(self.params.out_dir,
                                f"{self.params.embed_dim}d.emb")
        self.encoder.save_embeddings(embed_fn)
        print(f"Embeddings saved in {embed_fn}")

    def test_in_train(self):
        embeddings = self.encoder.Embeddings
        Euv = torch.matmul(embeddings, embeddings.T)
        Euu = torch.sum(embeddings * embeddings, dim=1)
        scores = []
        for i in range(self.test_edges.shape[1]):
            u, v = self.test_edges[0][i], self.test_edges[1][i]
            scores.append((Euv[u][v] / torch.sqrt(Euu[u]) / torch.sqrt(Euu[v])).item())
        return scores


if __name__ == '__main__':
    p_ = Params().parse_args()
    trainer = Trainer(p_)
    trainer.train()
