import os.path
import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as tf
from tqdm import tqdm
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score


def time_stmp() -> str:
    millis = int(round(time() * 1000))
    return hex(millis)[:4]


class Node2Vec:
    def __init__(self, G, test_edges, test_labels, dim=128, gpu_id=1, log_fn='log.txt', patience: int = 10):

        self.G = G
        self.dim = dim
        if torch.cuda.is_available() and gpu_id >= 0:
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device('cpu')
        self.Embeddings: torch.Tensor = torch.zeros(self.G.n, self.dim).to(self.device)
        self.Embeddings.requires_grad = False
        nn.init.normal_(self.Embeddings, mean=0.0, std=1.0 / (self.dim ** 0.5))
        self.UnigramTable = list()
        print("InitUnigramTable...")
        self.initUnigramTable()
        self.ridx = 0
        self.test_edges = test_edges
        self.test_labels = test_labels
        self.out_fp = open(log_fn, 'a')
        self.results = []
        self.patience = patience
        self.npatience = 0
        self.best_precision = 0.0
        self.best_auc = 0.0
        self.best_acc = 0.0
        self.best_embeds = torch.zeros_like(self.Embeddings)

    def __del__(self):
        self.out_fp.close()

    def initUnigramTable(self, size=10000000):
        count = [len(self.G.A[i]) + 1 for i in range(self.G.n)]
        count = np.array(count) ** 0.75
        total_pow = np.sum(count)
        i = 0
        prefix_sum = count[i] / total_pow
        for idx in tqdm(range(size)):
            self.UnigramTable.append(i)
            if idx > prefix_sum * size:
                i += 1
                if i >= self.G.n:
                    i = self.G.n - 1
                prefix_sum += count[i] / total_pow
        print('shuffling...')
        random.shuffle(self.UnigramTable)

    def training(self, max_iter=100, negative=100,
                 rewalk=True, weight_decay=1e-4):

        unisize = len(self.UnigramTable)
        lr = 0.01
        iter_gradient_norm = 0
        for iter in range(max_iter):
            iter_length = 0
            if rewalk:
                print('Re Random walking...')
                self.G.clean_walk()
                for i in tqdm(range(self.G.n)):
                    self.G.perform_walk(i, 2, 0.5, 20)
                iter_length = len(self.G.Walks)
            else:
                iter_length = self.G.n
            random.shuffle(self.G.Walks)

            for walk in self.G.Walks[:iter_length]:
                u = walk[0]
                embedu = self.Embeddings[u]

                idx_p = torch.tensor(walk[1:], dtype=torch.int64).to(self.device)

                embedvs_p = torch.index_select(self.Embeddings, 0, idx_p)
                # scores = torch.matmul(embedvs_p, embedu)

                p_p = 1.0 - torch.sigmoid(torch.matmul(embedvs_p, embedu))

                # negative sample
                neighbors = list()
                count = 0
                while count < negative:
                    v = self.UnigramTable[self.ridx]
                    if not v in self.G.A_set[u]:
                        neighbors.append(v)
                        count += 1
                    self.ridx += 1
                    if self.ridx >= len(self.UnigramTable):
                        self.ridx = 0

                idx_n = torch.tensor(neighbors, dtype=torch.int64).to(self.device)
                embedvs_n = torch.index_select(self.Embeddings, 0, idx_n)

                p_n = 1.0 - torch.sigmoid(-torch.matmul(embedvs_n, embedu))


                dEu = lr * (torch.matmul(p_p, embedvs_p) - torch.matmul(p_n, embedvs_n)) - weight_decay * embedu
                self.Embeddings[u] += dEu
                self.Embeddings.index_add_(0, idx_p, lr * (p_p.view(-1, 1) * (embedu - dEu)) - weight_decay * embedvs_p)
                self.Embeddings.index_add_(0, idx_n, -lr * (p_n.view(-1, 1) * (embedu - dEu)) - weight_decay * embedvs_n)

                if (iter + 1) % 10 == 0:
                    iter_gradient_norm += (torch.norm(dEu) / lr).item()

            if (iter + 1) % 25 == 0:
                lr *= 0.5
            if (iter + 1) % 10 == 0:
                print('iter : {} ; average_gradient_norm : {}'.format(iter + 1,
                                                                      iter_gradient_norm / iter_length))
                iter_gradient_norm = 0
            should_stop = self.test(iter)
            if should_stop:
                print("Early STOP !!!")
                break
        df = pd.DataFrame(self.results, columns=['niter', 'precision', 'AUC', 'ACC'])
        fn = 'n0.3-precision-auc-acc.xlsx'
        if os.path.exists(fn):
            fn = f'n0.3-precision-auc-acc-P{self.best_precision * 100:.3f}-AUC{self.best_auc * 100:.3f}-ACC{self.best_acc * 100:.3f}.xlsx'
        df.to_excel(fn)

    def test(self, epoch: int):

        embeds = self.Embeddings.detach()

        scores = embeds @ embeds.t()
        scores = scores.sigmoid().detach().cpu()
        preds = []
        all_scores = []

        test_edges = self.test_edges

        for (u, v) in test_edges:
            all_scores.append(scores[u, v].item())
            if scores[u, v] > 0.5:
                preds.append(1)
            else:
                preds.append(0)
        y_golden = self.test_labels

        acc = accuracy_score(y_golden, preds)
        precision = precision_score(y_golden, preds)
        auc = roc_auc_score(y_golden, all_scores)

        if auc < self.best_auc and precision < self.best_precision and acc < self.best_acc:
            self.npatience += 1
        else:
            if auc > self.best_auc:

                self.best_auc = auc
                self.npatience = 0
                self.best_embeds = self.Embeddings.detach().clone()
            elif precision > self.best_precision:
                self.best_precision = precision
                self.npatience = 0
            elif acc > self.best_acc:
                self.best_acc = acc
                self.npatience = 0

        self.results.append([epoch, precision, auc, acc])
        self.out_fp.write(f"Iter {epoch}|Precision: {precision:.6f}|AUC: {auc:.6f}|ACC: {acc:.6f}\n")
        print(f"Iter {epoch}|Precision: {precision:.6f}|AUC: {auc:.6f}|ACC: {acc:.6f}")

        if self.npatience > self.patience:

            return True
        else:
            return False

    def save_embeddings(self, pth):

        torch.save(self.best_embeds, pth)

    def load_embeddings(self, fn, device):
        res = torch.load(fn, map_location=device)
        self.Embeddings = res.clone()
