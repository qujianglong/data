# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATConv
from torch_geometric.utils import train_test_split_edges
from scipy import sparse
import pandas as pd
from torch_geometric import data as DATA
from pandas.core.frame import DataFrame
features=pd.read_csv('Feature(C).csv',encoding='gb18030')
features=features.iloc[:, 1:].values
adj=pd.read_csv('Adjacency(C).csv')
adj=adj.iloc[:, 1:].values
adj[adj>=0.5]=1
adj[adj<0.5]=0
adj = sparse.csr_matrix(adj).tocoo()
label =adj.data
adj = torch.LongTensor([adj.row.tolist(), adj.col.tolist()])
print(adj)
print(features)
GATData = DATA.Data(x=torch.FloatTensor(features),
                    edge_index=adj,
                    y=torch.FloatTensor(label)
                    )
data = train_test_split_edges(GATData)
import numpy as np
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    precision_list = TP/(TP+FP)
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    max_index = np.argmax(f1_score_list)
    precision = precision_list[max_index]
    return precision
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(data.num_features, 64)
        self.conv2 = GATConv(64, 32)
    def encode(self):
        x = F.dropout(self.conv1(data.x, data.train_pos_edge_index),0.4)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=10000,
        force_undirected=True,
    )
    neg_edge_index = neg_edge_index.to(device)
    optimizer.zero_grad()
    z = model.encode()
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test():
    model.eval()
    AUC = []
    precision=[]
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        AUC.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        link_probs=link_probs.cpu()

        precision.append(get_metrics(link_labels.cpu().numpy(), link_probs.numpy()))
    return AUC,precision
best_val_AUC= test_AUC = 0
T=[]
E=[]
for epoch in range(1, 100):
    train_loss = train()
    AUC,precision=test()
    val_AUC, tmp_test_AUC=AUC
    val_precision, tmp_test_precision = precision
    if val_AUC > best_val_AUC:
        best_val_AUC = val_AUC
        test_AUC= tmp_test_AUC
    log = 'Epoch: {:03d}, Loss: {:.4f}, best_Val: {:.4f}, Test: {:.4f},Test_precision: {:.4f}'
    print(log.format(epoch, train_loss, best_val_AUC, test_AUC,tmp_test_precision))
    T.append(test_AUC)
    E.append(epoch)
c={"Epoch" : E,
   "AUC" : T}
data=DataFrame(c)
print(data)
data.to_excel("result.xlsx",index=None)