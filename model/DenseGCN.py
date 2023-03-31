import dgl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader, Dataset


class DenseGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, lr=0.01, dropout=0.5, weight_decay=5e-4):
        super(DenseGCN, self).__init__()

        self.conv1 = dgl.nn.GraphConv(in_size, hid_size) # type: ignore
        self.conv2 = dgl.nn.GraphConv(hid_size, out_size) # type: ignore
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

    def forward(self, feat, adj):
        feat = self.conv1(adj, feat)
        feat = F.relu(feat)
        feat = F.dropout(feat, self.dropout, training=self.training)
        feat = self.conv2(adj, feat)

        return F.log_softmax(feat, dim=1).squeeze()

    def fit(self, feat, adj, labels, epochs: int, mask: torch.Tensor | None=None, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)



        for epoch in t:
            optimizer.zero_grad()
            predictions = self(feat, adj)
            if mask != None:
                loss = F.cross_entropy(predictions[mask], labels[mask])
            else:
                loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})

        return loss.item()
    
    def fit_samples(self, feats: list[torch.Tensor], adjs: list[torch.Tensor], 
                    labels: list[torch.Tensor], epochs: int, batch_size=16, 
                    verbose=True):
        
        # raw = []
        # for i in range(len(feats)):
        #     raw.append({
        #         "feat": feats[i],
        #         "adj": adjs[i],
        #         "label": labels[i]
        #     })

        class TDataset(Dataset):
            def __init__(self, feats: list[torch.Tensor], adjs: list[torch.Tensor], 
                    labels: list[torch.Tensor]):
                    self.feats = feats
                    self.adjs = adjs
                    self.labels = labels

            def __len__(self):
                    return len(self.feats)

            def __getitem__(self, index):
                    feat = feats[index]
                    adj = adjs[index]
                    label = labels[index]

                    return feat, adj, label

        dataloader = DataLoader(TDataset(feats, adjs, labels), batch_size=batch_size, shuffle=True) # type: ignore

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description("GCN Sample Training")
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for i in t:
            
            for feat, adj, label in dataloader:
                loss = 0
                for i in range(len(feat)):
                
                    pred = self(feat[i], adj[i])
                    loss += F.cross_entropy(pred, label[i])
                # acc += (pred.argmax(1) == sample["label"]).sum() / sample["label"].size[0]
                loss.backward() # type: ignore
                optimizer.step()
            
                with torch.no_grad():
                    t.set_postfix({
                        "loss": loss.item(), # type: ignore
                    })

    def fit_dgl_samples(self, samples: list[dgl.DGLGraph], 
                       epochs: int, batch_size=16, 
                       verbose=True):
        
        self.fit_samples(
            [x.ndata['feat'] for x in samples], # type: ignore
            [x.adj().to_dense() for x in samples], 
            [x.ndata['label'] for x in samples],  # type: ignore
            epochs, batch_size, verbose) 
        
    def eval_acc(self, feat, adj, labels) -> float:
         pred = self(feat, adj)
         acc = (labels == pred.argmax(1)).sum() / adj.shape[0]

         return acc.item()

    def eval_dgl_acc(self, graph: dgl.DGLGraph) -> float:
         return self.eval_acc(graph.ndata['feat'], graph.adj().to_dense(), graph.ndata['label'])
