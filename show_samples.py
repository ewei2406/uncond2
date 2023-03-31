import matplotlib.pyplot as plt
import torch
import dgl
# import numpy as np

def show_graphs(graphs: list[dgl.DGLGraph], size=16) -> None:
    show_adj([g.adj().to_dense() for g in graphs], size=size)


def show_adj(adjs: list[torch.Tensor], size=16) -> None:
    # def mirror(A: np.ndarray):
    #     return np.tril(A) + np.triu(A.T, 1)
    plt.figure()
    ct = len(adjs)
    f, axarr = plt.subplots(1, ct, figsize=(size,size))
    for i in range(ct):
        col = axarr[i].imshow(adjs[i], interpolation="nearest")
        axarr[i].axis('off')
        # f.colorbar(col, ax=axarr[-1])

if __name__ == "__main__":
    print("Hello world")
