import dgl
import random
import torch

def get_subgraphs(graph: dgl.DGLGraph, size: int, n: int) -> torch.Tensor:
    """
    Returns a tensor ~ `(n * size)` of node indices for `n` supgraphs of size `size`
    """
    res = torch.zeros([n, size])

    def get_subgraph():
        out = []
        stack = [random.randint(0, graph.num_nodes() - 1)]
        while len(out) < size:
            if len(stack) == 0:
                stack.append(random.randint(0, graph.num_nodes() - 1))
            curNode = stack.pop()
            if curNode not in out:
                out.append(curNode)
                children = graph.out_edges(curNode)[1].tolist()
                stack = children + stack
        return torch.tensor(out)
    
    for i in range(n):
        res[i] = get_subgraph()

    return res


def generate_samples(graph: dgl.DGLGraph, subgraph_size: int, n: int, method: str="neighbor") -> list[dgl.DGLGraph]:
    assert subgraph_size < graph.num_nodes()
    if method == "neighbor":
        return [get_neighborhood(graph, size=subgraph_size, start=random.randint(0, graph.num_nodes() - 1)) for _ in range(n)]
    
    raise NotImplementedError

def cast_to_raw(samples: list[dgl.DGLGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Feats, Adjs, Labels (in that order)
    """
    num_samples = len(samples)
    num_nodes = samples[0].num_nodes()
    feat_size = samples[0].ndata['feat'].shape[1] # type: ignore

    feats = torch.zeros(num_samples, 1, num_nodes, feat_size)
    adjs = torch.zeros(num_samples, 1, num_nodes, num_nodes)
    labels = torch.zeros(num_samples, 1, num_nodes, dtype=torch.long)

    for i, sample in enumerate(samples):
        feats[i] = sample.ndata['feat'] # type: ignore
        adjs[i] = sample.adj().to_dense().unsqueeze(0)
        labels[i] = sample.ndata['label'] # type: ignore

    return feats, adjs, labels


def get_neighborhood(graph: dgl.DGLGraph, size: int, start: int) -> dgl.DGLGraph:
    assert size < graph.num_nodes()
    assert start < graph.num_nodes()
    assert start >= 0
    out = []
    stack = [start]
    while len(out) < size:
        if len(stack) == 0:
            stack.append(random.randint(0, graph.num_nodes() - 1))
        curNode = stack.pop()
        if curNode not in out:
            out.append(curNode)
            children = graph.out_edges(curNode)[1].tolist()
            stack = children + stack
    return dgl.node_subgraph(graph, out)


if __name__ == "__main__":
    print("Hello world")
