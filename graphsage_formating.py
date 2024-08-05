import json
import sys
import time
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from pathlib import Path
import torch
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def get_nearest_neighbors(X: torch.Tensor, Y: torch.Tensor = None, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the
    k nearest neighbors of each data point

    Args:
        X:              Batch of data points
        Y (optional):   Defaults to None.
        k:              Number of nearest neighbors to calculate. Defaults to 3.

    Returns:
        Distances and indices of each datapoint
    """

    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(Y)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def compute_scale(Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function

    Args:
        Dis:        Distances of the k nearest neighbors of each data point.
        k:          Number of nearest neighbors. Defaults to 2.
        med:        Scale calculation method. Can be calculated by the median distance
                    from a data point to its neighbors, or by the maximum distance. 
        is_local:   Local distance (different for each data point), or global distance. Defaults to local.

    Returns:
        scale (global or local)
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True) -> torch.Tensor:   
    """
    Computes the Gaussian similarity function 
    according to a given distance matrix D and a given scale

    Args:
        D:      Distance matrix 
        scale:  scale
        Ids:    Indices of the k nearest neighbors of each sample
        device: Defaults to torch.device("cpu")
        is_local:  Determines whether the given scale is global or local 

    Returns:
        Matrix W with Gaussian similarities
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale ** 2))
    else:
        # local scales
        W = torch.exp(-torch.pow(D, 2).to(device) / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2))
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        
        # mask2 = torch.zeros([n, n]).to(device=device)
        
        rows = torch.arange(n).unsqueeze(1)
        mask[rows, Ids] = 1
        
        # for i in range(len(Ids)):
        #     mask2[i, Ids[i]] = 1
        
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.
    return sym_W



def get_affinity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the affinity matrix W

    Args:
        X (torch.Tensor):  Data

    Returns:
        torch.Tensor: Affinity matrix W
    """
    is_local = True
    n_neighbors = 8
    scale_k = 5
    Dx = torch.cdist(X,X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    scale = compute_scale(Dis, k=scale_k, is_local=is_local)
    W = get_gaussian_kernel(Dx, scale, indices, device=torch.device("cpu"), is_local=is_local)
    return W


def load_twomoons():
    n = 3000
    mean_deg = 3
    X, y = make_moons(n_samples=n, noise=0.05, random_state=42)

    # A = np.zeros((n, n))
    # num_edges = int(mean_deg * n / 2)

    # # Randomly add edges until the desired mean degree is achieved
    # edges_added = 0
    # while edges_added < num_edges:
    #     i = np.random.randint(0, n)
    #     j = np.random.randint(0, n)
    #     if i != j and A[i, j] == 0:
    #         A[i, j] = 1
    #         # A[j, i] = 1
    #         edges_added += 1
    # A_ = A + A.T
    # A_ = A_ + np.eye(n)
    # A_inv = np.linalg.pinv(A_)

    # X = A_inv @ X


    # A_r = A + A.T + np.eye(n)
    # X_r = A_r @ X
    
    # plt.scatter(X_r[:, 0], X_r[:, 1], c=y)
    # plt.savefig('twomoons.png')
    # plt.close
    # exit() 
    
    X = torch.FloatTensor(X)
    
    A = get_affinity_matrix(X)
    
    
    A = torch.FloatTensor(A)
    y = torch.LongTensor(y)
    
    
    train_size = 0.85 * n
    val_size = 0.05 * n
    idx = np.random.permutation(n)
    idx_train = idx[:int(train_size)]
    idx_val = idx[int(train_size):int(train_size + val_size)]
    idx_test = idx[int(train_size + val_size):]
    
    return A, X, y, idx_train, idx_val, idx_test


# def load_raw_data_cora():
#     paper_features_label = np.genfromtxt('./data/cora/cora.content', dtype=str)
#     features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
#     labels = paper_features_label[:, -1]
#     lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
#     labels = [lbl2idx[e] for e in labels]
#     papers = paper_features_label[:, 0].astype(np.int32)
#     paper2idx = {k: v for v, k in enumerate(papers)}
#     edges = np.genfromtxt('./data/cora/cora.cites', dtype=np.int32)
#     edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
#     adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                      shape=(len(labels), len(labels)), dtype=np.float32)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     # features = normalize(features)
#     # adj = normalize(adj + eye(adj.shape[0]))  # Normalizing for removing gradient vanishing and exploding problem
#     adj = torch.FloatTensor(adj.todense())
#     features = torch.FloatTensor(features.todense())
#     labels = torch.LongTensor(labels)
#     train_size = 0.85 * len(labels)
#     val_size = 0.05 * len(labels)
#     idx = np.random.permutation(len(labels))
#     idx_train = idx[:int(train_size)]
#     idx_val = idx[int(train_size):int(train_size + val_size)]
#     idx_test = idx[int(train_size + val_size):]
#     print("Number of training nodes: ", len(idx_train))
#     return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_raw_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)
    
    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))
    labels = torch.LongTensor(labels)
    
    idx = np.random.permutation(labels.shape[0])
    idx_train = idx[:int(0.85 * len(labels))]
    idx_val = idx[int(0.85 * len(labels)):int(0.9 * len(labels))]
    idx_test = idx[int(0.9 * len(labels)):]
    print("Number of training nodes: ", len(idx_train))
    print("Number of validation nodes: ", len(idx_val))
    print("Number of testing nodes: ", len(idx_test))
    
    
    return adj, features, labels, idx_train, idx_val, idx_test


def create_graphsage_data(dataset):
    # adj, features, labels, idx_train, idx_val, idx_test = load_raw_data(dataset)
    adj, features, labels, idx_train, idx_val, idx_test = load_twomoons()
    grapf_dict = {
        "directed": False,
        "graph": [],
        "nodes": [],
        "links": []
    }
    for node in range(adj.shape[0]):
        grapf_dict["nodes"].append({
            "id": str(node),
            "test": node in idx_test,
            "val": node in idx_val
        })

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] > 0:
                grapf_dict["links"].append({
                    "source": i,
                    "target": j
                })
                

    with open(f'./data/{dataset}/{dataset}-G.json', 'w') as f:
        json.dump(grapf_dict, f)
        f.close()

    with open(f'./data/{dataset}/{dataset}-feats.npy', 'wb') as f:
        np.save(f, features)
        f.close()

    id_map = {j: i for i, j in enumerate(range(adj.shape[0]))}
    with open(f'./data/{dataset}/{dataset}-id_map.json', 'w') as f:
        json.dump(id_map, f)
        f.close()

    class_map = {i: int(j) for i, j in enumerate(labels)}
    with open(f'./data/{dataset}/{dataset}-class_map.json', 'w') as f:
        json.dump(class_map, f)
        f.close()



create_graphsage_data('twomoons')
