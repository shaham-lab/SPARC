import sys
import pickle as pkl
from pathlib import Path
from numpy import linalg as LA
import networkx as nx
import scipy as sp
import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt, colors
from scipy.sparse import csr_matrix, coo_matrix, eye, diags
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from metrics import Metrics
from utils import plot_sorted_laplacian, plot_laplacian_eigenvectors, get_grassman_distance, get_nearest_neighbors, \
    compute_scale, get_gaussian_kernel


def split_data(x, y, adj, train_ratio):
    x_train, x_test, y_train, y_test, adj_train, adj_test = train_test_split(x, y, adj, train_size=train_ratio,
                                                                             random_state=0)
    return x_train, x_test, y_train, y_test, adj_train, adj_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_split(y, nclass, train_ratio=0.6, val_ratio=0.2):
    percls_trn = int(round(train_ratio * len(y) / nclass))
    val_lb = int(round(val_ratio * len(y)))

    indices = []
    for i in range(nclass):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index


def divide_graph_to_batches(V, y, A, batch_size):
    V = torch.tensor(V)
    y = torch.tensor(y)
    A = torch.tensor(A)

    # Step 1: Calculate label distribution
    label_counts = torch.bincount(y)

    # Step 2: Sort nodes by labels
    sorted_nodes = torch.argsort(y)

    # Step 3: Calculate node distances
    node_distances = torch.zeros_like(A)
    for i in range(len(V)):
        visited = torch.zeros(V.shape[0], dtype=torch.bool)
        queue = [i]
        visited[i] = True
        while queue:
            current_node = queue.pop(0)
            neighbors = torch.nonzero(A[current_node]).flatten()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    node_distances[i, neighbor] = node_distances[i, current_node] + 1
                    queue.append(neighbor)

    # Step 4: Sort nodes by distances
    sorted_nodes = sorted_nodes[torch.lexsort((node_distances[sorted_nodes].T, sorted_nodes))]

    # Step 5: Divide nodes into batches
    batches = []
    current_batch = []
    current_label_counts = torch.zeros_like(label_counts)
    for node in sorted_nodes:
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
            current_label_counts = torch.zeros_like(label_counts)
        if current_label_counts[y[node]] < label_counts[y[node]]:
            current_batch.append(node)
            current_label_counts[y[node]] += 1

    if current_batch:
        batches.append(current_batch)

    return batches

def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='../data',
                                train=True,
                                download=True,
                                transform=tensor_transform)
    test_set = datasets.MNIST(root='../data',
                                train=False,
                                download=True,
                                transform=tensor_transform)

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test



def load_raw_data_cora():
    path = Path('../data/cora')
    paper_features_label = np.genfromtxt('../data/cora/cora.content', dtype=str)
    features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
    labels = paper_features_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]
    papers = paper_features_label[:, 0].astype(np.int32)
    paper2idx = {k: v for v, k in enumerate(papers)}
    edges = np.genfromtxt(path / 'cora.cites', dtype=np.int32)
    edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
    adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(len(labels), len(labels)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # adj = normalize(adj + eye(adj.shape[0]))  # Normalizing for removing gradient vanishing and exploding problem
    adj = torch.FloatTensor(adj.todense())
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    return adj, features, labels


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_raw_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def get_affinities(x: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(x, x)
    W = torch.exp(-1 * 23 * (dists ** 2))
    W.fill_diagonal_(0)
    return W


def get_closest_k_neighbors_matrix(x: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(x, x)
    _, indices = torch.topk(dists, k + 1, dim=1, largest=False)
    indices = indices[:, 1:]
    W = torch.zeros_like(dists)
    W.scatter_(1, indices, 1)
    return W

def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def generate_data(dataset):
    if dataset in ['cora', 'citeseer']:

        adj, x, y = load_raw_data_cora()

        # adj = adj.todense()
        # x = x.todense()
        # x = feature_normalize(x)

        x = torch.FloatTensor(x)
        # y = torch.LongTensor(y)
        adj = torch.FloatTensor(adj)

        torch.save([x, y, adj],  '../data/cora/{}.pt'.format(dataset))


def load_data(dataset: str, nclass: int, train_ratio=0.6, valid_ratio=0.2) -> tuple:
    x, y, adj = torch.load('../data/{}/{}.pt'.format(dataset, dataset))

    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(dataset, y, nclass, train_ratio, valid_ratio)
    train, valid, test = map(torch.LongTensor, (train, valid, test))

    return x, y, adj, train, valid, test


def load_twomoon() -> tuple:
        data, y = make_moons(n_samples=2500, shuffle=True, noise=0.075, random_state=None)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
        x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
        y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
        return x_train, y_train, x_test, y_test

def load_reuters() -> tuple:
    with h5py.File('../data/Reuters/reutersidf_total.h5', 'r') as f:
        x = np.asarray(f.get('data'), dtype='float32')
        y = np.asarray(f.get('labels'), dtype='float32')

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test

def load_from_path(dpath: str, lpath: str = None) -> tuple:
    X = np.loadtxt(dpath, delimiter=',', dtype=np.float32)
    n_train = int(0.9 * len(X))

    x_train, x_test = X[:n_train], X[n_train:]
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    
    if lpath is not None:
        y = np.loadtxt(lpath, delimiter=',', dtype=np.float32)
        y_train, y_test = y[:n_train], y[n_train:]
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    else:
        y_train, y_test = None, None

    return x_train, y_train, x_test, y_test

def in1D(x, labels):
        """
        Sub-optimal equivalent to numpy.in1D().
        Hopefully this feature will be properly covered soon
        c.f. https://github.com/pytorch/pytorch/issues/3025
        Snippet by Aron Barreira Bordin
        Args:
            x (Tensor):             Tensor to search values in
            labels (Tensor/list):   1D array of values to search for

        Returns:
            Tensor: Boolean tensor y of same shape as x, with y[ind] = True if x[ind] in labels

        Example:
            in1D(torch.FloatTensor([1, 2, 0, 3]), [2, 3])
            FloatTensor([False, True, False, True])
        """
        mapping = torch.zeros(x.size()).byte()
        for label in labels:
            mapping = mapping | x.eq(label)
        return mapping


def compact1D(x):
    """
    "Compact" values 1D uint tensor, so that all values are in [0, max(unique(x))].
    Args:
        x (Tensor): uint Tensor

    Returns:
        Tensor: uint Tensor of same shape as x

    Example:
        densify1D(torch.ByteTensor([5, 8, 7, 3, 8, 42]))
        ByteTensor([1, 3, 2, 0, 3, 4])
    """
    x_sorted, x_sorted_ind = torch.sort(x, descending=True)
    x_sorted_unique, x_sorted_unique_ind = torch.unique(x_sorted, return_inverse=True)
    x[x_sorted_ind] = x_sorted_unique_ind
    return x

def symmetric_sparse_matrix(idx, sparse_matrix):
    adj = sparse_matrix.coalesce()
    i = adj.indices()
    v = adj.values()
    v_idx = in1D(i, idx).byte()
    v_idx = v_idx.sum(dim=0).squeeze() == i.size(0)  # or `v_idx.all(dim=1)` for pytorch 0.5+
    v_idx = v_idx.nonzero().squeeze()

    # Slicing `v` and `i` accordingly:
    v_sliced = v[v_idx]
    i_sliced = i.index_select(dim=1, index=v_idx)

    # Building sparse result tensor:
    i_sliced[0] = compact1D(i_sliced[0])
    i_sliced[1] = compact1D(i_sliced[1])

    # To make sure to have a square dense representation:
    size_sliced = torch.Size([len(idx), len(idx)])
    res = torch.sparse.FloatTensor(i_sliced, v_sliced, size_sliced)

    # Rearrange the matrix in the order of `idx`


    permutation = torch.argsort(torch.argsort(idx))
    res = res.to_dense()
    res = res[permutation, :][:, permutation]
    return res

def matrix_from_dict(indices, adj_dict):
        """
        This function creates a sparse matrix from a dictionary of indices.

        Args:
            indices (dict): The dictionary of indices
        """
        m = len(indices)
        matrix = torch.zeros(m, m)
        for i in range(m):
            for j in range(m):
                if indices[j].item() in adj_dict[indices[i].item()]:
                    matrix[i][j] = 1

        return matrix + torch.eye(m)

if __name__ == "__main__":
    # adj, features, labels = load_raw_data('cora')
    #
    # labels = labels.argmax(axis=1)
    # adj = adj.todense()
    # features = features.todense()
    # features = torch.from_numpy(features)
    # features_reshaped = features.view(-1, features.shape[-1])
    # unique_features = torch.unique(features_reshaped, dim=0)
    # for i in range(len(unique_features)):
    #     mask = torch.all(features_reshaped == unique_features[i], dim=1)
    #     if torch.count_nonzero(mask) > 1:
    #         print('Duplicate found')
    #         print(torch.nonzero(mask).squeeze())




    print(1)


    # A, X, y = load_raw_data('citeseer')
    #
    # A = A.todense()
    # X = X.todense()
    # y = y.argmax(axis=1)
    #
    # X = torch.from_numpy(X)
    # y = torch.from_numpy(y)
    # A = torch.from_numpy(A)
    #
    # indices = torch.arange(0, len(y))
    #
    # adj_dict = {}
    # for i in indices:
    #     adj_dict[i.item()] = set()
    #     for j in indices:
    #         if A[i][j] != 0:
    #             adj_dict[i.item()].add(j.item())
    #
    # torch.save(adj_dict, '../data/citeseer/adj_dict.pt')
    #
    # A = A + torch.eye(A.shape[0])
    # A = A.to_sparse()
    # torch.save(A, '../data/citeseer/adj_sparse.pt')
    #
    # torch.save([X, y, indices], '../data/citeseer/citeseer.pt')


    # adj = torch.load('../data/cora/adj_sparse.pt')
    #
    # adj = adj.to_dense()
    #
    # D = torch.diag(torch.sum(adj, dim=1))
    # D_inv = torch.inverse(D)
    # D_inv_sqrt = torch.sqrt(D_inv)
    #
    # L_sym = torch.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt
    #
    # V, U = np.linalg.eigh(L_sym.numpy())
    #
    # U = torch.from_numpy(U)
    # U = U[:, np.argsort(V)[:1024]]
    #
    # torch.save(U, '../data/cora/laplacian_eigen.pt')

    # adj, features, labels = load_raw_data('pubmed')
    #
    # indices = torch.arange(0, len(labels))
    #
    # features = features.todense()
    # features = torch.from_numpy(features)
    #
    # labels = torch.from_numpy(labels)
    # labels = torch.argmax(labels, dim=1)
    #
    # torch.save([features, labels, indices], '../data/pubmed/pubmed.pt')
    #
    # adj = adj.todense()
    # adj = torch.from_numpy(adj)
    # adj = adj + torch.eye(adj.size(0))
    # adj_sparse = adj.to_sparse()
    #
    # torch.save(adj_sparse, '../data/pubmed/adj_sparse.pt')
    #
    # adj_dict = {}
    #
    # for i in indices:
    #     adj_dict[i.item()] = set(torch.nonzero(adj[i]).squeeze().tolist())
    #
    # torch.save(adj_dict, '../data/pubmed/adj_dict.pt')




    #
    # adj_dict = torch.load('../data/cora/adj_dict.pt')
    # adj_sparse = torch.load('../data/cora/adj_sparse.pt')
    # adj = torch.load('../data/cora/adj_matrix.pt')
    # x, y, indices = torch.load('../data/cora/cora.pt')
    # adj = adj + torch.eye(adj.size(0))
    # adj_sparse =adj.to_sparse()
    # torch.save(adj_sparse, '../data/cora/adj_sparse.pt')
    #
    # # batch = indices[:8]
    # # while True:
    # #     neighbors = [adj_dict[i.item()] for i in batch]
    # #     # batch = torch.unique(torch.cat([batch, torch.cat(torch.tensor([list(adj_dict[i.item()]) for i in batch]))]))
    # #     if len(batch) > 128:
    # #         break
    # #
    # # print(len(batch))
    #
    # n = 2700
    # m = 1200
    #
    # # adj = torch.randint(0, 2, (n, n))
    # # adj = torch.arange(n * n).reshape(n, n)
    # # adj_sparse = adj.to_sparse()
    # # adj_dict = {}
    # # for i in range(n):
    # #     adj_dict[i] = set()
    # #     for j in range(n):
    # #         if adj[i][j] != 0:
    # #             adj_dict[i].add(j)
    # # indices = torch.arange(n)
    #
    # perm = torch.randperm(len(indices))
    # indices = indices[perm]
    # p = perm[:m]
    # c = indices[:m]
    # # p = torch.tensor([1, 1136, 1516, 2])
    #
    # #
    # print(1)
    # adj_1 = symmetric_sparse_matrix(p, adj_sparse)
    # print(2)
    # adj_2 = matrix_from_dict(p, adj_dict)
    # print(3)
    # adj_3 = adj[p, :][:, p]
    #
    # print(torch.count_nonzero(adj_1))
    # print(torch.count_nonzero(adj_2))
    # print(torch.count_nonzero(adj_3))
    #
    # print(torch.all(torch.eq(adj_1, adj_2)))
    # print(torch.all(torch.eq(adj_1, adj_3)))
    # print(torch.all(torch.eq(adj_2, adj_3)))
    #
    # print(torch.all(torch.eq(adj_1, adj_1.T)))
    # print(torch.all(torch.eq(adj_2, adj_2.T)))
    # print(torch.all(torch.eq(adj_3, adj_3.T)))

    # if torch.all(torch.eq(adj_1, adj_3)) == False:
    #     print(p)
    #     print(c)




    # W = adj.numpy() + np.eye(adj.shape[0])
    # D = np.diag(np.sum(W, axis=1))
    # D_inv = np.linalg.inv(D)
    # D_inv_sqrt = np.sqrt(D_inv)
    #
    # L = D - W
    # L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    # L_rw = np.eye(W.shape[0]) - D_inv @ W
    #
    # V, U = np.linalg.eigh(L)
    # V_sym, U_sym = np.linalg.eigh(L_sym)
    # V_rw, U_rw = np.linalg.eigh(L_rw)
    #
    # U = U[:, np.argsort(V)[:1024]]
    # U_sym = U_sym[:, np.argsort(V_sym)[:1024]]
    # U_rw = U_rw[:, np.argsort(V_rw)[:1024]]
    #
    # U = torch.from_numpy(U)
    # U_sym = torch.from_numpy(U_sym)
    # U_rw = torch.from_numpy(U_rw)
    # torch.save([U, U_sym, U_rw], '../data/cora/laplacian_eigen.pt')








    # adj_train = adj_train.float()
    # adj_train = adj_train.numpy()
    # # adj_train = adj_train - 0.1 * np.ones(adj_train.shape)
    # row_sums = adj_train.sum(axis=1)
    # degree_matrix = np.diag(row_sums)
    # laplacian_matrix = degree_matrix - adj_train
    # U, V = eigsh(laplacian_matrix, k=1, which='SM')
    #
    # # t = V @ np.diag(U) @ V.T
    # # U, V = largest_eigh(laplacian_matrix)
    # torch.save([x_train, torch.tensor(V), adj_train, y_train], 'embeddings_real.pt')
    # print(get_grassman_distance(embeddings, V))








