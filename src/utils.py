import json
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import sklearn
from networkx.readwrite import json_graph

# from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors

import partition_utils


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=bool)


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj

def normalize_adj_diag_enhance(adj, diag_lambda):
    """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (rowsum + 1e-20)
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
    return adj


def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

def sparse_to_tuple(sparse_mx):
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
    """Generate the batch for multiple clusters."""

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
        features_batches.append(torch.tensor(features[pt, :]).float())
        y_train_batches.append(torch.tensor(y_train[pt, :]).float())
        support_now = adj[pt, :][:, pt]
        if diag_lambda == -1:
            # support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
            support_batches.append(sparse_to_tuple(support_now))
        else:
            support_batches.append(
                sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (features_batches, support_batches, y_train_batches,
          train_mask_batches)

def preprocess(adj,
               features,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1):
    """Do graph partitioning and preprocessing for SGD training."""

    # Do graph partitioning
    part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
    if diag_lambda == -1:
        part_adj = normalize_adj(part_adj)
    else:
        part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
    parts = [np.array(pt) for pt in parts]

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    for pt in parts:
        features_batches.append(torch.tensor(features[pt, :]).float())
        now_part = part_adj[pt, :][:, pt]
        total_nnz += now_part.count_nonzero()
        support_batches.append(sparse_to_tuple(now_part))
        y_train_batches.append(torch.tensor(y_train[pt, :]).float())

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (parts, features_batches, support_batches, y_train_batches,
          train_mask_batches)

def load_graphsage_data(dataset_path, dataset_str, normalize=True):
    """Load GraphSAGE data."""
    start_time = time.time()

    current_path = os.getcwd()

    with open(f"{current_path}/{dataset_path}/{dataset_str}/{dataset_str}-G.json", "r") as file:
        graph_json = json.load(file)

    graph_nx = json_graph.node_link_graph(graph_json)

    with open(f"{current_path}/{dataset_path}/{dataset_str}/{dataset_str}-id_map.json", "r") as file:
        id_map = json.load(file)

    is_digit = list(id_map.keys())[0].isdigit()

    id_map = {k: int(v) for k, v in id_map.items()}
    id_map_re = {value: key for key, value in id_map.items()}
    with open(f"{current_path}/{dataset_path}/{dataset_str}/{dataset_str}-class_map.json", "r") as file:
        class_map = json.load(file)

    is_instance = isinstance(list(class_map.values())[0], list)

    broken_count = 0
    to_remove = []
    for node in graph_nx.nodes():
      if node not in id_map and node not in id_map_re:
        to_remove.append(node)
        broken_count += 1
    for node in to_remove:
      graph_nx.remove_node(node)
    # print(f'Removed {broken_count} nodes that lacked proper annotations due to networkx versioning issues')

    with open(f"{current_path}/{dataset_path}/{dataset_str}/{dataset_str}-feats.npy", "rb") as file:
        feats = np.load(file)

    # print('Loaded data in {:.2f} seconds'.format(time.time() - start_time))
    start_time = time.time()

    # print('num of nodes: ' + str(len(graph_nx.nodes())))

    edges = []

    for edge in graph_nx.edges():
        if edge[0] in id_map_re and edge[1] in id_map_re:
            edges.append((edge[0], edge[1]))
    num_data = len(id_map)

    # print('num of edges: ' + str(len(graph_nx.edges())))


    val_data = np.array(
        # [id_map_re[n] for n in graph_nx.nodes() if 'val' in graph_nx.nodes[n] and graph_nx.nodes[n]['val']],
        [id_map[n] for n in graph_nx.nodes() if 'val' in graph_nx.nodes[n] and graph_nx.nodes[n]['val']],
        dtype=np.int32)
    # print('done val')
    # print(len(val_data))
    test_data = np.array(
        # [id_map_re[n] for n in graph_nx.nodes() if 'test' in graph_nx.nodes[n] and graph_nx.nodes[n]['test']],
        [id_map[n] for n in graph_nx.nodes() if 'test' in graph_nx.nodes[n] and graph_nx.nodes[n]['test']],
        dtype=np.int32)
    
    is_train = np.ones((num_data), dtype=bool)
    # is_train[val_data] = False
    # is_train[test_data] = False
    train_data = np.array([n for n in range(num_data) if is_train[n]],
                          dtype=np.int32)

    train_edges = [
        (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
    ]
    edges = np.array(edges, dtype=np.int32)
    train_edges = np.array(train_edges, dtype=np.int32)

    # print('edges', len(edges))
    # print('train_edges', len(train_edges))

    # Process labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], class_map[k]] = 1

    if normalize:
        train_ids = np.array([
            # n
            id_map[n]
            for n in graph_nx.nodes()
            if 'val' in graph_nx.nodes[n] and
               not graph_nx.nodes[n]['val'] and not graph_nx.nodes[n]['test']
        ])
        train_feats = feats[train_ids]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    def _construct_adj(edges):
        # print(edges.shape)
        adj = sp.csr_matrix((np.ones(
            (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(num_data, num_data))
        adj += adj.transpose()
        adj += sp.eye(num_data)
        # adj = adj @ adj 
        # binarize the adjacency matrix to 1 and 0
        # adj[adj > 0] = 1
        return adj

    train_adj = _construct_adj(train_edges)
    full_adj = _construct_adj(edges)
    
    train_feats = feats[train_data]
    test_feats = feats

    # print('Loaded data in {:.2f} seconds'.format(time.time() - start_time))
    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data

def get_number_of_clusters(X: torch.Tensor,  n_samples: int, threshold: float) -> int:
    """
    Computes the number of clusters in the given dataset

    Args:
        X:          dataset
        n_samples:  number of samples to use for computing the number of clusters
        threshold:  threshold for the eigenvalues of the laplacian matrix. This 
                    threshold is used in order to find when the difference between 
                    the eigenvalues becomes large. 

    Returns:
        Number of clusters in the dataset
    """
    indices = torch.randperm(X.shape[0])[:n_samples]
    X = X[indices]
    
    W = get_affinity_matrix(X)
    L = get_laplacian(W)
    vals = get_eigenvalues(L)
    diffs = np.diff(vals)
    cutoff = np.argmax(diffs > threshold)
    num_clusters = cutoff + 1
    return num_clusters


def build_ann(X: torch.Tensor):
    """
    Builds approximate-nearest-neighbors object 
    that can be used to calculate the knn of a data-point

    Args:
        X:  dataset
    """
    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], 'euclidean')
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)

    t.build(50)
    t.save('ann_index.ann')


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes new batch of data points from the given batch (batch_x) 
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch

    Args:
        batch_x:    Batch of data points

    Returns:
        New batch of data points
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], 'euclidean')
    u.load('ann_index.ann')
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_laplacian(W: torch.Tensor) -> np.ndarray:
    """
    Computes the un-normalized Laplacian matrix, given the affinity matrix W

    Args:
        W (torch.Tensor):   Affinity matrix
    
    Returns:
        Laplacian matrix
    """

    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the columns and the rows of the laplacian by the true lablel in order
    to see whether the sorted laplacian is a block diagonal matrix

    Args:
        L:  Laplacian matrix
        y:  labels

    Returns:
        Sorted laplacian
    """

    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L


def sort_matrix_rows(A: np.ndarray , y: np.ndarray) -> np.ndarray:
    """
    Sorts the rows of a matrix by a given order y

    Args:
        A:  Numpy ndarray
        y:  True labels
    """

    i = np.argsort(y)
    A = A[i, :]
    return A


def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of a given matrix A and sorts them in increasing order

    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvalues
    """

    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals


def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues
    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvectors
    """

    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs


def plot_eigenvalues(vals: np.ndarray):
    """
    Plot the eigenvalues of the laplacian

    Args:
        vals:   Eigenvalues
    """

    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()


def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """
    Returns eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang


def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    """
    Plot the eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    # sort the rows of V
    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt


def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    """
    Plot the block diagonal matrix that is obtained from the sorted laplacian

    Args:
        W:  Affinity matrix
        y:  True labels
    """
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    plt.imshow(L, cmap='flag')
    plt.show()
    # save fig
    plt.savefig('block_diagonal.png')


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


# def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
#     """
#     Computes the Grassmann distance between the subspaces spanned by the columns of A and B
#
#     Args:
#         A:  Numpy ndarray
#         B:  Numpy ndarray
#     """
#
#     M = np.dot(np.transpose(A), B)
#     _, s, _ = np.linalg.svd(M, full_matrices=False)
#     s = 1 - np.square(s)
#     grassmann = np.sum(s)
#
#     return grassmann


def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the Grassmann distance between the subspaces spanned by the columns of A and B.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    B : np.ndarray
        Numpy ndarray.

    Returns
    -------
    float
        The Grassmann distance.
    """

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann

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
        rows = torch.arange(n).unsqueeze(1)
        mask[rows, Ids] = 1
        # for i in range(len(Ids)):
        #     mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.
    return sym_W


def plot_data_by_assignmets(X, assignments: np.ndarray):
    """
    Plots the data with the assignments obtained from SpectralNet.
    Relevant only for 2D data

    Args:
        X:                      Data
        cluster_assignments:    Cluster assignments 
    """

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()


def calculate_cost_matrix(C: np.ndarray , n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm

    Args:
        C (np.ndarray):     Confusion matrix
        n_clusters (int):   Number of clusters

    Returns:
        np.ndarray:        Cost matrix
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices

    Args:
        indices (np.ndarray):  Indices of the clusters

    Returns:
        np.ndarray:   Cluster labels
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def write_assignmets_to_file(assignments: np.ndarray):
    """
    Saves SpectralNet cluster assignments to a file

    Args:
        assignments (np.ndarray): The assignments that obtained from SpectralNet
    """

    np.savetxt("cluster_assignments.csv", assignments.astype(int), fmt='%i', delimiter=',')


def create_weights_dir():
    """
    Creates a directory for the weights of the Autoencoder and the Siamese network
    """
    if not os.path.exists('weights'):
        os.makedirs('weights')


def get_affinity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the affinity matrix W

    Args:
        X (torch.Tensor):  Data

    Returns:
        torch.Tensor: Affinity matrix W
    """
    is_local = False
    n_neighbors = 10
    scale_k = 5
    Dx = torch.cdist(X,X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    scale = compute_scale(Dis, k=scale_k, is_local=is_local)
    W = get_gaussian_kernel(Dx, scale, indices, device=torch.device("cpu"), is_local=is_local)
    return W

