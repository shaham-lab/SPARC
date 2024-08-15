import sys
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
import umap
import utils, partition_utils
from utils import *
from data import load_data
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering


SEED = 42

class InvalidMatrixException(Exception):
    pass

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(data_prefix, dataset_str, precalc):
    """Return the required data formats for GCN models."""
    (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
    train_data, val_data,
    test_data) = utils.load_graphsage_data(data_prefix, dataset_str)
    visible_data = train_data

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]

    train_mask = utils.sample_mask(train_data, labels.shape[0])
    val_mask = utils.sample_mask(val_data, labels.shape[0])
    test_mask = utils.sample_mask(test_data, labels.shape[0])

    if precalc:
        train_feats = train_adj.dot(feats)
        # train_feats = np.hstack((train_feats, feats))
        test_feats = full_adj.dot(feats)
        # test_feats = np.hstack((test_feats, feats))
    else:
        train_feats = feats
        test_feats = feats
        
    # plt.scatter(test_feats[:, 0], test_feats[:, 1], c=np.argmax(y_train, axis=1))
    # plt.savefig('train.png')
    # plt.clf()
    # exit()
    
    return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, train_data, val_data, test_data,
            num_data, visible_data)


def main():
    set_seed(SEED)
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset = config["dataset"]
    n_clusters = config["n_clusters"]
    bsize = config["spectral"]["bsize"]
    num_clusters = config["spectral"]["num_clusters"]
    num_clusters_val = config["spectral"]["num_clusters_val"]
    num_clusters_test = config["spectral"]["num_clusters_test"]
    diag_lambda = config["spectral"]["diag_lambda"]
    output_dim = config["spectral"]["architecture"]["output_dim"]

    
    print('Loading data...')
    
    (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, _, val_data, test_data, num_data,
     visible_data) = load_data('/data', dataset, False)

    try:
        parts = torch.load(f'./cached_partition/{dataset}/{SEED}/parts.pt')
    except FileNotFoundError: 
        if not os.path.exists(f'./cached_partition/{dataset}/{SEED}'):
            os.makedirs(f'./cached_partition/{dataset}/{SEED}')
        _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                               num_clusters)
        parts = [np.array(pt) for pt in parts]
        # torch.save(parts, f'./cached_partition/{dataset}/{SEED}/parts.pt')
    
    
    (features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.preprocess_multicluster(
        train_adj, parts, train_feats, y_train, train_mask,
        num_clusters, bsize, diag_lambda)

    (_, val_features_batches, val_support_batches, y_val_batches,
     val_mask_batches) = utils.preprocess(full_adj, test_feats, y_val, val_mask,
                                          np.arange(num_data),
                                          num_clusters_val,
                                          diag_lambda)

    (_, test_features_batches, test_support_batches, y_test_batches,
     test_mask_batches) = utils.preprocess(full_adj, test_feats, y_test,
                                           test_mask, np.arange(num_data),
                                           num_clusters_test,
                                           diag_lambda)
    idx_parts = list(range(len(parts)))

    train = (parts, features_batches, support_batches, y_train_batches, train_mask_batches)
    val = (idx_parts, val_features_batches, val_support_batches, y_val_batches, val_mask_batches)
    test = (idx_parts, test_features_batches, test_support_batches, y_test_batches, test_mask_batches)

    y_train = np.argmax(y_train, axis=1)
    unique, counts = np.unique(y_train, return_counts=True)
    print('Train:', unique, counts)
    y_val = np.argmax(y_val, axis=1)
    unique, counts = np.unique(y_val, return_counts=True)
    print('Val:', unique, counts)
    y_test = np.argmax(y_test, axis=1)
    unique, counts = np.unique(y_test, return_counts=True)
    print('Test:', unique, counts)
    



    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(train_feats)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=y_train)
    # plt.colorbar()
    # plt.savefig('umap.png')
    # plt.clf()
    # exit()
    
    # X = torch.tensor(train_feats, dtype=torch.float32)
    # A = full_adj.todense()
    # A = A + A @ A + A @ A @ A + A @ A @ A @ A
    # L = sort_laplacian(A, y_train)
    
    # L = np.log(L + 1)
    # L = F.normalize(torch.tensor(L, dtype=torch.float32), p=1, dim=1)
    
    # plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    # plt.colorbar()
    # plt.savefig('block_diagonal.png')
    # plt.clf()
    
    # # W = utils.get_affinity_matrix(X)
    # n_neighbors = 10
    # scale = 1.0
    # # Step 1: Compute nearest neighbors
    # nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X.numpy())
    # distances, indices = nn_model.kneighbors(X.numpy())

    # # Step 2: Construct affinity matrix
    # distances = torch.from_numpy(distances.astype(np.float32))
    # indices = torch.from_numpy(indices.astype(np.int64))
    # sigma2 = torch.mean(distances[:, -1]) ** 2
    # affinity_matrix = torch.exp(-distances.pow(2) / sigma2)
    # affinity_matrix = torch.zeros_like(affinity_matrix).scatter_(1, indices, affinity_matrix)

    # # Step 3: Normalize affinity matrix
    # row_sum = torch.sum(affinity_matrix, dim=1, keepdim=True)
    # normalized_affinity_matrix = affinity_matrix / row_sum

    # # Optionally scale the normalized affinity matrix
    # normalized_affinity_matrix *= scale
    # W = normalized_affinity_matrix
    
    # # W = F.normalize(W, p=1, dim=1)
    # W = sort_laplacian(W, y_train)
    
    # plt.imshow(W, cmap='hot', norm=colors.LogNorm())
    # plt.colorbar()
    # plt.savefig('affinity.png')
    # plt.clf()
    
    # exit()
    

    
    # X = torch.tensor(train_feats, dtype=torch.float32)
    # y = torch.tensor(y_train[train_mask], dtype=torch.long)
    # A = full_adj.todense()
    # A = torch.tensor(A, dtype=torch.float32)
    # A = A + torch.eye(A.shape[0])
    # A = A @ A
    #
    # # X = A @ X
    #
    # L = utils.get_affinity_matrix(X)
    # L = utils.sort_laplacian(L, y)
    # L = np.log(L + 1)
    # L = F.normalize(torch.tensor(L, dtype=torch.float32), p=1, dim=1)
    #
    # plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    # plt.colorbar()
    # plt.savefig('block_diagonal.png')
    # plt.show()
    # plt.clf()
    # exit(1)

    # k_s = [1, 5, 10, 20, 50, 100]
    # for k in k_s:
    #     idx = torch.argsort(dist, dim=1)
    #     acc = 0
    #     for i in range(dist.shape[0]):
    #         labels = y[idx[i, 1:k+1]]
    #         counts = torch.bincount(labels)
    #         cluster = torch.argmax(counts)
    #         if cluster == y[i]:
    #             acc += 1
    #     print(f'KNN-{k}:', acc / dist.shape[0])
    # # # knn clustering
    # # k = 100
    # # idx = torch.argsort(dist, dim=1)
    # # acc = 0
    # # for i in range(dist.shape[0]):
    # #     labels = y[idx[i, 1:k+1]]
    # #     counts = torch.bincount(labels)
    # #     cluster = torch.argmax(counts)
    # #     if cluster == y[i]:
    # #         acc += 1
    # # print('KNN:', acc / dist.shape[0])
    # exit()
    
    
    # min_dist = []
    # max_dist = []
    # # print each node closest node distance and furtherest node distance
    # for i in range(dist.shape[0]):
    #     idx = torch.argsort(dist[i])
    #     min_dist.append(dist[i, idx[1]])
    #     max_dist.append(dist[i, idx[-1]])
    #     for j in range(100):
    #         print(dist[i, idx][j], end=' ')
    #         print(y[idx][j], end=' ')
    #     print(f'Node {i}: Closest node distance: {dist[i, idx[1]]}, Furtherest node distance: {dist[i, idx[-1]]}')
    
    # print('Mean closest node distance:', np.mean(min_dist))
    # print('Mean furtherest node distance:', np.mean(max_dist))
    # exit()
    
    # acc = 0
    # # check if the closest node is the same class
    # for i in range(dist.shape[0]):
    #     idx = torch.argsort(dist[i])
    #     if y[i] == y[idx[1]]:
    #         acc += 1
    
    # print('Baseline:', acc / dist.shape[0])
    # exit()
    
        
    
    # X = torch.tensor(train_feats, dtype=torch.float32)
    # params = utils.find_params(X, y_train[train_mask])
    # print(params)
    # exit()
    
    # A = full_adj.todense()
    # # A = A + np.eye(A.shape[0])
    # # # A = np.power(A, 3)
    # # # A[A > 0] = 1
    

    
    # # # # print('mean degree:', np.mean(np.sum(A, axis=1)))
    # A = full_adj.todense()
    # A = torch.tensor(A, dtype=torch.float32)
    
    # # keep only 3 edges per node
    # for i in range(A.shape[0]):
    #     idx = np.argsort(A[i])
    #     A[i, idx[:-2]] = 0
    # # A = A @ A.T
    
    # A = A + torch.eye(A.shape[0])
    # A = A + A.T
    # A[A < 1] = 0
    # A = A @ A
    # # D_inv_sqrt = torch.diag(torch.pow(torch.sum(A, dim=1), -0.5))
    # # A = D_inv_sqrt @ A @ D_inv_sqrt
    # # gaussian kernel
    
    
    
    # X = torch.tensor(train_feats, dtype=torch.float32)
    # W = utils.get_affinity_matrix(A @ X)
    
    # A_ = sort_laplacian(A, y_train)
    # plt.imshow(A_, cmap='hot', norm=colors.LogNorm())
    # plt.show()
    # plt.savefig('adjacency.png')
    # plt.clf()
    
    

    # W_ = sort_laplacian(W, y_train)
    # plt.imshow(W_, cmap='hot', norm=colors.LogNorm())
    # plt.show()
    # plt.savefig('affinity.png')
    # plt.clf()
    
    # # # W = F.normalize(W, p=1, dim=1)
    # # # A = F.normalize(A, p=1, dim=1)
    
    # A = A + W
    
    # # # mask with probability 0.5
    # # mask = torch.rand(A.shape) < 0.9
    
    # # A[mask] = 0
    
    
    # # spectral clustering
    # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=SEED)
    # cluster_labels = spectral_clustering.fit_predict(A)
    
    
    # metrics = Metrics()
    
    # nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    # acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    # print(f'NMI: {nmi}, ACC: {acc}')
    # exit()
    
    
    # # X = A @ X
    # W = utils.get_affinity_matrix(X)
    
    # W = W + A
    # # # W[W > 0] = 1
    # # # W = A
    # # W = F.normalize(W, p=1, dim=1)
    # # W = sort_laplacian(W, y_train)
    # D = torch.diag(torch.sum(W, dim=1))
    # L = D - W
    # V, U = torch.linalg.eig(L)
    # V = V.real
    # U = U.real
    # idx = V.argsort()
    # V = V[idx]
    # U = U[:, idx]
    # print(V)
    # U = U[:, 1:output_dim+1]
    
    # kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    # kmeans.fit(U)
    # cluster_labels = kmeans.labels_
    
    # metrics = Metrics()
    
    # nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    # acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    # print(f'NMI: {nmi}, ACC: {acc}')
    # exit()
    
    # plt.imshow(W, cmap='hot', norm=colors.LogNorm())
    # plt.show()
    # plt.savefig('laplacian.png')
    
    
    # eigvals, eigvecs = eigsh(W, k=10, which='SM')
    # print(eigvals)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    # kmeans.fit(eigvecs)
    # cluster_labels = kmeans.labels_
    
    # metrics = Metrics()
    
    # nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    # acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    # print(f'NMI: {nmi}, ACC: {acc}')
    # exit()
    
    # L = sort_laplacian(W, y_train)
    
    # plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    # plt.imshow(L, cmap='flag')
    # plt.show()
    # plt.savefig('block_diagonal.png')
    # # exit()
    
    spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
    spectralnet.fit(train, val)


    X = torch.tensor(test_feats, dtype=torch.float32)
    # print(X.shape)
    Y = spectralnet.predict(X)
    Y = Y / np.sqrt(Y.shape[0])
    Y = Y / np.sqrt(np.sum(Y ** 2, axis=1, keepdims=True))
    
    # plot_laplacian_eigenvectors(Y, y_test)
    
    
    
    # np.save(f'./results/{dataset}/{output_dim}/{SEED}/spectralnet.npy', Y)
    
    kmeans = KMeans(n_clusters=70, random_state=SEED)
    kmeans.fit(Y)
    cluster_labels = kmeans.labels_
    outliares = 0
    for i in range(70):
        cluster = y_train[cluster_labels == i]
        print(np.bincount(cluster))
        outliares += np.sum(np.bincount(cluster)) - np.max(np.bincount(cluster))
    print(outliares)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(Y)
    cluster_labels = kmeans.labels_
    
    
    cm = clustering_metrics(cluster_labels[train_mask], y_train[train_mask])
    acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
    
    print('#################')
    
    cm = clustering_metrics(cluster_labels[test_mask], y_test[test_mask])
    acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
    
   
    # acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    # nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    # f1 = f1_score_optimal_permutation(y_train[train_mask], cluster_labels[train_mask], average='macro')
    # print(f'Train NMI: {nmi}, Train ACC: {acc}, Train F1: {f1}')
    

    
    
    # # plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
    # # plt.savefig('cluster.png')
    
    # print('##################')
    
    # nmi = metrics.nmi_score(cluster_labels[test_mask], y_test[test_mask])
    # acc = metrics.acc_score(cluster_labels[test_mask], y_test[test_mask], n_clusters)
    # f1 = f1_score_optimal_permutation(y_test[test_mask], cluster_labels[test_mask], average='macro')
    # print(f'Test NMI: {nmi}, Test ACC: {acc}, Test F1: {f1}')
    
    
    

    
if __name__ == "__main__":
    # print('Starting...')
    main()

