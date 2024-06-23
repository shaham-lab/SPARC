import sys
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

import utils, partition_utils
from utils import *
from data import load_data
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import torch.nn.functional as F

SEED = 10

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
    train_feats = np.hstack((train_feats, feats))
    test_feats = full_adj.dot(feats)
    test_feats = np.hstack((test_feats, feats))

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
     visible_data) = load_data('/data', dataset, True)

    try:
        parts = torch.load(f'./cached_partition/{dataset}/{SEED}/parts.pt')
    except FileNotFoundError: 
        if not os.path.exists(f'./cached_partition/{dataset}/{SEED}'):
            os.makedirs(f'./cached_partition/{dataset}/{SEED}')
        _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                               num_clusters)
        parts = [np.array(pt) for pt in parts]
        torch.save(parts, f'./cached_partition/{dataset}/{SEED}/parts.pt')
    
    
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
    
    
    A = full_adj.todense()
    A = A + np.eye(A.shape[0])
    # A = np.power(A, 3)
    A[A > 0] = 1
    

    
    # print('mean degree:', np.mean(np.sum(A, axis=1)))
    
    A = torch.tensor(A, dtype=torch.float32)
    X = torch.tensor(train_feats, dtype=torch.float32)
    # X = A @ X
    W = utils.get_affinity_matrix(X)
    
    # W = W + A
    # W[W > 0] = 1
    # W = A
    W = F.normalize(W, p=1, dim=1)
    
    D = torch.diag(torch.sum(W, dim=1))
    L = D - W
    L = L.numpy()
    eigvals, eigvecs = eigsh(L, k=10, which='SM')
    print(eigvals)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(eigvecs)
    cluster_labels = kmeans.labels_
    
    metrics = Metrics()
    
    nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    print(f'NMI: {nmi}, ACC: {acc}')
    exit()
    
    L = sort_laplacian(W, y_train)
    
    plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    plt.imshow(L, cmap='flag')
    plt.show()
    plt.savefig('block_diagonal.png')
    # exit()
    
    spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
    spectralnet.fit(train, val)


    X = torch.tensor(test_feats, dtype=torch.float32)
    # print(X.shape)
    Y = spectralnet.predict(X)
    Y = Y / np.sqrt(Y.shape[0])
    
    
    # np.save(f'./results/{dataset}/{output_dim}/{SEED}/spectralnet.npy', Y)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(Y)
    cluster_labels = kmeans.labels_
    
    metrics = Metrics()
    
    nmi = metrics.nmi_score(cluster_labels[train_mask], y_train[train_mask])
    acc = metrics.acc_score(cluster_labels[train_mask], y_train[train_mask], n_clusters)
    print(f'NMI: {nmi}, ACC: {acc}')
    
    print('##################')
    
    nmi = metrics.nmi_score(cluster_labels[test_mask], y_test[test_mask])
    acc = metrics.acc_score(cluster_labels[test_mask], y_test[test_mask], n_clusters)
    print(f'Test NMI: {nmi}, Test ACC: {acc}')
    
    # print('##################')
    # nmi = metrics.nmi_score(y_test, y_test)
    # acc = metrics.acc_score(y_test, y_test, n_clusters)
    # print(f'Perfect NMI: {nmi}, Perfect ACC: {acc}')
    
    
    
    
    
if __name__ == "__main__":
    # print('Starting...')
    main()

