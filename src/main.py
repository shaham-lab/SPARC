import sys
import json
import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
import numpy as np
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse import csgraph
from sklearn.neighbors import KNeighborsClassifier
import utils, partition_utils
from utils import *
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.sparse import save_npz
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


def load_data(data_prefix, dataset_str):
    """Return the required data formats for GCN models."""
    (num_data, train_adj, full_adj, feats, train_feats, test_feats, 
        labels, train_data, val_data, test_data) = utils.load_graphsage_data(data_prefix, dataset_str)
    
    visible_data = train_data


    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test = labels

    train_mask = utils.sample_mask(train_data, labels.shape[0])
    val_mask = utils.sample_mask(val_data, labels.shape[0])
    test_mask = utils.sample_mask(test_data, labels.shape[0])

    train_feats = feats
    test_feats = feats

    return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test, 
            train_mask, val_mask, test_mask, train_data, val_data, test_data, num_data, visible_data)
  

def main(seed=0):
    set_seed(seed)
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
     visible_data) = load_data('/data', dataset)
        

    _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                               num_clusters)
    parts = [np.array(pt) for pt in parts] 
    
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
    
    X = torch.tensor(test_feats, dtype=torch.float32)
    spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
    spectralnet.fit(train, val)


    Y = spectralnet.predict(X)
    Y = Y / np.sqrt(Y.shape[0])
    
    X_embedded = spectralnet.embed(X) 
    labels = np.argmax(y_test, axis=1)

    print('Saving results...')
    # Define the directory path
    directory = f'./sparc_results/{dataset}'

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    
    np.save(os.path.join(directory, f'embeddings.npy'), Y)
    # np.save(os.path.join(directory, 'features.npy'), X)
    # np.save(os.path.join(directory, 'labels.npy'), labels)
    # np.save(os.path.join(directory, f'test_mask.npy'), test_mask)
    # np.save(os.path.join(directory, f'val_mask.npy'), val_mask)
    # np.save(os.path.join(directory, f'train_mask.npy'), train_mask)
    # np.save(os.path.join(directory, f'X_embedded.npy'), X_embedded)
    

if __name__ == "__main__":
    main()

