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



# def load_raw_data(dataset):
#     if dataset == 'cora':
#         return load_raw_data_cora()
#     else:
#         raise ValueError(f"Dataset {dataset} not supported.")

def create_graphsage_data(dataset):
    adj, features, labels, idx_train, idx_val, idx_test = load_raw_data(dataset)
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



create_graphsage_data('pubmed')
