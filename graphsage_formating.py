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



def load_raw_data_cora():
    paper_features_label = np.genfromtxt('./data/cora/cora.content', dtype=str)
    features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
    labels = paper_features_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]
    papers = paper_features_label[:, 0].astype(np.int32)
    paper2idx = {k: v for v, k in enumerate(papers)}
    edges = np.genfromtxt('./data/cora/cora.cites', dtype=np.int32)
    edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
    adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(len(labels), len(labels)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # adj = normalize(adj + eye(adj.shape[0]))  # Normalizing for removing gradient vanishing and exploding problem
    adj = torch.FloatTensor(adj.todense())
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    train_size = 0.9 * len(labels)
    val_size = 0.05 * len(labels)
    idx = np.random.permutation(len(labels))
    idx_train = idx[:int(train_size)]
    idx_val = idx[int(train_size):int(train_size + val_size)]
    idx_test = idx[int(train_size + val_size):]
    print("Number of training nodes: ", len(idx_train))
    return adj, features, labels, idx_train, idx_val, idx_test


def create_data_cora():
    adj, features, labels, idx_train, idx_val, idx_test = load_raw_data_cora()
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

    with open('./data/cora/cora-G.json', 'w') as f:
        json.dump(grapf_dict, f)
        f.close()

    with open('./data/cora/cora-feats.npy', 'wb') as f:
        np.save(f, features)
        f.close()

    id_map = {j: i for i, j in enumerate(range(adj.shape[0]))}
    with open('./data/cora/cora-id_map.json', 'w') as f:
        json.dump(id_map, f)
        f.close()

    class_map = {i: int(j) for i, j in enumerate(labels)}
    with open('./data/cora/cora-class_map.json', 'w') as f:
        json.dump(class_map, f)
        f.close()


def random_orthogonal_matrix(n):
    # Generate a random matrix
    A = np.random.rand(n, n)

    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    # Ensure that the determinant is 1 (orthogonal matrix)
    det = np.linalg.det(Q)
    if det < 0:
        Q[:, 0] = -Q[:, 0]

    return Q



# reddit_emb = np.load('C:/Users/jacob/PycharmProjects/SpectralNet/reddit_emb/1024-3/reddit_embeddings.npy')





create_data_cora()

# # Create an undirected, unweighted graph with n=10 nodes and random edges
# n = 100
# G = nx.gnp_random_graph(n, p=0.1, seed=42)  # Adjust the probability 'p' as needed
#
# # Compute the Laplacian matrix
# laplacian_matrix = nx.normalized_laplacian_matrix(G).toarray()
#
# # Perform the eigendecomposition of the Laplacian matrix
# vecs, vals, _ = np.linalg.svd(laplacian_matrix)
# vecs = vecs[:, np.argsort(vals)]
# U = vecs[:, :16]
#
# # random rotation matrix
# theta = np.random.rand(16, 16)
#
# Q = random_orthogonal_matrix(16)
#
# print(get_grassman_distance(U, -U))
# print(get_grassman_distance(U, np.dot(U,  theta)))
# print(get_grassman_distance(U, np.dot(U, np.dot(theta, theta.T))))

