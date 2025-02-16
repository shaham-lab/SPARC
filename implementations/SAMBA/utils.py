import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(
        torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    # adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(
        L, k=pos_enc_dim+1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()

    return lap_pos_enc

def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix

def re_features_spectral_diffusion_distance_avarage_seq(graph, features, K, pe_dim=10, t = 2):
     # Compute  symmetric Laplacian matrix
    A = graph.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    D = sp.diags(dgl.backend.asnumpy(
        graph.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(graph.number_of_nodes()) - D * A * D

    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pe_dim, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    EigVal = EigVal[EigVal.argsort()]

    EigVec = np.real(EigVec)
    EigVal = np.real(EigVal)

    # Compute the modified diffusion distances (1 - eigenvalue)^2
    modified_distances = (1 - EigVal) + (1 - EigVal) ** 2 + (1-EigVal)**6 + (1-EigVal)**4
    
    # Scale each eigenvector column by the corresponding diffusion distance
    scaled_eigenvectors = EigVec * modified_distances

    # Use approximate k-NN to find the most similar nodes based on scaled eigenvectors
   
    knn = NearestNeighbors(n_neighbors=1024, algorithm='auto').fit(scaled_eigenvectors)
    distances, indices = knn.kneighbors(scaled_eigenvectors)
    
    # Create a tensor to hold the node features
    nodes_features = torch.empty((scaled_eigenvectors.shape[0], K+1, features.shape[1] ))
    # Get indices of the most similar nodes (excluding the node itself)
    similar_indices = indices[:, 1:]  # skip the first one because it's the node itself

  
    # Stack the features for each node and its K most similar nodes
    nodes_features[:, 0, :] = torch.tensor(features)  # current node's features

    starts_idx = [0,2,4,8,16,32,64,128,256,512,1024]
    #starts_idx = [0,10,20,30,40,50,60,70,80,90,100]

    
    # Calculate the average of different sets of 10 closest nodes for positions 1 to 10
    for i in range(1,K+1):
        # Get the indices of the next set of 10 closest nodes   
        start_idx = starts_idx[i-1]
        end_idx = starts_idx[i]
        #start_idx = (i-1) * seq_len
        #end_idx = (i) * seq_len

        closest_indices = similar_indices[:, :end_idx]
        distances = distances[:, :end_idx]
        distance = distances.mean(axis=1)

        closest_indices = torch.tensor(closest_indices)
        features = torch.tensor(features)
       
        # Calculate the average features of these 10 closest nodes
        avg_features = features[closest_indices].mean(dim=1) / distance[:, np.newaxis]
        nodes_features[:, i, :] = avg_features 

    return nodes_features

def re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding(spectral_encoding, features, K, train_mask, val_mask):
    # Create a tensor to hold the node features
    nodes_features = torch.empty((spectral_encoding.shape[0], K+1, features.shape[1] ))
    # Get indices of the most similar nodes (excluding the node itself)
    # Stack the features for each node and its K most similar nodes
    nodes_features[:, 0, :] = torch.tensor(features)  # current node's features
    # starts_idx = [0,2,4,8,16,32,64,128,256,512,1024]
    #starts_idx = [0,10,20,50,100,200,500,1000,2000,3000,5000]
    starts_idx = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096]

    # Use approximate k-NN to find the most similar nodes based on euclidean distance on the spectral encoding
    knn = NearestNeighbors(n_neighbors=2000, algorithm='auto').fit(spectral_encoding)
    distances, indices = knn.kneighbors(spectral_encoding)

    '''
    for node in tqdm(range(indices.shape[0])):
        for k in range(1,indices.shape[1]):
            if not train_mask[indices[node][k]].any() or not val_mask[indices[node][k]].any():
                np.delete(indices[node], k)
    '''
    # Efficient filtering of indices using numpy vectorized operations
    for node in tqdm(range(indices.shape[0])):
        # Get all valid neighbors for this node by combining the train and val mask conditions
        valid_neighbors = indices[node][(train_mask[indices[node]] | val_mask[indices[node]])]
        if len(valid_neighbors) < 2** K:
            # If there are not enough valid neighbors, pad with the last valid neighbor
            valid_neighbors = np.pad(valid_neighbors, (node, 2**K - len(valid_neighbors)), mode='edge')
        # Update the indices array with the valid neighbors
        indices[node, 1:len(valid_neighbors)] = valid_neighbors[:len(valid_neighbors)-1]


    for i in tqdm(range(1,K+1)):
        end_idx = starts_idx[i]
        closest_indices = indices[:, :end_idx+1]

        closest_indices = torch.tensor(closest_indices)
        features = torch.tensor(features)
       
        # Calculate the average features of these 10 closest nodes
        avg_features = features[closest_indices].mean(dim=1)
        nodes_features[:, i, :] = avg_features 

    return nodes_features


def scipy_to_torch_sparse_coo(scipy_matrix):
    """
    Converts a scipy sparse matrix to a PyTorch sparse COO tensor.
    Ensures the matrix is in COO format before conversion.
    
    Args:
        scipy_matrix (scipy.sparse matrix): The input scipy sparse matrix.
    
    Returns:
        torch.sparse_coo_tensor: The converted PyTorch sparse COO tensor.
    """
    # Ensure the matrix is in COO format
    if not sp.isspmatrix_coo(scipy_matrix):
        scipy_matrix = scipy_matrix.tocoo()

    # Extract the row, col, and data from the scipy sparse COO matrix
    rows = torch.tensor(scipy_matrix.row, dtype=torch.long)
    cols = torch.tensor(scipy_matrix.col, dtype=torch.long)
    values = torch.tensor(scipy_matrix.data, dtype=torch.float32)

    # Stack the row and column indices to match the format required by PyTorch
    indices = torch.stack([rows, cols])

    # Create the PyTorch sparse COO tensor with the same shape as the scipy matrix
    torch_sparse_coo = torch.sparse_coo_tensor(indices, values, scipy_matrix.shape)

    return torch_sparse_coo




def re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(spectral_encoding, features, K, train_mask, val_mask, test_mask):
    # Use approximate k-NN to find the most similar nodes based on euclidean distance
    knn = NearestNeighbors(n_neighbors=min(2**14, len(train_mask)), algorithm='auto')
    # knn.fit(spectral_encoding[train_mask])
    knn.fit(spectral_encoding)
    distances, indices = knn.kneighbors(spectral_encoding)

    # Predefine node features tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    n_nodes, n_features = features_tensor.shape
    result_shape = (n_nodes, K + 1, n_features)

    # Initialize nodes_features lazily to save memory
    nodes_features = torch.empty(result_shape, dtype=torch.float32)
    nodes_features[:, 0, :] = features_tensor  # Set current node's features

    # Precompute slice ranges for efficient use
    starts_idx = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Efficient slicing to get closest features
    for j in tqdm(range(K)):
        end_idx = starts_idx[j + 1] if j + 1 < len(starts_idx) else indices.shape[1]
        closest_indices = indices[:, :end_idx]

        # Compute mean of closest features in chunks to save memory
        for i in range(0, n_nodes, 1024):  # Process in chunks of 1024
            chunk_indices = closest_indices[i : i + 1024]
            chunk_features = torch.mean(features_tensor[chunk_indices], dim=1)
            nodes_features[i : i + 1024, j + 1, :] = chunk_features

    return nodes_features





