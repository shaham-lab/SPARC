# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of preprocessing functions for different graph formats."""

import json
import time

from networkx.readwrite import json_graph
import numpy as np
import partition_utils
import scipy.sparse as sp
import sklearn.metrics
import sklearn.preprocessing
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile


def parse_index_file(filename):
  """Parse index file."""
  index = []
  for line in gfile.Open(filename):
    index.append(int(line.strip()))
  return index


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=bool)


def sym_normalize_adj(adj):
  """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)) + 1e-20
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
  adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
  return adj


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


def sparse_to_tuple(sparse_mx):
  """Convert sparse matrix to tuple representation."""

  def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
      mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

  if isinstance(sparse_mx, list):
    for i in range(len(sparse_mx)):
      sparse_mx[i] = to_tuple(sparse_mx[i])
  else:
    sparse_mx = to_tuple(sparse_mx)

  return sparse_mx


def calc_f1(y_pred, y_true, multilabel):
  if multilabel:
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
  else:
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
  return sklearn.metrics.f1_score(
      y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
          y_true, y_pred, average='macro')


def construct_feed_dict(features,eigen_vecs, support, labels, labels_mask, placeholders):
  """Construct feed dictionary."""
  feed_dict = dict()
  feed_dict.update({placeholders['labels']: labels})
  feed_dict.update({placeholders['labels_mask']: labels_mask})
  feed_dict.update({placeholders['features']: features})
  feed_dict.update({placeholders['eigen_vecs']: eigen_vecs})
  feed_dict.update({placeholders['support']: support})
  feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
  return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            eigen_vecs,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
  """Generate the batch for multiple clusters."""

  print('features shape: ' + str(features.shape))
  print('eigen_vecs shape: ' + str(eigen_vecs.shape))
  features_batches = []
  eigen_vecs_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  np.random.shuffle(parts)
  for _, st in enumerate(range(0, num_clusters, block_size)):
    pt = parts[st]
    for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
      pt = np.concatenate((pt, parts[pt_idx]), axis=0)
    features_batches.append(features[pt, :])
    eigen_vecs_batches.append(eigen_vecs[pt, :])
    y_train_batches.append(y_train[pt, :])
    support_now = adj[pt, :][:, pt]
    if diag_lambda == -1:
      support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
    else:
      support_batches.append(
          sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
    total_nnz += support_now.count_nonzero()

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (features_batches, eigen_vecs_batches, support_batches, y_train_batches,
          train_mask_batches)


def preprocess(adj,
               features,
               eigen_vecs,
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
  eigen_vecs_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  for pt in parts:
    features_batches.append(features[pt, :])
    eigen_vecs_batches.append(eigen_vecs[pt, :])
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_train_batches.append(y_train[pt, :])

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (parts, features_batches, eigen_vecs_batches, support_batches, y_train_batches,
          train_mask_batches)


def load_graphsage_data(dataset_path, dataset_str, normalize=True):
  """Load GraphSAGE data."""
  start_time = time.time()

  graph_json = json.load(
      gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                          dataset_str)))
  graph_nx = json_graph.node_link_graph(graph_json)

  # print(graph_nx.nodes['2hk2a7'])
  # print(graph_nx.nodes[13])

  id_map = json.load(
      gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                               dataset_str)))
  is_digit = list(id_map.keys())[0].isdigit()
  # id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
  id_map = {k: int(v) for k, v in id_map.items()}
  id_map_re = {value: key for key, value in id_map.items()}
  class_map = json.load(
      gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                  dataset_str)))

  is_instance = isinstance(list(class_map.values())[0], list)
  
  # class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
  #              for k, v in class_map.items()}
  # class_map = {k: (v if is_instance else int(v)) 
  #              for k, v in class_map.items()}

  # co = 0
  # for node in graph_nx.nodes():
  #   if 'val' in graph_nx.nodes[node]:
  #     co += 1
  # print(co)

  print(len(graph_nx.nodes()))
  # broken_count = 0
  # to_remove = []
  # for node in graph_nx.nodes():
  #   if node not in id_map and node not in id_map_re:
  #     to_remove.append(node)
  #     broken_count += 1
  # for node in to_remove:
  #   graph_nx.remove_node(node)
  # tf.logging.info(
  #     'Removed %d nodes that lacked proper annotations due to networkx versioning issues',
  #     broken_count)

  feats = np.load(
      gfile.Open(
          '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
          'rb')).astype(np.float32)
  
  
  eigen_vecs = np.load(
      gfile.Open(
          '{}/{}/spectralnet.npy'.format(dataset_path, dataset_str, dataset_str),
          'rb')).astype(np.float32)
  
  print('eigem_vecs shape: ' + str(eigen_vecs.shape))
  print('feats shape: ' + str(feats.shape))

  tf.logging.info('Loaded data (%f seconds).. now preprocessing..',
                  time.time() - start_time)
  start_time = time.time()

  print('num of nodes: ' + str(len(graph_nx.nodes())))

  # co = 0
  # for node in graph_nx.nodes():
  #   if 'val' in graph_nx.nodes[node]:
  #     co += 1
  # print(co)

  # co = 0
  # for node in graph_nx.nodes():
  #   print( graph_nx.nodes[node])
  #   if co == 10:
  #     break
  #   co += 1

  edges = []
  
  for edge in graph_nx.edges():
    if edge[0] in id_map_re and edge[1] in id_map_re:
      edges.append((edge[0], edge[1]))
  num_data = len(id_map)
  
  print('num of edges: ' + str(len(graph_nx.edges())))
  
  # print(graph_nx.nodes['2hk2a7'])
  # print(graph_nx.nodes[13])
  val_data = np.array(
    [id_map[n] for n in graph_nx.nodes() if 'val' in graph_nx.nodes[n] and graph_nx.nodes[n]['val']],
    dtype=np.int32)
  print('done val')
  print(len(val_data))
  test_data = np.array(
      [id_map[n] for n in graph_nx.nodes() if 'test' in graph_nx.nodes[n] and graph_nx.nodes[n]['test']],
      dtype=np.int32)
  print('test size: ' + str(len(test_data)))
  is_train = np.ones((num_data), dtype=bool)
  is_train[val_data] = False
  is_train[test_data] = False
  train_data = np.array([n for n in range(num_data) if is_train[n]],
                        dtype=np.int32)

  train_edges = [
      (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
  ]
  edges = np.array(edges, dtype=np.int32)
  train_edges = np.array(train_edges, dtype=np.int32)

  print('edges', len(edges))
  print('train_edges', len(train_edges))


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
    print(edges.shape)
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_data, num_data))
    adj += adj.transpose()
    return adj

  train_adj = _construct_adj(train_edges)
  full_adj = _construct_adj(edges)
  

  # train_feats = feats[train_data]
  train_feats = feats
  test_feats = feats
  
  train_eigen_vecs = eigen_vecs
  test_eigen_vecs = eigen_vecs
  
  # full_adj = full_adj + sp.eye(full_adj.shape[0])
  # conv_feats = full_adj @ feats
  # print('conv_feats shape: ' + str(conv_feats.shape))
  # np.save('train_mask.npy', train_data)
  # np.save('val_mask.npy', val_data)
  # np.save('test_mask.npy', test_data)
  # np.save('feats.npy', feats)
  # np.save('conv_feats.npy', conv_feats)
  # np.save('labels.npy', labels)
  
  # print('DONE SAVING NEW DATASET')

  tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
  return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data, train_eigen_vecs, test_eigen_vecs