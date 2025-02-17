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
from tqdm import tqdm


import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_in_chunks(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


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
    idx_train = idx[:int(0.94 * len(labels))]
    idx_val = idx[int(0.94 * len(labels)):int(0.97 * len(labels))]
    idx_test = idx[int(0.97 * len(labels)):]
    print("Number of training nodes: ", len(idx_train))
    print("Number of validation nodes: ", len(idx_val))
    print("Number of testing nodes: ", len(idx_test))
    
    
    return adj, features, labels, idx_train, idx_val, idx_test


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


"""
https://github.com/THUDM/GRAND-plus/blob/main/utils/make_dataset.py
"""
def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def get_dataset_NAGphormer(dataset, split_seed=2, file_dir='./data/'):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:
        file_path = file_dir + dataset+".pt"

        data_list = torch.load(file_path)

        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]
        
        adj = csr_matrix(adj.to_dense())
        
        adj = adj + adj.T + sp.eye(adj.shape[0])
        adj[adj > 1] = 1

        # if dataset == "pubmed":
        #     graph = PubmedGraphDataset()[0]
        # elif dataset == "corafull":
        #     graph = CoraFullDataset()[0]
        # elif dataset == "computer":
        #     graph = AmazonCoBuyComputerDataset()[0]
        # elif dataset == "photo":
        #     graph = AmazonCoBuyPhotoDataset()[0]
        # elif dataset == "cs":
        #     graph = CoauthorCSDataset()[0]
        # elif dataset == "physics":
        #     graph = CoauthorPhysicsDataset()[0]
        # elif dataset == "cora":
        #     graph = CoraGraphDataset()[0]
        # elif dataset == "citeseer":
        #     graph = CiteseerGraphDataset()[0]

        # graph = dgl.to_bidirected(graph)


    elif dataset in {"aminer", "reddit", "Amazon2M"}:
        file_dir = file_dir + dataset + '/'
        file_path = file_dir + dataset + '.pt'
        if os.path.exists(file_path):
            data_list = torch.load(file_path)

            #adj, features, labels, idx_train, idx_val, idx_test

            adj = data_list[0]
            features = data_list[1]
            labels = data_list[2]
            idx_train = data_list[3]
            idx_val = data_list[4]
            idx_test = data_list[5]
        else:
            import pickle as pkl
            if dataset == 'aminer':
            
                adj = pkl.load(open(os.path.join(file_dir, "{}.adj.sp.pkl".format(dataset)), "rb"))
                features = pkl.load(
                    open(os.path.join(file_dir, "{}.features.pkl".format(dataset)), "rb"))
                labels = pkl.load(
                    open(os.path.join(file_dir, "{}.labels.pkl".format(dataset)), "rb"))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
                idx_unlabel = np.concatenate((idx_val, idx_test))
                features = col_normalize(features)
            elif dataset in ['reddit']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset))) 
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
                idx_unlabel = np.concatenate((idx_val, idx_test))
                print(dataset, features.shape)
            
            elif dataset in ['Amazon2M']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset)))
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                class_num = labels.shape[1]
                idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20*class_num, val_size=30 * class_num)
                idx_unlabel = np.concatenate((idx_val, idx_test))


            # adj = adj + sp.eye(adj.shape[0])
            # D1 = np.array(adj.sum(axis=1))**(-0.5)
            # D2 = np.array(adj.sum(axis=0))**(-0.5)
            # D1 = sp.diags(D1[:, 0], format='csr')
            # D2 = sp.diags(D2[0, :], format='csr')

            # A = adj.dot(D1)
            # A = D2.dot(A)
            # adj = A
            
            adj = adj + adj.T + sp.eye(adj.shape[0])
            adj[adj > 1] = 1
            # labels = torch.argmax(labels, -1)
            
    print("Number of nodes: ", adj.shape[0])
    print("Number of edges: ", int(adj.sum()/2))

    features = torch.tensor(features)
    labels = torch.tensor(labels)
    idx_train = torch.tensor(idx_train)
    idx_val = torch.tensor(idx_val)
    idx_test = torch.tensor(idx_test)
    
    labels = torch.argmax(labels, -1)
    
    # split the dataset 0.94 train, 0.03 val, 0.03 test
    idx = np.random.permutation(labels.shape[0])
    idx_train = idx[:int(0.94 * len(labels))]
    idx_val = idx[int(0.94 * len(labels)):int(0.97 * len(labels))]
    idx_test = idx[int(0.97 * len(labels)):]
    
    D1 = np.array(adj.sum(axis=1))**(-0.5)
    D2 = np.array(adj.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    
    norm_adj = adj.dot(D1)
    norm_adj = D2.dot(norm_adj)
        
    features = torch.tensor(features, dtype=torch.float32)
    
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)

            

    return  adj, features, labels, idx_train, idx_val, idx_test



def load_gragsage_bad_data(dataset, file_dir):
    
    with open(f"{file_dir}/{dataset}/{dataset}-G.json", 'r') as f:
        grapf_dict = json.load(f)
        f.close()
    
    with open(f"{file_dir}/{dataset}/{dataset}-feats.npy", 'rb') as f:
        features = np.load(f)
        f.close()
    
    with open(f"{file_dir}/{dataset}/{dataset}-id_map.json", 'r') as f:
        id_map = json.load(f)
        f.close()
    
    with open(f"{file_dir}/{dataset}/{dataset}-class_map.json", 'r') as f:
        class_map = json.load(f)
        f.close()
    

    # labels
    labels = np.zeros((len(id_map), 1))
    for key, value in class_map.items():
        labels[id_map[key]] = value
    
    # adj
    links = grapf_dict['links']
    num_nodes = len(id_map)
    adj = sp.coo_matrix((np.ones(len(links)), ([link['source'] for link in links], [link['target'] for link in links])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    # features
    features = torch.tensor(features)
    
    


def create_graphsage_data(dataset, file_dir):
    start_time = time.time()
    adj, features, labels, idx_train, idx_val, idx_test = get_dataset_NAGphormer(dataset, 0, file_dir)
    print("done load dataset")
    grapf_dict = {
        "directed": False,
        "graph": [],
        "nodes": [],
        "links": []
    }
    print("start node saving")
    # Convert lists to sets for faster lookup
    idx_test_set = set(idx_test)
    idx_val_set = set(idx_val)
    # Using a list comprehension to generate the list of node dictionaries
    nodes = [
        {
            "id": str(node),
            "test": node in idx_test_set,
            "val": node in idx_val_set
        }
        for node in range(adj.shape[0])
    ]
    grapf_dict["nodes"].extend(nodes)

    print("done node saving")
    print("start edge saving")
    adj_c = adj.coalesce()
    ijs = adj_c.indices()

    # Using a list comprehension to generate the list of link dictionaries
    links = []
    chunk_size = 100000  # Adjust chunk size as needed
    for chunk in tqdm(process_in_chunks(ijs.t(), chunk_size), desc="Edges"):
        for ij in chunk:
            links.append({
                "source": int(ij[0]),
                "target": int(ij[1])
            })

    grapf_dict["links"].extend(links)
    print("done edge saving")
    print(f"Number of linkes {len(links)}")
    print("start graph saving")
    Path(f'{file_dir}/{dataset}').mkdir(parents=True, exist_ok=True)

    with open(f'{file_dir}/{dataset}/{dataset}-G.json', 'w') as f:
        json.dump(grapf_dict, f)
        f.close()
    print("done graph saving")

    print("start feature saving")
    with open(f'{file_dir}/{dataset}/{dataset}-feats.npy', 'wb') as f:
        np.save(f, features)
        f.close()
    print("done feature saving")

    print("start id saving")
    id_map = {j: i for i, j in enumerate(range(adj.shape[0]))}
    # mkdir if not exist
    with open(f'{file_dir}/{dataset}/{dataset}-id_map.json', 'w') as f:
        json.dump(id_map, f)
        f.close()
    print("done id saving")

    print("start class saving")

    class_map = {i: int(j) for i, j in enumerate(labels)}
    with open(f'{file_dir}/{dataset}/{dataset}-class_map.json', 'w') as f:
        json.dump(class_map, f)
        f.close()
    print("done class saving")
    print("done saving")

    print("Time: ", time.time() - start_time)

file_dir = './data/'
create_graphsage_data('reddit', file_dir)
