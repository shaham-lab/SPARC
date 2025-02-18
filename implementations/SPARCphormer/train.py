from data import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import TransformerModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse
import dgl
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=1,
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--pe_dim', type=int, default=3,
                        help='Positional encoding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    
    parser.add_argument('--space', type=str, default='spectral',
                        help='space')


    return parser.parse_args()

def convert_adj_matrix(adjacency_matrix):
    # Convert the sparse matrix to COO format and make sure it's of type float32
    adjacency_matrix_coo = adjacency_matrix.tocoo().astype(np.float32)
    # Convert the SciPy sparse COO matrix to a PyTorch sparse tensor
    row = torch.tensor(adjacency_matrix_coo.row, dtype=torch.long)
    col = torch.tensor(adjacency_matrix_coo.col, dtype=torch.long)
    value = torch.tensor(adjacency_matrix_coo.data, dtype=torch.float32)  # Convert to float32
    # Create the PyTorch sparse tensor
    adjacency_matrix_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), value, adjacency_matrix_coo.shape)
    adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    D1 = np.array(adjacency_matrix.sum(axis=1))**(-0.5)
    D2 = np.array(adjacency_matrix.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    A = adjacency_matrix.dot(D1)
    A = D2.dot(A)

    
    adj = A
    # adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def load_data(dataset, seed=0):
    directory = '../../sparc_results/' + dataset + '/'
    spectral_encoding = np.load(os.path.join(directory, f'embeddings.npy'))
    features = np.load(os.path.join(directory, 'features.npy'))
    labels = np.load(os.path.join(directory, 'labels.npy'))
    train_mask = np.load(os.path.join(directory, f'train_mask.npy'))
    val_mask = np.load(os.path.join(directory, f'val_mask.npy'))
    test_mask = np.load(os.path.join(directory, f'test_mask.npy'))
    X_embedded = np.load(os.path.join(directory, f'X_embedded.npy'))
    adj = None
    features = features.astype(np.float32)
    features = torch.tensor(features, dtype=torch.float32)
    
    # concat the spectral encoding to the features
    features = np.concatenate((features, X_embedded), axis=1)
    
    labels = torch.tensor(labels)
    labels = labels.to(device)
    return spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask

def laplacian_eigenvectors(A, dim):
    """
    Compute Graph Laplacian eigenvectors.
    
    Parameters:
        A (torch.Tensor or scipy.sparse matrix): Adjacency matrix of the graph. If a torch.Tensor, it will be converted to a sparse matrix.
        dim (int): Number of eigenvectors to compute.
        
    Returns:
        torch.Tensor: Matrix of Laplacian eigenvectors.
    """
    # Convert A to a sparse matrix if it's a torch.Tensor
    if isinstance(A, torch.Tensor):
        # If it's a sparse tensor, convert it to dense first
        if A.is_sparse:
            A = A.to_dense()
        A = A.numpy()  # Convert to numpy array
        A = sp.csr_matrix(A)  # Convert to scipy sparse matrix

    # Ensure the adjacency matrix is in CSR format
    A = A.tocsr().astype(float)
        
    # Degree matrix with normalized values
    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees = np.clip(degrees, 1, None)  # Avoid division by zero
    N = sp.diags(degrees ** -0.5, dtype=float)
    
    # Compute the normalized Laplacian matrix
    L = sp.eye(A.shape[0]) - N @ A @ N

    # Compute eigenvalues and eigenvectors of the Laplacian
    EigVal, EigVec = sp.linalg.eigs(L, k=dim + 1, which='SR', tol=1e-2)
    
    # Sort eigenvectors by ascending eigenvalue order
    EigVec = EigVec[:, EigVal.argsort()]
    
    # Select the top positional encoding dimensions
    laplace_eigenvectors = torch.from_numpy(EigVec[:, :dim]).float()

    return laplace_eigenvectors


def create_data_loaders(processed_features, labels, train_mask, val_mask, test_mask, batch_size):
    batch_data_train = Data.TensorDataset(  
        processed_features[train_mask], labels[train_mask])
    batch_data_val = Data.TensorDataset(
        processed_features[val_mask], labels[val_mask])
    batch_data_test = Data.TensorDataset(
        processed_features[test_mask], labels[test_mask])


    train_data_loader = Data.DataLoader(
        batch_data_train, batch_size=args.batch_size, shuffle=True)
    val_data_loader = Data.DataLoader(
        batch_data_val, batch_size=args.batch_size, shuffle=True)
    test_data_loader = Data.DataLoader(
        batch_data_test, batch_size=args.batch_size, shuffle=True)
    
    return train_data_loader, val_data_loader, test_data_loader


def grassmann_distance(A, B):
    """
    Compute the Grassmann distance between two matrices A and B.
    
    Parameters:
        A (ndarray): Matrix representing the first subspace (m x n).
        B (ndarray): Matrix representing the second subspace (m x n).
        
    Returns:
        float: The Grassmann distance.
    """
    # Compute the SVD of A.T @ B
    U, singular_values, Vh = svd(A.T @ B)
    
    # Compute the principal angles (theta)
    theta = np.arccos(np.clip(singular_values, -1.0, 1.0))
    
    # Grassmann distance is the sum of the squares of the sines of the principal angles
    distance = np.sqrt(np.sum(np.sin(theta) ** 2))
    return distance


def train_valid_epoch(epoch):

    model.train()
    loss_train_b = 0
    acc_train_b = 0
    for _, item in enumerate(train_data_loader):
        
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        optimizer.zero_grad()
        output = model(nodes_features)
        loss_train = F.nll_loss(output, labels)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()
        

        loss_train_b += loss_train.item()
        acc_train = utils.accuracy_batch(output, labels)
        acc_train_b += acc_train.item()

    model.eval()
    loss_val_b = 0
    acc_val_b = 0
    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        output = model(nodes_features)
        loss_val = F.nll_loss(output, labels)
        
        loss_val_b += loss_val.item()
        acc_val = utils.accuracy_batch(output, labels)
        acc_val_b += acc_val.item()


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train_b),
          'acc_train: {:.4f}'.format(acc_train_b/np.sum(train_mask)),
          'loss_val: {:.4f}'.format(loss_val_b),
          'acc_val: {:.4f}'.format(acc_val_b/np.sum(val_mask)))

    return loss_val_b, acc_val_b

def test():

    loss_test = 0
    acc_test = 0
    loss_val =0
    acc_val = 0
    for _, item in enumerate(test_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        model.eval()

        output = model(nodes_features)
        loss_test += F.nll_loss(output, labels).item()
        acc_test += utils.accuracy_batch(output, labels).item()

    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        model.eval()

        output = model(nodes_features)
        loss_val += F.nll_loss(output, labels).item()
        acc_val += utils.accuracy_batch(output, labels).item()

    print("Test accuracy = {:.4f}".format(acc_test/np.sum(test_mask)))
    
    # print("CS Val accuracy = {:.4f}".format(acc_val/np.sum(val_mask)))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parse_args()
seed = args.seed
set_seed(seed)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')



# Load data from spectralnet preprocess
spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask = load_data(args.dataset, seed)

if args.space == 'spectral':
    print('spectral')
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(spectral_encoding, features, args.hops, train_mask, val_mask, test_mask)
elif args.space == 'features':
    print('features')
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(features, features, args.hops, train_mask, val_mask, test_mask)
elif args.space == 'computed':
    print('computed')
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(spectral_encoding_computed, features, args.hops, train_mask, val_mask, test_mask)
elif args.space == 'hops':
    print('hops')
    processed_features = utils.re_features_hops(adj, features, args.hops)     

# creat data loaders
train_data_loader, val_data_loader, test_data_loader = create_data_loaders(processed_features, labels,train_mask, val_mask, test_mask, args.batch_size)

# model configuration
model = TransformerModel(hops=args.hops,
                        n_class=labels.max().item() + 1,
                        input_dim=(features.shape[1]),
                        n_layers=args.n_layers,
                        num_heads=args.n_heads,
                        hidden_dim=args.hidden_dim,
                        ffn_dim=args.ffn_dim,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.attention_dropout).to(device)

# print(model)
# print('total params:', sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
    optimizer,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    lr=args.peak_lr,
    end_lr=args.end_lr,
    power=1.0,
)


print("training...")

t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
for epoch in range(args.epochs):
    loss_val, acc_val = train_valid_epoch(epoch)
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
# print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

print("testing...")
test()

    