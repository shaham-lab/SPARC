from data import get_dataset
import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import MambaModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import dgl
import scipy.sparse as sp
from scipy.linalg import svd

def parse_args():
    parser = argparse.ArgumentParser()

    # Main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pubmed', help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=1, help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    # Model parameters
    parser.add_argument('--hops', type=int, default=7, help='Hop of neighbors to be calculated')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of Mamba layers')
    parser.add_argument('--d_state', type=int, default=16, help='State size for SSM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Mamba-specific parameters
    parser.add_argument('--ssm_size', type=int, default=64, help='Size of SSM hidden state')
    parser.add_argument('--expand_factor', type=int, default=2, help='Expansion factor for SSM')
    parser.add_argument('--dt_min', type=float, default=0.001, help='Minimum delta step size')
    parser.add_argument('--dt_max', type=float, default=0.1, help='Maximum delta step size')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('--tot_updates', type=int, default=1000, help='Total updates for LR scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400, help='Warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, help='Peak learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, help='End learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--space', type=str, default='spectral', help='Space')

    return parser.parse_args()

def setup_training(args):
    """Setup training environment with proper seeding and device configuration"""
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return device

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
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def load_data(dataset):
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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_valid_epoch(epoch, model, train_data_loader, val_data_loader, optimizer, 
                     lr_scheduler, device, args, train_mask, val_mask):
    # Training metrics
    loss_train_b = 0
    acc_train_b = 0

    model.train()
    for _, item in enumerate(train_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(nodes_features)
        loss = F.nll_loss(output, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        lr_scheduler.step()

        # Update metrics
        loss_train_b += loss.item()
        acc_train = utils.accuracy_batch(output, labels)
        acc_train_b += acc_train.item()

    # Validation phase
    loss_val_b = 0
    acc_val_b = 0
    
    model.eval()
    with torch.no_grad():
        for _, item in enumerate(val_data_loader):
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            output = model(nodes_features)
            loss = F.nll_loss(output, labels)
            
            loss_val_b += loss.item()
            acc_val = utils.accuracy_batch(output, labels)
            acc_val_b += acc_val.item()


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train_b),
          'acc_train: {:.4f}'.format(acc_train_b/np.sum(train_mask)),
          'loss_val: {:.4f}'.format(loss_val_b),
          'acc_val: {:.4f}'.format(acc_val_b/np.sum(val_mask)))

    return loss_val_b, acc_val_b

def test(model, test_data_loader, val_data_loader, device, test_mask, val_mask):
    model.eval()
    loss_test = 0
    acc_test = 0
    loss_val =0
    acc_val = 0
    
    with torch.no_grad():
        # Test set evaluation
        for _, item in enumerate(test_data_loader):
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            output = model(nodes_features)
            loss = F.nll_loss(output, labels)
            loss_test += loss.item()
            acc_test += utils.accuracy_batch(output, labels).item()

        # Validation set evaluation
        for _, item in enumerate(val_data_loader):
            nodes_features = item[0].to(device)
            labels = item[1].to(device)

            output = model(nodes_features)
            loss = F.nll_loss(output, labels)
            loss_val += loss.item()
            acc_val += utils.accuracy_batch(output, labels).item()

    print('Test Accuracy: {:.4f}'.format(acc_test/np.sum(test_mask)))
    print('Validation Accuracy: {:.4f}'.format(acc_val/np.sum(val_mask)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


args = parse_args()
seed = args.seed
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# Load and process data
# print('Loading data from spectralnet preprocess...')
spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask = load_data(args.dataset)

print("Processing features...")
if args.space == 'spectral':
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        spectral_encoding, features, args.hops, train_mask, val_mask, test_mask
    )
elif args.space == 'features':
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        features, features, args.hops, train_mask, val_mask, test_mask
    )

# Create data loaders
print('Creating data loaders...')
train_data_loader, val_data_loader, test_data_loader = create_data_loaders(processed_features, labels, train_mask, val_mask, test_mask, args.batch_size)


# Initialize model
print('Initializing model...')
model = MambaModel(
    hops=args.hops,
    n_class=labels.max().item() + 1,
    input_dim=features.shape[1],
    n_layers=args.n_layers,
    hidden_dim=args.hidden_dim,
    d_state=args.d_state,
    dropout_rate=args.dropout
).to(device)

print(model)
print('Total parameters:', sum(p.numel() for p in model.parameters()))

# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.peak_lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95)  # Modified betas for better training stability
)

lr_scheduler = PolynomialDecayLR(
    optimizer,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    lr=args.peak_lr,
    end_lr=args.end_lr,
    power=1.0,
)

# Training loop
print("Training...")
t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)

for epoch in range(args.epochs):
    loss_val, acc_val = train_valid_epoch(
        epoch, model, train_data_loader, val_data_loader,
        optimizer, lr_scheduler, device, args, train_mask, val_mask
    )
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")
print(f"Total training time: {time.time() - t_total:.4f}s")

# Load best model and test
print(f'Loading {early_stopping.best_epoch + 1}th epoch')
model.load_state_dict(early_stopping.best_state)

print("Testing...")
test(model, test_data_loader, val_data_loader, device, test_mask, val_mask)


