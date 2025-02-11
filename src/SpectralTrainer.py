import random
import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch import sigmoid
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

import utils
from utils import *
from torch.utils.data import DataLoader, random_split, TensorDataset


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int, device: torch.device):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.device = device
        self.orthonorm_weights = torch.eye(self.input_dim).to(self.device)
        self.rotation_matrix = None
        

        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "n_layers":
                continue
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                self.layers.append(layer)
                current_dim = next_dim


    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies the Cholesky decomposition to orthonormalize the output (`Y`) of the network.
        The orthonormalized output is returned as a tensor.
        """

        m = Y.shape[0]

        _, R = torch.linalg.qr(Y)
        D = torch.diag(torch.sign(torch.diag(R)))
        R = R @ D

        orthonorm_weights = np.sqrt(m) * torch.inverse(R)

        return orthonorm_weights


    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the Cholesky decomposition. The output tensor is returned.
        """

        for layer in self.layers:
            x = layer(x)
            
        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        return Y


    def set_rotation_matrix(self, rotation_matrix):
        self.rotation_matrix = rotation_matrix


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()


    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W, 
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        
        if is_normalized:
            d = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(d[:, None])

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        
        return loss



class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False, siamese_net: nn.Module = None):
        """
        This class is responsible for training the SpectralNet model.

        Args:
            config (dict):                  The configuration dictionary
            device (torch.device):          The device to use for training
            is_sparse (bool, optional):     Whether the graph-laplacian obtained from a mini-batch is sparse or not.
                                            In case it is sparse, we build the batch by taking 1/5 of the original random batch,
                                            and then we add to each sample 4 of its nearest neighbors. Defaults to False.
        """

        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config["spectral"]
        self.dataset = config["dataset"]
        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.architecture = self.spectral_config["architecture"]
        self.n_clusters = self.spectral_config["n_clusters"]
        self.weights_path = f"./weights/{self.dataset}/spectralnet_weights.pth"


    def train(self, train, val) -> SpectralNetModel:
        """
        This function trains the SpectralNet model.

        Args:
            train (tuple):                  The training data
            adj_dict (dict):                The adjacency dictionary
            sparse_adj (torch.sparse.Tensor): The sparse adjacency matrix
            siamese_net (nn.Module, optional): The siamese network to use for training. Defaults to None.
        Returns:
            SpectralNetModel: The trained SpectralNet model
        """
        self.parts, self.features_batches, self.support_batches, self.y_train_batches, self.train_mask_batches = train
        self.idx_parts, self.val_features_batches, self.val_support_batches, self.y_val_batches, self.val_mask_batches = val
        self.idx_parts = list(range(len(self.parts)))
        self.counter = 0
        self.criterion = SpectralNetLoss()
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=self.features_batches[0].shape[1], device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              factor=self.lr_decay,
                                                              patience=self.patience)


        min_val_loss = 1000000000
        batches = len(self.features_batches)
        batches_val = len(self.val_features_batches)
        grassmans = []
        print("Training SpectralNet:")
        for epoch in range(self.epochs):
            train_loss = 0.0

            for pid in range(batches):
                pid_2 = np.random.choice(batches)

                # Use preprocessed batch data
                features_b = self.features_batches[pid].to(self.device)                
                support_b = self.support_batches[pid]
                # y_train_b = self.y_train_batches[pid].to(self.device)
                W = self._get_support_matrix(support_b)
                perm = np.random.permutation(len(features_b))
                features_b = features_b[perm]
                W = W[perm][:, perm]
                # y_train_b = y_train_b[perm]
                
                features_b_2 = self.features_batches[pid_2].to(self.device)

                perm = np.random.permutation(len(features_b_2))
                features_b_2 = features_b_2[perm]

                # Orthogonality step
                self.spectral_net.eval()
                self.spectral_net(features_b_2, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()


                Y = self.spectral_net(features_b, should_update_orth_weights=False)

                # Affinity matrix for loss computation
                W = self._get_valid_affinity_matrix(W, features_b)

                loss = self.criterion(W, Y, is_normalized=True)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation step
            valid_loss = 0.0
            self.spectral_net.eval()
            with torch.no_grad():
                for pid in range(batches_val):
                    features_b = self.val_features_batches[pid].to(self.device)
                    support_b = self.val_support_batches[pid]
                    y_val_b = self.y_val_batches[pid]
                    val_mask_b = self.val_mask_batches[pid]

                    W = self._get_support_matrix(support_b)

                    Y = self.spectral_net(features_b, should_update_orth_weights=False)
                    
                    W = self._get_valid_affinity_matrix(W, features_b)

                    loss = self.criterion(W, Y, is_normalized=True)
                    valid_loss += loss.item()

            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]: break
            print(f"Epoch: {epoch +1}/{self.epochs}, Train Loss: {train_loss / batches:.7f}, Valid Loss: {valid_loss / batches_val:.7f}, LR: {current_lr:.6f}" )


            if valid_loss <= min_val_loss:
                min_val_loss = valid_loss
                self.save_model()
                

        return self.spectral_net


    def _get_support_matrix(self, support):
        indices = torch.LongTensor(support[0])
        values = torch.FloatTensor(support[1])
        size = support[2]
        W = torch.sparse.FloatTensor(indices.t(), values, size).to_dense()
        W = W.to(device=self.device)
        return W


    def _get_valid_affinity_matrix(self, W, X):
        W_affinity = self._get_affinity_matrix(X)
        W = F.normalize(W, p=1, dim=1) + 0.1 * F.normalize(W_affinity, p=1, dim=1)
        # # zero diagonal
        # W = W + torch.ones_like(W) * 1e-6
        W = W + W @ W + W @ W @ W
        W.fill_diagonal_(0)
        W = F.normalize(W, p=2, dim=1) # + 0.1 * F.normalize(W_affinity, p=2, dim=1)
        # W = W.to(self.device)
        return W
       
       
    def _get_rotation_matrix(self, batches):
        l = None     
        for pid in range(batches):
            features_b = self.features_batches[pid].to(self.device)                
            support_b = self.support_batches[pid]
            W = self._get_support_matrix(support_b)
            Y = self.spectral_net(features_b, should_update_orth_weights=False)
            
            if l is None:
                l = Y.T @ W @ Y
            else:
                l += Y.T @ W @ Y
        l = l / batches
        l = (l + l.T) / 2
        
        l = l.detach().cpu().numpy()
        ortho_matrix, eigenvalues_pred, _ = np.linalg.svd(l)
        eigenvalues_pred = eigenvalues_pred.real
        self.ortho_matrix = np.array(ortho_matrix.real)
        indices = np.argsort(eigenvalues_pred)
        ortho_matrix = np.array(self.ortho_matrix[:, indices])
        self.spectral_net.set_rotation_matrix(ortho_matrix)


    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.
        Args:
            X (torch.Tensor):   The input data
        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.spectral_config["is_local_scale"]
        n_neighbors = self.spectral_config["n_neighbors"]
        scale_k = self.spectral_config["scale_k"]
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(Dx, scale, indices, device=self.device, is_local=is_local)
        return W


    def save_model(self):
        """
        This function saves the model.
        """
        torch.save(self.spectral_net.state_dict(), self.weights_path)



class ReduceLROnAvgLossPlateau(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0, verbose=False, min_delta=1e-4):
        """
        Custom ReduceLROnPlateau scheduler that uses the average loss instead of the loss of the last epoch.

        Args:
            optimizer (_type_):             The optimizer
            factor (float, optional):       factor by which the learning rate will be reduced. 
                                            new_lr = lr * factor. Defaults to 0.1.
            patience (int, optional):       number of epochs with no average improvement after 
                                            which learning rate will be reduced.
            min_lr (int, optional):         A lower bound on the learning rate of all param groups.
            verbose (bool, optional):       If True, prints a message to stdout for each update.
            min_delta (_type_, optional):   threshold for measuring the new optimum, to only focus on
                                            significant changes. Defaults to 1e-4.
        """

        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = 1e5
        self.avg_losses = []
        self.min_lr = min_lr
        super(ReduceLROnAvgLossPlateau, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.factor ** self.num_bad_epochs
                for base_lr in self.base_lrs]

    def step(self, loss=1.0, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        current_loss = loss
        if len(self.avg_losses) < self.patience:
            self.avg_losses.append(current_loss)
        else:
            self.avg_losses.pop(0)
            self.avg_losses.append(current_loss)
        avg_loss = sum(self.avg_losses) / len(self.avg_losses)
        if avg_loss < self.best - self.min_delta:
            self.best = avg_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = float(param_group['lr'])
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'Epoch {epoch}: reducing learning rate to {new_lr}.')
                self.wait = 0
            self.wait += 1
