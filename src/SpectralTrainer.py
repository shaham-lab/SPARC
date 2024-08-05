import random
import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch import sigmoid
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import utils
# from src.data import divide_graph_to_batches
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

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = True) -> torch.Tensor:
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
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        return loss
        
        # try:
        #     if random.random() < 0.01:
        #         print('========')
                
        #         print('========')
        # except:
        #     pass
            
            

        # D_inv_sqrt = torch.diag(torch.sum(W, dim=1) ** (-0.5))
        # L_sys = torch.eye(W.shape[0]).to(W.device) - D_inv_sqrt @ W @ D_inv_sqrt
        # loss = torch.trace(Y.t() @ L_sys @ Y) / (m * 2)
        # return loss

        # D_inv = torch.diag(torch.sum(W, dim=1) ** (-1))
        # L_rw = (torch.eye(W.shape[0]).to(W.device) - D_inv @ W) + torch.ones(W.shape[0]) * 0.01
        # # L_rw = torch.pow(L_rw, 2)
        # loss = torch.trace(Y.t() @ L_rw @ Y) / m
        # return loss

        # # if is_normalized:
        # d = torch.sum(W, dim=1)
        # Y = Y / torch.sqrt(d[:, None])

        # Dy = torch.cdist(Y, Y)
        # loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        # return loss


        # D_inv_sqrt = torch.diag(torch.sum(W, dim=1) ** (-0.5))
        # L_rw = torch.eye(W.shape[0]).to(W.device) - D_inv_sqrt @ W @ D_inv_sqrt
        #
        # loss = Y.t() @ L_rw @ Y
        # return torch.trace(loss) / loss.shape[0]




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
        self.siamese_net = siamese_net
        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.architecture = self.spectral_config["architecture"]
        self.n_clusters = self.spectral_config["n_clusters"]
        self.weights_path = "./weights/spectralnet_weights.pth"

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
        # self.X, self.y, self.indices, self.cluster = train
        # adj = sparse_adj.coalesce()
        # self.A = self.get_adjacency_matrix(self.indices)
        # self.adj_indices = adj.indices()
        # self.adj_values = adj.values()
        # self.adj_dict = adj_dict
        self.counter = 0
        self.criterion = SpectralNetLoss()
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=self.features_batches[0].shape[1], device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              factor=self.lr_decay,
                                                              patience=self.patience)

        # if os.path.exists(self.weights_path):
            # self.spectral_net.load_state_dict(torch.load(self.weights_path))
            # print("Loaded SpectralNet weights")
        #     return self.spectral_net

        # train_loader, ortho_loader, valid_loader = self._get_data_loader()
        # valid_indices = torch.tensor(valid_loader.dataset.indices)
        #
        # D = torch.diag(torch.sum(self.A, dim=1))
        # L = D - self.A
        # L_sys = torch.eye(self.A.shape[0]).to(self.device) - torch.diag(torch.sum(self.A, dim=1) ** (-0.5)) @ self.A @ torch.diag(torch.sum(self.A, dim=1) ** (-0.5))
        # L_rw = torch.eye(self.A.shape[0]).to(self.device) - torch.diag(torch.sum(self.A, dim=1) ** (-1)) @ self.A
        #
        # V, U = torch.linalg.eig(L)
        # V_sys, U_sys = torch.linalg.eig(L_sys)
        # V_rw, U_rw = torch.linalg.eig(L_rw)
        #
        # U = U[:, :1024]
        # U_sys = U_sys[:, :1024]
        # U_rw = U_rw[:, :1024]

        min_val_loss = 1000000000
        batches = len(self.features_batches)
        batches_val = len(self.val_features_batches)
        grassmans = []
        print("Training SpectralNet:")
        for epoch in range(self.epochs):
            train_loss = 0.0

            # if epoch % 20 == 1:
            #     reducer = umap.UMAP()
            #     features = []
            #     labels = []
            #     for i in range(batches):
            #         features.append(self.features_batches[i].detach().cpu().numpy())
            #         labels.append(self.y_train_batches[i].detach().cpu().numpy())
            #     features = np.concatenate(features, axis=0)
            #     labels = np.concatenate(labels, axis=0)
            #     labels = np.argmax(labels, axis=1)
            #     Y = self.spectral_net(torch.FloatTensor(features).to(self.device), should_update_orth_weights=False)
            #     embedding = reducer.fit_transform(Y.detach().cpu().numpy())
            #     plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
            #     plt.title('UMAP projection of the dataset', fontsize=24)
            #     plt.show()
            #     plt.savefig('umap.png')

            for pid in range(batches):
                pid_2 = np.random.choice(batches)

                # Use preprocessed batch data
                features_b = self.features_batches[pid].to(self.device)                
                support_b = self.support_batches[pid]
                y_train_b = self.y_train_batches[pid]
                
                y_train_b = torch.argmax(y_train_b, dim=1)
                # train_mask_b = self.train_mask_batches[pid]
                # indices = torch.LongTensor(support_b[0])
                # values = torch.FloatTensor(support_b[1])
                # size = support_b[2]
                # W = torch.sparse.FloatTensor(indices.t(), values, size).to_dense().to(self.device) + torch.ones(size) * 0.01
                
                perm = np.random.permutation(len(features_b))
                features_b = features_b[perm]


                features_b_2 = self.features_batches[pid_2].to(self.device)

                perm_2 = np.random.permutation(len(features_b_2))
                features_b_2 = features_b_2[perm_2]



                # Orthogonality step
                self.spectral_net.eval()
                self.spectral_net(features_b_2, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                W = self._get_support_matrix(support_b)
                W = W[perm][:, perm]
                
                
                # W = W + 0.1 * (W @ W) + 0.01 * (W @ W @ W) + 0.001 * (W @ W @ W @ W) + 0.0001 * (W @ W @ W @ W @ W)
                # W = W + W @ W + W @ W @ W + W @ W @ W @ W + W @ W @ W @ W @ W
                
                
                # W_ = W.detach().cpu().numpy()
                
                # # # W_[W_ > 0] = 1
                # W_ = np.log(W_ + 1)
                # W_ = sort_laplacian(W_, y_train_b[perm])
                # sorted_y = np.sort(y_train_b[perm])

                # separation_indices = []
                # for i in range(1, len(sorted_y)):
                #     if sorted_y[i] != sorted_y[i - 1]:
                #         separation_indices.append(i)

                # plt.imshow(W_, cmap='hot', norm=colors.LogNorm())

                # # Adding vertical and horizontal lines to separate classes
                # for sep_idx in separation_indices:
                #     plt.axvline(x=sep_idx + 0.5, color='blue', linestyle='--')
                #     plt.axhline(y=sep_idx + 0.5, color='blue', linestyle='--')

                # plt.colorbar()
                # plt.title('Laplacian Matrix with Class Separation')
                # plt.show()

                # # Save the figure
                # plt.savefig('block_diagonal_2b_.png')
                # plt.clf()
                
                
                # L = sort_laplacian(W_, y_train_b[perm])
                # import matplotlib.pyplot as plt
                # plt.imshow(L, cmap='hot', norm=colors.LogNorm())
                # plt.imshow(L, cmap='flag')
                # plt.show()
                # plt.savefig('block_diagonal.png')
                

                Y = self.spectral_net(features_b, should_update_orth_weights=False)
                

                
                if self.siamese_net is not None:
                    with torch.no_grad():
                        features_b = self.siamese_net.forward_once(features_b)
                
                
                # W_ = W + W @ W + W @ W @ W + W @ W @ W @ W + W @ W @ W @ W @ W + W @ W @ W @ W @ W @ W + W @ W @ W @ W @ W @ W @ W + W @ W @ W @ W @ W @ W @ W @ W + W @ W @ W @ W @ W @ W @ W @ W @ W + W @ W @ W @ W @ W @ W @ W @ W @ W @ W
                # zeros in diag
                # W_ = W_ - torch.diag(torch.diag(W_))
                
                # W_ = F.normalize(W, p=2, dim=1) + F.normalize(W @ W, p=2, dim=1) + F.normalize(W @ W @ W, p=2, dim=1) + F.normalize(W @ W @ W @ W, p=2, dim=1) + F.normalize(W @ W @ W @ W @ W, p=2, dim=1)
                W_ = symmetric_normalize(W) + symmetric_normalize(W @ W) + symmetric_normalize(W @ W @ W) + symmetric_normalize(W @ W @ W @ W) + symmetric_normalize(W @ W @ W @ W @ W)
                W_ = F.normalize(W_, p=2, dim=1)
                W_2 = self._get_affinity_matrix(W_ @ features_b)
                W_2 = F.normalize(W_2, p=2, dim=1)
                
                
                # W_2 = self._get_affinity_matrix(features_b)
                W += W_2
                
                # W_input = W @ features_b
                # W_input = W_input.detach().cpu().numpy()
                
                # plt.scatter(W_input[:, 0], W_input[:, 1], c=y_train_b[perm])
                # plt.show()
                # plt.savefig('input_space.png')
                # plt.clf()
                
                
                # W_2_ = W_2.detach().cpu().numpy()
                # W_2_ = np.log(W_2_ + 1)
                # # W_2_[W_2_ > 0] = 1
                # W_2_ = sort_laplacian(W_2_, y_train_b[perm])
                # plt.imshow(W_2_, cmap='hot', norm=colors.LogNorm())
                # separation_indices = []
                # sorted_y = np.sort(y_train_b[perm])
                # for i in range(1, len(sorted_y)):
                #     if sorted_y[i] != sorted_y[i - 1]:
                #         separation_indices.append(i)
                
                # for sep_idx in separation_indices:
                #     plt.axvline(x=sep_idx + 0.5, color='blue', linestyle='--')
                #     plt.axhline(y=sep_idx + 0.5, color='blue', linestyle='--')
                # plt.show()
                # plt.savefig('block_diagonal_2_b_affinity.png')
                # plt.clf()
                
                
                # W = 0.1* W + 0.9 *  W_2
                # W = 0.9 * W + 0.1 * W_2
                # W = 0.1* (W @ W) + 0.9 *  (W_2)
                
                # W = F.normalize(W, p=2, dim=1)
                # W_2 = F.normalize(W_2, p=2, dim=1)
                
                
                
                # W = W_2
                
                # W_ = W.detach().cpu().numpy()
                # W_ = np.log(W_ + 1)
                # W_ = sort_laplacian(W_, y_train_b[perm])
                # plt.imshow(W_, cmap='hot', norm=colors.LogNorm())
                
                # separation_indices = []
                # sorted_y = np.sort(y_train_b[perm])
                # for i in range(1, len(sorted_y)):
                #     if sorted_y[i] != sorted_y[i - 1]:
                #         separation_indices.append(i)
                # # Adding vertical and horizontal lines to separate classes
                # for sep_idx in separation_indices:
                #     plt.axvline(x=sep_idx + 0.5, color='blue', linestyle='--')
                #     plt.axhline(y=sep_idx + 0.5, color='blue', linestyle='--')
                    
                # plt.colorbar()
                # plt.title('Laplacian Matrix with Class Separation')
                # plt.show()
                # plt.savefig('block_diagonal_X.png')
                # plt.clf()
                
                


                loss = self.criterion(W, Y, is_normalized=False)

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

                    Y = self.spectral_net(features_b, should_update_orth_weights=False)
                    
                    W = self._get_support_matrix(support_b)
                    
                    if self.siamese_net is not None:
                        with torch.no_grad():
                            features_b = self.siamese_net.forward_once(features_b)
                    
                    W_2 = self._get_affinity_matrix(W @ features_b)
                    W = W_2

                    

                    loss = self.criterion(W, Y, is_normalized=True)
                    valid_loss += loss.item()

            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]: break
            print(f"Epoch: {epoch +1}/{self.epochs}, Train Loss: {train_loss / batches:.7f}, Valid Loss: {valid_loss / batches_val:.7f}, LR: {current_lr:.6f}" )



            if valid_loss <= min_val_loss:
                min_val_loss = valid_loss
                self.save_model()
                
        
        Q = self._get_rotation_matrix(batches)

            
        
        np.save('grassmans_0_2.npy', np.array(grassmans))
        return self.spectral_net

    def validate(self, valid_loader: DataLoader) -> float:
        """
        This function validates the SpectralNet model during the training process.

        Args:
            valid_loader (DataLoader):  The validation data loader

        Returns:
            float: The validation loss
        """

        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y, indices = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                Y = self.spectral_net(X, should_update_orth_weights=False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.forward_once(X)

                W = self.get_adjacency_matrix(indices)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()

        self.counter += 1

        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_support_matrix(self, support):
        indices = torch.LongTensor(support[0])
        values = torch.FloatTensor(support[1])
        size = support[2]
        W = torch.sparse.FloatTensor(indices.t(), values, size).to_dense() # + torch.ones(size) * 1e-5
        # W[W > 0] = 1
        # W = W - torch.diag(torch.diag(W)) + torch.eye(W.shape[0]) * 0.001
        # W = W + W.T
        # W = W @ W
        # normalize
        # W = F.normalize(W, p=2, dim=1)
        
        W = W.to(device=self.device)
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

        
        
    def _get_adjacency_matrix(self, batch_indices):
        m = len(batch_indices)
        sub_matrix = torch.eye(m)
        sub_matrix = sub_matrix + self.A[batch_indices][:, batch_indices]
        sub_matrix = F.normalize(sub_matrix, p=2, dim=1)  # + torch.eye(m)
        # sub_matrix = torch.zeros((m, m))
        # for i in range(m):
        #     sub_matrix[i][i] = 1
        #     for j in range(m):
        #         a = batch_indices[i]
        #         b = batch_indices[j]
        #         sub_matrix[i][j] = self.A[a][b]
        # # write to txt file
        # np.savetxt('adj_matrix.txt', sub_matrix.numpy())
        return sub_matrix

    def get_adjacency_matrix(self, indices):
        m = len(indices)
        sub_matrix = torch.zeros((m, m)).to(self.device)
        for i in range(m):
            sub_matrix[i][i] = 1
            s = []
            try:
                s = list(self.adj_dict[indices[i].item()])
            except:
                pass
            indices_i = torch.where(torch.tensor([elem in s for elem in indices]))
            for j in indices_i[0]:
                sub_matrix[i][j] = 1
        return sub_matrix

    def in1D(self, x, labels):
        """
        Sub-optimal equivalent to numpy.in1D().
        Hopefully this feature will be properly covered soon
        c.f. https://github.com/pytorch/pytorch/issues/3025
        Snippet by Aron Barreira Bordin
        Args:
            x (Tensor):             Tensor to search values in
            labels (Tensor/list):   1D array of values to search for

        Returns:
            Tensor: Boolean tensor y of same shape as x, with y[ind] = True if x[ind] in labels

        Example:
            in1D(torch.FloatTensor([1, 2, 0, 3]), [2, 3])
            FloatTensor([False, True, False, True])
        """
        mapping = torch.zeros(x.size()).byte()
        for label in labels:
            mapping = mapping | x.eq(label)
        return mapping

    def compact1D(self, x):
        """
        "Compact" values 1D uint tensor, so that all values are in [0, max(unique(x))].
        Args:
            x (Tensor): uint Tensor

        Returns:
            Tensor: uint Tensor of same shape as x

        Example:
            densify1D(torch.ByteTensor([5, 8, 7, 3, 8, 42]))
            ByteTensor([1, 3, 2, 0, 3, 4])
        """
        x_sorted, x_sorted_ind = torch.sort(x, descending=True)
        x_sorted_unique, x_sorted_unique_ind = torch.unique(x_sorted, return_inverse=True)
        x[x_sorted_ind] = x_sorted_unique_ind
        return x

    def symmetric_sparse_matrix(self, idx):
        v_idx = self.in1D(self.adj_indices, idx).byte()
        v_idx = v_idx.sum(dim=0).squeeze() == self.adj_indices.size(0)  # or `v_idx.all(dim=1)` for pytorch 0.5+
        v_idx = v_idx.nonzero().squeeze()

        # Slicing `v` and `i` accordingly:
        v_sliced = self.adj_values[v_idx]
        i_sliced = self.adj_indices.index_select(dim=1, index=v_idx)

        # Building sparse result tensor:
        i_sliced[0] = self.compact1D(i_sliced[0])
        i_sliced[1] = self.compact1D(i_sliced[1])

        # To make sure to have a square dense representation:
        size_sliced = torch.Size([len(idx), len(idx)])
        res = torch.sparse.FloatTensor(i_sliced, v_sliced, size_sliced)

        # Rearrange the matrix in the order of `idx`
        permutation = torch.argsort(torch.argsort(idx))
        res = res.to_dense()
        res = res[permutation, :][:, permutation]
        return res


    def get_data_from_indices(self, indices):
        x = self.X[indices]
        y = self.y[indices]
        adj = self.adj[indices, :][:, indices]
        return x, adj, y

    def get_affinities(self, X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, X)
        W = torch.exp(-1 * 23 * (dists ** 2))
        W.fill_diagonal_(0)
        return W

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

    def matrix_from_dict(self, indices):
        """
        This function creates a sparse matrix from a dictionary of indices.

        Args:
            indices (dict): The dictionary of indices
        """
        m = len(indices)
        matrix = torch.zeros(m, m)
        for i in range(m):
            matrix[i][i] = 1
            for j in range(m):
                if indices[j].item() in self.adj_dict[indices[i].item()]:
                    matrix[i][j] = 1

        return matrix

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        train_size = int(self.X.shape[0] * 0.9)
        valid_size = self.X.shape[0] - train_size

        dataset = TensorDataset(self.X, self.y, self.indices)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        ortho_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, ortho_loader, valid_loader

    def save_model(self):
        """
        This function saves the model.
        """
        # torch.save(self.spectral_net.state_dict(), self.weights_path)
        pass

    def _get_neighbors(self, indices_grad):
        """
        This function returns the neighbors of the given indices.

        Args:
            indices_grad (torch.Tensor):    The indices

        Returns:
            torch.Tensor:   The neighbors
        """
        neighbors = set()
        for i in range(len(indices_grad)):
            neighbors = neighbors.union(self.adj_dict[indices_grad[i].item()])
        return torch.tensor(list(neighbors))



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
