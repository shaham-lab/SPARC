import torch
import numpy as np

from AETrainer import *
from SiameseTrainer import *
from SpectralTrainer import *
from sklearn.cluster import KMeans


class SpectralNet:
    def __init__(self, n_clusters: int, config: dict):
        """
        Args:
            n_clusters (int):   The dimension of the projection subspace
            config (dict):      The configuration dictionary
        """

        self.n_clusters = n_clusters
        self.config = config
        self.embeddings_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Using {self.device} device")

    def fit(self, train, val):
        """ train, adj_dict, sparse_adj
        Performs the main training loop for the SpectralNet model.

        Args:
            X (torch.Tensor):   Data to train the networks on
            y (torch.Tensor):   Labels in case there are any. Defaults to None.
            adj (torch.Tensor): Adjacency matrix of the graph. Defaults to None.
        """

        should_use_ae = self.config["should_use_ae"]
        should_use_siamese = self.config["should_use_siamese"]
        create_weights_dir()
        parts, features_batches, support_batches, y_train_batches, train_mask_batches = train
        idx_parts, val_features_batches, val_support_batches, y_val_batches, val_mask_batches = val
        if should_use_ae:
            
            ae_trainer = AETrainer(self.config, self.device)
            self.ae_net = ae_trainer.train(features_batches, val_features_batches)
            features_batches, val_features_batches = ae_trainer.embed(features_batches, val_features_batches)
            train = parts, features_batches, support_batches, y_train_batches, train_mask_batches
            val = idx_parts, val_features_batches, val_support_batches, y_val_batches, val_mask_batches
        
        
        if should_use_siamese:
            train_mask = train_mask_batches[0]
            # val_mask = val_mask_batches[0]
            # x = torch.cat((features_batches[0][train_mask], val_features_batches[0][val_mask]), 0)
            x = features_batches[0][train_mask]
            siamese_trainer = SiameseTrainer(self.config, self.device)
            self.siamese_net = siamese_trainer.train(x)
        else:
            self.siamese_net = None

        # is_sparse = self.config["is_sparse_graph"]
        # if is_sparse:
        #     build_ann(torch.cat((x_train, x_valid), 0))
        is_sparse = False
        spectral_trainer = SpectralTrainer(self.config, self.device, is_sparse=is_sparse, siamese_net=self.siamese_net)
        self.spec_net = spectral_trainer.train(train, val)
        
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predicts the cluster assignments for the given data.
        
        Args:
            X (torch.Tensor):   Data to be clustered
        
        Returns:
            np.ndarray:  The cluster assignments for the given data

        """      
        # X = X.view(X.size(0), -1)
        X = X.to(self.device)
        should_use_ae = self.config["should_use_ae"]
        if should_use_ae:
            X = self.ae_net.encoder(X)
        X = X.to(self.device)
        self.embeddings_ = self.spec_net(X, should_update_orth_weights = False).detach().cpu().numpy()
        # self.embeddings_ = self.embeddings_ @ self.spec_net.rotation_matrix
        
        # cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_)
        # return cluster_assignments
        return self.embeddings_

    
    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Performs k-means clustering on the spectral-embedding space.

        Args:
            embeddings (np.ndarray):   the spectral-embedding space

        Returns:
            np.ndarray:  the cluster assignments for the given data
        """
        kmeans = KMeans(n_clusters=self.n_clusters).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
    
