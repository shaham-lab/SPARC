import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

class AE(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AE, self).__init__()
        self.architecture = architecture

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.architecture["hidden_dim1"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim1"], self.architecture["hidden_dim2"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim2"], self.architecture["hidden_dim3"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim3"], self.architecture["output_dim"]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.architecture["output_dim"], self.architecture["hidden_dim3"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim3"], self.architecture["hidden_dim2"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim2"], self.architecture["hidden_dim1"]),
            nn.ReLU(),
            nn.Linear(self.architecture["hidden_dim1"], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embed_single(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function embeds a single data point using the trained autoencoder.
        Args:
            X (torch.Tensor):  the data point to embed
        Returns:
            torch.Tensor: the embedded data point
        """
        return self.encoder(X)

class AETrainer:
    def __init__(self, config: dict, device: torch.device):
        """

        Args:
            config (dict):          the configuration dictionary
            device (torch.device):  the device to train on
        """
        self.device = device
        self.dataset = config["dataset"]
        self.ae_config = config["ae"]
        self.lr = self.ae_config["lr"]
        self.epochs = self.ae_config["epochs"]
        self.lr_decay = self.ae_config["lr_decay"]
        self.patience = self.ae_config["patience"]
        self.n_samples = self.ae_config["n_samples"]
        self.batch_size = self.ae_config["batch_size"]
        self.architecture = self.ae_config["architecture"]
        self.output_dim = self.architecture["output_dim"]
        self.weights_path = f"./weights/{self.dataset}/ae_weights_{self.output_dim}.pth"
    
    def train(self, features_batches, val_features_batches) -> AE:
        """
        This function trains the autoencoder on the given data.

        Args:
            X (torch.Tensor):   the data to train on

        Returns:
            AE: the trained autoencoder

        """
        self.features_batches = features_batches
        self.val_features_batches = val_features_batches
        self.criterion = nn.MSELoss()
        self.ae_net = AE(self.architecture, input_dim=self.features_batches[0].shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode="min", 
                                                              factor=self.lr_decay, 
                                                              patience=self.patience)

        if os.path.exists(self.weights_path):
            self.ae_net.load_state_dict(torch.load(self.weights_path))
            return self.ae_net
        
        batches = len(self.features_batches)
        batches_val = len(self.val_features_batches)
        print("Training Autoencoder:")
        for epoch in range(self.epochs):
            train_loss = 0.0
            for pid in range(batches):
                batch_x = self.features_batches[pid]
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= batches
            valid_loss = self.validate(self.val_features_batches)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.ae_config["min_lr"]: break
            print("Epoch: {}/{}, Train Loss: {:.4f}, Valid Loss: {:.4f}, LR: {:.6f}".
            format(epoch + 1, self.epochs, train_loss, valid_loss, current_lr))
        
        torch.save(self.ae_net.state_dict(), self.weights_path)
        return self.ae_net
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        This function validates the autoencoder on the given data during the training process.

        Args:
            valid_loader (DataLoader):  the data to validate on

        Returns:
            float: the validation loss
        """

        self.ae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for pid in range(len(valid_loader)):
                batch_x = self.val_features_batches[pid]
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def embed(self, features_batches, val_features_batches):
        """
        This function embeds the given data using the trained autoencoder.

        Args:
            X (torch.Tensor):  the data to embed

        Returns:
            torch.Tensor: the embedded data
        """

        print("Embedding data ...")
        self.ae_net.eval()
        with torch.no_grad():
            encoded_features_batches = []
            for pid in range(len(features_batches)):
                batch_x = features_batches[pid]
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                encoded_features_batches.append(self.ae_net.encoder(batch_x))
            
            encoded_val_features_batches = []
            for pid in range(len(val_features_batches)):
                batch_x = val_features_batches[pid]
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                encoded_val_features_batches.append(self.ae_net.encoder(batch_x))
        return encoded_features_batches, encoded_val_features_batches


    def embed_single(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function embeds a single data point using the trained autoencoder.
        Args:
            X (torch.Tensor):  the data point to embed
        Returns:
            torch.Tensor: the embedded data point
        """
        self.ae_net.eval()
        with torch.no_grad():
            X = X.to(self.device)
            return self.ae_net.encoder(X)


    def _get_data_loader(self) -> tuple:
        """
        This function splits the data into train and validation sets 
        and returns the corresponding data loaders.

        Returns:
            tuple: the train and validation data loaders
        """
        
        X = self.X[:self.n_samples]
        trainset_len = int(len(X) * 0.9)
        validset_len = len(X) - trainset_len
        trainset, validset = random_split(X, [trainset_len, validset_len])
        train_loader = DataLoader(trainset, batch_size=self.ae_config["batch_size"], shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.ae_config["batch_size"], shuffle=False)
        return train_loader, valid_loader