# GraphSpectralNet

This PyTorch implementation a framework of Graph datasets embedding that leverages low-dimensional embeddings of graph Laplacians to address scalability and generalization challenges, enabling efficient operations on large datasets and new nodes, effective mini-batching for SGD-based training.
This framework is implementated in "SpecNetGCN: Scalable and Generalizable Spectral Graph Convolutional Network"


## Requirements

To run GraphSpectralNet, you'll need Python 3.x and to run the requirements.txt file as follows: <br>

```bash
pip3 install -r requirements.txt
```
Install [METIS](https://stackoverflow.com/questions/54326406/how-to-install-metis-library-for-python)

#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.

## Usage

To use GraphSpectralNet on Cora, Pubmed, or the citeseer dataset, `cd` to src directory and run the following:

```bash
python3 main.py config/cora|pubmed|citeseer.json
```
If you want to use your own dataset, you should provide a json config file that looks like the following:

```json
{
    "dataset": "cora",
    "n_clusters": 7,
    "is_sparse_graph": false,
    "should_use_ae": false,
    "should_use_siamese": false,
    "should_check_generalization": true,
    "ae": {
        "architecture": {
            "hidden_dim1": 512,
            "hidden_dim2": 512,
            "hidden_dim3": 2048,
            "output_dim": 10
        },
        "epochs": 100,
        "n_samples": 70000,
        "lr": 1e-3,
        "lr_decay": 0.1,
        "min_lr": 1e-7,
        "patience": 5,
        "batch_size": 256
    },
    "spectral": {
        "architecture": {
            "n_layers": 4,
            "hidden_dim1": 1024,
            "hidden_dim2": 1024,
            "hidden_dim3": 2048,
            "output_dim": 512
        },
        "epochs": 250,
        "lr": 1e-3,
        "lr_decay": 0.05,
        "min_lr": 1e-10,
        "batch_size": 10000,
        "n_neighbors": 8,
        "scale_k": 2,
        "is_local_scale": false,
        "n_clusters": 7,
        "patience": 10,
        "bsize": 10000,
        "num_clusters": 1,
        "num_clusters_val": 1,
        "num_clusters_test": 1,
        "diag_lambda": -1
    }
}

```

