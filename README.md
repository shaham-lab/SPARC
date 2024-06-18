# GraphSpectralNet

Link to the paper - [SpectralNet](https://openreview.net/pdf?id=HJ_aoCyRZ)

## Requirements

To run GraphSpectralNet, you'll need Python 3.x and to run the requirements.txt file as follows: <br>

```bash
pip3 install -r req.txt
```


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
        "epochs": 600,
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

