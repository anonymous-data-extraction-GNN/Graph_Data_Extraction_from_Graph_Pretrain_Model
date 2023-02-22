# Graph Data Extraction from Graph Pre-trained Model 

This project is the implementation of the paper "Graph Data Extraction from Graph Pre-trained Model". This paper targets the graph pre-trained model trained by context prediction task and proposes a novel framework for graph data extraction from the graph pre-trained model.

This repo contains the codes for the models in the paper.

## Dependencies

The project is tested on `Python 3.7`. 
```
pytorch                1.0.1
torch-cluster          1.2.4
torch-geometric        1.0.3
torch-scatter          1.1.2
torch-sparse           0.2.4
torch-spline-conv      1.0.6
rdkit                  2019.03.1.0
tqdm                   4.31.1
tensorboardX           1.6
```

## Pre-training
We use context prediction task to pre-train GNN.
```
python pretrain_contextpred.py --dataset zinc_standard_agent_pretrain --output_model_file saved/pretrain.pth
```

## Graph Data Extraction Attack
### Estimate the graph context model
```
python esitmate_graph_context_model.py --dataset zinc_standard_agent_attack --output_model_file saved/estimated.pth
```

### Graph data extraction
### 
```
python 
```







## Acknowledgements

Part of this code, especially the GNN model' pre-training, is based on Weihua Hu et al.'s [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns).