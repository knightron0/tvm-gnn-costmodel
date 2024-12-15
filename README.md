# GNN-based Cost Model for TVM's AutoScheduler

A graph neural network (GNN)-based learned cost model for tensor program optimization. We capture global semantics and dependencies by embedding the abstract syntax tree (AST) of [TVM's TensorIR](https://arxiv.org/abs/2207.04296) programs into a GNN to predict precise execution costs. Currently, this model is specific to NVIDIA V100s and is trained using [TLM's](https://github.com/zhaiyi000/tlm?tab=readme-ov-file) version of the [TenSet](https://github.com/tlc-pack/tenset) dataset.

This is a staging repository we used to run experiments, we've integrated our model into TVM [in this fork](https://github.com/dwijenchawra/tvm).

## Repository Structure
### Feature Extraction
Scripts to generate embeddings for each node in the TIR AST. The `extract.py` script decomposes high-level Relay networks into workloads (in TE) and lowers them to TIR. Using pre-order and post-order hooks, it builds a graph representation, from which uniform random walks are sampled to create the sentences for our corpus. `word2vec.py` and `fasttext.py` are used to fit the corresponding `gensim` models (and visualize the embeddings).

### Dataset
Converting the original TLM dataset's `<MeasureInput, MeasureOutput>` samples to [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) objects. `lower_meas_recs.py` reads JSONs with measure records from TLM, lowers the TE workloads to TensorIR sketches, and builds/saves graphs for the same. `make_pt.py` takes the graphs and converts them into PyTorch Geometric objects, complete with embeddings from FastText. `split.py` is for splitting the workloads into train/test/val sets. 

### Preprocessing
`fix_data.py` does Z-Score normalization and `data_analysis.py` was used to create visuals for the report. 

### Models
`gcn.py` is the training script for our primary model (3-layer GCN with TopK pooling). `exps/` contains a bunch of other models/configurations we tried.

### Evals
Contains direct evaluation scripts on the test dataset. `train_xgboost.py` extracts features from workloads included in the training dataset and uses them to train an XGBoost model. `eval_xgb.py` and `eval_gcn.py` run the corresponding models through the test set. 
