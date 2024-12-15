# GNN-based Cost Model for TVM's AutoScheduler

A graph neural network (GNN)-based learned cost model for tensor program optimization. We capture global semantics and dependencies by embedding the abstract syntax tree (AST) of [TVM's TensorIR](https://arxiv.org/abs/2207.04296) programs into a GNN to predict precise execution costs. Currently, this model is specific to NVIDIA V100s and is trained using [TLM's](https://github.com/zhaiyi000/tlm?tab=readme-ov-file) version of the [TenSet](https://github.com/tlc-pack/tenset) dataset.

This is a staging repository we used to run experiments, we've integrated our model into TVM [in this fork](https://github.com/dwijenchawra/tvm).

## Repository Structure
### [Feature Extraction](https://github.com/knightron0/tvm-gnn-costmodel/tree/main/feature-extraction)
Scripts to generate embeddings for each node in the TIR AST:
- The `extract.py` script decomposes high-level Relay networks into workloads (in TE) and lowers them to TIR. Using pre-order and post-order hooks, it builds a graph representation, from which uniform random walks are sampled to create the sentences for our corpus.
- `word2vec.py` and `fasttext.py` are used to fit the corresponding `gensim` models (and visualize the embeddings).
### [Dataset](https://github.com/knightron0/tvm-gnn-costmodel/tree/main/dataset)

### [Preprocessing](https://github.com/knightron0/tvm-gnn-costmodel/tree/main/dataset)
### [Models](https://github.com/knightron0/tvm-gnn-costmodel/tree/main/models)
### [Eval](https://github.com/knightron0/tvm-gnn-costmodel/tree/main/eval)
