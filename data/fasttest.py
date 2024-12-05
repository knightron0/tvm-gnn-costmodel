from gensim.models import fasttext
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

types = ["LetStmt", "AttrStmt", "IfThenElse", "For", "While", "Allocate", "AllocateConst", "DeclBuffer", "BufferStore", "BufferRealize", "AssertStmt", "ProducerStore", "ProducerRealize", "Prefetch", "SeqStmt", "Evaluate", "Block", "BlockRealize", "Var", "SizeVar", "BufferLoad", "ProducerLoad", "Let", "Call", "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "Min", "Max", "EQ", "NE", "LT", "LE", "GT", "GE", "And", "Or", "Reduce", "Cast", "Not", "Select", "Ramp", "Broadcast", "Shuffle", "IntImm", "FloatImm", "StringImm", "Any"]

with open('traces.txt', 'r') as f:
    sentences = [line.strip().split() for line in f]

# model = fasttext.FastText(sentences=sentences, vector_size=18, window=5, min_count=1, workers=4)
# model.build_vocab(sentences)
# model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
# model.save("fasttext_embed.model")


model = fasttext.FastText.load("fasttext_embed.model")

# Get embeddings for each type
node_embeddings = []
node_targets = []
for type_name in types:
    try:
        node_embeddings.append(model.wv[type_name])
        node_targets.append(type_name)
    except KeyError:
        print(f"Warning: '{type_name}' not found in model vocabulary")

node_embeddings = np.array(node_embeddings)
print(f"Embedding shape: {node_embeddings.shape}")

transform = TSNE

trans = transform(n_components=2, perplexity=10)
node_embeddings_2d = trans.fit_transform(node_embeddings)
label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
node_colours = [label_map[target] for target in node_targets]

plt.figure(figsize=(20, 16))
plt.axes().set(aspect="equal")
scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=node_colours, alpha=0.3)

# Add labels for each point
for i, txt in enumerate(node_targets):
    plt.annotate(txt, (node_embeddings_2d[i, 0], node_embeddings_2d[i, 1]), fontsize=8)

plt.title("{} visualization of type embeddings".format(transform.__name__))
plt.savefig('type_embeddings.png')
plt.close()
