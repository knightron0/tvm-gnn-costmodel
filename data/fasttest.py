from gensim.models import fasttext
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

types = ["LetStmt", "AttrStmt", "IfThenElse", "For", "While", "Allocate", "AllocateConst", "DeclBuffer", "BufferStore", "BufferRealize", "AssertStmt", "ProducerStore", "ProducerRealize", "Prefetch", "SeqStmt", "Evaluate", "Block", "BlockRealize", "Var", "SizeVar", "BufferLoad", "ProducerLoad", "Let", "Call", "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "Min", "Max", "EQ", "NE", "LT", "LE", "GT", "GE", "And", "Or", "Reduce", "Cast", "Not", "Select", "Ramp", "Broadcast", "Shuffle", "IntImm", "FloatImm", "StringImm", "Any"]

# with open('traces.txt', 'r') as f:
#     sentences = [line.strip().split() for line in f]

# model = fasttext.FastText(sentences=sentences, vector_size=18, window=5, min_count=1, workers=4)
# model.build_vocab(sentences)
# model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
# model.save("fasttext_embed.model")


model = fasttext.FastText.load("/scratch/gilbreth/mangla/ast-models/fasttext_embed.model")

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

perplexity = 20
trans = transform(n_components=2, perplexity=perplexity)
node_embeddings_2d = trans.fit_transform(node_embeddings)
label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
node_colours = [label_map[target] for target in node_targets]
plt.figure(figsize=(16, 12), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
plt.axis('off')

scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                     c=node_colours,
                     cmap='tab20', 
                     s=300,
                     alpha=1.0)

for i, txt in enumerate(node_targets):
    x, y = node_embeddings_2d[i]
    plt.annotate(txt,
                xy=(x, y), 
                xytext=(x, y-0.08),
                fontsize=10,
                fontweight='normal',
                bbox=dict(facecolor='white',
                         edgecolor='gray', 
                         alpha=0.7,
                         pad=0.5),
                ha='center',
                va='center')

plt.title("t-SNE Visualization of TensorIR Node Types",
          fontsize=16,
          pad=20, 
          fontweight='bold')

plt.margins(0.05)

plt.savefig(f'type_embeddings_ppx_new_off_{perplexity}.png',
            dpi=300,
            bbox_inches='tight', 
            facecolor='white',
            pad_inches=0.2)
plt.close()