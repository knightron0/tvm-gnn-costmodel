from gensim.models import word2vec
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

types = ["LetStmt", "AttrStmt", "IfThenElse", "For", "While", "Allocate", "AllocateConst", "DeclBuffer", "BufferStore", "BufferRealize", "AssertStmt", "ProducerStore", "ProducerRealize", "Prefetch", "SeqStmt", "Evaluate", "Block", "BlockRealize", "Var", "SizeVar", "BufferLoad", "ProducerLoad", "Let", "Call", "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "Min", "Max", "EQ", "NE", "LT", "LE", "GT", "GE", "And", "Or", "Reduce", "Cast", "Not", "Select", "Ramp", "Broadcast", "Shuffle", "IntImm", "FloatImm", "StringImm", "Any"]

sentences = word2vec.LineSentence('traces.txt')

# model = word2vec.Word2Vec(sentences, vector_size=18, window=5, min_count=1, workers=4)
# model.build_vocab(sentences,update=True)
# model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
# model.save("full_new_ast_embed.model")


model = word2vec.Word2Vec.load("full_new_ast_embed.model")
print(model.wv.vectors.shape)

node_ids = model.wv.index_to_key
node_embeddings = (
    model.wv.vectors
)

node_targets = [node_id for node_id in node_ids]


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

plt.title("{} visualization of node embeddings".format(transform.__name__))
plt.savefig('node_embeddings.png')
plt.close()

