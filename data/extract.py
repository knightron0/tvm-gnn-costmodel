import sys
import numpy as np
import tqdm
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import networkx as nx
from pyvis.network import Network
import uuid


class Graph:
    def __init__(self):
        self.nodes = {}
        self.root = None

    def add_node(self, id, data, par=-1, root=False, neighbors=[]):
        self.nodes[id] = {
            'data': data,
            'root': root,
            'neighbors': neighbors.copy(),
            'parent': par
        }
        if root:
            self.root = id

    def get_node(self, id):
        return self.nodes[id]

    def dfs_trace(self, start_id):
        visited = set()
        trace = []
        
        def _dfs(node_id):
            if node_id not in visited:
                visited.add(node_id)
                node = self.nodes[node_id]
                trace.append(node)
                for neighbor in node['neighbors']:
                    if neighbor not in visited:
                        _dfs(neighbor)

        _dfs(start_id)
        return trace

    def bfs_trace(self, start_id):
        visited = set()
        trace = []
        queue = [start_id]
        
        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                node = self.nodes[node_id]
                trace.append(node)
                queue.extend(node['neighbors'])
                
        return trace

    def uniform_random_walk(self, start_id, steps):
        walk = [self.nodes[start_id]]
        curr_id = start_id
        
        for _ in range(steps):
            curr_node = self.nodes[curr_id]
            if not curr_node['neighbors']:
                break
            next_id = np.random.choice(curr_node['neighbors'])
            next_node = self.nodes[next_id]
            walk.append(next_node)
            curr_id = next_id
            
        return walk

    
      
def vizgraph(graph: nx.DiGraph):
    nt = Network('100%', '100%', directed=True, notebook=False)
    nt.show_buttons(filter_=['physics'])
    nt.options.physics.use_repulsion = True
    nt.from_nx(graph)
    # Save to file directly instead of using show() which requires template rendering
    nt.save_graph("graph" + str(uuid.uuid4()) + ".html")


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "densenet-121":
        mod, params = relay.testing.densenet.get_workload()
    elif name == "vgg-16":
        mod, params = relay.testing.vgg.get_workload(
            num_layers=16,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "vgg-19":
        mod, params = relay.testing.vgg.get_workload(
            num_layers=19,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    # elif name == "alexnet":
    #     mod, params = relay.testing.alexnet.get_workload(
    #         batch_size=batch_size,
    #         dtype=dtype,
    #         image_shape=image_shape,
    #     )
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size,
            dtype=dtype
        )
    elif name == "lstm":
        mod, params = relay.testing.lstm.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            iterations=10,
            num_hidden=1024,
        )
    elif name == "dcgan":
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size,
            dtype=dtype
        )
    return mod, params, input_shape, output_shape


def get_available_networks():
    """Returns a list of available networks and their configurations"""
    return [
        ("resnet-18", "NHWC"),
        # ("resnet-34", "NHWC"), 
        # ("resnet-50", "NHWC"),
        # ("mobilenet", "NHWC"),
        # ("inception_v3", "NHWC"),
        # ("squeezenet_v1.1", "NCHW"),
        # ("densenet-121", "NHWC"),
        # ("vgg-16", "NHWC"),
        # ("vgg-19", "NHWC"),
        # ("mlp", "NHWC"),
        # ("lstm", "NHWC"),
        # ("dcgan", "NHWC"),
    ]

def process_all_networks(target_str="llvm -mcpu=core-avx2", batch_size=1, dtype="float32"):
    """Process multiple networks and generate graphs for each task"""
    target = tvm.target.Target(target_str)
    networks = get_available_networks()
    
    all_graphs = {}  # Store graphs by network and task
    
    for network_name, layout in networks:
        print(f"\n=== Processing {network_name} ===")
        data = relay.var("data", shape=(batch_size, 1024), dtype=dtype)
        weight = relay.var("weight", shape=(1024, 1024), dtype=dtype)
        mod = tvm.IRModule.from_expr(relay.Function([data, weight], relay.nn.matmul(data, weight)))
        params = {}
        input_shape = (batch_size, 1024)
        output_shape = (batch_size, 1024)
        
        # Extract tasks
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        
        network_graphs = []
        for idx, task in tqdm.tqdm(enumerate(tasks), total=len(tasks)):
            print(f"Processing Task {idx} (workload key: {task.workload_key})")
            graph_out = build_graph(task, task.compute_dag.get_init_state())
            
            # Write traces to file
            # dt = " ".join([type(x['data']).__name__ for x in graph_out.dfs_trace(graph_out.root)])
            # bt = " ".join([type(x['data']).__name__ for x in graph_out.bfs_trace(graph_out.root)])
            # with open("traces.txt", "a") as f:
            #     f.write(dt + "\n" + bt + "\n")
            
            # for node_id in graph_out.nodes:
            #     for _ in range(3):
            #         with open("traces.txt", "a") as f:
            #             f.write(" ".join([type(x['data']).__name__ for x in graph_out.uniform_random_walk(node_id, 50)]) + "\n")
            #     node = graph_out.nodes[node_id]
            #     if len(node['neighbors']) > 1:
            #         neighbor_types = [type(node['data']).__name__] + [type(graph_out.nodes[n]['data']).__name__ for n in node['neighbors'] if n != node['parent']]
            #         with open("traces.txt", "a") as f:
            #             f.write(" ".join(neighbor_types) + "\n")
            
            network_graphs.append({
                'task_id': idx,
                'workload_key': task.workload_key,
                'graph': graph_out
            })
            # Visualize the graph structure
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Create NetworkX directed graph for tree structure
            G = nx.DiGraph()
            
            # Add nodes and edges starting from root
            def add_nodes_from_root(node_id):
                node_data = graph_out.nodes[node_id]
                G.add_node(node_id, data=type(node_data['data']).__name__)
                for neighbor in node_data['neighbors']:
                    if neighbor != node_data['parent']:  # Only process children
                        G.add_edge(node_id, neighbor)
                        add_nodes_from_root(neighbor)
                        
            add_nodes_from_root(graph_out.root)
            
            # Set up the plot
            plt.figure(figsize=(20,15))
            
            # Use hierarchical layout to show tree structure
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                 node_color='lightblue',
                                 node_size=2000,
                                 alpha=0.7)
            
            # Draw edges with arrows to show hierarchy
            nx.draw_networkx_edges(G, pos,
                                 edge_color='gray', 
                                 width=2,
                                 alpha=0.5,
                                 arrows=True,
                                 arrowsize=20)
            
            # Add labels with node type information
            labels = nx.get_node_attributes(G, 'data')
            nx.draw_networkx_labels(G, pos, labels,
                                  font_size=8,
                                  font_weight='bold')
            
            plt.title(f"Tree Structure for Task {idx}\nWorkload: {task.workload_key}",
                     pad=20, fontsize=12)
            plt.axis('off')
            
            # Save the plot
            plt.savefig(f'task_{idx}_tree.png',
                       bbox_inches='tight', 
                       dpi=300)
            plt.close()
            break
        all_graphs[network_name] = network_graphs
    
    return all_graphs

def build_graph(task, state):
    graph = Graph()
    parent_stack = []

    def preorder(node):
        current_node_id = len(graph.nodes)
        graph.add_node(current_node_id, node, par=parent_stack[-1] if parent_stack else -1, root=(current_node_id == 0))

        if parent_stack:
            parent = parent_stack[-1]
            graph.nodes[parent]['neighbors'].append(current_node_id)
            graph.nodes[current_node_id]['neighbors'].append(parent)
        
        parent_stack.append(current_node_id)

        return None

    def postorder(node):
        if parent_stack:
            parent_stack.pop()
        return None

    @tvm.tir.transform.prim_func_pass(opt_level=1)
    def ast_extractor(f, mod, ctx):
        tvm.tir.stmt_functor.ir_transform(f.body, preorder, postorder)
        return f

    schedule, args = task.compute_dag.apply_steps_from_state(state)

    with tvm.transform.PassContext(opt_level=1):
        mod = tvm.lower(schedule, args)
        tvm.tir.stmt_functor.ir_transform(mod["main"].body, preorder, postorder)
        
    
    return graph

if __name__ == "__main__":
    # Process all networks and generate graphs
    print("Starting network processing...")
    all_graphs = process_all_networks()
    
    # Print summary
    print("\n=== Processing Graphs ===")
    for network_name, network_graphs in all_graphs.items():
        print(f"\n{network_name}:")
        print(f"Total tasks processed: {len(network_graphs)}")
        for task_data in network_graphs:
            print(f"- Task {task_data['task_id']}: {task_data['workload_key']}")
            print(f"  Nodes: {len(task_data['graph'].nodes)}")
