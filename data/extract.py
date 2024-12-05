import sys
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import networkx as nx
from pyvis.network import Network
import uuid


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
    return mod, params, input_shape, output_shape


def get_available_networks():
    """Returns a list of available networks and their configurations"""
    return [
        ("resnet-18", "NHWC"),
        # ("resnet-34", "NHWC"),
        # ("resnet-50", "NHWC"),
        # ("mobilenet", "NHWC"),
        # ("inception_v3", "NHWC"),
        # ("squeezenet_v1.1", "NCHW"),  # Only supports NCHW
    ]

def process_all_networks(target_str="llvm -mcpu=core-avx2", batch_size=1, dtype="float32"):
    """Process multiple networks and generate graphs for each task"""
    target = tvm.target.Target(target_str)
    networks = get_available_networks()
    
    all_graphs = {}  # Store graphs by network and task
    
    for network_name, layout in networks:
        print(f"\n=== Processing {network_name} ===")
        mod, params, input_shape, output_shape = get_network(
            network_name, batch_size, layout, dtype
        )
        
        # Extract tasks
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        
        network_graphs = []
        for idx, task in enumerate(tasks):
          print(f"Processing Task {idx} (workload key: {task.workload_key})")
          graph = gnn_feature_extractor(task, task.compute_dag.get_init_state())
          # network_graphs.append({
          #     'task_id': idx,
          #     'workload_key': task.workload_key,
          #     'graph': graph
          # })
          break
        all_graphs[network_name] = network_graphs
    
    return all_graphs

def gnn_feature_extractor(task, state):
    graph = nx.DiGraph()
    parent_stack = []
    types = []

    def preorder(node):
        current_node_id = str(graph.number_of_nodes()) + str(type(node))
        current_node_content = str(node)
        graph.add_node(current_node_id, title=current_node_content)

        if parent_stack:
            parent = parent_stack[-1]
            graph.add_edge(current_node_id, str(parent))
        parent_stack.append(current_node_id)

        return None

    def postorder(node):
        if parent_stack:
            parent_stack.pop()

        return None
    @tvm.tir.transform.prim_func_pass(opt_level=1)
    def ast_extractor(f, mod, ctx):
        graph.clear()
        parent_stack.clear()

        graph.add_node("root")
        parent_stack.append("root")
        
        tvm.tir.stmt_functor.ir_transform(f.body, preorder, postorder)
        return f

    schedule, args = task.compute_dag.apply_steps_from_state(state)

    with tvm.transform.PassContext(opt_level=1):
        mod = tvm.lower(schedule, args)
        print("TVM IR:")
        print(mod.astext())
        print(type(mod))
        print("Pythonic TensorIR:")
        print(mod.script()) 
        tvm.tir.stmt_functor.ir_transform(mod["main"].body, preorder, postorder)
    
    vizgraph(graph)
    return graph

if __name__ == "__main__":
    # Process all networks and generate graphs
    print("Starting network processing...")
    all_graphs = process_all_networks()
    
    # Print summary
    # print("\n=== Processing Summary ===")
    # for network_name, network_graphs in all_graphs.items():
    #     print(f"\n{network_name}:")
    #     print(f"Total tasks processed: {len(network_graphs)}")
    #     for task_data in network_graphs:
    #         print(f"- Task {task_data['task_id']}: {task_data['workload_key']}")
    #         print(f"  Nodes: {task_data['graph'].number_of_nodes()}")
    #         print(f"  Edges: {task_data['graph'].number_of_edges()}")
    #         break
    #     break