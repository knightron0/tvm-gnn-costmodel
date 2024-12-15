import tempfile

import numpy as np
import os
import tvm
from tvm import auto_scheduler
import xgboost as xgb
from tvm.testing.auto_scheduler import matmul_auto_scheduler_test
import argparse
from tvm.auto_scheduler.measure import MeasureInput, MeasureResult
from tvm.auto_scheduler.search_task import SearchTask
import pickle
import tvm.target
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Add tqdm import
import json
import torch
import torch.nn.functional as F

NETWORK_INFO_FOLDER = None
TO_MEASURE_PROGRAM_FOLDER = None
MEASURE_RECORD_FOLDER = None
HARDWARE_PLATFORM = None

def register_data_path(target_str):
    assert isinstance(target_str, str)
    model_list = ["i7", "v100", "a100", "2080", "None"]
    for model in model_list:
        if model in target_str:
            break
    assert model != "None"

    xgb.set_config(verbosity=2)
    print(f"register data path: {model}")
    global NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER, HARDWARE_PLATFORM
    NETWORK_INFO_FOLDER = f"/scratch/gilbreth/mangla/tlm_dataset/gen/dataset/network_info/{model}"
    TO_MEASURE_PROGRAM_FOLDER = f"/scratch/gilbreth/mangla/tlm_dataset/gen/dataset/to_measure_programs/{model}"
    MEASURE_RECORD_FOLDER = f"/scratch/gilbreth/mangla/tlm_dataset/gen/dataset/measure_records/{model}"
    HARDWARE_PLATFORM = model

def load_and_register_tasks():
    assert NETWORK_INFO_FOLDER is not None
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors
        )

    return tasks


def get_sample_records(number):
    """Generate a list of random MeasureInput and MeasureResult pairs"""
    N = 128
    task = auto_scheduler.SearchTask(func=matmul_auto_scheduler_test, args=(N, N, N), target="llvm")
    policy = auto_scheduler.SketchPolicy(task, verbose=0)
    states = policy.sample_initial_population()[:number]

    inputs = [auto_scheduler.MeasureInput(task, s) for s in states]
    results = [
        auto_scheduler.MeasureResult([np.random.uniform(0.5, 1.0)], 0, "", 0.1, 0)
        for _ in range(len(inputs))
    ]

    return task, inputs, results


def test_random_model():
    task, inputs, results = get_sample_records(50)

    model = auto_scheduler.RandomModel()
    model.update(inputs, results)
    scores = model.predict(task, [x.state for x in inputs])
    assert len(scores) == len(inputs)


def train_xgb_model(filename):
    model.save('xgb.model')


def load_dataset(args, test_workloads):
    try:
        inputs, res = auto_scheduler.RecordReader(args).read_lines()
    except Exception as e:
        print(f"Could not read file {args}, error: {e}")
        exit()
    test_inputs = [] 
    test_res = []
    # print(type(inputs[0]), type(res[0]))
    # exit()
    assert(len(inputs) == len(res))
    for i in range(len(inputs)):
        hash_str = inputs[i].task.workload_key.split('"')[1].split('"')[0]
        if hash_str not in test_workloads:
            continue
        test_inputs.append(inputs[i])
        test_res.append(res[i])
    return test_inputs, test_res

if __name__ == "__main__":
    import argparse
    test_workloads = open('test_workloads.txt', 'r').readlines()
    test_workloads = [x.strip() for x in test_workloads]

    directory_path = "/scratch/gilbreth/mangla/tlm_dataset/gen/gen_data/measure_data_v100/"

    # Load task registry
    register_data_path("nvidia/nvidia-v100")
    print("Load all tasks...")
    tasks = load_and_register_tasks()

    files = os.listdir(directory_path)
    model = auto_scheduler.XGBModel(verbose_eval=1000, num_warmup_sample=-1, model_file='xgb.model')
    model.load("/scratch/gilbreth/mangla/xgb-bestmodel.model")
    print('Loaded Model')

    name = None
    for i, filename in enumerate(files):
        if "graph" in filename:
            continue
        inputs, results = load_dataset(directory_path + filename, test_workloads)
        if len(inputs) == 0:
          continue

        
        # # for inp in inputs:
        # #   # score = model.predict(inp.task, [inp.state])
        # #   print(inp.task)
        # #   print(inp.state)
        # #   print(score)
        # #   print(results[0])
        # #   break
        # # break
        # # scores = model.predict([x.task for x in inputs], [x.state for x in inputs])

        # actual_values = torch.tensor([result.cost for result in results], dtype=torch.float32)
        # predicted_values = torch.tensor(scores, dtype=torch.float32)
        # print(actual_values.shape, predicted_values.shape)

        # huber_loss = F.huber_loss(predicted_values, actual_values)

        # print(f"Huber Loss for {filename}: {huber_loss.item()}")
        # break