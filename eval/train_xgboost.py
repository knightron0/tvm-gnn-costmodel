# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Test cost models"""

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


def load_dataset(args, train_workloads):
    try:
        inputs, res = auto_scheduler.RecordReader(args).read_lines()
    except Exception as e:
        print(f"Could not read file {args}, error: {e}")
        exit()
    train_inputs = [] 
    train_res = []
    assert(len(inputs) == len(res))
    for i in range(len(inputs)):
        hash_str = inputs[i].task.workload_key.split('"')[1].split('"')[0]
        if hash_str not in train_workloads:
            continue
        train_inputs.append(inputs[i])
        train_res.append(res[i])
    return train_inputs, train_res

if __name__ == "__main__":
    import argparse
    train_workloads = open('train_workloads.txt', 'r').readlines()
    train_workloads = [x.strip() for x in train_workloads]

    directory_path = "/scratch/gilbreth/mangla/tlm_dataset/gen/gen_data/measure_data_v100/"

    # Load task registry
    register_data_path("nvidia/nvidia-v100")
    print("Load all tasks...")
    tasks = load_and_register_tasks()

    files = os.listdir(directory_path)
    name = None
    for i, filename in enumerate(files):
        if "graph" in filename:
            continue
        inputs, results = load_dataset(directory_path + filename, train_workloads)
        if len(inputs) == 0:
            print("Nothing found in", filename)
            continue

        print("Updating XGBoost with", filename, len(inputs))

        model = auto_scheduler.XGBModel(verbose_eval=1000, num_warmup_sample=-1, model_file='xgb.model')
        if name:
            model.load(name)
        model.update(inputs, results)
        
        name = '/scratch/gilbreth/mangla/xgb-models-new/xgboost_' + filename + str(i) + '.model'
        model.save(name)
        
