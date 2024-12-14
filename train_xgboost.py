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

import tvm
from tvm import auto_scheduler

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

NETWORK_INFO_FOLDER = './tlm_dataset/gen/dataset/network_info/v100'
TO_MEASURE_PROGRAM_FOLDER = None
MEASURE_RECORD_FOLDER = './tlm_dataset/gen/gen_data/measure_data_v100/'
HARDWARE_PLATFORM = None

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


def test_xgb_model():
    task, inputs, results = get_sample_records(50)

    model = auto_scheduler.XGBModel(num_warmup_sample=-1)
    model.update(inputs, results)
    preds = model.predict(task, [x.state for x in inputs])
    assert len(preds) == len(inputs)

    costs = [np.mean([x.value for x in res.costs]) for res in results]
    throughputs = np.min(costs) / costs

    # test regression quality
    rmse = np.sqrt(np.mean([np.square(pred - label) for pred, label in zip(preds, throughputs)]))
    assert rmse <= 0.3

    # test loading a record file
    tmpdir = tvm.contrib.utils.tempdir()
    tmpfile = tmpdir.relpath("test1")
    auto_scheduler.save_records(tmpfile, inputs, results)
    model.update_from_file(tmpfile)

    # test model serialization
    tmpfile = tmpdir.relpath("test2")
    model.save(tmpfile)
    model.load(tmpfile)


if __name__ == "__main__":
    load_and_register_tasks()
