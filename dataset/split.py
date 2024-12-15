import json
import random

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import MeasureInput, MeasureResult
from tvm.auto_scheduler.search_task import SearchTask
import pickle
import tqdm
import os
import tqdm
from concurrent.futures import ProcessPoolExecutor

file_name = "/scratch/gilbreth/mangla/bigdataset_metadata.txt"

def split_workloads():
  with open(file_name, "r") as f:
    lines = f.readlines()

  workloads = {}

  for i in range(len(lines)):
    line = lines[i]
    parts = line.split(",", 1)
    hash_str = parts[1].strip()[2:-2].split('"')[0]
    if hash_str not in workloads:
      workloads[hash_str] = 1
    else:
      workloads[hash_str] += 1

  x = []

  for workload in workloads:
    x.append(workload)

  random.shuffle(x)
  train_x = x[:int(len(x) * 0.8)]
  test_x = x[int(len(x) * 0.8):int(len(x) * 0.9)]
  val_x = x[int(len(x) * 0.9):]
  print(len(train_x), len(test_x), len(val_x))
  train_x_sum = sum([workloads[workload] for workload in train_x])
  test_x_sum = sum([workloads[workload] for workload in test_x])
  val_x_sum = sum([workloads[workload] for workload in val_x])
  print(train_x_sum, test_x_sum, val_x_sum)

  with open('train_workloads.txt', 'w') as f:
    for workload in train_x:
      f.write(f"{workload}\n")

  with open('test_workloads.txt', 'w') as f:
    for workload in test_x:
      f.write(f"{workload}\n")

  with open('val_workloads.txt', 'w') as f:
    for workload in val_x:
      f.write(f"{workload}\n")

  print("Saved workload keys to train_workloads.txt, test_workloads.txt, and val_workloads.txt")


def process_line(line, train_workloads, test_workloads, val_workloads):
  file_path = line.split(",", 1)[0].strip()
  workload_hash = line.split(",", 1)[1].strip()[2:-2].split('"')[0]
  set_name = "train" if workload_hash in train_workloads else "test" if workload_hash in test_workloads else "val"
  new_path = f"/scratch/gilbreth/mangla/newdataset/{set_name}"
  
  os.system(f"cp {file_path} {new_path}")
  dest_path = os.path.join(new_path, os.path.basename(file_path))
  return dest_path, file_path

def move_files():
  with open(file_name, "r") as f:
    lines = f.readlines()
  train_workloads = []
  test_workloads = []
  val_workloads = []

  with open('train_workloads.txt', 'r') as f:
    train_workloads = [line.strip() for line in f.readlines()]

  with open('test_workloads.txt', 'r') as f:
    test_workloads = [line.strip() for line in f.readlines()]

  with open('val_workloads.txt', 'r') as f:
    val_workloads = [line.strip() for line in f.readlines()]

  path_mapping = {}

  with ProcessPoolExecutor(max_workers=8) as executor:
    futures = []
    for line in lines:
      futures.append(executor.submit(process_line, line, train_workloads, test_workloads, val_workloads))

    for future in tqdm.tqdm(futures):
      dest_path, file_path = future.result()
      path_mapping[dest_path] = file_path

  with open('/scratch/gilbreth/mangla/path_mapping.json', 'w') as f:
      json.dump(path_mapping, f, indent=2)

move_files()