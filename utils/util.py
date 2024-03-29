import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import subprocess as sp
import logger
from mlflow import MlflowClient
import mlflow
import time

def get_device(mode="single"):
    # multi mode
    if mode == "multi":
        print(f"Using GPU 0,1")
        return "0,1"
    # single mode
    else:
        try:
            def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]
            
            ACCEPTABLE_AVAILABLE_MEMORY = 16000  # MB
            COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = _output_to_list(
                sp.check_output(COMMAND.split()))[1:]
            memory_free_values = [int(x.split()[0])
                                for i, x in enumerate(memory_free_info)]
            # print(memory_free_values)
            if memory_free_values[0] > ACCEPTABLE_AVAILABLE_MEMORY:
                print(f"GPU 0 with {memory_free_values[0]} free momory")
                return "0"
            elif memory_free_values[1] > ACCEPTABLE_AVAILABLE_MEMORY:
                print(f"GPU 1 with {memory_free_values[1]} free momory")
                return "1"
            else:
                print(f"No free GPU")
                return ""
        except:
            print(f"No free GPU")
            return ""

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
    
## wrappers ##

def calculate_running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(
            f"\nTempo di esecuzione per {func.__name__}:  {int(minutes)}m {int(seconds)}s")
        return result
    return wrapper

## mlflow
def init_project_mlflow(model_type="Classification"):
    try:
        client = MlflowClient(tracking_uri="http://172.18.0.49:5050") 
        all_experiments = client.search_experiments()
        print(all_experiments)
        default_experiment = [
            {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
            for experiment in all_experiments
            if experiment.name == "Default"
        ][0]
        print(default_experiment)
        
        #Provide an Experiment description that will appear in the UI
        experiment_description = (
            f"This is the Signals {model_type} project. "
            "This experiment contains models for Signals."
        )

        # Provide searchable tags that define characteristics of the Runs that
        # will be in this Experiment
        experiment_tags = {
            "project_name": f"Signals {model_type}",
            "store_dept": f"{model_type}",
            "team": "ml",
            "project_quarter": "Q3-2023",
            "mlflow.note.content": experiment_description,
        }

        # Create the Experiment, providing a unique name
        classification_signals_experiment = client.create_experiment(
            name=f"Signals_{model_type}", tags=experiment_tags
        )
        print("MLFlow Project Created!")
    except:
        print("Error, MLFlow Project Already Exists!")

def create_exp_mlflow(model_name, model_type="Classification"):
    # Sets the current active experiment
    mlflow.set_tracking_uri("http://172.18.0.49:5050")
    semseg_barrier_experiment = mlflow.set_experiment(f"Signals_{model_type}")
    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = str(model_name).split("/")[-2]

    # Define an artifact path that the model will be saved to.
    #artifact_path = f"barrier_{args.model}"
    return run_name