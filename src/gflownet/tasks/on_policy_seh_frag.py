import ast
import copy
import os
import shutil
import socket
import wandb
import heapq
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
import scipy.stats as stats
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN, ThompsonSamplingGraphTransformerGFN
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import GFNTrainer
from gflownet.train import RewardScalar
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.utils.transforms import thermometer

import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


if __name__ == '__main__':
    hps = {
        'log_dir': "./logs/debug_run",
        'overwrite_existing_exp': True,
        'qm9_h5_path': '/data/chem/qm9/qm9.h5',
        'log_dir': './logs/debug_run',
        'num_training_steps': 10_000,
        'temperature_sample_dist': 'constant',
        'temperature_dist_params': (32.),
        'validate_every': 20000,
        'lr_decay': 20000,
        'sampling_tau': 0.,#0.99,
        'num_data_loader_workers': 0,
        'do_thompson_sampling_bootstrap': False,
        'random_action_prob': 0.0,
        'thompson_sampling_bootstrap_prob': 0.0,
        'num_top_k_to_track': 10,
        'mode_reward_threshold': 0.87,
        'do_ts': False,
        'group_name': 'cool_group',
        'reward_threshs': [0.875, .9375, 1.0, 1.125],
        'k_vals': [10, 100],
        'learning_rate': 0.00043521031934084793,
        'Z_learning_rate': 0.002716724069009233,
        'weight_decay': 1.0260546404209485e-7,
    }
    if os.path.exists(hps['log_dir']):
        if hps['overwrite_existing_exp']:
            shutil.rmtree(hps['log_dir'])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps['log_dir'])

    ray.init(
        num_cpus=6,
        num_gpus=torch.cuda.device_count(),
    )

    trainable  = tune.with_resources(SEHFragTrainer, {'cpu': 6.0, 'gpu': 1.0})
    tuner = tune.Tuner(
        trainable,
        param_space=hps,
        run_config=ray.air.config.RunConfig(
            local_dir='~/scratch/ray_results',
            name='ts_tuning_seh_frag',
            stop={'training_iteration': 30000},
            verbose=3,
            callbacks=[WandbLoggerCallback(
                project='gfn_exploration',
                group='on_policy_tuning_seh_frag',
                dir='~/scratch/wandb',
                log_config=True
            )]
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,
        )
    )

    tuner.fit()

