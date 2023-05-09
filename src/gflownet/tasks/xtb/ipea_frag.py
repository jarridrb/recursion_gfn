import os
from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from rdkit.Chem.rdchem import Mol as RDMol

from gflownet.train import FlatRewards
from gflownet.tasks.seh_frag import SEHTask
from gflownet.tasks.xtb.ipea_xtb import XTB_IPEA

'''
To run, install xtb via conda, or via source (+ export PATH="~/xtb/bin/:$PATH")

A given, closed-shell organic molecule has paired electrons. 
Energy is required to kick out an electron, energy will be gained when you add an electron
The lowest energy required to kick out an electron is the ionization potential (IP)
The maximum energy gained upon receiving an electron is the electron affinity (EA)

By optimizing IP/EA, you can tailor a molecule to be a suitable (photo)catalyst/semiconductor/electrolyte. 
We here use two simple tasks to show by proof-of-concept that you can design the electronics of conjugated molecules

There are multiple different oracles implemented here, using simplest one right now.
Oracle 1: RDKIT MMFF geometry opt + XTB IPEA single point (neutral, vertical)
Oracle 2: RDKIT MMFF geometry opt + XTB geometry optimization (neutral) + XTB IPEA single point (vertical) 
Oracle 3: RDKIT MMFF geometry opt + XTB geometry optimization (neutral) + XTB geometry optimization (ionic)

There are 2 different tasks:
# Task 1: minimize adiabatic IP # in GFN we maximize -IP
# Task 2: minimize adiabatic EA # in GFN we maximize -EA

We are taking multiple short-cuts here, e.g. not searching for conformers properly. To be written.

To distribute the number of threads reasonable in the OpenMP section for XTB it is recommended to use 
export OMP_NUM_THREADS=<ncores>,1
export OMP_STACKSIZE=4G
'''

# FIXME add new FRAGMENTS
# FIXME add parallelization

default_ipea_config = {
    'task': 'ip', # or ea
    'oracle_config': {
        'log_dir': os.getcwd(),
        "moltocoord_config":{
            'conformer_config': {
                "num_conf": 2,
                "maxattempts": 100,
                "randomcoords": True,
                "prunermsthres": 1.5,
            },
        },
        "conformer_ladder": 0, # only applicable for multifidelity active learning
        "remove_scratch": True,
        'ff': 'mmff',
        'semiempirical': 'xtb',
        'mol_repr': 'mols',
    }
}

class IPEATask(SEHTask):
    """Sets up a task where the reward is computed using a semiempirical quantum chemical method
     for ionization potential and electron affinity
    """
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float, float],
                 num_thermometer_dim: int, rng: np.random.Generator = None,
                 wrap_model: Callable[[nn.Module], nn.Module] = None,
                 ipea_oracle_config: dict = default_ipea_config):
        self._wrap_model = wrap_model
        self.rng = rng
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.ipea_oracle = XTB_IPEA(**ipea_oracle_config)

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        scores = [-self.ipea_oracle(mol, oracle_level = 1) for mol in mols] # originally minimize, now maximizes
        is_valid = torch.tensor([not np.isnan(score) for score in scores]).bool() # fixme isnan isn't always invalid mol
        scores = torch.Tensor([-np.inf if np.isnan(score) else score for score in scores])
        scores = self.flat_reward_transform(scores).clip(-10, 10).reshape((-1, 1))
        return FlatRewards(scores), is_valid

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 1) # to be changed

    def inverse_flat_reward_transform(self, rp):
        return rp * 1 # to be changed