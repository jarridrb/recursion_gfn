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
from gflownet.utils.metrics import compute_diverse_top_k
from gflownet.utils.transforms import thermometer

import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import wandb


_HP_KEYS_TO_LOG_AS_GROUP_NAME = (
    'do_ts',
    'ensemble_size',
    'do_thompson_sampling_bootstrap',
    'thompson_sampling_bootstrap_prob',
    'learning_rate',
    'Z_learning_rate',
    'global_batch_size',
    'random_action_prob',
    'sampling_tau',
)

_HP_KEYS_TO_LOG_AS_RUN_NAME = _HP_KEYS_TO_LOG_AS_GROUP_NAME + ('seed',)

_DEFAULT_HPS = {
    'hostname': socket.gethostname(),
    'bootstrap_own_reward': False,
    'learning_rate': 1e-4,
    'Z_learning_rate': 1e-4,
    'global_batch_size': 64,
    'num_emb': 128,
    'num_layers': 4,
    'tb_epsilon': None,
    'illegal_action_logreward': -75,
    'reward_loss_multiplier': 1,
    'temperature_sample_dist': 'uniform',
    'temperature_dist_params': (.5, 32.),
    'weight_decay': 1e-8,
    'num_data_loader_workers': 8,
    'momentum': 0.9,
    'adam_eps': 1e-8,
    'lr_decay': 20000,
    'Z_lr_decay': 20000,
    'clip_grad_type': 'norm',
    'clip_grad_param': 10,
    'random_action_prob': 0.,
    'valid_random_action_prob': 0.,
    'sampling_tau': 0.,
    'num_thermometer_dim': 32,
}

class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching (TODO: port to this repo).
    """
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float, float],
                 num_thermometer_dim: int, rng: np.random.Generator = None, wrap_model: Callable[[nn.Module],
                                                                                                 nn.Module] = None):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == 'constant':
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == 'gamma':
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == 'uniform':
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'loguniform':
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == 'beta':
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {'beta': beta, 'encoding': beta_enc}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(cond_info['beta'].shape), \
            f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models['seh'](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class SEHFragTrainer(GFNTrainer):
    def setup(self, hps: Dict[str, Any]):
        super().setup(hps)
        self.wandb_client = get_wandb_client(self.hps)

        self.reward_threshs = hps['reward_threshs']
        for thresh in self.reward_threshs:
            setattr(self, f'mode_inchis_thresh_{thresh}', set())

        self.k_vals = hps['k_vals']
        for k in self.k_vals:
            setattr(self, f'top_{k}_heap', [])
            setattr(self, f'top_{k}_mols', set())
            setattr(self, f'top_{k}_diverse', [])


    def default_hps(self) -> Dict[str, Any]:
        return {
            'hostname': socket.gethostname(),
            'bootstrap_own_reward': False,
            'learning_rate': 1e-4,
            'Z_learning_rate': 1e-4,
            'global_batch_size': 64,
            'num_emb': 128,
            'num_layers': 4,
            'tb_epsilon': None,
            'illegal_action_logreward': -75,
            'reward_loss_multiplier': 1,
            'temperature_sample_dist': 'uniform',
            'temperature_dist_params': (.5, 32.),
            'weight_decay': 1e-8,
            'num_data_loader_workers': 8,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20000,
            'Z_lr_decay': 20000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': 0.,
            'valid_random_action_prob': 0.,
            'sampling_tau': 0.,
            'num_thermometer_dim': 32,
        }

    def setup_algo(self):
        self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, self.hps, max_nodes=9)

    def setup_task(self):
        self.task = SEHTask(dataset=self.training_data, temperature_distribution=self.hps['temperature_sample_dist'],
                            temperature_parameters=self.hps['temperature_dist_params'],
                            num_thermometer_dim=self.hps['num_thermometer_dim'], wrap_model=self._wrap_model_mp)

    def setup_model(self):
        model_type = GraphTransformerGFN if not self.hps['do_ts'] else ThompsonSamplingGraphTransformerGFN

        init_kwargs = {
            'env_ctx': self.ctx,
            'num_emb': self.hps['num_emb'],
            'num_layers': self.hps['num_layers']
        }

        if self.hps['do_ts']:
            init_kwargs['ensemble_size'] = self.hps['ensemble_size']
            init_kwargs['prior_weight'] = self.hps['prior_weight']

        self.model = model_type(**init_kwargs)

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=self.hps['num_thermometer_dim'])

    def _get_log_info(self, batch, info):
        self._update_modes_list(batch)
        self._update_top_k_mols(batch)

        ret_dict = {
            'batch_reward_mean': batch.rewards[batch.num_offline:].mean().item(),
        }

        for thresh in self.reward_threshs:
            ret_dict[f'num_modes_found_thresh_{thresh}'] = len(getattr(self, f'mode_inchis_thresh_{thresh}'))

        for k in self.k_vals:
            ret_dict[f'top_{k}_mean_reward'] = np.array(
                getattr(self, f'top_{k}_heap')
            ).mean().item()

            ret_dict[f'top_{k}_diverse'] = self._update_top_k_diverse(batch, k)

        return ret_dict

    def _update_top_k_diverse(self, batch, k):
        online_rewards = batch.rewards[batch.num_offline:]
        online_mols = batch.mols[batch.num_offline:]

        top_k_diverse = getattr(self, f'top_{k}_diverse')
        top_k_diverse.extend(list(zip(online_rewards, online_mols)))

        top_k_diverse_val, top_k_diverse = compute_diverse_top_k(
            list(map(lambda x: x[1], top_k_diverse)),
            list(map(lambda x: x[0], top_k_diverse)),
            k=k
        )

        setattr(self, f'top_{k}_diverse', top_k_diverse)

        return top_k_diverse_val

    def _update_top_k_mols(self, batch):
        online_mols = batch.mols[batch.num_offline:]
        online_rewards = batch.rewards[batch.num_offline:]

        unique_mols_rewards = [
            (reward, mol)
            for reward, mol in zip(online_rewards, online_mols)
            #if Chem.CanonSmiles(mol) not in self.top_k_mols
        ]

        for reward, mol in unique_mols_rewards:
            for k in self.k_vals:
                heap = getattr(self, f'top_{k}_heap')
                if len(heap) < k:
                    heap_op = heapq.heappush
                else:
                    heap_op = heapq.heappushpop

                heap_op(heap, reward.item())


    def _update_modes_list(self, batch):
        online_rewards = batch.rewards[batch.num_offline:]
        online_mols = batch.mols[batch.num_offline:]

        for reward_thresh in self.reward_threshs:
            is_mode_ind = online_rewards > reward_thresh
            if is_mode_ind.sum() == 0:
                continue

            new_mode_mols = [online_mols[i] for i in is_mode_ind.nonzero().flatten()]

            new_inchi = list(map(Chem.MolToInchi, new_mode_mols))
            getattr(self, f'mode_inchis_thresh_{reward_thresh}').update(new_inchi)

    def _setup(self):
        hps = self.hps
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.setup_env_context()
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(non_Z_params, hps['learning_rate'], (hps['momentum'], 0.999),
                                    weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
        self.opt_Z = torch.optim.Adam(Z_params, hps['Z_learning_rate'], (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / hps['lr_decay']))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2**(-steps / hps['Z_lr_decay']))

        self.sampling_tau = hps['sampling_tau']
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model
        eps = hps['tb_epsilon']
        hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.mb_size = hps['global_batch_size']
        self.clip_grad_param = hps['clip_grad_param']
        self.clip_grad_callback = {
            'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            'none': (lambda x: None)
        }[hps['clip_grad_type']]

    def _loss_step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

    def log(self, info, index, key):
        super().log(info, index, key)
        self.wandb_client.log({
            f'{key}_{dict_key}': val
            for dict_key, val in info.items()
        })


def get_wandb_client(hps):
    group_name = hps.get('group_name', 'default_gfn_mols')
    run_name = '-'.join([f'{key}={hps.get(key, None)}' for key in _HP_KEYS_TO_LOG_AS_RUN_NAME])

    return wandb.init(
        group=group_name,
        name=run_name,
        config=hps,
        dir='~/scratch/wandb'
    )


def main():
    """Example of how this model can be run outside of Determined"""
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
        'do_thompson_sampling_bootstrap': True,
        'thompson_sampling_bootstrap_prob': tune.uniform(0.1, 0.99),
        'reward_threshs': [0.875, .9375, 1.0, 1.125],
        'k_vals': [10, 100],
        'do_ts': True,
        'group_name': 'seh_frag_ts_tune',
        'learning_rate': tune.loguniform(1e-4, 1e-2),
        'Z_learning_rate': tune.loguniform(1e-4, 1e-2),
        'weight_decay': tune.loguniform(1e-8, 1e-4),
        'ensemble_size': tune.randint(5, 1000),
        'prior_weight': tune.uniform(1e-1, 50)
    }
    if os.path.exists(hps['log_dir']):
        if hps['overwrite_existing_exp']:
            shutil.rmtree(hps['log_dir'])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps['log_dir'])

    ray.init(
        num_cpus=10,
        num_gpus=torch.cuda.device_count(),
    )

    trainable  = tune.with_resources(SEHFragTrainer, {'cpu': 3.0, 'gpu': 1.0})
    tuner = tune.Tuner(
        trainable,
        param_space=hps,
        run_config=ray.air.config.RunConfig(
            local_dir='~/scratch/ray_results',
            name='ts_tuning_seh_frag',
            stop={'training_iteration': 2000},
            verbose=3,
            callbacks=[WandbLoggerCallback(
                project='gfn_exploration',
                group='ts_tuning_seh_frag',
                dir='~/scratch/wandb',
                log_config=True,
                settings=wandb.Settings(start_method='thread')
            )]
        ),
        tune_config=tune.TuneConfig(
            num_samples=-1,
            search_alg=OptunaSearch(
                metric='num_modes_found_thresh_1.0',
                mode='max',
                points_to_evaluate=[
                    {
                        'thompson_sampling_bootstrap_prob': 0.22294915559486975,
                        'learning_rate': 0.0001713234294140119,
                        'Z_learning_rate': 0.0037773883660782138,
                        'weight_decay': 0.0000018980069295545816,
                        'prior_weight': 45.41677490817762,
                        'ensemble_size': 308
                    },
                    {
                        'thompson_sampling_bootstrap_prob': 0.20981209575944224,
                        'learning_rate': 0.0012070125235745886,
                        'Z_learning_rate': 0.00015990127057048908,
                        'weight_decay': 0.0000546478846488304,
                        'prior_weight': 45.026691341420225,
                        'ensemble_size': 737
                    },
                    {
                        'thompson_sampling_bootstrap_prob': 0.21973241272230065,
                        'learning_rate': 0.0004667175129532691,
                        'Z_learning_rate': 0.0057056782791636475,
                        'weight_decay': 3.320175301882085e-7,
                        'prior_weight': 29.209968698933498,
                        'ensemble_size': 295
                    },
                ],
                evaluated_rewards=[0.0, 1.0, 0.0]
            ),
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                metric='num_modes_found_thresh_1.0',
                mode='max',
                grace_period=300,
                max_t=2000
            )
        )
    )

    tuner.fit()

    #wandb_client = get_wandb_client(hps)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #trial = SEHFragTrainer(hps, device, wandb_client)
    #trial.verbose = True
    #trial.run()


if __name__ == '__main__':
    main()
