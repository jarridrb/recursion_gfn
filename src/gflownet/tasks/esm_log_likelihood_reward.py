from esm_reward.lm_design import Designer
from dreamfold.utils.vocabs import AMINO_ACID_VOCAB
from pathlib import Path
from typing import Dict, Tuple
from copy import deepcopy
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd
import torch

AMINO_ACID_VOCAB = list('ILVAGMFYWEDQNHCRKSTP')
_vocab = deepcopy(AMINO_ACID_VOCAB)
_vocab.remove('C')

_REPO_IDX_TO_CHAR = {idx: char for idx, char in enumerate(_vocab)}
_SUPPRESS_AAS = {'C'}

def _expand_states(states, padding_token, eos_token):
    actions, seqs = [], []
    arange_idx = torch.arange(len(states))
    for idx in range(states.shape[1] - 1, -1, -1):
        actions.append(states[:, idx].clone())

        if idx < states.shape[1] - 1:
            states[:, idx + 1] = padding_token

        states[:, idx] = eos_token
        seqs.append(states.clone())

    seqs = torch.stack(tuple(reversed(seqs)), dim=1)
    actions = torch.stack(tuple(reversed(actions)), dim=1)

    return seqs, actions

class ESMRewardModelWrapper(Designer):
    def __init__(self):
        self.allowed_AA = ''.join(
            AA
            for AA in self.standard_AA
            if not AA in _SUPPRESS_AAS
        )

        self.device = get_device()
        self._init_models()

class ESMLogLikelihoodTask(GFNTask):
    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator
    ):
        self.temperature_conditional = TemperatureConditional(cfg, rng)

        self.esm_reward_calculator = ESMRewardModelWrapper()

        self.language_model_energy_term_weight = cfg.task.language_model_energy_term_weight

        self.ngram_energy_term_weight = cfg.task.ngram_energy_term_weight
        self.ngram_orders = cfg.task.ngram_orders

        self.all_esm_toks = self.esm_reward_calculator.vocab.all_toks
        self.esm_vocab_char_to_idx = {
            char: idx
            for idx, char in enumerate(self.all_esm_toks)
            if char in self.esm_reward_calculator.allowed_AA
        }

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, objs: List[str]) -> Tuple[FlatRewards, Tensor]:
        log_rewards = self.get_log_reward(objs)

        return FlatRewards(log_rewards[:, None]), torch.ones(len(objs), dtype=torch.bool)

    def _encode(
        self,
        sequences: 'TensorType["batch_size", "seq_len", int]'
    ) -> 'TensorType["batch_size", "seq_len", "num_aa_types", int]':
        def convert(token):
            return self.esm_vocab_char_to_idx[token]

        # A token could be invalid, specifically if the token is an
        # end-of-sentence or padding token
        def is_valid(token):
            return token in self.esm_vocab_char_to_idx

        big_list = [
            [convert(tkn) for tkn in seq if is_valid(tkn)]
            for seq in sequences
        ]

        int_esm_encoded_seqs = torch.tensor([
            [convert(tkn) for tkn in seq if is_valid(tkn)]
            for seq in sequences
        ]).to(device=get_device())

        return F.one_hot(int_esm_encoded_seqs, len(self.all_esm_toks)).float()

    def get_log_rewards(
        self,
        sequences: List[str]
    ) -> TensorType["batch_size", float]:
        rewards, _ = self.esm_reward_calculator.calc_total_loss(
            x=self._encode(sequences),
            mask=None,
            LM_w=self.language_model_energy_term_weight,
            struct_w=False,
            ngram_w=self.ngram_energy_term_weight,
            ngram_orders=self.ngram_orders
        )

        return -rewards

class ESMLogLikelihoodTrainer(StandardOnlineTrainer):
    task: ESMLogLikelihoodTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 10
        cfg.algo.max_len = 10
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.task.language_model_energy_term_weight = 1.0
        cfg.task.ngram_energy_term_weight = 0.5
        cfg.task.ngram_orders = [1, 2, 3]

    def setup_model(self):
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
        )

    def setup_task(self):
        self.task = ESMLogLikelihoodTask(
            ["aa", "bb", "cc"],
            cfg=self.cfg,
            rng=self.rng,
        )

    def setup_env_context(self):
        self.env = SeqBuildingEnv(None)
        self.ctx = AutoregressiveSeqBuildingContext(
            "abc",
            self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        # If the algo implements it, avoid giving, ["A", "AB", "ABC", ...] as a sequence of inputs, and instead give
        # "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using a transformer with
        # causal self-attention.
        self.algo.model_is_autoregressive = True


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run_toy_seq",
        "device": "cuda",
        "overwrite_existing_exp": True,
        "num_training_steps": 2_000,
        "checkpoint_every": 200,
        "num_workers": 4,
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [2.0],
                "num_thermometer_dim": 1,
            }
        },
        "algo": {"train_random_action_prob": 0.05},
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = ToySeqTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
