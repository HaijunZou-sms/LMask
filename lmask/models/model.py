from typing import Any, Union
import torch.nn as nn
from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models import REINFORCE
from rl4co.utils.ops import unbatchify
from lightning.pytorch.utilities import grad_norm

from lmask.utils.metric_utils import get_filtered_max


class LMaskPenaltyModel(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        policy_kwargs: dict = {},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_samples: int = None,
        rho_c: float = 1.0,
        rho_n: float = 1.0,
        entropy_coef: float = 0.0,
        round_eps: float = 1e-5,
        penalty_type: str = "combined_penalty",
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=["env", "policy"])
        assert baseline == "shared", "Only shared baseline is supported for LMaskPenaltyModel"
        super(LMaskPenaltyModel, self).__init__(env, policy, baseline, **kwargs)
        self.num_samples = num_samples
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None
        self.rho_c, self.rho_n = rho_c, rho_n
        self.entropy_coef = entropy_coef
        self.round_eps = round_eps
        self.penalty_type = penalty_type

    def shared_step(self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None):
        td = self.env.reset(batch)
        n_aug, n_sample = self.num_augment, self.num_samples
        n_sample = td["locs"].size(-2) if n_sample is None else n_sample

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            n_sample = 1
            td = self.augment(td)

        # Evaluate policy
        
        out = self.policy(td, self.env, phase=phase, num_samples=n_sample, return_entropy=True)

        # Unbatchify reward to [batch_size, num_samples](during training phase) or [batch_size, num_augment, num_samples].
        reward_td = unbatchify(out["reward"], (n_aug, n_sample))
        reward = reward_td["negative_length"]
        total_constraint_violation = reward_td["total_constraint_violation"]
        violated_node_count = reward_td["violated_node_count"]

        if self.penalty_type == "combined_penalty":
            penalized_reward = reward - self.rho_c * total_constraint_violation - self.rho_n * violated_node_count  # [B, S]
        elif self.penalty_type == "constraint_viol_penalty":
            penalized_reward = reward - self.rho_c * total_constraint_violation
        elif self.penalty_type == "viol_node_count_penalty":
            penalized_reward = reward - self.rho_n * violated_node_count

        sol_feas = total_constraint_violation < self.round_eps  # [B, S] or [B, A, S]
        ins_feas = sol_feas.any(dim=tuple(range(1, sol_feas.dim())))  # [B]
        out.update(
            {
                "reward": reward.mean().item(),
                "violated_node_count": reward_td["violated_node_count"],
                "total_constraint_violation": reward_td["total_constraint_violation"],
                "penalized_reward": penalized_reward,
                "ins_feas_rate": ins_feas.float().mean().item() * 100,
                "sol_feas_rate": sol_feas.float().mean().item() * 100,
            }
        )

        # Training phase
        if phase == "train":
            assert n_sample > 1, "num_samples must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], n_sample)
            entropy = unbatchify(out["entropy"], n_sample)
            advantage = penalized_reward - penalized_reward.mean(dim=-1, keepdim=True)
            loss = -(log_likelihood * advantage).mean() - self.entropy_coef * entropy.mean()
            out.update({"loss": loss})
            out.update({"max_reward": get_filtered_max(reward, sol_feas)})

        # Validation and Test phase
        else:
            if n_sample > 1:
                # Calculate max_reward (no augmentation or first augmentation slice)
                no_aug_reward = reward if n_aug == 0 else reward[:, 0]
                no_aug_sol_feas = sol_feas if n_aug == 0 else sol_feas[:, 0]
                out.update({"max_reward": get_filtered_max(no_aug_reward, no_aug_sol_feas)})

                # Calculate max_aug_reward (using all augmentations)
            if n_aug > 1:
                out.update({"max_aug_reward": get_filtered_max(reward, sol_feas)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
    #         norms = grad_norm(self.policy, norm_type=2)
    #         self.log_dict(norms)


class LMaskBacktrackingPenaltyModel(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        policy_kwargs: dict = {},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_samples: int = None,
        rho_c: float = 1.0,
        rho_n: float = 1.0,
        entropy_coef: float = 0.0,
        round_eps: float = 1e-5,
        penalty_type: str = "combined_penalty",
        switch_mode: bool = False,
        switch_epoch: int = 100,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        self.save_hyperparameters(ignore=["env", "policy"])
        assert baseline == "shared", "Only shared baseline is supported for LMaskPenaltyModel"
        super(LMaskBacktrackingPenaltyModel, self).__init__(env, policy, baseline, **kwargs)
        self.num_samples = num_samples
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None
        self.rho_c, self.rho_n = rho_c, rho_n
        self.entropy_coef = entropy_coef
        self.round_eps = round_eps
        self.penalty_type = penalty_type
        self.switch_mode = switch_mode
        self.switch_epoch = switch_epoch

    def shared_step(self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None):
        self.env.phase = phase
        td = self.env.reset(batch)

        if phase == "train":
            n_aug = 0
            n_sample = td["locs"].size(-2) if self.num_samples is None else self.num_samples
            decode_type = "sampling"
        else:
            n_aug, n_sample = self.num_augment, 0
            decode_type = "greedy"
        
        # out tensordict has been internally unbatchified to [B, A, S] or [B, S]
        # To record scalar values, we need to convert it to a dict
        out = self.env.rollout(td, self.policy, num_samples=n_sample, num_augment=n_aug, decode_type=decode_type, device=td.device) 
        out = out.to_dict()  

        reward = out["reward"]
        total_constraint_violation = out["total_constraint_violation"]
        violated_node_count = out["violated_node_count"]

        if self.penalty_type == "combined_penalty":
            penalized_reward = reward - self.rho_c * total_constraint_violation - self.rho_n * violated_node_count  # [B, S]
        elif self.penalty_type == "constraint_viol_penalty":
            penalized_reward = reward - self.rho_c * total_constraint_violation
        elif self.penalty_type == "viol_node_count_penalty":
            penalized_reward = reward - self.rho_n * violated_node_count

        sol_feas = total_constraint_violation < self.round_eps  # [B, S] or [B, A, S]
        ins_feas = sol_feas.any(dim=tuple(range(1, sol_feas.dim())))  # [B]
        
        out.update(
            {
                "reward": reward,
                "penalized_reward": penalized_reward,
                "ins_feas_rate": ins_feas.float() * 100,
                "sol_feas_rate": sol_feas.float() * 100,
            }
        )

        # Training phase
        if phase == "train":
            assert n_sample > 1, "num_samples must be > 1 during training"
            log_likelihood, entropy = out["log_likelihood"], out["entropy"]
            advantage = penalized_reward - penalized_reward.mean(dim=-1, keepdim=True)
            loss = -(log_likelihood * advantage).mean() - self.entropy_coef * entropy.mean()
            out.update({"loss": loss})
            out.update({"max_reward": get_filtered_max(reward, sol_feas)})

        # Validation and Test phase
        else:
            out.update({"max_aug_reward": get_filtered_max(reward, sol_feas)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
    
    # def on_train_epoch_end(self):
    #     if self.switch_mode and self.current_epoch == self.switch_epoch:
    #         self.env.disable_backtracking = False
    #         self.env.max_backtracking_steps = 100
    #         self.rho_c = 0.5
    #         self.rho_n = 0.5
            
