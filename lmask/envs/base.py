from abc import abstractmethod
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.decoding import get_log_likelihood
from rl4co.utils.ops import batchify, unbatchify, calculate_entropy
from rl4co.data.transforms import StateAugmentation
from ..utils.ops import get_action_from_logits_and_mask, slice_cache


class LazyMaskEnvBase(RL4COEnvBase):
    """
    Base class for lazy mask environments that support backtracking during decoding.

    This class provides common functionality for environments that use lazy mask
    with backtracking, such as TSPTW and TSPDL. Problem-specific logic should be
    implemented in subclasses.
    """

    def __init__(self, **kwargs):
        self.get_reward_by_distance = kwargs.pop("get_reward_by_distance", False)
        self.disable_backtracking = kwargs.pop("disable_backtracking", False)
        self.phase = kwargs.pop("phase", "test")
        self.max_backtrack_steps = kwargs.pop("max_backtrack_steps", 300)
        self.lookahead_step = kwargs.pop("lookahead_step", 2)
        super().__init__(check_solution=False, **kwargs)
        self.round_error_epsilon = 1e-5

    @abstractmethod
    def get_action_mask(self, td):
        """Get action mask based on problem-specific constraints."""
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, td, actions):
        """Calculate reward based on problem-specific objectives."""
        raise NotImplementedError

    @abstractmethod
    def _get_violations(self, td):
        """Calculate constraint violations based on problem-specific constraints."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, td):
        """Update environment state after taking an action."""
        raise NotImplementedError

    @abstractmethod
    def backtrack(self, td):
        """Backtrack to previous state."""
        raise NotImplementedError

    def _decode_base(self, td, hidden, policy, num_samples, decode_type, logits_fn):
        """A wrapper to handle the decoding process."""
        if self.phase == "test":
            with Progress(
                TextColumn(f"[bold cyan]⚡ {'Multi' if num_samples > 1 else 'Fast'} Decode", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TextColumn("[green]✓{task.fields[done]:>4}"),
                TextColumn("[yellow]→{task.fields[forward]:>4}"),
                TextColumn("[red]←{task.fields[backtrack]:>4}"),
                TextColumn("[magenta]Invalid{task.fields[invalid]:>4}"),
                TimeElapsedColumn(),
                console=Console(),
                refresh_per_second=10,
            ) as progress:
                task = progress.add_task("decode", total=td["locs"].size(0), done=0, forward=0, backtrack=0, invalid=0)
                td, logprobs_stack = self._decode_core(
                    td, hidden, policy, num_samples, decode_type, logits_fn, progress, task
                )
        else:
            td, logprobs_stack = self._decode_core(td, hidden, policy, num_samples, decode_type, logits_fn)
        return td, logprobs_stack

    def _decode_core(self, td, hidden, policy, num_samples, decode_type, logits_fn, progress=None, task=None):
        """ "Core LazyMask decoding loop."""
        if self.phase == "train":
            batch_size, num_nodes, _ = td["locs"].size()
            logprobs_stack = torch.zeros(
                (batch_size, num_nodes, num_nodes), dtype=torch.float32, device=td.device, requires_grad=True
            )

        while not td["done"].all():
            active = ~td["done"]
            has_feas = td["action_mask"].any(dim=-1)
            at_depot = td["step_idx"] == 0
            # print(f"current partial solution: {td['node_stack'][0, :td['step_idx'][0]+1]}")

            step_foward_mask = active & has_feas
            backtrack_mask = active & (~has_feas) & (~at_depot)
            invalid_mask = active & (~has_feas) & (at_depot)

            if step_foward_mask.any():
                step_forward_idx = step_foward_mask.nonzero(as_tuple=True)[0]
                logits, mask = logits_fn(td, hidden, step_forward_idx, num_samples)

                logprobs, td["action"][step_forward_idx] = get_action_from_logits_and_mask(
                    logits, mask, decode_type=decode_type, return_logprobs=True
                )
                td[step_forward_idx] = self._step(td[step_forward_idx])

                if self.phase == "train":
                    step_idx = td["step_idx"][step_forward_idx]
                    logprobs_stack = logprobs_stack.index_put(
                        indices=(step_forward_idx, step_idx), values=logprobs, accumulate=False
                    )

            if backtrack_mask.any():
                backtrack_idx = backtrack_mask.nonzero(as_tuple=True)[0]
                td[backtrack_idx] = self.backtrack(td[backtrack_idx])

            if invalid_mask.any():
                print("Invalid instances detected, marking as done.")
                invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
                td["done"][invalid_idx] = True
                td["invalid"][invalid_idx] = True

            if progress is not None and task is not None:
                progress.update(
                    task,
                    completed=td["done"].sum().item(),
                    done=td["done"].sum().item(),
                    forward=step_foward_mask.sum().item(),
                    backtrack=backtrack_mask.sum().item(),
                    invalid=invalid_mask.sum().item(),
                )

        return (td, logprobs_stack) if self.phase == "train" else (td, None)

    def decode_multi(self, td, hidden, policy, num_samples=1, decode_type="sampling"):
        def get_logits_multi(td, hidden, step_forward_idx, num_samples):
            """Multi decode: compute logits for all instances at once."""
            logits, mask = policy.decoder(td, hidden, num_samples)
            return logits[step_forward_idx], mask[step_forward_idx]

        return self._decode_base(td, hidden, policy, num_samples, decode_type, get_logits_multi)

    def decode_fast(self, td, hidden, policy, num_samples=1, decode_type="greedy"):
        assert num_samples <= 1, "This fast implementation of lazymask decoding only supports num_samples=1"

        def get_logits_fast(td, hidden, step_forward_idx, num_samples):
            """Fast decode: compute logits only for stepforward instances."""
            td_step = td[step_forward_idx]
            cache_step = slice_cache(hidden, step_forward_idx)
            return policy.decoder(td_step, cache_step)

        return self._decode_base(td, hidden, policy, num_samples, decode_type, get_logits_fast)

    def rollout(
        self, td, policy, num_samples=1, num_augment=8, decode_type="greedy", device="cuda", return_td=False, **kwargs
    ):
        """A wrapper to handle the existence of context manager at inference phase."""
        if self.phase == "train":
            return self._rollout_core(td, policy, num_samples, num_augment, decode_type, device, return_td, **kwargs)
        else:
            with torch.inference_mode():
                with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                    return self._rollout_core(
                        td, policy, num_samples, num_augment, decode_type, device, return_td, **kwargs
                    )

    def _rollout_core(
        self, td, policy, num_samples=1, num_augment=8, decode_type="greedy", device="cuda", return_td=False, **kwargs
    ):
        td = td.to(device)
        td = self.reset(td)

        if self.phase == "train":
            num_augment = 0
            decode_type = "sampling"
        num_samples = 0 if decode_type == "greedy" else num_samples
        if num_augment > 1:
            td = StateAugmentation(num_augment=num_augment, augment_fn="dihedral8")(td)

        # RL4COEnvBase internally modify td["done"] to 2-dim tensor, so we need to squeeze it back to 1-dim
        td["done"].squeeze_(-1)
        hidden, _ = policy.encoder(td)
        _, _, hidden = policy.decoder.pre_decoder_hook(td, self, hidden, num_samples)

        if num_samples <= 1:
            td, logprobs_stack = self.decode_fast(td, hidden, policy, num_samples, decode_type)
        else:
            td = batchify(td, num_samples)
            td, logprobs_stack = self.decode_multi(td, hidden, policy, num_samples, decode_type)

        out = self.get_rollout_result(td)
        if self.phase == "train":
            out["log_likelihood"] = get_log_likelihood(logprobs_stack, td["node_stack"])
            out["entropy"] = calculate_entropy(logprobs_stack)
        out = unbatchify(out, (num_augment, num_samples))

        return (out, td) if return_td else out

    def get_rollout_result(self, td):
        actions = td["node_stack"][:, 1:]
        negative_length = self._get_reward(td, actions)
        sol_feas = td["step_idx"] == (td["locs"].size(1) - 1)
        out = TensorDict(
            {
                "reward": negative_length,
                "actions": actions,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size,
            device=td.device,
        )

        if self.phase != "test":
            node_constraint_violations = self._get_violations(td, actions)
            total_constraint_violation = node_constraint_violations.sum(dim=-1)
            violated_node_count = (node_constraint_violations > self.round_error_epsilon).float().sum(dim=-1)
            sol_feas = total_constraint_violation < self.round_error_epsilon
            out.update(
                {
                    "total_constraint_violation": total_constraint_violation,
                    "violated_node_count": violated_node_count,
                    "sol_feas": sol_feas,
                }
            )
        return out

    def check_permutation(self, actions):
        """
        Check if the actions form a valid permutation of node indices.
        """
        sorted_actions = actions.sort(dim=-1).values
        is_permutation = (sorted_actions == torch.arange(actions.size(-1), device=actions.device)).all(dim=-1)
        return is_permutation
