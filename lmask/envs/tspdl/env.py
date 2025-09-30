import torch
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from lmask.envs.tspdl.generator import TSPDLGenerator
from ..base import LazyMaskEnvBase


def get_action_mask(td, lookahead_step=2, round_error_epsilon=1e-5):
    if lookahead_step == 1:
        load_on_arrival = td["current_load"].unsqueeze(-1) + td["demand"]
        meets_draft_limit = load_on_arrival <= (td["draft_limit"] + round_error_epsilon)
        unvisited = ~td["visited"]
        can_visit_local = unvisited & meets_draft_limit

        any_draft_limit_viol = (unvisited & ~meets_draft_limit).any(dim=-1, keepdim=True)  # [B, 1]
        can_visit = torch.where(any_draft_limit_viol, torch.zeros_like(can_visit_local), can_visit_local)

    elif lookahead_step == 2:
        load_succ = td["current_load"].unsqueeze(-1) + td["demand"]  # [B, n+1]
        load_grandsucc = load_succ.unsqueeze(-1) + td["demand"].unsqueeze(1)  # [B, n+1, n+1]

        succ_feasible = load_succ <= (td["draft_limit"] + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = load_grandsucc <= (td["draft_limit"].unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]

        eye = torch.eye(td["locs"].size(1), dtype=torch.bool, device=td.device).unsqueeze(0)
        skip_mask = td["visited"].unsqueeze(1) | eye  # [B, n+1, n+1]
        grandsucc_check = (grandsucc_feasible | skip_mask).all(dim=-1)

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n+1]

    elif lookahead_step == 3:
        load_succ = td["current_load"].unsqueeze(-1) + td["demand"]  # [B, n+1]
        load_grandsucc = load_succ.unsqueeze(-1) + td["demand"].unsqueeze(1)  # [B, n+1, n+1]
        load_greatgrandsucc = load_grandsucc.unsqueeze(-1) + td["demand"].unsqueeze(1).unsqueeze(
            1
        )  # [B, n+1, n+1, n+1]

        succ_feasible = load_succ <= (td["draft_limit"] + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = load_grandsucc <= (td["draft_limit"].unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]
        greatgrandsucc_feasible = load_greatgrandsucc <= (
            td["draft_limit"].unsqueeze(1).unsqueeze(1) + round_error_epsilon
        )  # [B, n+1, n+1, n+1]

        # Create skip mask
        # - grand_skip_mask: [B, n+1, n+1] such that grand_skip_mask[b, i, j] = True if node j is visited in batch b or j = i
        # - grandsucc_skip_mask: [B, n+1, n+1, n+1] such that grandsucc_skip_mask[b, i, j, k] = True if node k is visited in batch b or k=i or k = j
        num_nodes = td["locs"].size(1)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=td.device)
        eye_ij = eye.unsqueeze(0)  # [1, n+1, n+1]
        eye_jk = eye.unsqueeze(0).unsqueeze(0)  # [1, 1, n+1, n+1]
        eye_ik = eye.unsqueeze(0).unsqueeze(2)  # [1, n+1, 1, n+1]

        assert eye_ij.shape == (1, num_nodes, num_nodes)
        assert eye_jk.shape == (1, 1, num_nodes, num_nodes)
        assert eye_ik.shape == (1, num_nodes, 1, num_nodes)

        grand_skip_mask = td["visited"].unsqueeze(1) | eye_ij  # [B, n+1, n+1]
        greatgrand_skip_mask = td["visited"].unsqueeze(1).unsqueeze(1) | eye_jk | eye_ik  # [B, n+1, n+1, n+1]

        # check for great grand successors and grand successors
        greatgrandsucc_check = (greatgrandsucc_feasible | greatgrand_skip_mask).all(dim=-1)  # [B, n+1, n+1]
        grandsucc_check = (grandsucc_feasible & greatgrandsucc_check & ~grand_skip_mask).any(dim=-1)  # [B, n+1]

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n+1]

    return can_visit


class TSPDLEnv(RL4COEnvBase):
    def __init__(self, generator=TSPDLGenerator, generator_params={}, **kwargs):
        self.lookahead_step = kwargs.pop("lookahead_step", 2)
        kwargs.pop("max_backtrack_steps", None)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td=None, batch_size=None):
        visited = torch.zeros((*batch_size, td["locs"].size(1)), dtype=torch.bool, device=td.device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "draft_limit": td["draft_limit"],
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=td.device),
                "current_load": torch.zeros(*batch_size, dtype=torch.float32, device=td.device),
                "draft_limit_violation": torch.zeros_like(visited, dtype=torch.float32),
                "visited": visited,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        can_visit = get_action_mask(
            td, lookahead_step=self.lookahead_step, round_error_epsilon=self.round_error_epsilon
        )  # [B, n+1]
        action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        return action_mask

    def _step(self, td):
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        current_node = td["action"]
        current_load = td["current_load"] + gather_by_index(td["demand"], current_node)
        current_draft_limit = gather_by_index(td["draft_limit"], current_node)
        td["draft_limit_violation"][batch_idx, current_node] = (current_load - current_draft_limit).clamp_(min=0.0)

        visited = td["visited"].scatter_(1, current_node.unsqueeze(1), 1)
        done = visited.sum(1) == visited.size(1)
        reward = torch.zeros_like(done, dtype=torch.float32)
        td.update(
            {
                "visited": visited,
                "current_node": current_node,
                "current_load": current_load,
                "reward": reward,
                "done": done,
            }
        )
        num_unvisited = (~td["visited"][0]).sum().item()
        action_mask = self.get_action_mask(td) if num_unvisited > 1 else ~visited
        td.set("action_mask", action_mask)
        return td

    def _get_reward(self, td, actions):

        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)

        draft_limit_viol = td["draft_limit_violation"]  # [B, n+1]
        total_constraint_violation = draft_limit_viol.sum(dim=1)  # [B]
        violated_node_count = (draft_limit_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

        return TensorDict(
            {
                "negative_length": -tour_length,
                "total_constraint_violation": total_constraint_violation,
                "violated_node_count": violated_node_count,
            },
            batch_size=td["locs"].size(0),
            device=td.device,
        )

    def rollout(self, td, policy, num_samples=1, decode_type="greedy", device="cuda", **kwargs):
        with torch.inference_mode():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                td = td.to(device)
                td = self.reset(td)
                td_aug = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td)
                num_samples = 0 if decode_type == "greedy" else num_samples

                out = policy(td_aug, self, decode_type=decode_type, num_samples=num_samples)
                actions = unbatchify(out["actions"], (8, num_samples))

                reward_td = unbatchify(out["reward"], (8, num_samples))
                reward, total_constraint_violation = (
                    reward_td["negative_length"],
                    reward_td["total_constraint_violation"],
                )
                sol_feas = total_constraint_violation < self.round_error_epsilon

        return TensorDict(
            {
                "actions": actions,
                "reward": reward,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size[0],
            device=td.device,
        )


class TSPDLLazyMaskEnv(LazyMaskEnvBase):
    def __init__(self, generator=TSPDLGenerator, generator_params={}, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator(**generator_params)

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_nodes = td["locs"].size(1)
        visited = torch.zeros((*batch_size, num_nodes), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "draft_limit": td["draft_limit"],
                "current_load": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "backtrack_steps": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "step_idx": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "load_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.float32, device=device),
                "node_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.int64, device=device),
                "refine_intensity_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.int64, device=device),
                "invalid": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "backtrack_budget_reached": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
            device=td.device,
        )
        if self.disable_backtracking:
            td_reset["backtrack_budget_reached"] = torch.ones_like(td_reset["backtrack_budget_reached"])
        action_mask = self.get_action_mask(td_reset)
        mask_stack = torch.zeros((*batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        mask_stack[:, 0] = action_mask
        td.update({"action_mask": action_mask, "mask_stack": mask_stack})
        return td_reset

    def get_action_mask(self, td):
        can_visit = get_action_mask(
            td, lookahead_step=self.lookahead_step, round_error_epsilon=self.round_error_epsilon
        )

        if self.phase == "test":
            action_mask = can_visit
        else:
            no_valid_action = ~can_visit.any(dim=-1, keepdim=True)
            budget_reached = td["backtrack_budget_reached"].unsqueeze(-1)  # [B, 1]
            no_valid_and_budget_reached = no_valid_action & budget_reached
            action_mask = torch.where(no_valid_and_budget_reached, ~td["visited"], can_visit)

        return action_mask

    def _get_reward(self, td, actions):
        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)
        return -tour_length

    def _get_violations(self, td, actions):
        draft_limit_tour = gather_by_index(td["draft_limit"], actions)
        draft_limit_violations = torch.clamp(td["load_stack"][:, 1:] - draft_limit_tour, min=0.0)
        return draft_limit_violations

    def _step(self, td):
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        new_node = td["action"]
        new_load = td["current_load"] + td["demand"][batch_idx, new_node]

        visited = td["visited"].scatter_(1, new_node[..., None], True)
        new_step_idx = td["step_idx"] + 1
        td["done"] = visited.all(dim=-1)

        td["reward"] = torch.zeros_like(td["done"], dtype=torch.float32)
        td["load_stack"][batch_idx, new_step_idx] = new_load
        td["node_stack"][batch_idx, new_step_idx] = new_node
        # Reset refine intensity for the new step
        td["refine_intensity_stack"][batch_idx, new_step_idx] = 0
        td.update(
            {
                "step_idx": new_step_idx,
                "visited": visited,
                "current_load": new_load,
                "current_node": new_node,
            }
        )

        action_mask = self.get_action_mask(td)
        td.set("action_mask", action_mask)
        td["mask_stack"][batch_idx, new_step_idx] = action_mask
        return td

    def backtrack(self, td):
        step_idx = td["step_idx"]
        new_step_idx = step_idx - 1

        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        deleted_node = td["node_stack"][batch_idx, step_idx]
        td["visited"].scatter_(1, deleted_node[..., None], False)

        td["backtrack_steps"] += 1
        td["backtrack_budget_reached"] = td["backtrack_steps"] >= self.max_backtrack_steps
        if self.phase == "test":
            td["done"] = td["backtrack_budget_reached"]

        td["mask_stack"][batch_idx, new_step_idx, deleted_node] = False
        td["refine_intensity_stack"][batch_idx, new_step_idx] = (
            td["refine_intensity_stack"][batch_idx, new_step_idx] + 1
        )
        td.update(
            {
                "step_idx": new_step_idx,
                "action_mask": td["mask_stack"][batch_idx, new_step_idx],
                "current_load": td["load_stack"][batch_idx, new_step_idx],
                "current_node": td["node_stack"][batch_idx, new_step_idx],
            }
        )
        return td
