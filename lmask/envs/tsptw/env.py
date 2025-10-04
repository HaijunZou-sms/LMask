import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import get_distance, gather_by_index, unbatchify
from rl4co.data.transforms import StateAugmentation
from .generator import TSPTWGenerator
from ..base import LazyMaskEnvBase
from ...utils.ops import get_tour_length


def get_action_mask(td, lookahead_step=2, round_error_epsilon=1e-5):
    """
    Initialize the overestimation set for the current partial route.
    1. For TSL (`lookahead_step = 1`), there are only two cases:
       - The current node can reach all the unvisited nodes within their time windows.
       - Otherwise, the groud-truth potential set is empty, and we must backtrack.
    2. For SSL (`lookahead_step = 2`), a successor node is available if:
       - The arrival time at it from the current node is within its time window.
       - The successor node can reach all its successors within their time windows.

    Note: For efficiency, feasibility checks are computed for all nodes initially,
    then filtered using skip_mask to exclude nodes that don't require validation.
    This approach also enables handling heterogeneous instances with varying numbers
    of remaining unvisited nodes across different instances in the batch.
    """
    if lookahead_step == 1:
        curr_node = td["current_node"]

        d_ij = get_distance(gather_by_index(td["locs"], curr_node)[:, None, :], td["locs"])  # [B, n+1]
        arrival_time = td["current_time"][:, None] + d_ij
        can_reach_in_time = arrival_time <= (td["time_windows"][..., 1] + round_error_epsilon)  # [B, n+1]

        all_succ_feasible = (td["visited"] | can_reach_in_time).all(dim=-1, keepdim=True)  # [B, 1]
        unvisited = ~td["visited"]
        can_visit = torch.where(all_succ_feasible, unvisited, torch.zeros_like(unvisited))

    elif lookahead_step == 2:
        batch_size, num_nodes, _ = td["locs"].shape
        batch_idx = torch.arange(batch_size, device=td.device)  # [B, ]

        tw_early, tw_late = td["time_windows"].unbind(-1)

        dur_cur_succ = td["duration_matrix"][batch_idx, td["current_node"], :]

        service_start_time_succ = torch.max(td["current_time"].unsqueeze(1) + dur_cur_succ, tw_early)
        service_start_time_grandsucc = torch.max(service_start_time_succ.unsqueeze(-1) + td["duration_matrix"], tw_early.unsqueeze(1))

        succ_feasible = service_start_time_succ <= (tw_late + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = service_start_time_grandsucc <= ( tw_late.unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]

        eye = torch.eye(num_nodes, dtype=torch.bool, device=td.device).unsqueeze(0)
        visited_step1_per_choice = td["visited"].unsqueeze(1) | eye  # [B, n+1, n+1]
        grandsucc_check = (grandsucc_feasible | visited_step1_per_choice).all(dim=-1)  # [B, n+1]

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n+1]

    elif lookahead_step == 3:
        batch_size, num_nodes, _ = td["locs"].shape
        batch_idx = torch.arange(batch_size, device=td.device)  # [B, ]

        tw_early, tw_late = td["time_windows"].unbind(-1)

        dur_cur_succ = td["duration_matrix"][batch_idx, td["current_node"], :]
        service_start_time_succ = torch.max(td["current_time"].unsqueeze(1) + dur_cur_succ, tw_early)

        service_start_time_grandsucc = torch.max(service_start_time_succ.unsqueeze(-1) + td["duration_matrix"], tw_early.unsqueeze(1))  # [B, n+1, n+1]
        service_start_time_greatgrandsucc = torch.max(service_start_time_grandsucc.unsqueeze(-1) + td["duration_matrix"].unsqueeze(1), tw_early.unsqueeze(1).unsqueeze(1))  # [B, n+1, n+1, n+1]

        succ_feasible = service_start_time_succ <= (tw_late + round_error_epsilon)  # [B, n+1]
        grandsucc_feasible = service_start_time_grandsucc <= (tw_late.unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1]
        greatgrandsucc_feasible = service_start_time_greatgrandsucc <= (tw_late.unsqueeze(1).unsqueeze(1) + round_error_epsilon)  # [B, n+1, n+1, n+1]


        # visited_step1_per_choice [B, n+1, n+1]: for each step-1 candidate, the visited set after taking that single step.
        # visited_step2_per_choice_pair [B, n+1, n+1, n+1]: for each (step-1, step-2) candidate pair, the visited set after taking two steps in sequence.
        eye = torch.eye(num_nodes, dtype=torch.bool, device=td.device)
        eye_ij = eye.unsqueeze(0)  # [1, n+1, n+1]
        eye_jk = eye.unsqueeze(0).unsqueeze(0)  # [1, 1, n+1, n+1]
        eye_ik = eye.unsqueeze(0).unsqueeze(2)  # [1, n+1, 1, n+1]

        visited_step1_per_choice = td["visited"].unsqueeze(1) | eye_ij  # [B, n+1, n+1]
        visited_step2_per_choice_pair = td["visited"].unsqueeze(1).unsqueeze(1) | eye_jk | eye_ik  # [B, n+1, n+1, n+1]

        grandsucc_check = (grandsucc_feasible | visited_step1_per_choice).all(dim=-1)  # [B, n+1]

        all_greatgrand_feasible = (greatgrandsucc_feasible | visited_step2_per_choice_pair).all(dim=-1)  # [B, n+1, n+1]
        greatgrand_check = (all_greatgrand_feasible & ~visited_step1_per_choice).any(dim=-1)  # [B, n+1]
        has_no_unvisited_grandchild = visited_step1_per_choice.all(dim=-1)
        greatgrand_check = greatgrand_check | has_no_unvisited_grandchild

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check & greatgrand_check  # [B, n+1]

    return can_visit


class TSPTWEnv(RL4COEnvBase):
    def __init__(self, generator=TSPTWGenerator, generator_params={}, **kwargs):
        self.lookahead_step = kwargs.pop("lookahead_step", 2)
        kwargs.pop("max_backtrack_steps", None)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td=None, batch_size=None):
        device = td.device
        num_nodes = td["locs"].size(1)

        visited = torch.zeros((*batch_size, num_nodes), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "service_start_time_cache": torch.zeros_like(visited),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "time_window_violation": torch.zeros((*batch_size, num_nodes), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.lookahead_step >= 2:
            td_reset["distance_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
            td_reset["duration_matrix"] = td_reset["distance_matrix"] + td["service_time"][:, :, None]
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        if self.lookahead_step == 1:
            return ~td["visited"]
        elif self.lookahead_step >= 2:
            unvisited = ~td["visited"]
            can_visit = get_action_mask(
                td, lookahead_step=self.lookahead_step, round_error_epsilon=self.round_error_epsilon
            )
            action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        return action_mask

    def _step(self, td):
        """
        update the state of the environment, including
        current_node, current_time, time_window_violation, visited and action_mask
        """
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        curr_node, new_node = td["current_node"], td["action"]
        curr_loc, new_loc = (
            td["locs"][batch_idx, curr_node],
            td["locs"][batch_idx, new_node],
        )  # [B, 2]

        travel_time = get_distance(curr_loc, new_loc)  # [B,]

        arrival_time = td["current_time"] + travel_time
        tw_early_new, tw_late_new = (td["time_windows"][batch_idx, new_node]).unbind(-1)
        service_time = td["service_time"][batch_idx, new_node]
        new_time = torch.max(arrival_time, tw_early_new) + service_time
        td["time_window_violation"][batch_idx, new_node] = torch.clamp(arrival_time - tw_late_new, min=0.0)

        visited = td["visited"].scatter_(1, new_node[..., None], True)
        done = visited.sum(dim=-1) == visited.size(-1)
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "current_time": new_time,
                "current_node": new_node,
                "visited": visited,
                "done": done,
                "reward": reward,
            }
        )

        num_unvisited = (~td["visited"][0]).sum().item()
        action_mask = self.get_action_mask(td) if num_unvisited > 1 else ~visited
        td.set("action_mask", action_mask)
        return td

    def _get_reward(self, td, actions):
        tour_length = get_tour_length(td, actions)

        tw_viol = td["time_window_violation"]  # [B, n+1]
        total_constraint_violation = tw_viol.sum(dim=1)  # [B]
        violated_node_count = (tw_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

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
                num_samples = 0 if decode_type == "greedy" else num_samples
                td = td.to(device)
                td = self.reset(td)
                td_aug = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td)

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


class TSPTWLazyMaskEnv(LazyMaskEnvBase):
    def __init__(self, generator=TSPTWGenerator, generator_params={}, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator(**generator_params)

    def _reset(self, td=None, batch_size=None):
        """
        Reset the environment state for lazy mask backtracking algorithm.

        The algorithm maintains three core stacks (time_stack, node_stack, mask_stack) to enable
        backtracking during tour construction. Since different instances in a batch may have
        different step_idx values, these stacks are implemented as tensors indexed by step_idx
        rather than traditional list-based stacks.

        Key state variables:
        - time_stack: [B, n+1] - Stores the current_time at each step of the tour construction.
                    time_stack[:, i] contains the time when we finish visiting the i-th node in the tour.
        - node_stack: [B, n + 1] - Stores the visited nodes at each step of the tour construction.
                    node_stack[:, i] contains the node ID of the i-th visited node in the tour.
        - mask_stack: [B, n + 1, n + 1] - Stores the action_mask at each step.
                    mask_stack[:, i] contains the valid action mask when we are at step i.
                    During backtracking, previously failed nodes are excluded from the mask.
        - invalid: [B] - Boolean flag indicating whether an instance is confirmed as infeasible.
                        Set to True when backtracking reaches the depot (step_idx=0) and no valid actions exist.

        Also note that the compute mode "donot_use_mm_for_euclid_dist" for the distance matrix is necessary here to  to ensure precision.
        """
        device = td.device
        num_nodes = td["locs"].size(1)  # Includes depot + customers
        visited = torch.zeros((*batch_size, num_nodes), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "backtrack_steps": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "step_idx": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "time_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.float32, device=device),
                "node_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.int64, device=device),
                "refine_intensity_stack": torch.zeros((*batch_size, num_nodes), dtype=torch.int64, device=device),
                "invalid": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "backtrack_budget_reached": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "done": torch.zeros(*batch_size, dtype=torch.bool, device=device),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
            device=device,
        )

        if self.lookahead_step >= 2:
            if "duration_matrix" in td.keys():
                td_reset["duration_matrix"] = td["duration_matrix"]
                td_reset["distance_matrix"] = td["distance_matrix"]
            else:
                td_reset["distance_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
                td_reset["duration_matrix"] = td_reset["distance_matrix"]

        if self.disable_backtracking:
            td_reset["backtrack_budget_reached"] = torch.ones_like(td_reset["backtrack_budget_reached"])

        action_mask = self.get_action_mask(td_reset)
        mask_stack = torch.zeros((*batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        mask_stack[:, 0] = action_mask
        td_reset.update({"action_mask": action_mask, "mask_stack": mask_stack})
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
        if self.get_reward_by_distance:
            batch_size = actions.shape[0]
            batch_idx = torch.arange(batch_size, device=actions.device).unsqueeze(-1)
            tours = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=actions.device), actions], dim=-1)
            shifted_tours = torch.roll(tours, -1)
            arc_lengths = td["distance_matrix"][batch_idx, tours, shifted_tours]
            tour_length = arc_lengths.sum(dim=-1)
        else:
            ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
            diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
            tour_length = diff.norm(dim=-1).sum(-1)
        return -tour_length

    def _get_violations(self, td, actions):
        tw_late = td["time_windows"][..., 1]
        tw_late_tour = gather_by_index(tw_late, actions)
        tw_viol = torch.clamp(td["time_stack"][:, 1:] - tw_late_tour, min=0.0)  # [B, n]
        return tw_viol

    def _step(self, td):

        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        curr_node, new_node = td["current_node"], td["action"]
        if self.get_reward_by_distance:
            travel_time = td["distance_matrix"][batch_idx, curr_node, new_node]
        else:
            curr_loc, new_loc = gather_by_index(td["locs"], curr_node), gather_by_index(td["locs"], new_node)
            travel_time = get_distance(curr_loc, new_loc)

        arrival_time = td["current_time"] + travel_time
        tw_early_new = gather_by_index(td["time_windows"][..., 0], new_node)
        service_time = gather_by_index(td["service_time"], new_node)
        new_time = torch.max(arrival_time, tw_early_new) + service_time

        visited = td["visited"].scatter_(1, new_node[..., None], True)

        new_step_idx = td["step_idx"] + 1
        td["done"] = visited.all(dim=-1)

        td["reward"] = torch.zeros_like(td["done"], dtype=torch.float32)
        td["time_stack"][batch_idx, new_step_idx] = new_time
        td["node_stack"][batch_idx, new_step_idx] = new_node
        # Reset refine intensity for the new step
        td["refine_intensity_stack"][batch_idx, new_step_idx] = 0
        td.update(
            {
                "step_idx": new_step_idx,
                "visited": visited,
                "current_time": new_time,
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

        td["mask_stack"][batch_idx, new_step_idx, deleted_node] = False
        td["refine_intensity_stack"][batch_idx, new_step_idx] = (
            td["refine_intensity_stack"][batch_idx, new_step_idx] + 1
        )

        if self.phase == "test":
            td["done"] = td["backtrack_budget_reached"]

        td.update(
            {
                "step_idx": new_step_idx,
                "action_mask": td["mask_stack"][batch_idx, new_step_idx],
                "current_time": td["time_stack"][batch_idx, new_step_idx],
                "current_node": td["node_stack"][batch_idx, new_step_idx],
            }
        )
        return td
