import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import Generator


class TSPTWGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 100.0,
        max_tw_width: float = 100.0,
        hardness: str = "hard",
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.max_tw_width = max_tw_width
        self.hardness = hardness
        self.normalize = normalize

    def _generate(self, batch_size, return_permutation=False):
        # Sample locations
        locs = (
            torch.rand((*batch_size, self.num_loc + 1, 2)) * (self.max_loc - self.min_loc)
            + self.min_loc
        )

        # Sample Time windows
        if self.hardness == "easy":
            tw_early, tw_late = self.generate_random_time_windows(
                locs, 55 * (self.num_loc + 1), 0.5, 0.75
            )

        elif self.hardness == "medium":
            tw_early, tw_late = self.generate_random_time_windows(
                locs, 55 * (self.num_loc + 1), 0.1, 0.2
            )

        elif self.hardness == "hard":
            # Generate random permutations over the customer nodes
            randperm = torch.stack([torch.randperm(self.num_loc) + 1 for _ in range(*batch_size)])
            randperm = torch.cat(
                [torch.zeros((*batch_size, 1), dtype=torch.long), randperm], dim=-1
            )
            tw_early, tw_late = self.generate_time_windows_from_randomperm(
                locs, randperm, self.max_tw_width
            )

        if self.normalize:
            scaler = self.max_loc - self.min_loc
            locs, tw_early, tw_late = locs / scaler, tw_early / scaler, tw_late / scaler

        td = TensorDict(
            {
                "locs": locs,
                "time_windows": torch.stack([tw_early, tw_late], dim=-1),
                "service_time": torch.zeros((*batch_size, self.num_loc + 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

        if self.hardness == "hard" and return_permutation:
            td.set("randperm", randperm)
        return td

    def _get_depot_windows(self, locs, tw_early, tw_late):
        d_i0 = (locs[:, 1:] - locs[:, 0:1]).norm(dim=-1)
        depot_late = torch.max(tw_late + d_i0, dim=-1, keepdim=True).values
        tw_early = torch.cat([torch.zeros_like(depot_late), tw_early], dim=-1)
        tw_late = torch.cat([depot_late, tw_late], dim=-1)
        return tw_early, tw_late

    def generate_random_time_windows(self, locs, expected_distance, alpha, beta):
        batch_size, num_loc = locs.size(0), locs.size(1) - 1  # no depot

        node_tw_early = torch.randint(0, expected_distance, (batch_size, num_loc))

        epsilon = torch.rand(batch_size, num_loc) * (beta - alpha) + alpha
        tw_width = torch.round(epsilon * expected_distance)
        node_tw_late = node_tw_early + tw_width

        tw_early, tw_late = self._get_depot_windows(
            locs, node_tw_early.float(), node_tw_late.float()
        )

        return tw_early, tw_late

    def generate_time_windows_from_randomperm(self, locs, randperm, max_tw_width):
        # e_i = d_i - U[0, w/2], l_i = d_i + U[0, w/2]
        batch_size, num_loc = locs.size(0), locs.size(1) - 1  # no depot

        locs_perm = locs.gather(1, randperm.unsqueeze(-1).expand(-1, -1, 2))
        diff = locs_perm[:, :-1] - locs_perm[:, 1:]  # [B, N, 2]
        arc_length = torch.norm(diff, dim=-1)  # [B, N]
        arrival_time = torch.cumsum(arc_length, dim=-1)

        node_tw_early_perm = torch.clamp(
            arrival_time - torch.rand((batch_size, num_loc)) * max_tw_width / 2, min=0
        )
        node_tw_late_perm = arrival_time + torch.rand((batch_size, num_loc)) * max_tw_width / 2
        tw_early_perm, tw_late_perm = self._get_depot_windows(
            locs_perm, node_tw_early_perm, node_tw_late_perm
        )

        tw_early, tw_late = torch.zeros_like(tw_early_perm), torch.zeros_like(tw_late_perm)
        tw_early.scatter_(1, randperm, tw_early_perm)
        tw_late.scatter_(1, randperm, tw_late_perm)
        return tw_early, tw_late
