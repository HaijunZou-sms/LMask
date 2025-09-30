from rl4co.envs.common.base import RL4COEnvBase
from lmask.envs.tsptw.env import TSPTWEnv, TSPTWLazyMaskEnv
from lmask.envs.tspdl.env import TSPDLEnv, TSPDLLazyMaskEnv

ENV_REGISTRY = {
    "tsptw": TSPTWEnv,
    "tsptw-lazymask": TSPTWLazyMaskEnv,
    "tspdl": TSPDLEnv,
    "tspdl-lazymask": TSPDLLazyMaskEnv,
}


def get_env(env_name: str, generator_params={}, **kwargs) -> RL4COEnvBase:
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(f"Unknown environment {env_name}")
    return env_cls(generator_params=generator_params, **kwargs)
    
