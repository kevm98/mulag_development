import importlib
import torch.nn as nn
from stable_baselines3.common.utils import constant_fn
from typing import Any


def _import_from_string(path: str):
    # "package.module:ClassName"
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def process_sb3_cfg(cfg: dict, num_envs: int) -> dict:
    def update_dict(hyperparams: dict[str, Any], depth: int) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value, depth + 1)

            if isinstance(value, str):
                # existing behavior: convert "nn.ELU" -> nn.ELU
                if value.startswith("nn."):
                    hyperparams[key] = getattr(nn, value[3:])

                # NEW: import custom classes from "module:attr"
                # Only do this for keys where SB3 expects a class object
                if key in ["features_extractor_class", "policy_class"] and ":" in value:
                    hyperparams[key] = _import_from_string(value)

            if depth == 0:
                if key in ["learning_rate", "clip_range", "clip_range_vf"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

        # if "n_minibatches" in hyperparams:
        #     hyperparams["batch_size"] = (hyperparams.get("n_steps", 2048) * num_envs) // hyperparams["n_minibatches"]
        #     del hyperparams["n_minibatches"]

        return hyperparams

    return update_dict(cfg, depth=0)
