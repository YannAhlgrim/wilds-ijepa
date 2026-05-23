import argparse
import itertools
import os
import sys
import tempfile

import yaml
import submitit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_supervised import main as app_main


def _set_by_dotted_key(config, dotted_key, value):
    keys = dotted_key.split(".")
    cur = config
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _deep_update(base, updates):
    for key, value in updates.items():
        _set_by_dotted_key(base, key, value)
    return base


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


class GridTrainer:
    def __init__(self, fname):
        self.fname = fname

    def __call__(self):
        with open(self.fname, "r") as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
        app_main(args=params, resume_preempt=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="grid yaml file")
    parser.add_argument("--folder", type=str, help="location to save submitit logs")
    parser.add_argument("--partition", type=str, help="cluster partition to submit jobs on")
    parser.add_argument("--nodes", type=int, default=1, help="num. nodes to request for job")
    parser.add_argument(
        "--tasks-per-node", type=int, default=1, help="num. procs to per node"
    )
    parser.add_argument("--time", type=int, default=4300, help="time in minutes to run job")
    args = parser.parse_args()

    grid_cfg = _load_yaml(args.grid)
    base_config_path = grid_cfg["base_config"]
    base_params = _load_yaml(base_config_path)

    constants = grid_cfg.get("constants", {})
    grid = grid_cfg.get("grid", {})
    launch = grid_cfg.get("launch", {})

    params_list = []
    for overrides in _expand_grid(grid):
        params = yaml.safe_load(yaml.dump(base_params))
        _deep_update(params, constants)
        _deep_update(params, overrides)
        params_list.append(params)

    log_folder = args.folder or launch.get("folder")
    if not log_folder:
        raise ValueError("submitit folder required via --folder or grid launch.folder")

    executor = submitit.SlurmExecutor(
        folder=os.path.join(log_folder, "job_%j"), max_num_timeout=20
    )
    executor.update_parameters(
        partition=args.partition or launch.get("partition"),
        mem_per_gpu=launch.get("mem_per_gpu", "55G"),
        time=args.time or int(launch.get("time", 4300)),
        nodes=args.nodes or int(launch.get("nodes", 1)),
        ntasks_per_node=args.tasks_per_node or int(launch.get("tasks_per_node", 1)),
        cpus_per_task=int(launch.get("cpus_per_task", 10)),
        gpus_per_node=args.tasks_per_node or int(launch.get("tasks_per_node", 1)),
    )

    temp_dir = tempfile.mkdtemp(prefix="grid_configs_")

    jobs = []
    with executor.batch():
        for idx, params in enumerate(params_list):
            tmp_path = os.path.join(temp_dir, f"grid_{idx}.yaml")
            with open(tmp_path, "w") as f:
                yaml.dump(params, f)
            job = executor.submit(GridTrainer(tmp_path))
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == "__main__":
    main()
