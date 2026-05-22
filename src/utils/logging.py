# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def _format_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_run_name(args):
    meta_args = args.get("meta", {})
    data_args = args.get("data", {})
    opt_args = args.get("optimization", {})
    mask_args = args.get("mask", {})

    model_name = meta_args.get("model_name", "model")
    patch_size = mask_args.get("patch_size", meta_args.get("patch_size"))
    crop_size = data_args.get("crop_size", meta_args.get("crop_size"))
    batch_size = data_args.get("batch_size")
    optimizer = opt_args.get("optimizer", "opt")
    lr = opt_args.get("lr")
    weight_decay = opt_args.get("weight_decay")
    epochs = opt_args.get("epochs")

    parts = [
        model_name,
        f"p{_format_value(patch_size)}" if patch_size is not None else None,
        f"c{_format_value(crop_size)}" if crop_size is not None else None,
        f"bs{_format_value(batch_size)}" if batch_size is not None else None,
        str(optimizer).lower(),
        f"lr{_format_value(lr)}" if lr is not None else None,
        f"wd{_format_value(weight_decay)}" if weight_decay is not None else None,
        f"ep{_format_value(epochs)}" if epochs is not None else None,
    ]
    return "-".join([p for p in parts if p])


def _extract_run_name_from_checkpoint(meta_args):
    checkpoint_path = meta_args.get("checkpoint_path")
    if checkpoint_path:
        folder = os.path.dirname(checkpoint_path)
    else:
        folder = meta_args.get("checkpoint_folder")
    if not folder:
        return None
    return os.path.basename(os.path.normpath(folder))


def resolve_log_dir(args, stage="train"):
    log_args = args.setdefault("logging", {})
    auto_folder = log_args.get("auto_folder", True)
    if auto_folder or not log_args.get("folder"):
        run_name = log_args.get("run_name")
        if stage == "eval" and not run_name:
            meta_args = args.get("meta", {})
            run_name = _extract_run_name_from_checkpoint(meta_args)
        if not run_name:
            run_name = build_run_name(args)
        base_dir = "experiment_logs"
        if stage == "eval":
            folder = os.path.join(base_dir, "eval-wilds", run_name)
        else:
            folder = os.path.join(base_dir, run_name)
        log_args["folder"] = folder
    os.makedirs(log_args["folder"], exist_ok=True)
    return log_args["folder"]


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats
