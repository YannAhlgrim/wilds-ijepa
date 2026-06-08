# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import socket

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def init_distributed(port=None, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            return 1, 0

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port if port is not None else _find_free_port())
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_NET_GIB_EXTRA_IFS'] = 'lo'

    try:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        logger.warning(f'NCCL init failed ({e}); falling back to single-process')
        return 1, 0

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
