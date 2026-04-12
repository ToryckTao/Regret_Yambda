from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def set_random_seed(seed):
    """输入随机种子；输出为设置 Python / NumPy / Torch 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print('dir "' + fpath + '" existed')
    else:
        try:
            os.mkdir(fpath)
        except Exception:
            print('error when creating "' + fpath + '"')


def setup_path(fpath, is_dir=True):
    dirs = [p for p in fpath.split("/")]
    cur_path = ""
    dirs = dirs[:-1] if not is_dir else dirs
    for p in dirs:
        cur_path += p
        check_folder_exist(cur_path)
        cur_path += "/"


def padding_and_clip(sequence, max_len, padding_direction="left"):
    """输入 list 和最大长度；输出 padding / 截断后的 list。"""
    if len(sequence) < max_len:
        if padding_direction == "left":
            sequence = [0] * (max_len - len(sequence)) + sequence
        else:
            sequence = sequence + [0] * (max_len - len(sequence))
    return sequence[-max_len:] if padding_direction == "left" else sequence[:max_len]


def wrap_batch(batch, device):
    """输入一个 batch dict；输出移动到指定 device 后的 batch dict。"""
    for key, value in batch.items():
        if type(value).__module__ == np.__name__:
            batch[key] = torch.from_numpy(value)
        elif torch.is_tensor(value):
            batch[key] = value
        elif type(value) is list:
            batch[key] = torch.tensor(value)
        else:
            continue
        if batch[key].type() == "torch.DoubleTensor":
            batch[key] = batch[key].float()
        batch[key] = batch[key].to(device)
    return batch


def get_regularization(*modules):
    reg = 0
    for module in modules:
        for param in module.parameters():
            reg = torch.mean(param * param) + reg
    return reg


def init_weights(module):
    if "Linear" in str(type(module)):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.normal_(module.bias, mean=0.0, std=0.01)
    elif "Embedding" in str(type(module)):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)


def sample_categorical_action(
    action_prob,
    candidate_ids,
    slate_size,
    with_replacement=True,
    batch_wise=False,
    return_idx=False,
):
    """输入候选概率；输出采样得到的 action item id。"""
    if with_replacement:
        indices = Categorical(action_prob).sample(sample_shape=(slate_size,))
        indices = torch.transpose(indices, 0, 1)
    else:
        indices = torch.cat(
            [torch.multinomial(prob, slate_size, replacement=False).view(1, -1) for prob in action_prob],
            dim=0,
        )
    action = torch.gather(candidate_ids, 1, indices) if batch_wise else candidate_ids[indices]
    if return_idx:
        return action.detach(), indices.detach()
    return action.detach()


class LinearScheduler:
    """线性 epsilon scheduler；schedule_timesteps 至少为 1，避免除零。"""

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = max(int(schedule_timesteps), 1)
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class SinScheduler:
    """sin-shaped scheduler；schedule_timesteps 至少为 1，避免除零。"""

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = max(int(schedule_timesteps), 1)
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = np.sin(min(float(t) / self.schedule_timesteps, 1.0) * np.pi * 0.5)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
