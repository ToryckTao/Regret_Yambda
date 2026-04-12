from __future__ import annotations

import torch


def sum_with_cost(feedback, zero_reward_cost=0.1):
    """输入 feedback=[B,L]；输出每个 slate 的求和 reward。"""
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    return torch.sum(feedback + cost, dim=-1)


def sigmoid_sum_with_cost(feedback, zero_reward_cost=0.1):
    return torch.sigmoid(sum_with_cost(feedback, zero_reward_cost))


def log_sum_with_cost(feedback, zero_reward_cost=0.1):
    reward = sum_with_cost(feedback, zero_reward_cost)
    reward[reward > 0] = (reward[reward > 0] + 1).log()
    return torch.sigmoid(reward)


def mean_with_cost(feedback, zero_reward_cost=0.1):
    """输入 feedback=[B,L]；输出每个 slate 的平均 reward。"""
    cost = torch.zeros_like(feedback)
    cost[feedback == 0] = -zero_reward_cost
    return torch.mean(feedback + cost, dim=-1)


def mean_advance_with_cost(feedback, zero_reward_cost=0.1, offset=0.5):
    return mean_with_cost(feedback, zero_reward_cost) - offset
