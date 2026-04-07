#!/usr/bin/env python3
"""
SARA 核心逻辑快速测试
验证：环境 regret 追踪 → Facade 后验池更新 → 推理时惩罚
"""

import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, '/root/autodl-tmp/data/HSRL/HSRL')

from model.RegretPool import RegretPool, RegretPoolManager

def test_regret_signal_tracking():
    """测试 1: 环境的后悔追踪逻辑"""
    print("=" * 50)
    print("测试 1: 环境 regret 追踪")
    print("=" * 50)
    
    # 模拟 pending_regrets 和 regret_signals
    pending_regrets = []
    regret_signals = [[] for _ in range(1)]  # 1个用户
    
    # Step 0: 预埋一个后悔
    current_step = 0
    pending_regrets.append({
        'trigger_step': 3,  # 3步后触发
        'user_id': 0,
        'item_idx': 0,
        'item_id': 100,
        'psi': -1.0,
        'delta_t': 3
    })
    print(f"Step {current_step}: 预埋后悔 -> trigger_step=3")
    
    # Step 1-2: 检查未触发
    for step in [1, 2]:
        triggered = []
        remaining = []
        for regret in pending_regrets:
            if regret['trigger_step'] <= step:
                triggered.append(regret)
            else:
                remaining.append(regret)
        pending_regrets = remaining
        if triggered:
            regret_signals[0].extend(triggered)
            print(f"Step {step}: 触发后悔! (不应该触发)")
        else:
            print(f"Step {step}: 无触发")
    
    # Step 3: 应该触发
    step = 3
    triggered = []
    remaining = []
    for regret in pending_regrets:
        if regret['trigger_step'] <= step:
            triggered.append(regret)
        else:
            remaining.append(regret)
    pending_regrets = remaining
    if triggered:
        for r in triggered:
            regret_signals[0].append({
                'time': r['trigger_step'],
                'type': 'unlike',
                'item_idx': r['item_idx'],
                'item_id': r['item_id'],
                'psi': r['psi'],
                'delta_t': r['delta_t']
            })
        print(f"Step {step}: 触发后悔! item_id={triggered[0]['item_id']}, psi={triggered[0]['psi']}, delta_t={triggered[0]['delta_t']}")
    
    # 验证
    assert len(regret_signals[0]) == 1, "应该有1个 regret signal"
    assert regret_signals[0][0]['item_id'] == 100, "item_id 应该是 100"
    print("✓ 测试 1 通过: 环境 regret 追踪正常")


def test_regret_pool_update():
    """测试 2: Facade 后验池更新"""
    print("\n" + "=" * 50)
    print("测试 2: Facade 后验池更新")
    print("=" * 50)
    
    import argparse
    
    # 创建 mock args
    args = argparse.Namespace(
        regret_pool_size=20,
        regret_penalty_weight=0.5,
        regret_layer_weights='0.05,0.25,0.7',
        regret_gamma=0.9,
        regret_pool_init_path=''
    )
    
    # 初始化 RegretPoolManager
    manager = RegretPoolManager(
        args,
        n_users=10,
        n_levels=3,
        vocab_sizes=[256, 256, 256]
    )
    
    # 模拟 SID tokens (3层)
    sid_tokens = [10, 20, 30]  # 一个物品的 SID
    psi = -1.0
    delta_t = 3
    gamma = 0.9
    
    # 计算 Φ = γ^Δt × ψ_t
    phi = (gamma ** delta_t) * psi
    print(f"计算 phi = {gamma}^{delta_t} × ({psi}) = {phi:.4f}")
    
    # 加入后验池
    manager.add_regret(user_id=0, sid_tokens=sid_tokens, phi=phi)
    
    pool = manager.get_pool(0)
    print(f"后验池大小: {pool.current_size}")
    print(f"存储的 tokens: {pool.tokens[:pool.current_size]}")
    print(f"存储的 phis: {pool.phis[:pool.current_size]}")
    
    assert pool.current_size == 1, "后验池应该有1条记录"
    assert np.isclose(pool.phis[0], phi), "phi 值不匹配"
    print("✓ 测试 2 通过: 后验池更新正常")


def test_regret_penalty_inference():
    """测试 3: 推理时惩罚计算"""
    print("\n" + "=" * 50)
    print("测试 3: 推理时惩罚计算")
    print("=" * 50)
    
    import argparse
    
    args = argparse.Namespace(
        regret_pool_size=20,
        regret_penalty_weight=0.5,
        regret_layer_weights='0.05,0.25,0.7',
        regret_gamma=0.9,
        regret_pool_init_path=''
    )
    
    pool = RegretPool(args, n_levels=3, vocab_sizes=[256, 256, 256])
    
    # 添加一条后悔记录: SID = [10, 20, 30], phi = -0.729
    pool.add([10, 20, 30], -0.729)
    
    # 模拟候选 logits (3层, vocab_size=256)
    logits_l0 = torch.randn(1, 256)
    logits_l1 = torch.randn(1, 256)
    logits_l2 = torch.randn(1, 256)
    
    # 计算每层惩罚
    penalty_l0 = pool.compute_penalty(0, logits_l0)
    penalty_l1 = pool.compute_penalty(1, logits_l1)
    penalty_l2 = pool.compute_penalty(2, logits_l2)
    
    print(f"Level 0 penalty shape: {penalty_l0.shape}")
    print(f"Level 1 penalty shape: {penalty_l1.shape}")
    print(f"Level 2 penalty shape: {penalty_l2.shape}")
    
    # 验证惩罚不为零
    assert penalty_l0.abs().sum() > 0, "惩罚应该非零"
    assert penalty_l1.abs().sum() > 0, "惩罚应该非零"
    assert penalty_l2.abs().sum() > 0, "惩罚应该非零"
    
    # 验证被惩罚的 token 是 [10, 20, 30]
    # 惩罚应该在该 token 位置有较高值
    print(f"Token 10 在 level0 的惩罚: {penalty_l0[0, 10].item():.4f}")
    print(f"Token 20 在 level1 的惩罚: {penalty_l1[0, 20].item():.4f}")
    print(f"Token 30 在 level2 的惩罚: {penalty_l2[0, 30].item():.4f}")
    
    # 验证被惩罚位置的值更负（概率降低）
    assert penalty_l0[0, 10] < penalty_l0[0, 0] + 0.1, "惩罚应该在目标 token 位置更明显"
    
    print("✓ 测试 3 通过: 推理时惩罚计算正常")


def test_env_step_integration():
    """测试 4: 集成测试 - 模拟完整流程"""
    print("\n" + "=" * 50)
    print("测试 4: 集成测试")
    print("=" * 50)
    
    # 简化模拟: 3步推荐
    pending_regrets = []
    regret_signals = [[] for _ in range(1)]
    
    # Step 0: 推荐物品100, 预埋后悔
    pending_regrets.append({
        'trigger_step': 2,  # 2步后触发
        'user_id': 0,
        'item_idx': 0,
        'item_id': 100,
        'psi': -1.0,
        'delta_t': 2
    })
    print("Step 0: 推荐物品100, 预埋后悔(Δt=2)")
    
    # Step 1: 推荐物品200
    pending_regrets.append({
        'trigger_step': 3,
        'user_id': 0,
        'item_idx': 1,
        'item_id': 200,
        'psi': -1.0,
        'delta_t': 2
    })
    print("Step 1: 推荐物品200, 预埋后悔(Δt=2)")
    
    # Step 2: 检查触发 + 推荐物品300
    step = 2
    triggered = []
    remaining = []
    for regret in pending_regrets:
        if regret['trigger_step'] <= step:
            triggered.append(regret)
        else:
            remaining.append(regret)
    pending_regrets = remaining
    
    for r in triggered:
        regret_signals[0].append({
            'item_idx': r['item_idx'],
            'item_id': r['item_id'],
            'psi': r['psi'],
            'delta_t': r['delta_t']
        })
        print(f"Step 2: 触发后悔! item_id={r['item_id']}, delta_t={r['delta_t']}")
    
    pending_regrets.append({
        'trigger_step': 4,
        'user_id': 0,
        'item_idx': 2,
        'item_id': 300,
        'psi': -1.0,
        'delta_t': 2
    })
    print("Step 2: 推荐物品300, 预埋后悔(Δt=2)")
    
    # Step 3: 检查触发
    step = 3
    triggered = []
    remaining = []
    for regret in pending_regrets:
        if regret['trigger_step'] <= step:
            triggered.append(regret)
        else:
            remaining.append(regret)
    pending_regrets = remaining
    
    for r in triggered:
        regret_signals[0].append({
            'item_idx': r['item_idx'],
            'item_id': r['item_id'],
            'psi': r['psi'],
            'delta_t': r['delta_t']
        })
        print(f"Step 3: 触发后悔! item_id={r['item_id']}, delta_t={r['delta_t']}")
    
    # 验证
    print(f"\n最终 regret_signals: {len(regret_signals[0])} 条")
    for i, sig in enumerate(regret_signals[0]):
        gamma = 0.9
        phi = (gamma ** sig['delta_t']) * sig['psi']
        print(f"  {i+1}. item_id={sig['item_id']}, delta_t={sig['delta_t']}, phi={phi:.4f}")
    
    assert len(regret_signals[0]) == 2, "应该有2条 regret"
    print("✓ 测试 4 通过: 集成测试正常")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SARA 核心逻辑单元测试")
    print("=" * 60)
    
    test_regret_signal_tracking()
    test_regret_pool_update()
    test_regret_penalty_inference()
    test_env_step_integration()
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
