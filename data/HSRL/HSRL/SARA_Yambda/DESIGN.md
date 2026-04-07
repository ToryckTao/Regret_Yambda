# SARA (Regret-Aware Recommendation) 开发日志

## 项目概述

SARA 是基于层级推荐策略（SID）的 regret-aware 增强版本，核心创新是在推荐过程中引入用户后悔机制，减少推荐中的"定时炸弹"问题。

## 核心创新

### 1. 后悔惩罚 D（推理时）

**公式**：D(z_l) = Σ_j Sim(z_l, z_j) × Φ_j

其中：
- z_l: 当前层级 l 的候选 token embedding
- z_j: 后悔池中第 j 个物品的 token embedding
- Φ_j = γ^Δt × ψ_t: 惩罚值
  - γ: 时间折扣因子（默认 0.9）
  - Δt: 后悔延迟步数
  - ψ_t: 即时惩罚 (unlike = -1, undislike = +1)

**实现位置**：`SARAFacade_credit.py` 的 `apply_policy` 方法

**效果**：推理时，对类似历史后悔物品的 token 施加惩罚，降低其被选概率。

### 2. 后悔信号检测（真正的延迟后悔）

**核心机制**：
1. **Step t**：推荐物品产生正面反馈 (preds > 0) 时，模拟器以概率 p = exp(-pred * 2) 预埋一个"定时炸弹"
2. **采样延迟**：从 Poisson(4) + 1 采样延迟步数 Δt
3. **Step t + Δt**：后悔被触发，计算惩罚 Φ = γ^Δt × ψ_t (ψ_t = -λ_unlike)
4. 将 (SID_tokens, Φ) 加入后悔池

**实现位置**：
- 环境：`SARAEnvironment_GPU.py` 的 `step` 方法
  - 预埋后悔：`pending_regrets` 列表
  - 触发检查：每步检查 `pending_regrets`，将触发的 regret 转入 `regret_signals`
- Facade：`SARAFacade_credit.py` 的 `_update_regret_pool` 方法

**数据流**：
```
Step t:
  推荐物品 → URM预测 → 如果preds>0:
    → 计算反悔概率 p = exp(-pred*2)
    → 掷骰子决定是否预埋后悔
    → 如果预埋: 记录 (trigger_step=t+Δt, user_id, item_id, ψ=-λ)

Step t+Δt:
  检查 pending_regrets
  → 如果 trigger_step == current_step:
    → 触发后悔，加入 regret_signals
    → Facade 将其加入后悔池
```

### 3. 追溯更新 Advantage（待完善）

**原计划**：A' = A + γ^Δt × ψ_t

**当前状态**：由于训练数据中没有 unlike→like 的配对信息，暂时使用 B 方式替代。

**未来方向**：如果有包含完整后悔标签的数据，可以实现 A 方式。

## 实现细节

### RegretPool 数据结构

```python
class RegretPool:
    - tokens: (max_size, n_levels)  # 存储 SID token 序列
    - phis: (max_size,)            # 存储惩罚值 Φ
    - current_size: int             # 当前存储数量
```

### 层级权重

- Level 1: w1 = 0.05
- Level 2: w2 = 0.25  
- Level 3: w3 = 0.70

累积权重用于计算不同层级的惩罚强度。

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| regret_pool_size | 20 | 后悔池大小 |
| regret_penalty_weight | 0.5 | 推理时惩罚系数 η |
| regret_layer_weights | 0.05,0.25,0.7 | 层级权重 |
| regret_gamma | 0.9 | 时间折扣因子 γ |
| lambda_unlike | 1.0 | unlike 惩罚权重 |

## 文件结构

```
SARA_Yambda/
├── env/
│   └── SARAEnvironment_GPU.py    # 环境：模拟用户反馈和 regret 信号
├── model/
│   ├── RegretPool.py             # 后悔池数据结构
│   └── facade/
│       └── SARAFacade_credit.py  # Facade：集成 regret 惩罚
├── preprocess_regret.py          # 预处理后悔池数据
├── regret_config.py              # 配置参数
├── train_sara.sh                 # 训练脚本
└── DESIGN.md                     # 本文档
```

## 实现总结 (2026-03-14)

### 已完成功能

1. **延迟后悔检测机制** (真正的 regret)
   - Step t: 预埋后悔 (物品ID, Δt) 到 `pending_regrets`
   - Step t+Δt: 触发后悔，更新 `regret_signals`

2. **后悔池更新**
   - Facade 从 `regret_signals` 获取触发后的后悔
   - 计算 Φ = γ^Δt × ψ_t
   - 将 (SID_tokens, Φ) 加入用户的后悔池

3. **推理时惩罚**
   - 应用 D(z_l) = Σ_j Sim(z_l, z_j) × Φ_j
   - 降低类似后悔物品的 token 概率

### 数据流

```
Step t:
  推荐 → URM预测
    → preds > 0? 是 → 掷骰子决定是否预埋后悔
      → 预埋: pending_regrets.append({trigger_step, item_id, ψ, Δt})

Step t+Δt:
  检查 pending_regrets
    → 触发: regret_signals.append({item_id, ψ, Δt})
    → Facade: add_regret(user_id, sid_tokens, Φ)

推理时:
  apply_policy → 计算惩罚 → 降低后悔物品概率
```

### 待测试
