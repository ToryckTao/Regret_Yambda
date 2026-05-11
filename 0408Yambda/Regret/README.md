# Regret-Yambda 工作目录

本目录用于新的 regret-aware HSRL/Yambda 实验，不直接修改 `../HSRL` baseline。

`../Repersentation` 负责语义 ID 生成与诊断；`Regret` 只消费它产出的 SID/mapping，不在这里重复训练 codebook。

## 推荐目录职责

先不要从 `../HSRL` 直接整目录 copy。第一步只建立清晰边界：

```text
Regret/
  README.md
  configs/
    pipeline.yaml              # 路径、SID variant、reward 权重、A2C 参数
  regret_core/
    data/                      # transition split / streaming dataset
    env/                       # UserResponse 环境
    model/                     # SID actor、critic、UserResponse
    rl/                        # A2C trainer / rollout storage
    eval/                      # ranking / rollout / regret metrics
  scripts/
    01_prepare_sid_mapping.py
    02_split_transitions.py
    03_train_user_response.py
    04_train_a2c.py
    05_eval.py
```

真正复用 `../HSRL` 时，只按模块搬必要逻辑，避免把 warm-start、DDPG、旧 TSV 假设一起带进来。

## 当前主线判断

### SID 方案

第一版主线使用 `raw_rqkmeans`，同时保留 `rqvae` 作为强 baseline。

原因来自现有全量诊断：

| variant | full collision | recon cosine error | L1 locality | L12 locality | hamming |
|---|---:|---:|---:|---:|---:|
| raw_rqkmeans | 0.380324 | 0.0820865 | 0.70212 | 0.25726 | 2.74782 |
| rqvae | 0.192233 | 0.0616737 | 0.69844 | 0.17960 | 3.02106 |

解释：

- `rqvae` 的 full-SID collision 更低，重构也更好。
- `raw_rqkmeans` 的二级前缀局部性更强，embedding 近邻更容易共享 SID 前缀。
- Regret pool / policy intervention 要利用“失败 semantic path 的前缀重叠”做 soft penalty，因此前缀局部性比单纯低碰撞更关键。

因此主实验先用 `raw_rqkmeans`，避免把 regret 干预建立在前缀语义较弱的 SID 上。RQVAE 不删，作为对照组报告。

## Transition 数据定义

不要把 state 定义成“用户对当前 item 的历史”。这会让历史很短，并丢掉用户全局兴趣上下文。

每条样本应定义成一个 user-item episode transition：

```text
state/history_before:
  episode 开始前，这个用户的全局交互历史

action:
  当前 episode 对应的 target item

episode_events:
  用户在该时间窗口内对 action item 的所有事件
  例如 listen / like / dislike / unlike / undislike

reward:
  从 episode_events 聚合出的标量反馈

next_state/history_after:
  episode 结束后，这个用户的全局历史
```

关键点：

- `episode_events` 用来算 reward 和 regret，不是替代全局 history。
- `history_before` 才是 actor/critic/UserResponse 看到的 state。
- `next_history_after` 才是 RL transition 的 next_state。

实现时不需要把完整 `episode_events` list 写进磁盘。只保留聚合摘要即可：

```text
n_events
n_listen
max_play_ratio
mean_play_ratio
has_like
has_dislike
has_unlike
has_undislike
first_event_time
last_event_time
regret_type
regret_strength
reward_raw
reward_scaled
```

这样不会显著增加内存或磁盘压力。内存中也只需要对当前用户流式处理；不需要全局保存所有用户的 episode_events。

## 为什么不能只读当前 history 里 target item 的全部历史算 reward

可以把“当前 state 里这个用户过去对 target item 的交互历史”作为 feature 或 prior，但不能直接替代当前 action 的 reward。

原因有三点：

1. reward 应该描述 `action` 之后发生的结果。如果只用 history 里已经发生的 target-item 事件，得到的是过去偏好，不是这次推荐后的反馈。
2. 如果你为了算 reward 读取这个用户对 target item 的全局全部交互，很容易读到未来事件，造成 time leakage。
3. 全局 user-item 聚合会抹掉用户态度变化。例如早期 like、后期 unlike，全局聚合只剩一个混合分数，RL 无法知道“什么时候推荐是好、什么时候推荐会后悔”。

推荐折中：

```text
history_target_stats:
  当前 state 之前，用户对 target item 的历史统计。作为 UserResponse/critic 的可选 feature。

episode_reward:
  当前 action episode 内发生的反馈聚合。作为 reward label。
```

也就是说，“过去同 item 历史”可以加入 state；“当前 episode 反馈”才是 reward。

## Reward 在哪里起作用

`episode_events -> reward` 有两个作用层次。

第一层：训练 UserResponse。

```text
history_before + action_item -> reward
```

UserResponse 学的是：在某个用户状态下推荐这个 item，预计能拿到多少 reward。这里的 reward 就来自 `episode_events` 聚合。

第二层：RL rollout 时产生 buffer reward。

```text
actor 选 action_item
UserResponse(history, action_item) 预测 reward
env.step 返回 reward
facade.update_buffer 写入 replay buffer
critic 用 reward 计算 TD target
actor 用 advantage 调整 SID path 概率
```

也就是说，真实日志里的聚合 reward 不一定直接进入 online-style replay buffer；它先训练 UserResponse。之后 actor 在模拟环境里选到任意候选 item 时，UserResponse 才给出相应 reward。

如果后续做 offline RL，可以额外把 transition 文件直接预填进 replay buffer：

```text
buffer.add(history_before, logged_action, reward_scaled, next_history_after, done)
```

这属于另一个实验开关，和 simulator rollout 分开比较。

如果正式做 A2C，则不使用 replay buffer 的离线随机采样训练方式。A2C 是 on-policy：

```text
rollout N step
  -> 每步存 state/action/logprob/reward/value/done
  -> 计算 return 和 advantage
  -> 更新 actor 和 critic
  -> 丢弃这批 rollout
```

因此 Regret 里如果命名为 `A2C`，就要实现 rollout storage，而不是沿用 DDPG 的 replay buffer。可以保留一个 optional logged-transition pretrain，但它不应混同于 A2C 主训练。

## Reward 初版

先保留 `reward_raw` 和 `reward_scaled` 两套值：

```text
reward_raw =
  1.0 * max_play_ratio
  + 1.0 * I(like)
  - 1.0 * I(dislike)
  - 1.5 * I(unlike)
  + 0.5 * I(undislike)

reward_scaled = clip(reward_raw, -1.0, 2.0)
```

`reward_raw` 用于分析分布；`reward_scaled` 用于 UserResponse 和 RL。

## 第一批实验

| id | SID | split/reward | actor init | reward source | 目的 |
|---|---|---|---|---|---|
| A | raw_rqkmeans | transition split + regret reward | random | UserResponse | 无 warmstart A2C 主线 |
| B | raw_rqkmeans | transition split + regret reward | random | logged transition pretrain + UserResponse | 看 logged transition 是否稳定早期训练 |
| C | rqvae | transition split + regret reward | random | UserResponse | SID 对照 |
| D | pca64_nowhiten | transition split + regret reward | random | UserResponse | PCA 对照 |
| E | raw_rqkmeans | transition split + regret reward | random | UserResponse + regret pool | regret intervention 主方法 |

## 需要实现

1. SID mapping adapter：把 `../Repersentation/artifacts_full_train/<variant>/item_ids.npy + sid.npy` 转成 Regret 训练需要的 dense mapping。
2. 新 split：输出 `train_transitions/val_transitions/test_transitions`，建议 Parquet shards。
3. Reward diagnostics：训练前输出 reward 分布、负样本比例、regret 类型比例、rows/events/episodes per user。
4. 新 UserResponse：输入全局 history item embedding、history feedback/event type、可选 history_target_stats、action item embedding，预测 `reward_scaled`。
5. A2C trainer：不依赖 HPN warmstart；actor/critic 从随机初始化或指定 checkpoint 开始。
6. Evaluation：candidate ranking、simulator rollout total reward、regret event rate、negative reward rate。
7. Regret pool：从 negative regret episode 中维护 semantic path memory，并在 actor SID logits 上做 soft penalty。

## 当前已写到 UserResponse

```text
configs/pipeline.example.json
regret_core/data/schema.py
regret_core/data/transition_dataset.py
regret_core/model/user_response.py
regret_core/env/user_response_env.py
scripts/01_prepare_sid_mapping.py
scripts/02_split_transitions.py
scripts/03_train_user_response.py
```

### 运行顺序

第一步，把 `Repersentation` 的 SID artifact 转成 Regret dense mapping。训练 UserResponse 需要 item feature 矩阵，因此要加 `--write_item_features`：

```bash
cd /root/autodl-tmp/0408Yambda/Regret

python3 scripts/01_prepare_sid_mapping.py \
  --variant raw_rqkmeans \
  --out_dir artifacts/mappings/raw_rqkmeans \
  --write_item_features
```

第二步，切 transition。这里不会保存完整 `episode_events`，只保存聚合摘要和 history/next_history：

```bash
python3 scripts/02_split_transitions.py \
  --orig2dense_npy artifacts/mappings/raw_rqkmeans/orig2dense_item_id.npy \
  --out_root artifacts/transitions/raw_rqkmeans
```

第三步，训练 UserResponse：

```bash
python3 scripts/03_train_user_response.py \
  --train_path artifacts/transitions/raw_rqkmeans/train \
  --val_path artifacts/transitions/raw_rqkmeans/val \
  --dense_item_features_npy artifacts/mappings/raw_rqkmeans/dense_item_features.npy \
  --save_path artifacts/env/regret_user_response.pt
```

调试时可以加：

```bash
--max_users 1000
--max_train_rows 100000
--max_val_rows 20000
```

### 当前 UserResponse 输入

```text
history_features
history_feedbacks
history_event_type_ids
history_mask
action_features
history_target_stats
```

输出：

```text
reward_scaled
```
