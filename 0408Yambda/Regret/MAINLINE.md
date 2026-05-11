# main_v1 正式主线

这个文件只记录当前正式主线。旧实验文件和长名字先保留，但后续汇报、复现、继续开发都优先看这里。

## 一句话流程

```text
原始事件流 -> 按 1 小时切 session -> 连续同 item 聚合成 RL step -> 训练 simulator -> 训练 HSAC 策略 -> 用 simulator rollout 评估 reward / depth / negative rate
```

## 以后只用这 5 个入口

```bash
./scripts/01_split_data.sh
./scripts/02_train_simulator.sh
./scripts/03_train_policy.sh
./scripts/04_eval_policy.sh
./scripts/05_sweep_eta.sh
```

长任务用 screen：

```bash
screen -dmS main_policy bash -lc 'cd /root/autodl-tmp/0408Yambda/Regret && ./scripts/03_train_policy.sh'
screen -r main_policy
```

主线配置：

```text
configs/main.env
```

## 短路径

当前已验证产物通过短路径访问：

```text
artifacts/current/data
artifacts/current/simulator
artifacts/current/policy_actor
artifacts/current/policy_critic
artifacts/current/policy_meta.json
artifacts/current/eval_eta010_test1000.json
artifacts/current/eta_sweep_test500.tsv
```

这些短路径是软链接，指向历史长名字产物。这样不破坏旧日志，同时日常不用再记长名字。

如果重新跑实验，默认输出到：

```text
artifacts/transitions/main_v1_data
artifacts/user_response/main_v1_simulator
artifacts/models/policy_hsac_main_actor
artifacts/models/policy_hsac_main_critic
artifacts/models/policy_hsac_main.meta.json
```

确认新结果更好后，再手动更新 `artifacts/current/*` 软链接。不要直接覆盖当前已验证结果。

## 数据定义

原始 timestamp 单位是 5 秒。session 切分规则：

```text
相邻事件时间差 > 3600 秒，则断开成新 session
单个 session 最大跨度 21600 秒
连续同 item 最多聚合 100 个原始事件
```

RL step 定义：

```text
同一个 session 内，连续出现的同一个 item 聚合成一个 step
例如 AAABBBCDE -> A, B, C, D, E 共 5 个 step
每个 step 的时间戳取该连续 run 的最后一个原始事件时间
```

这样做的目的：让“系统推荐一个 item”和“用户对这个 item 的一组连续反馈”一一对应。

## RRCA 和 RAPI

RRCA 是训练时的回顾式 reward 修正：

```text
base reward = listen + like - dislike
psi = - unlike + undislike
effective reward = base reward + gamma^Delta * psi
```

当前主线是 `rrca_apply_to=effective_reward`，也就是把后悔/修正信号写进有效 reward，再影响优势函数。

RAPI 是生成时的失败记忆池介入：

```text
失败记忆池记录用户历史里失败的语义路径
当前主线把 low_play / dislike / unlike 都放进失败记忆池
生成下一个语义 token 时，对和失败路径重叠的 token 加 soft mask
```

旧文件名里的 `brev` 就是论文里的 `B_rev`，以后中文统一叫“失败记忆池”。旧文件名里的 `af` 是 `all_failed`，表示 low_play / dislike / unlike 都进入失败记忆池。

## 当前固定参数

```text
history_len=50
session_gap_seconds=3600
max_session_span_seconds=21600
max_run_events=100
reward_version=v2

regret_memory_size=20
regret_memory_gamma=0.9
regret_memory_scope=all_failed

simulator=train_rows=1m, epochs=3, decoupled_reward_model

hsac_episodes=100000
hsac_epochs=1
hsac_max_steps=10
hsac_decode_top_k=4
hsac_gamma=0.9
rrca_apply_to=effective_reward

eval_eta=0.10
eval_episodes=1000
eval_max_steps=20
eval_decode_top_k=8
```

## 当前主结果

1000 episodes simulator rollout，RAPI eta=0.10：

```text
base_reward=7.2940
rapi_reward=7.3314
delta_reward=+0.0374

base_step=19.737
rapi_step=19.785
delta_step=+0.048

base_neg=0.3004
rapi_neg=0.3006
delta_neg=+0.00018
```

结论：

```text
RAPI 对累计 reward 和交互 depth 有小幅正收益。
negative rate 没有明显下降，说明失败记忆池目前主要在改善整体收益，不是稳定抑制负反馈。
```

## 和论文仍未完全一致的地方

```text
1. simulator 目前一次推荐主要给一个聚合反馈，不是真实用户可能连续多动作的完整过程。
2. RRCA 已写入 effective reward，但 lambda / omega 仍是超参数，不是可学习权重。
3. RAPI 的失败记忆池已包含 low_play / dislike / unlike，但介入强度 eta 仍需要 sweep。
4. 当前评估依赖 simulator，不能等同于真实在线 A/B。
5. 旧 offline SID/replay 训练仍保留做消融，不是当前主线。
```

## 脚本说明

详细脚本用途看：

```text
scripts/README.md
```

## 旧文件处理规则

```text
run_*.sh：旧包装入口已删除；当前主线只用 01-05 短脚本
scripts/User-response/：已合并进 08_train_yambda_simulator.py 并删除
06_train_sid_offline.py 和 07_eval_regret_pool_replay.py：旧 offline/replay 消融，不是主线
```

后续不要再新增 `run_*` 包装脚本；需要新入口时沿用 `NN_中文可读动作.sh` 的命名方式。
