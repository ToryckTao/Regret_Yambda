# scripts 目录说明

这个目录现在分成三类：主线入口、底层 Python、历史兼容脚本。日常只用主线入口。

## 主线入口

```text
01_split_data.sh       数据切分：事件流 -> session -> 连续同 item 聚合成 RL step
02_train_simulator.sh  训练 simulator：历史状态 + 候选 item -> 预测反馈 / reward 近似
03_train_policy.sh     训练策略：HSAC + simulator rollout + RRCA 有效 reward
04_eval_policy.sh      固定 eta=0.10 评估：reward、depth、negative rate
05_sweep_eta.sh        扫描 eta：观察 RAPI 介入强度是否稳定
```

后台运行示例：

```bash
screen -dmS policy_train bash -lc 'cd /root/autodl-tmp/0408Yambda/Regret && ./scripts/03_train_policy.sh'
screen -r policy_train
```

## 底层 Python

```text
01_prepare_sid_mapping.py       准备 dense item 和语义 ID 映射；重建 RQKmeans 映射时才用
02_split_transitions.py         主线数据切分核心；必须保留
04_analyze_event_distribution.py 原始事件分布分析；论文数据描述用
06_train_sid_offline.py         旧的离线 SID/actor-critic 训练；现在不作为主线，但保留做消融
07_eval_regret_pool_replay.py   旧的 replay 式 RAPI 评估；现在不是主评估
08_train_yambda_simulator.py    主线 simulator 训练核心；必须保留
09_eval_simulator_rollout.py    主线 rollout 评估核心；必须保留
10_train_hsac_simulator_rollout.py 主线策略训练核心；必须保留
11_analyze_session_run_stats.py 新 session/step 数据统计；论文数据描述用
```

## 历史脚本状态

```text
旧 run_*.sh 包装脚本已删除。
旧 User-response/ 目录已合并进 08_train_yambda_simulator.py 并删除。
```

当前处理原则：

```text
新实验：只从 01_split_data.sh 到 05_sweep_eta.sh 进入
旧复现：查看 MAINLINE.md、实验日志和当前 01-05 脚本参数，不再依赖旧 run_* 文件
清理原则：不再新增 run_* 包装脚本
```

不建议现在直接改名底层 Python。原因是多个旧 meta、日志里仍写死了这些脚本名；直接重命名会让旧实验不可复现。现在采取“短入口 + 中文说明 + 保留底层名”的折中方案。

## 命名规则

以后对外只用 `main_v1`：

```text
configs/main.env
artifacts/current/data
artifacts/current/simulator
artifacts/current/policy_actor
artifacts/current/policy_critic
```

旧名里的含义：

```text
rqkm      raw_rqkmeans 语义 ID
srun      session_run，也就是按 session 和连续同 item run 构造 step
brev      B_rev，论文里的失败记忆池；以后中文叫“失败记忆池”
af        all_failed，low_play / dislike / unlike 都进入失败记忆池
rrcaeff   RRCA 写入 effective reward
eta010    RAPI 介入强度 eta=0.10
```
