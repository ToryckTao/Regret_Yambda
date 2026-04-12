# 0408Yambda: Yambda-HSRL Baseline

目标：在 Yambda 50M sequential 数据上先跑通一个最小 HSRL baseline。当前版本只保留正式全量链路，不再保留 debug/smoke 专用脚本。

## 目录原则

- `01-07_*.py`: 正式流程文件，按编号顺序执行。
- `pipeline_config.sh`: 全流程唯一 shell 配置入口，集中管理路径和训练超参。
- `run_stage.sh`: 唯一 shell 运行入口，读取 `pipeline_config.sh` 后调用对应 Python 文件。
- `hsrl_core/`: 项目内置的最小 HSRL 主干代码，不依赖 `adapter/`。
- `adapter/`: Yambda 专用适配层，覆盖 reader / env / agent / facade 等需要改的模块。
- `artifacts/`: 运行后产物目录，包括 codebook、mapping、split、模型 checkpoint；已在 `.gitignore` 中，不应提交。
- `preprocess/`: 预处理缓存目录；已在 `.gitignore` 中，不应提交。

`hsrl_core/` 和 `adapter/` 是解耦的：`hsrl_core` 提供基础 HSRL 结构，`adapter/bootstrap.py` 会把 `hsrl_core` 和 `adapter` 加入 import path，并让 `adapter` 中同名模块优先覆盖。这样项目运行不依赖外部 `data/HSRL/HSRL` 源码目录。

## 前期准备

需要本地已有数据：

```text
/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet
/Users/Toryck/Coding/DATASET/Yambda/sequential/50m/multi_event.parquet
```

安装依赖：

```bash
cd /Users/Toryck/Coding/Regret_Yambda/0408Yambda
python3 -m pip install -r requirements.txt
```

如果数据不在默认位置，运行时覆盖：

```bash
YAMBA_DATA_DIR=/path/to/Yambda ./run_stage.sh 01
```

## 正式运行顺序

单阶段运行：

```bash
cd /Users/Toryck/Coding/Regret_Yambda/0408Yambda
./run_stage.sh 01
./run_stage.sh 02
./run_stage.sh 03
./run_stage.sh 04
./run_stage.sh 05
./run_stage.sh 06
./run_stage.sh 07
```

组合运行：

```bash
./run_stage.sh preprocess   # 01, 02, 03
./run_stage.sh train        # 04, 05, 06
./run_stage.sh all          # 01-07 全流程
```

不建议直接一口气 `all` 跑云服务器首次实验；更稳的是每个阶段确认输出存在后再进入下一阶段。

## 参数配置

所有正式流程的 shell 参数都在 `pipeline_config.sh`。最重要的默认值：

```bash
CODEBOOK_N_LEVELS=4
CODEBOOK_SIZE=256
CODEBOOK_SAMPLE_SIZE=200000
MAX_USERS=0
HPN_MAX_TRAIN_ROWS=0
URM_MAX_TRAIN_ROWS=0
SID_N_ITER=10000
SID_MAX_CANDIDATE_ITEMS=50000
EVAL_MAX_ROWS=0
```

`0` 的含义通常是全量，例如 `MAX_USERS=0` 表示处理全部用户，`HPN_MAX_TRAIN_ROWS=0` 表示读取全量训练 TSV。

可以临时覆盖单个参数：

```bash
SID_N_ITER=20000 SID_MAX_CANDIDATE_ITEMS=100000 ./run_stage.sh 06
```

直接执行 `python 04_train_hpn_warmstart.py` 也可以使用 Python 文件内置默认路径，但不会自动读取 `pipeline_config.sh` 里的 shell 覆盖值。正式流程建议统一使用 `./run_stage.sh <stage>`。

## 文件说明

`01_build_codebook.py`

- 输入：`embeddings.parquet` 中的 `normalized_embed` 或 `embed`。
- 输出：`artifacts/codebook/yambda_rq_codebook.npz` 和 meta json。
- 功能：训练 HSRL 风格的 residual balanced KMeans codebook。正式默认是 `4 x 256`。注意：codebook 训练默认对 embedding 做 reservoir sampling；后续 `02` 会对全量 item 编码。

`02_build_item_sid.py`

- 输入：`embeddings.parquet`、`01` 产出的 codebook。
- 输出：`orig2dense_item_id.npy`、`dense2orig_item_id.npy`、`dense_item2sid.npy`。
- 功能：给所有 embedding-covered item 分配 dense id，并把每个 item 编成 SID token path。

`03_split_data.py`

- 输入：`multi_event.parquet`、`02` 产出的 `orig2dense`。
- 输出：`artifacts/processed/train.tsv`、`val.tsv`、`test.tsv`、`split.meta.json`。
- 功能：流式读取用户事件，按 user-item episode 构造决策样本，再按用户时间顺序切 train/val/test。
- 样本核心字段：`user_mid_history` 是历史 item dense id 序列，`user_click_history` 是历史事件信号，`target_dense_item_id` / `slate_of_items` 是当前 action item，`user_clicks` 是聚合 reward，`feedback_label` 是二值反馈。

`04_train_hpn_warmstart.py`

- 输入：`train.tsv`、`val.tsv`、`dense_item2sid.npy`、`embeddings.parquet`。
- 输出：`artifacts/models/hpn_warmstart.pt` 和 meta json。
- 功能：BC warm-start。输入历史窗口张量约为 `[B, H, 128]`，target SID 为 `[B, L]`，正式默认 `H=50, L=4`。

`05_train_user_response.py`

- 输入：`train.tsv`、`val.tsv`、`test.tsv`、`embeddings.parquet`。
- 输出：`yambda_user_env.model.checkpoint`、`yambda_user_env.model.log`、feature cache、meta json。
- 功能：训练 `history + exposure item -> reward` 的近似 UserResponse，给下一步 offline RL 环境提供反馈。

`06_train_yambda_sid.py`

- 输入：`05` 的 UserResponse log、`02` 的 `dense_item2sid.npy`、`04` 的 HPN checkpoint。
- 输出：`artifacts/models/yambda_sid_actor`、`yambda_sid_critic`、optimizer checkpoint、meta json。
- 功能：跑最小 SID actor-critic。Actor 逐层输出 SID token 分布，Facade 在候选 item 集内选 item，UserResponse 给 reward，Critic 做 value/TD 更新。

`07_eval_candidate_ranking.py`

- 输入：`test.tsv`、`dense_item2sid.npy`、actor checkpoint、`embeddings.parquet`。
- 输出：`artifacts/models/candidate_ranking.meta.json`。
- 功能：没有完整 simulator 时的离线候选集排序检查，输出 HR@K、NDCG@K、MRR、target rank、token accuracy。

## 数据流例子

原始用户事件：

```text
uid=7
[(item=3, listen, ratio=0.30), (item=5, like), (item=192, listen, ratio=0.95), (item=192, like)]
```

经过 `02` 后，item 会变成 dense id 和 SID：

```text
orig item 192 -> dense item 1185194 -> SID [12, 88, 7, 203]
```

经过 `03` 后，一条样本可以是：

```text
history: [dense(3), dense(5)]
history signals: [0.30, 1.0]
target/action item: dense(192)
target SID: [12, 88, 7, 203]
reward: max_play_ratio(0.95) + like(1.0) = 1.95
feedback_label: 1
next_history: [dense(3), dense(5), dense(192), dense(192)]
```

训练时 `04` 学的是：给定历史上下文，把 target item 的 SID token path 概率抬高。`06` 学的是：Actor 生成/打分 SID action，环境用 `05` 训练出的 UserResponse 近似返回 reward，Critic 再用 reward 做 TD 更新。

## 当前保留文件

正式流程只需要这些顶层入口：

```text
01_build_codebook.py
02_build_item_sid.py
03_split_data.py
04_train_hpn_warmstart.py
05_train_user_response.py
06_train_yambda_sid.py
07_eval_candidate_ranking.py
pipeline_config.sh
run_stage.sh
requirements.txt
README.md
```

已删除的旧入口包括：`03_build_item_meta.py`、`04_build_decision_table.py`、`08_smoke_sid_rl_step.py`、`run_preprocess.sh`、`run_full_pipeline.sh`、`train_yambda_sid.sh`、`Discovery.md`。
