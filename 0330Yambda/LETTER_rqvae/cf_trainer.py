#!/usr/bin/env python3
"""
SASRec CF Embedding Extractor

基于 SASRec（pmixer/SASRec.pytorch）训练协同过滤模型，
并从 `item_emb` 中提取 item CF embeddings，用于 LETTER 的 Collaborative Regularization。

流程：
  1. 将 interaction parquet 转换为 SASRec txt 格式
  2. 训练轻量 SASRec（hidden_units=64, num_blocks=2）
  3. 提取 `model.item_emb.weight`（去掉 padding）→ CF embeddings
  4. 保存为 npz: {item_ids, cf_embeddings}，与 embeddings.npz 对齐

用法:
  # 方式一：使用 likes.parquet
  python cf_trainer.py \
      --input ../data/sequential-50m/likes.parquet \
      --embeddings ../data/embeddings.npz \
      --output ../data/cf_embeddings.npz \
      --dataset_name likes \
      --device cuda:0

  # 方式二：使用 multi_event.parquet（更丰富）
  python cf_trainer.py \
      --input ../data/sequential-50m/multi_event.parquet \
      --embeddings ../data/embeddings.npz \
      --output ../data/cf_embeddings.npz \
      --dataset_name multi_event \
      --device cuda:0

  # 方式三：合并多个 event 文件
  python cf_trainer.py \
      --input_combo likes,listens,multi_event \
      --data_dir ../data/sequential-50m \
      --embeddings ../data/embeddings.npz \
      --output ../data/cf_embeddings.npz \
      --dataset_name combined \
      --device cuda:0
"""
import argparse
import os
import time
import copy
import random
import shutil
import tempfile
import numpy as np
import torch
from itertools import chain
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  SASRec Model（from pmixer/SASRec.pytorch，轻量适配版）
# ─────────────────────────────────────────────────────────────────────────────

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))
        ))
        return outputs.transpose(-1, -2)


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.2, maxlen=50, norm_first=True):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.norm_first = norm_first

        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(hidden_units, num_heads,
                                             dropout=dropout_rate,
                                             batch_first=False))
            self.forward_layernorms.append(
                torch.nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            try:
                torch.nn.init.xavier_normal_(p.data)
            except Exception:
                pass
        self.pos_emb.weight.data[0, :] = 0
        self.item_emb.weight.data[0, :] = 0

    def _log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.hidden_units ** 0.5

        positions = np.tile(
            np.arange(1, log_seqs.shape[1] + 1),
            [log_seqs.shape[0], 1]
        )
        positions = torch.LongTensor(positions).to(log_seqs.device)
        positions = positions * (log_seqs != 0).long()
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = ~torch.tril(
            torch.ones((seqs.shape[1], seqs.shape[1]),
                       dtype=torch.bool, device=seqs.device)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_out, _ = self.attention_layers[i](x, x, x,
                                                       attn_mask=timeline_mask)
                seqs = seqs + mha_out
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs))
            else:
                mha_out, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = self.attention_layernorms[i](seqs + mha_out)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs))

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self._log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self._log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  Data Utilities（from SASRec.utils.py，适配版）
# ─────────────────────────────────────────────────────────────────────────────

def data_partition_from_parquet(item_interactions):
    """
    从 item_interactions DataFrame 构建 train/val/test 划分。

    Args:
        item_interactions: DataFrame with ['uid', 'item_id', 'timestamp']

    Returns:
        [user_train, user_valid, user_test, usernum, itemnum]
    """
    # 用 groupby 向量化构建（比 iterrows 快 100 倍）
    grouped = item_interactions.sort_values('timestamp').groupby('uid')['item_id'].apply(list)
    User = grouped.to_dict()
    user_ids = list(User.keys())
    usernum = len(user_ids)

    user_train = {}
    user_valid = {}
    user_test = {}
    itemnum = 0

    for uid in user_ids:
        items = User[uid]
        itemnum = max(itemnum, max(items))
        if len(items) < 4:
            user_train[uid] = items
            user_valid[uid] = []
            user_test[uid] = []
        else:
            user_train[uid] = items[:-2]
            user_valid[uid] = [items[-2]]
            user_test[uid] = [items[-1]]

    return [user_train, user_valid, user_test, user_ids, usernum, itemnum]


def random_neq(l, r, s):
    t = random.randint(l, r)
    while t in s:
        t = random.randint(l, r)
    return t


class WarpSampler:
    def __init__(self, User, user_ids, itemnum, batch_size=128, maxlen=50, n_workers=4):
        self.User = User
        self.user_ids = np.array(user_ids, dtype=np.int32)
        self.usernum = len(user_ids)
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.queue = __import__('queue').Queue(maxsize=n_workers * 10)
        self.processes = []
        for _ in range(n_workers):
            p = __import__('multiprocessing').Process(
                target=_sample_user,
                args=(self.User, self.user_ids, itemnum, batch_size, maxlen,
                      self.queue, random.randint(0, 2 ** 31))
            )
            p.daemon = True
            p.start()
            self.processes.append(p)

    def next_batch(self):
        return self.queue.get()

    def close(self):
        for p in self.processes:
            p.terminate()
            p.join()


def _sample_user(User, user_ids, itemnum, batch_size, maxlen, result_queue, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    user_ids = list(user_ids)   # numpy array → Python list for stable indexing
    usernum = len(user_ids)
    counter = 0
    while True:
        if counter % usernum == 0:
            random.shuffle(user_ids)
        batch = []
        for _ in range(batch_size):
            uid = user_ids[counter % usernum]
            while len(User[uid]) <= 1:
                uid = random.choice(user_ids)
            seq = np.zeros(maxlen, dtype=np.int32)
            pos = np.zeros(maxlen, dtype=np.int32)
            neg = np.zeros(maxlen, dtype=np.int32)
            nxt = User[uid][-1]
            idx = maxlen - 1
            ts = set(User[uid])
            for i in reversed(User[uid][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1:
                    break
            batch.append((uid, seq, pos, neg))
            counter += 1
        result_queue.put(list(zip(*batch)))


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, dataset, args, k=10):
    [train, valid, test, user_ids, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    HIT = 0.0
    valid_users = 0

    eval_users = random.sample(user_ids, min(10000, usernum))
    for u in eval_users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros(args.maxlen, dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = random.randint(1, itemnum)
            while t in rated:
                t = random.randint(1, itemnum)
            item_idx.append(t)

        with torch.no_grad():
            predictions = -model.predict(
                *[torch.LongTensor(l).to(args.device) for l in [[u], [seq], item_idx]]
            )[0].cpu().numpy()

        rank = predictions.argsort().argsort()[0].item()
        valid_users += 1
        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    return NDCG / valid_users, HIT / valid_users


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_sasrec(
    item_interactions,          # DataFrame: uid, item_id, timestamp
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.2,
    maxlen=50,
    num_epochs=200,
    batch_size=256,
    lr=0.001,
    device='cuda:0',
    eval_every=20,
    patience=3,
    save_dir=None,
):
    dataset = data_partition_from_parquet(item_interactions)
    [user_train, user_valid, user_test, user_ids, usernum, itemnum] = dataset

    print(f"  SASRec data: {usernum} users, {itemnum} items")
    avg_len = np.mean([len(user_train[u]) for u in user_train])
    print(f"  Avg seq length: {avg_len:.1f}")

    sampler = WarpSampler(user_train, user_ids, itemnum,
                          batch_size=batch_size, maxlen=maxlen)
    model = SASRec(
        usernum, itemnum,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        maxlen=maxlen,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SASRec params: {n_params:,}")

    adam = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_ndcg = 0.0
    no_improve = 0
    best_state = None

    class _DummyArgs:
        pass
    args_d = _DummyArgs()
    args_d.maxlen = maxlen
    args_d.device = device

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in tqdm(range(len(user_train) // batch_size),
                       ncols=60, desc=f"SASRec E{epoch}", leave=False):
            u, seq, pos, neg = sampler.next_batch()
            u = np.array(u)
            seq = np.array(seq)
            pos = np.array(pos)
            neg = np.array(neg)

            pos_logits, neg_logits = model(
                torch.LongTensor(u).to(device),
                torch.LongTensor(seq).to(device),
                torch.LongTensor(pos).to(device),
                torch.LongTensor(neg).to(device),
            )
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)

            indices = np.where(pos != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])

            adam.zero_grad()
            loss.backward()
            adam.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        if epoch % eval_every == 0 or epoch == num_epochs:
            model.eval()
            ndcg, hit = evaluate(model, dataset, args_d, k=10)
            print(f"  SASRec E{epoch:3d} | loss={avg_loss:.4f} | "
                  f"NDCG@10={ndcg:.4f} HIT@10={hit:.4f}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience and epoch >= num_epochs // 2:
                print(f"  Early stop at epoch {epoch}")
                break

    sampler.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, usernum, itemnum


def extract_cf_embeddings(
    model,
    itemnum,
    device='cuda:0',
):
    """从训练好的 SASRec 中提取 item embeddings（去掉 padding）。"""
    with torch.no_grad():
        item_embs = model.item_emb.weight.cpu().numpy()  # (itemnum+1, hidden)
    cf_embs = item_embs[1:]  # 去掉 padding (item_id 从 1 开始)
    item_ids = np.arange(1, itemnum + 1, dtype=np.uint32)
    return item_ids, cf_embs


def align_with_semantic_embeddings(
    cf_item_ids,
    cf_embs,
    semantic_npz_path,
):
    """
    将 CF embeddings 与 semantic embeddings.npz 对齐。

    有些 item 在 CF 模型中未出现（冷启动），这些 item 的 CF embedding
    用 CF embedding 的均值填充。

    Returns:
        aligned_cf_embs: (N_items, cf_dim) 与 embeddings.npz 顺序一致
    """
    sem_data = np.load(semantic_npz_path, allow_pickle=True)
    sem_item_ids = sem_data['item_ids']  # (N,)
    N = len(sem_item_ids)

    # 构建 item_id → CF embedding 的映射
    cf_map = {int(item_id): cf_embs[i]
              for i, item_id in enumerate(cf_item_ids)}

    cf_dim = cf_embs.shape[1]
    aligned = np.zeros((N, cf_dim), dtype=np.float32)

    # 均值填充
    cf_mean = cf_embs.mean(axis=0)
    n_found = 0
    n_missing = 0

    for i in range(N):
        sid = int(sem_item_ids[i])
        if sid in cf_map:
            aligned[i] = cf_map[sid]
            n_found += 1
        else:
            aligned[i] = cf_mean
            n_missing += 1

    print(f"  CF embedding alignment: {n_found}/{N} items found, "
          f"{n_missing}/{N} filled with mean")

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SASRec CF Embedding Extractor")
    parser.add_argument('--input', type=str, default=None,
                        help='Single parquet file path (uid, item_id, timestamp)')
    parser.add_argument('--input_combo', type=str, default=None,
                        help='Comma-separated list: likes,listens,multi_event')
    parser.add_argument('--data_dir', type=str,
                        default='../data/sequential-50m',
                        help='Directory containing parquet files')
    parser.add_argument('--embeddings', type=str,
                        default='../data/embeddings.npz',
                        help='Semantic embeddings npz path')
    parser.add_argument('--output', type=str,
                        default='../data/cf_embeddings.npz',
                        help='Output npz path')
    parser.add_argument('--dataset_name', type=str, default='likes')

    # SASRec 超参数
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ── 1. 加载/合并 interaction data ─────────────────────────────────────

    if args.input_combo:
        dfs = []
        for name in args.input_combo.split(','):
            fname = os.path.join(args.data_dir, f'{name.strip()}.parquet')
            print(f"Loading {fname} ...")
            pf = pq.ParquetFile(fname)
            table = pf.read(columns=['uid', 'timestamp', 'item_id'])

            uid_arr  = table['uid'].to_pylist()
            ts_arr   = table['timestamp'].to_pylist()
            item_arr = table['item_id'].to_pylist()
            n_per_row = [len(x) for x in item_arr]

            flat_ts   = np.fromiter(chain.from_iterable(ts_arr),   dtype=np.uint32)
            flat_item = np.fromiter(chain.from_iterable(item_arr), dtype=np.uint32)
            uid_exp   = np.repeat(np.array(uid_arr, dtype=np.uint32), n_per_row)

            df_sub = pd.DataFrame({
                'uid':       uid_exp,
                'timestamp': flat_ts,
                'item_id':   flat_item,
            })
            dfs.append(df_sub)
            print(f"  -> {len(df_sub):,} interactions from {table.num_rows:,} users")

        # 合并去重（同一用户对同一 item 取最早 interaction）
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['uid', 'item_id'], keep='first')
        print(f"  Combined: {len(df):,} interactions, "
              f"{df['uid'].nunique():,} users, {df['item_id'].nunique():,} items")

    elif args.input:
        print(f"Loading {args.input} ...")
        pf = pq.ParquetFile(args.input)
        table = pf.read(columns=['uid', 'timestamp', 'item_id'])
        uid_arr  = table['uid'].to_pylist()
        ts_arr   = table['timestamp'].to_pylist()
        item_arr = table['item_id'].to_pylist()
        n_per_row = [len(x) for x in item_arr]

        flat_ts   = np.fromiter(chain.from_iterable(ts_arr),   dtype=np.uint32)
        flat_item = np.fromiter(chain.from_iterable(item_arr), dtype=np.uint32)
        uid_exp   = np.repeat(np.array(uid_arr, dtype=np.uint32), n_per_row)

        df = pd.DataFrame({
            'uid':       uid_exp,
            'timestamp': flat_ts,
            'item_id':   flat_item,
        })
        print(f"  -> {len(df):,} interactions from {table.num_rows:,} users")
    else:
        raise ValueError("必须指定 --input 或 --input_combo")

    # 保留必要列并排序
    required_cols = ['uid', 'item_id', 'timestamp']
    df = df[required_cols].copy()
    df = df.sort_values(['uid', 'timestamp'])
    print(f"  Total interactions: {len(df)}")

    # ── 2. 训练 SASRec ──────────────────────────────────────────────────────
    print(f"\nTraining SASRec on {args.dataset_name} ...")
    t0 = time.time()
    model, usernum, itemnum = train_sasrec(
        df,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        maxlen=args.maxlen,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        eval_every=args.eval_every,
        patience=args.patience,
    )
    print(f"  SASRec training done in {time.time() - t0:.1f}s")

    # ── 3. 提取 CF embeddings ───────────────────────────────────────────────
    print("Extracting CF embeddings ...")
    cf_item_ids, cf_embs_raw = extract_cf_embeddings(model, itemnum, args.device)
    print(f"  Raw CF embeddings: {cf_embs_raw.shape}")

    # ── 4. 与 semantic embeddings 对齐 ─────────────────────────────────────
    print("Aligning with semantic embeddings ...")
    cf_embs_aligned = align_with_semantic_embeddings(
        cf_item_ids, cf_embs_raw, args.embeddings
    )
    print(f"  Aligned CF embeddings: {cf_embs_aligned.shape}")

    # ── 5. 保存 ─────────────────────────────────────────────────────────────
    np.savez(args.output,
             cf_embeddings=cf_embs_aligned,
             item_ids=np.load(args.embeddings)['item_ids'])
    print(f"\nSaved CF embeddings to: {args.output}")
    print(f"  Shape: {cf_embs_aligned.shape}, dtype: {cf_embs_aligned.dtype}")


if __name__ == "__main__":
    main()
