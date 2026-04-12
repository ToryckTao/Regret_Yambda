import ast

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import padding_and_clip


class YambdaFeatureCacheReader(Dataset):
    """
    轻量版 Yambda reader。

    输入：
    - 03_split_data.py 产出的 train / val / test TSV
    - 05_train_user_response.py 产出的 feature_cache_npz

    输出：
    - 与 YambdaDataReader 兼容的 batch 字段
    """

    @staticmethod
    def parse_data_args(parser):
        """输入：argparse parser。输出：追加参数后的 parser。"""
        parser.add_argument("--train_file", type=str, required=True)
        parser.add_argument("--val_file", type=str, default="")
        parser.add_argument("--test_file", type=str, default="")
        parser.add_argument("--feature_cache_npz", type=str, required=True)
        parser.add_argument("--max_seq_len", type=int, default=50)
        parser.add_argument("--max_train_rows", type=int, default=200000)
        parser.add_argument("--max_val_rows", type=int, default=50000)
        parser.add_argument("--data_separator", type=str, default="\t")
        parser.add_argument("--n_worker", type=int, default=0)
        parser.add_argument("--n_item", type=int, default=0)
        return parser

    def __init__(self, args):
        self.phase = "train"
        self.max_seq_len = int(args.max_seq_len)
        self.n_worker = int(getattr(args, "n_worker", 0))
        self.data_separator = getattr(args, "data_separator", "\t")
        self._read_data(args)
        self._read_feature_cache(args.feature_cache_npz)
        self.item_vec_size = int(self.features.shape[1]) if self.features.size > 0 else int(getattr(args, "embedding_dim", 128))
        self.portrait_len = 0
        self.n_item = int(getattr(args, "n_item", 0)) or int(self.dense_ids.max(initial=0))

    def _read_data(self, args):
        """输入：args。输出：填充 self.data。"""
        train_nrows = None if int(getattr(args, "max_train_rows", 0)) <= 0 else int(args.max_train_rows)
        val_nrows = None if int(getattr(args, "max_val_rows", 0)) <= 0 else int(args.max_val_rows)
        self.data = {
            "train": pd.read_table(args.train_file, sep=self.data_separator, engine="python", nrows=train_nrows),
        }
        self.data["val"] = (
            pd.read_table(args.val_file, sep=self.data_separator, engine="python", nrows=val_nrows)
            if len(getattr(args, "val_file", "")) > 0
            else self.data["train"]
        )
        self.data["test"] = (
            pd.read_table(args.test_file, sep=self.data_separator, engine="python", nrows=val_nrows)
            if len(getattr(args, "test_file", "")) > 0
            else self.data["val"]
        )

    def _read_feature_cache(self, feature_cache_npz):
        """输入：feature_cache_npz。输出：dense_id -> feature 的轻量查表结构。"""
        cache = np.load(feature_cache_npz)
        self.dense_ids = cache["dense_ids"].astype(np.int64)
        self.features = cache["features"].astype(np.float32)
        self.id_to_row = {int(iid): idx for idx, iid in enumerate(self.dense_ids.tolist())}
        self.zero_feature = np.zeros(self.features.shape[1], dtype=np.float32)

    def set_phase(self, phase):
        """输入：train / val / test。输出：切换当前 phase。"""
        assert phase in ["train", "val", "test"]
        self.phase = phase

    def __len__(self):
        return len(self.data[self.phase])

    def __getitem__(self, idx):
        """
        输入：样本下标。
        输出：与 YambdaDataReader 兼容的一条 record。
        """
        row = self.data[self.phase].iloc[idx]
        exposure = [int(row["target_dense_item_id"])]
        history = self._parse_list_cell(row["user_mid_history"])
        history_feedback = self._parse_list_cell(row["user_click_history"], as_float=True)
        hist_length = min(len(history), self.max_seq_len)
        history = padding_and_clip(history, self.max_seq_len)
        history_feedback = padding_and_clip(history_feedback, self.max_seq_len)
        feedback = [float(row["user_clicks"])]

        record = {
            "timestamp": int(row["sequence_id"]),
            "exposure": np.asarray(exposure, dtype=np.int64),
            "exposure_features": self.get_item_list_meta(exposure).astype(float),
            "feedback": np.asarray(feedback, dtype=np.float32),
            "history": np.asarray(history, dtype=np.int64),
            "history_features": self.get_item_list_meta(history).astype(float),
            "history_length": hist_length,
            "history_mask": np.asarray(padding_and_clip([1] * hist_length, self.max_seq_len), dtype=np.float32),
            "user_profile": np.zeros(1, dtype=np.float32),
        }
        return record

    @staticmethod
    def _parse_list_cell(cell, as_float=False):
        """输入：TSV 的 list 字符串。输出：Python list。"""
        if isinstance(cell, list):
            value = cell
        elif isinstance(cell, str):
            value = [] if cell == "" else ast.literal_eval(cell)
        elif pd.isna(cell):
            value = []
        else:
            value = [cell]
        if as_float:
            return [float(x) for x in value]
        return [int(x) for x in value]

    def get_item_list_meta(self, iid_list, from_idx=False):
        """
        输入：
        - iid_list: dense item id 列表

        输出：
        - item feature 矩阵，形状 [len(iid_list), item_vec_size]
        """
        features = []
        for iid in iid_list:
            iid = int(iid)
            if iid == 0:
                features.append(self.zero_feature)
                continue
            row_idx = self.id_to_row.get(iid)
            features.append(self.features[row_idx] if row_idx is not None else self.zero_feature)
        return np.asarray(features, dtype=np.float32)

    def get_statistics(self):
        """输入：无。输出：环境和模型初始化需要的统计信息。"""
        return {
            "length": len(self),
            "n_item": self.n_item,
            "item_vec_size": self.item_vec_size,
            "user_portrait_len": self.portrait_len,
            "max_seq_len": self.max_seq_len,
            "n_feedback": 2,
        }
