"""
RegretPool: 存储用户后悔记录的数据结构
用于 SARA 的 Regret-Aware Policy Generation

存储结构:
- tokens: (max_size, n_levels) 存储每个后悔物品的SID token序列
- phis: (max_size,) 存储每个后悔物品的惩罚值 Φ = γ^Δt * ψ_t
"""

import torch
import torch.nn as nn


class RegretPool(nn.Module):
    """
    后悔池模块
    - 初始化时从用户历史挖掘"喜欢→取消"的转折点
    - 推理时计算当前层各token的惩罚值
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--regret_pool_size', type=int, default=20,
                            help='后悔池大小 N')
        parser.add_argument('--regret_penalty_weight', type=float, default=0.5,
                            help='推理时惩罚系数 η')
        parser.add_argument('--regret_layer_weights', type=str, default='0.05,0.25,0.7',
                            help='层级权重 W=[w1,w2,w3]，用逗号分隔')
        parser.add_argument('--regret_gamma', type=float, default=0.9,
                            help='时间折扣因子 γ')
        return parser

    def __init__(self, args, n_levels=3, vocab_sizes=None):
        super().__init__()
        self.max_size = args.regret_pool_size
        self.n_levels = n_levels
        self.penalty_weight = args.regret_penalty_weight
        self.gamma = args.regret_gamma

        # 解析层级权重 W=[w1,w2,w3]
        layer_weights_str = getattr(args, 'regret_layer_weights', '0.05,0.25,0.7')
        self.layer_weights = torch.tensor(
            [float(w) for w in layer_weights_str.split(',')],
            dtype=torch.float32
        )
        # 确保长度匹配
        if len(self.layer_weights) < n_levels:
            self.layer_weights = torch.cat([
                self.layer_weights,
                torch.ones(n_levels - len(self.layer_weights))
            ])
        self.layer_weights = self.layer_weights[:n_levels]

        # vocab_sizes 用于独热编码（可选，当前实现用命中惩罚）
        self.vocab_sizes = vocab_sizes or [256] * n_levels

        # 存储结构
        # tokens: (max_size, n_levels) - 存储SID token序列
        self.register_buffer('tokens', torch.zeros(self.max_size, self.n_levels, dtype=torch.long))
        # phis: (max_size,) - 存储惩罚值 Φ
        self.register_buffer('phis', torch.zeros(self.max_size, dtype=torch.float32))

        self.current_size = 0  # 当前存储数量

    def reset(self):
        """重置池子（每个episode开始时）"""
        self.tokens.zero_()
        self.phis.zero_()
        self.current_size = 0

    def add(self, sid_tokens, phi):
        """
        添加一条后悔记录（使用FIFO替换）

        Args:
            sid_tokens: list或tensor, 长度n_levels, 每个元素是token ID
            phi: float, 惩罚值 Φ = γ^Δt * ψ_t
        """
        if not isinstance(sid_tokens, torch.Tensor):
            sid_tokens = torch.tensor(sid_tokens, dtype=torch.long)

        # 找到插入位置（轮转）
        idx = self.current_size % self.max_size if self.current_size < self.max_size else 0
        self.tokens[idx] = sid_tokens[:self.n_levels]
        self.phis[idx] = phi

        if self.current_size < self.max_size:
            self.current_size += 1

    def init_from_history(self, history_item_ids, history_feedback, item2sid):
        """
        从用户历史初始化后悔池
        扫描"喜欢→取消"的转折点

        Args:
            history_item_ids: list, 用户历史物品ID序列
            history_feedback: list, 对应反馈 (+1喜欢, -1取消, 0无反馈)
            item2sid: dict, 物品ID -> SID序列的映射
        """
        self.reset()

        # 找到所有"喜欢→取消"的转折点
        regret_items = []  # (item_id, 喜欢的时间, 取消的时间)

        for t in range(1, len(history_feedback)):
            # 找到取消的时间点
            if history_feedback[t] == -1:  # 当前是取消
                # 回溯找到最近的喜欢时间
                for prev_t in range(t - 1, -1, -1):
                    if history_feedback[prev_t] == 1:  # 找到喜欢
                        item_id = history_item_ids[t]
                        if item_id in item2sid:
                            delta_t = t - prev_t  # 时间差
                            regret_items.append((item_id, delta_t))
                        break

        # 存入后悔池
        for item_id, delta_t in regret_items[:self.max_size]:
            sid_tokens = item2sid[item_id]  # (n_levels,)
            phi = (self.gamma ** delta_t) * 1.0  # ψ_t 暂时设为1.0（后面环境会传具体的）
            self.add(sid_tokens, phi)

    def compute_penalty(self, level, candidate_logits):
        """
        计算当前层的惩罚向量

        Args:
            level: int, 当前层级 (0-indexed)
            candidate_logits: (B, V_l) 原始logits

        Returns:
            penalty: (B, V_l) 惩罚值
        """
        B, V = candidate_logits.shape

        if self.current_size == 0:
            return torch.zeros_like(candidate_logits)

        # 获取后悔池中第level层的token (current_size,)
        pool_tokens = self.tokens[:self.current_size, level]
        pool_phis = self.phis[:self.current_size]

        # 层级累积权重: W[l] = sum(w_1 to w_l)
        # 例如 level=0 -> [1,0,0], level=1 -> [1,1,0], level=2 -> [1,1,1]
        level_cum_weights = torch.cumsum(self.layer_weights, dim=0)  # (n_levels,)
        current_level_weight = level_cum_weights[level]  # 当前层的累积权重

        # 计算惩罚: 对于每个候选token v
        # penalty[b, v] = sum(Φ_j * W[l] if pool_tokens[j] == v)

        # 扩展维度用于广播
        # pool_tokens: (1, current_size) -> (V, current_size)
        # candidate v: (V, 1)
        # 比较: (V, current_size) == pool_token -> (V, current_size)

        pool_tokens_exp = pool_tokens.unsqueeze(0).expand(V, -1)  # (V, current_size)
        v_range = torch.arange(V, device=candidate_logits.device).unsqueeze(1)  # (V, 1)
        matches = (pool_tokens_exp == v_range).float()  # (V, current_size)

        # 惩罚 = matches * phis -> sum over current_size -> (V,)
        penalty_per_v = torch.matmul(matches, pool_phis)  # (V,)

        # 应用层级权重
        penalty_per_v = penalty_per_v * current_level_weight

        # 扩展到batch: (V,) -> (B, V)
        penalty = penalty_per_v.unsqueeze(0).expand(B, -1)

        return penalty

    def get_regret_info(self):
        """
        获取当前后悔池信息（用于debug或可视化）
        """
        return {
            'size': self.current_size,
            'tokens': self.tokens[:self.current_size].cpu().numpy(),
            'phis': self.phis[:self.current_size].cpu().numpy(),
            'layer_weights': self.layer_weights.cpu().numpy()
        }


class RegretPoolManager:
    """
    管理所有用户的后悔池
    为每个用户维护一个独立的RegretPool实例
    """

    def __init__(self, args, n_users, n_levels=3, vocab_sizes=None):
        self.n_users = n_users
        self.n_levels = n_levels
        self.args = args

        # 为每个用户创建一个RegretPool
        # 注意：这里不直接存储，因为用户数可能很大
        # 使用字典按需创建
        self.pools = {}

        # 公共的item2sid映射（所有用户共享）
        self.item2sid = None

    def set_item2sid(self, item2sid):
        """设置物品ID到SID的映射"""
        self.item2sid = item2sid

    def get_pool(self, user_id):
        """获取指定用户的后悔池"""
        if user_id not in self.pools:
            self.pools[user_id] = RegretPool(
                self.args,
                n_levels=self.n_levels
            )
        return self.pools[user_id]

    def init_user_pool(self, user_id, history_item_ids, history_feedback):
        """初始化指定用户的后悔池"""
        if self.item2sid is None:
            raise ValueError("item2sid not set, call set_item2sid first")

        pool = self.get_pool(user_id)
        pool.init_from_history(history_item_ids, history_feedback, self.item2sid)
        return pool

    def add_regret(self, user_id, sid_tokens, phi):
        """为指定用户添加后悔记录"""
        pool = self.get_pool(user_id)
        pool.add(sid_tokens, phi)

    def compute_penalty(self, user_id, level, candidate_logits):
        """计算指定用户当前层的惩罚"""
        pool = self.get_pool(user_id)
        return pool.compute_penalty(level, candidate_logits)
