"""
SARA 后悔池机制超参数配置

所有与 Regret Pool 相关的超参数统一在这里管理
"""

# ==================== 奖励函数参数 ====================
# r_t = ω_listen * played_ratio + ω_like * I_like - ω_dislike * I_dislike
OMEGA_LISTEN = 1.0      # 播放权重
OMEGA_LIKE = 5.0       # 喜欢权重
OMEGA_DISLIKE = 1.0    # 不喜欢权重

# ==================== 后悔信号参数 ====================
# ψ_t = -λ_unlike * I_unlike + λ_undislike * I_undislike
# Φ = γ^Δt * ψ_t
LAMBDA_UNLIKE = 1.0    # 后悔惩罚强度 (like -> dislike)
LAMBDA_UNDISLIKE = 0.5 # 取消后悔奖励强度 (dislike -> neutral)
REGRET_GAMMA = 0.9    # 时间衰减因子

# ==================== 后悔池参数 ====================
REGRET_POOL_SIZE = 20     # 每个用户的后悔池最大容量 N
REGRET_POOL_DROP = True   # 是否支持动态更新（满了之后丢弃最老的）

# ==================== 后悔池推理参数 ====================
# p_θ(z_l=k | c_{l-1}) = softmax(W_l c_{l-1} + η · D(z_l=k))
REGRET_PENALTY_WEIGHT = 0.1   # η: 惩罚强度系数

# ==================== W 层权重 (固定) ====================
# W = [w1, w2, w3, ...] 对应各层的权重
# 当前层 l 的掩码: [1,1,...,1,0,0,...] (l个1)
# W_l = mask ⊙ W (Hadamard积)
W_LEVEL_WEIGHTS = [0.05, 0.25, 0.70]  # L=3层，对应 w1, w2, w3

# ==================== 相似度计算参数 ====================
SIMILARITY_TYPE = 'token_overlap'  # 'token_overlap' | 'embedding_cosine'
# token_overlap: 计算前缀与后悔路径的 token 重叠率
# embedding_cosine: 需要 token embeddings，计算余弦相似度
