# model/policy/SIDPolicy.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_regularization

class SIDPolicy_credit(nn.Module):
    """
    层级残差版 SID：
    forward 返回
      - 'sid_logits': [ (B,V1), (B,V2), ... ]   # 逐层条件（残差）得到
      - 'state_emb' : (B, d_model)
      - 'seq_emb'   : (B, H, d_model)
      - 'reg'       : scalar
    残差流程（可微）：
      context_0 = state_emb
      for l in 1..L:
        logits_l = head_l(context_{l-1})                 # (B, V_l)
        probs_l  = softmax(logits_l / tau)               # (B, V_l)
        e_l      = probs_l @ token_emb_l.weight          # (B, d_model)
        context_l = LayerNorm(context_{l-1} - e_l)       # 残差，供下一层使用
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--sasrec_n_layer', type=int, default=2)
        parser.add_argument('--sasrec_d_model', type=int, default=64)
        parser.add_argument('--sasrec_d_forward', type=int, default=128)
        parser.add_argument('--sasrec_n_head', type=int, default=4)
        parser.add_argument('--sasrec_dropout', type=float, default=0.1)
        parser.add_argument('--sid_levels', type=int, default=3)
            # 修改这里：直接输入一个整数，比如 64
        parser.add_argument('--sid_vocab_sizes', type=int, default=256,
                            help='每一层的vocab大小，将自动扩展为 [v, v, v]')
        parser.add_argument('--sid_temp', type=float, default=1.0, help='softmax 温度，>1 更平缓，<1 更尖锐')
        
        # SARA 专属推断参数
        parser.add_argument('--sara_eta', type=float, default=0.5, help='SARA 惩罚系数 η')
        parser.add_argument('--sara_layer_weights', type=str, default='0.05,0.25,0.70', help='各层前缀匹配权重')
        return parser

    def __init__(self, args, environment):
        super().__init__()
        # === 基本超参 ===
        self.n_layer   = args.sasrec_n_layer
        self.d_model   = args.sasrec_d_model
        self.n_head    = args.sasrec_n_head
        self.dropout   = args.sasrec_dropout
        self.sid_temp  = float(getattr(args, 'sid_temp', 1.0))

        # === 空间信息 ===
        self.n_item    = environment.action_space['item_id'][1]
        self.item_dim  = environment.action_space['item_feature'][1]
        self.maxlen    = environment.observation_space['history'][1]

        # 兼容旧字段
        self.state_dim  = self.d_model
        self.action_dim = self.d_model

        # === 编码器（SASRec 风格）===
        self.item_map  = nn.Linear(self.item_dim, self.d_model)
        self.pos_emb   = nn.Embedding(self.maxlen, self.d_model)
        self.emb_drop  = nn.Dropout(self.dropout)
        self.emb_norm  = nn.LayerNorm(self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=args.sasrec_d_forward,
            nhead=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layer)

        # 注册 buffer：位置索引 + 全局下三角 mask
        self.register_buffer('pos_idx', torch.arange(self.maxlen, dtype=torch.long), persistent=False)
        full = torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.register_buffer('attn_mask_full', ~full, persistent=False)

        # === SID 层级 ===
        self.sid_levels = int(args.sid_levels)
        
        # SARA 拦截参数
        self.sara_eta = getattr(args, 'sara_eta', 0.5)
        weights_str = getattr(args, 'sara_layer_weights', '0.05,0.25,0.70')
        self.sara_layer_weights = [float(w) for w in weights_str.split(',')]

        self.sid_vocab_sizes = [args.sid_vocab_sizes] * self.sid_levels

        # 每层分类头：context -> logits_l
        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, v) for v in self.sid_vocab_sizes])

        # 每层 codebook（token embedding）
        self.sid_token_embeds = nn.ModuleList([
            nn.Embedding(v, self.d_model) for v in self.sid_vocab_sizes
        ])

        # 每层一个 LayerNorm，用于 residual 稳定
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

    def get_sara_logits(self, feed_dict, pool_tokens=None, pool_phis=None):
        """
        SARA 张量拦截魔法：环境交互专用！
        在生成过程中，将生成的软概率 (C_curr) 与全局后悔池 (pool_tokens) 进行前缀匹配。
        若撞上黑名单，直接在 Logits 上施加惩罚 D。
        """
        enc = self.encode_state(feed_dict)
        context = enc['state_emb']
        B = context.shape[0]
        
        sid_logits = []
        context_list = [context]
        
        V_dim = self.sid_vocab_sizes[0]
        L = self.sid_levels
        tau = self.sid_temp
        
        # C_curr: 记录当前 Batch 已经生成的每层软概率前缀 [B, L, V]
        C_curr = torch.zeros(B, L, V_dim, device=context.device)
        
        # 将池子里的历史失败路径转为 One-Hot 张量供矩阵乘法使用
        if pool_tokens is not None and pool_tokens.shape[0] > 0:
            N = pool_tokens.shape[0]
            # pool_tokens: [N, L] -> B_pool: [N, L, V]
            B_pool = F.one_hot(pool_tokens, num_classes=V_dim).float()
        else:
            pool_tokens = None
            
        # 层级权重张量
        W = torch.tensor(self.sara_layer_weights, device=context.device)

        for l in range(L):
            logits_l = self.sid_heads[l](context)  # [B, V]
            
            # 如果后悔池里有东西，开始计算拦截惩罚
            if pool_tokens is not None:
                if l == 0:
                    # 预测第 0 层时，所有历史教训无条件处于"激活监控"状态
                    similarity_vec = torch.ones(B, N, device=context.device)
                else:
                    # 获取已生成的前缀 C 和池子里的前缀 B
                    C_slice = C_curr[:, :l, :]    # [B, l, V]
                    B_slice = B_pool[:, :l, :]    # [N, l, V]
                    
                    # 广播计算点乘匹配度 -> hit_matrix: [B, N, l]
                    hit_matrix = (C_slice.unsqueeze(1) * B_slice.unsqueeze(0)).sum(dim=-1)
                    
                    # 乘上对应的层级权重后求和 -> similarity_vec: [B, N]
                    W_active = W[:l].unsqueeze(0).unsqueeze(0)  # [1, 1, l]
                    similarity_vec = (hit_matrix * W_active).sum(dim=-1)
                
                # 计算激活惩罚度 (匹配度 * 炸弹威力) -> active_penalty: [B, N]
                active_penalty = similarity_vec * pool_phis.unsqueeze(0)
                
                # 将惩罚度映射到当前层的词表维度上 -> D: [B, V]
                B_curr_level = B_pool[:, l, :]  # [N, V]
                D = torch.matmul(active_penalty, B_curr_level)
                
                # 核心：在 Logits 上执行无情扣减
                logits_l = logits_l - self.sara_eta * D

            sid_logits.append(logits_l)
            
            # 用扣减后的 logits 算出新的软概率
            probs_l = F.softmax(logits_l / tau, dim=-1)
            
            # 将生成的软概率填入前缀记录中，供下一层循环使用
            C_curr[:, l, :] = probs_l
            
            # 步进 Context
            emb_tbl = self.sid_token_embeds[l].weight
            exp_emb = torch.matmul(probs_l, emb_tbl)
            context = self.sid_res_norms[l](context - exp_emb)
            context_list.append(context)
            
        return {
            'sid_logits': sid_logits,
            'context_list': torch.stack(context_list, dim=1),
            'output_seq': enc['output_seq']
        }

    def encode_state(self, feed_dict):
        """
        入: feed_dict['history_features'] -> (B, H, item_dim)
        出: {'output_seq': (B, H, d_model), 'state_emb': (B, d_model)}
        """
        hist = feed_dict['history_features']               # (B, H, item_dim)
        B, H, _ = hist.shape

        pos = self.pos_emb(self.pos_idx[:H])               # (H, d_model) → (B, H, d_model)
        pos = pos.unsqueeze(0).expand(B, H, -1)

        x   = self.item_map(hist)                          # (B, H, d_model)
        x   = self.emb_norm(self.emb_drop(x + pos))

        attn_mask = self.attn_mask_full[:H, :H]            # (H, H)
        out_seq   = self.transformer(x, mask=attn_mask)    # (B, H, d_model)
        state     = out_seq[:, -1, :]                      # (B, d_model)
        return {'output_seq': out_seq, 'state_emb': state}


    def forward(self, feed_dict):
        enc = self.encode_state(feed_dict)
        context = enc['state_emb']  # (B, d_model)

        sid_logits = []
        context_list = [context]  # 保存初始 context

        tau = self.sid_temp if self.sid_temp is not None else 1.0

        for l in range(self.sid_levels):
            logits_l = self.sid_heads[l](context)  # (B, V_l)
            sid_logits.append(logits_l)

            probs_l = torch.softmax(logits_l / tau, dim=-1)

            emb_tbl = self.sid_token_embeds[l].weight  # (V_l, d_model)
            exp_emb = probs_l @ emb_tbl  # (B, d_model)

            # 残差更新
            context = self.sid_res_norms[l](context - exp_emb)

            # 保存每一步的 context
            context_list.append(context)

        # === 新增 ===
        # 变成 tensor (B, L+1, d_model)，方便直接存到 buffer
        context_tensor = torch.stack(context_list, dim=1)

        reg = get_regularization(self.item_map, self.transformer)
        return {
            'sid_logits': sid_logits,           # list[(B, V_l)]
            'context_list': context_tensor,     # tensor (B, L+1, d_model)
            # 'state_emb' : enc['state_emb'],
            'seq_emb': enc['output_seq'],
            'reg': reg
        }


