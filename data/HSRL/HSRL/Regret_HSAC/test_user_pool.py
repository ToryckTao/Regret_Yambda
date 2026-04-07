import torch

# ==========================================
# 测试：用户独立后悔池
# ==========================================
def test_user_regret_pool():
    B = 4  # batch_size (用户数)
    N = 10  # 池大小
    L = 4   # HAC 层数
    
    # 模拟用户后悔池
    user_regret_pool = torch.full((B, N, L + 1), -1, dtype=torch.long)
    
    print("=== 初始化 ===")
    print(f"user_regret_pool shape: {user_regret_pool.shape}")
    print(f"所有 -1: { (user_regret_pool == -1).all() }")
    
    # 模拟一些后悔事件
    # 用户 0 在 slot 0 加入一个后悔: sid=[1,2,3,4], phi=50
    user_regret_pool[0, 0, :L] = torch.tensor([1, 2, 3, 4])
    user_regret_pool[0, 0, L] = 50
    
    # 用户 1 在 slot 1 加入一个后悔
    user_regret_pool[1, 1, :L] = torch.tensor([5, 6, 7, 8])
    user_regret_pool[1, 1, L] = 80
    
    print("\n=== 加入后悔后 ===")
    print(f"用户 0: {user_regret_pool[0]}")
    print(f"用户 1: {user_regret_pool[1]}")
    
    # 模拟同步到 Facade
    pool_list = []
    phi_list = []
    
    for b in range(B):
        pool_b = user_regret_pool[b]  # (N, L+1)
        valid_mask = pool_b[:, -1] != -1
        if valid_mask.sum() > 0:
            tokens = pool_b[valid_mask, :L]
            phis = pool_b[valid_mask, L].float() / 100
            pool_list.append(tokens)
            phi_list.append(phis)
    
    if len(pool_list) > 0:
        all_tokens = torch.cat(pool_list, dim=0)
        all_phis = torch.cat(phi_list, dim=0)
        print(f"\n=== 同步到 Facade ===")
        print(f"pool_tokens: {all_tokens}")
        print(f"pool_phis: {all_phis}")
    
    # 测试：用户 0 新增一个后悔
    # 找空槽
    pool_0 = user_regret_pool[0]
    empty_slots = (pool_0[:, -1] == -1).nonzero(as_tuple=True)[0]
    print(f"\n用户 0 的空槽: {empty_slots}")
    
    # 加入
    slot_idx = empty_slots[0]
    user_regret_pool[0, slot_idx, :L] = torch.tensor([9, 10, 11, 12])
    user_regret_pool[0, slot_idx, L] = 30
    
    print(f"用户 0 更新后: {user_regret_pool[0]}")
    
    print("\n=== 测试通过 ===")

if __name__ == "__main__":
    test_user_regret_pool()
