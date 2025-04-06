import random

def generate_nearby_states(goal_state, max_steps=5, num_states=5):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
    nearby_states = []

    for _ in range(num_states):
        state = [list(row) for row in goal_state]  # 初始为目标状态
        steps = random.randint(1, max_steps)       # 随机选择步数（1到max_steps）

        for _ in range(steps):
            # 找到 0 的位置
            zero_i, zero_j = next((i, j) for i, row in enumerate(state)
                                for j, val in enumerate(row) if val == 0)
            # 随机选择一个合法移动
            valid_moves = []
            for di, dj in directions:
                new_i, new_j = zero_i + di, zero_j + dj
                if 0 <= new_i < 3 and 0 <= new_j < 3:
                    valid_moves.append((di, dj))
            if not valid_moves:
                break
            di, dj = random.choice(valid_moves)
            new_i, new_j = zero_i + di, zero_j + dj
            # 交换 0 和相邻数字
            state[zero_i][zero_j], state[new_i][new_j] = state[new_i][new_j], state[zero_i][zero_j]

        nearby_states.append(state)  # 保存状态

    return nearby_states