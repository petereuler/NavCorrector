import numpy as np

def generate_trajectory_2d(init_p, init_h, delta_l_list, delta_h_list):
    """
    [修改] 使用步长（Δl）与绝对航向角（ψ）在平面内重建轨迹
    不再累加航向变化，直接使用预测的绝对航向
    """
    trajectory = [init_p.copy()]
    current_p = init_p.copy()
    # [修改] 对于绝对航向，不再需要维护current_h

    delta_l_list = np.squeeze(delta_l_list)      # (N, 1) → (N,)
    delta_h_list = np.squeeze(delta_h_list)  # (N, 1) → (N,) 现在这是绝对航向

    for dl, abs_h in zip(delta_l_list, delta_h_list):
        # 若仍是 array([x])，则 item() 提取纯标量
        if hasattr(dl, 'item'):
            dl = dl.item()
        if hasattr(abs_h, 'item'):
            abs_h = abs_h.item()

        # [修改] 直接使用绝对航向，不再累加
        dx = dl * np.cos(abs_h)
        dy = dl * np.sin(abs_h)

        current_p = current_p + np.array([dx, dy])
        trajectory.append(current_p.copy())

    return np.array(trajectory)

def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    """
    使用四元数增量在6D空间中重建轨迹
    """
    cur_p = np.array(init_p)
    cur_q = np.array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for y_delta_p, y_delta_q in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(cur_q.rotation_matrix, y_delta_p.T).T
        cur_q = cur_q * Quaternion(y_delta_q)
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))
    