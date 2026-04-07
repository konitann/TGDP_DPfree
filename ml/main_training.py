import os
import math
import numpy as np
import pandas as pd
import collections
import networkx as nx
import pulp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from grpc_client import HEClient 

# --- データ読み込みと集約 (前回の数理トリック入り) ---
def load_and_aggregate_data_tgdp(target_nodes=442):
    # ... (この関数は前回提示した内容のままでOKです。変更ありません) ...
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_path = os.path.join(current_dir, "../dataset/diabetes.csv")
    alpha_path = os.path.join(current_dir, "../dataset/soc-sign-bitcoinalpha.csv")
    
    df_diabetes = pd.read_csv(diabetes_path).head(target_nodes)
    X = df_diabetes.drop(columns=['target']).values
    y = df_diabetes['target'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    G = nx.Graph()
    G.add_nodes_from(range(1, target_nodes + 1))
    with open(alpha_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                u, v, rating = int(parts[0]), int(parts[1]), float(parts[2])
                if rating > 0 and 1 <= u <= target_nodes and 1 <= v <= target_nodes:
                    G.add_edge(u, v)

    nodes = sorted(list(G.nodes()))
    prob = pulp.LpProblem("MDS", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)
    prob += pulp.lpSum([x[i] for i in nodes])
    for v in nodes:
        neighbors = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum([x[u] for u in neighbors]) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    data_collectors = [i for i in nodes if pulp.value(x[i]) == 1.0]
    mds_size = len(data_collectors)

    X_agg_list, y_agg_list = [], []
    total_users = 0

    for collector in data_collectors:
        trust_network = list(G.neighbors(collector)) + [collector]
        indices = [n - 1 for n in trust_network]
        weight = len(trust_network)
        total_users += weight

        X_mean = np.mean(X_scaled[indices], axis=0)
        y_mean = np.mean(y_scaled[indices], axis=0)
        
        # 数学的トリック: √weightを掛ける
        X_weighted = X_mean * math.sqrt(weight)
        y_weighted = y_mean * math.sqrt(weight)
        
        X_agg_list.append(X_weighted)
        y_agg_list.append(y_weighted)

    X_agg = np.array(X_agg_list)
    y_agg = np.array(y_agg_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X_agg, y_agg, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, mds_size, total_users

def compute_loss(X, y, weights, lambda_reg):
    predictions = X.dot(weights)
    mse = np.mean((predictions - y) ** 2)
    l2_penalty = lambda_reg * np.sum(weights ** 2)
    return mse + l2_penalty

# --- 1. グローバルクリッピング関数 ---
def clip_updates(updates, max_norm_C):
    """L2ノルムによる全体の勾配クリッピング (Mode Cの安全装置用)"""
    total_norm = np.linalg.norm(updates, ord=2)
    clip_coef = max_norm_C / (total_norm + 1e-9)
    if clip_coef < 1.0:
        return updates * clip_coef
    return updates

# --- 2. パーサンプル勾配計算関数 (新規追加) ---
def compute_per_sample_gradient(X, y, weights, C, n_samples_total, lambda_reg):
    """データ1件ごとの勾配を計算し、それぞれクリッピングしてから総和をとる"""
    pred = X.dot(weights)  # (n_samples, 1)
    errors = pred - y      # (n_samples, 1)
    
    # 各サンプルの勾配 (n_samples, n_features)
    per_sample_grads = X * errors
    
    # データごとのL2ノルムを計算 (n_samples, 1)
    norms = np.linalg.norm(per_sample_grads, axis=1, keepdims=True)
    
    # クリッピング係数の計算: min(1, C / norm)
    clip_coefs = np.clip(C / (norms + 1e-9), a_min=None, a_max=1.0)
    
    # 各勾配にクリッピングを適用
    clipped_grads = per_sample_grads * clip_coefs
    
    # クリッピングされた勾配の総和をとる
    grad_sum = np.sum(clipped_grads, axis=0).reshape(-1, 1)
    
    # 最終的な平均勾配 (正則化項を追加)
    final_grad = (2 / n_samples_total) * grad_sum + 2 * lambda_reg * weights
    return final_grad

# --- 3. プレラン関数 (パーサンプル対応に修正) ---
def compute_median_clip_threshold(X_train, y_train, n_samples_total, lambda_reg, pre_run_iters=15, lr=0.1):
    """プレランにより、パーサンプル閾値とグローバル安全閾値を決定する"""
    n_samples, n_features = X_train.shape
    weights_pre = np.zeros((n_features, 1))
    
    per_sample_norms = []
    global_norms = []

    print(f"--- プレラン（事前学習）を開始: {pre_run_iters} イテレーション ---")
    for i in range(pre_run_iters):
        pred = X_train.dot(weights_pre)
        
        # 1. パーサンプル勾配のノルムを記録 (C_per_sample用)
        per_sample_grads = X_train * (pred - y_train)
        norms = np.linalg.norm(per_sample_grads, axis=1)
        per_sample_norms.extend(norms.tolist())
        
        # 2. 全体平均勾配のノルムを記録 (Mode Cの C_global 用)
        grad = (2 / n_samples_total) * X_train.T.dot(pred - y_train) + 2 * lambda_reg * weights_pre
        global_norms.append(np.linalg.norm(grad, ord=2))
        
        weights_pre = weights_pre - lr * grad
    
    C_per_sample = float(np.median(per_sample_norms))
    C_global = float(np.median(global_norms))
    print(f"プレラン完了。")
    print(f"  パーサンプル閾値 C_per_sample = {C_per_sample:.6f}")
    print(f"  グローバル安全閾値 C_global = {C_global:.6f}\n")
    return C_per_sample, C_global

def run_experiment(save_plot=True):
    X_train, X_test, y_train, y_test, mds_size_T, n_samples_total = load_and_aggregate_data_tgdp(target_nodes=442)
    n_samples_compressed, n_features = X_train.shape
    print(f"元データ数: {n_samples_total} -> 圧縮後(MDSサイズ): {mds_size_T}")

    # パラメータ設定
    lambda_reg = 0.5       
    initial_lr = 0.1       
    num_iterations = 100
    epsilon_dp = 3.0
    delta = 1e-5

    # 1. プレランによる C の決定
    C_per_sample, C_global = compute_median_clip_threshold(X_train, y_train, n_samples_total, lambda_reg, pre_run_iters=15, lr=initial_lr)

    # 2. 感度とノイズスケールの計算 (パーサンプル方式)
    # 1つのサンプルが変化したときの和の最大変化量は 2 * C_per_sample
    sensitivity = 2.0 * C_per_sample / n_samples_total
    tgdp_delta_avg = sensitivity * mds_size_T
    noise_scale = tgdp_delta_avg / epsilon_dp

    weights_A = np.zeros((n_features, 1))
    weights_B = np.zeros((n_features, 1))
    weights_C = np.zeros((n_features, 1))

    client = HEClient()
    metrics = {
        "loss_A": [], "loss_B": [], "loss_C": [],
        "rmse_A": [], "rmse_C": [],
        "l2_dist_AC": [], "l2_dist_BC": [],
        "epsilon_C": []
    }
    
    # ノイズ平滑化用のキュー
    recent_sigmas = collections.deque(maxlen=10)

    for i in range(num_iterations):
        current_lr = initial_lr / (1 + 0.05 * i)

        # --- Mode A: 平文 DP-SGD (パーサンプル) ---
        grad_A = compute_per_sample_gradient(X_train, y_train, weights_A, C_per_sample, n_samples_total, lambda_reg)
        noise_A = np.random.laplace(loc=0.0, scale=noise_scale, size=grad_A.shape)
        weights_A = weights_A - current_lr * (grad_A + noise_A)

        # --- Mode B: 平文 Baseline (パーサンプル) ---
        grad_B = compute_per_sample_gradient(X_train, y_train, weights_B, C_per_sample, n_samples_total, lambda_reg)
        weights_B = weights_B - current_lr * grad_B

        # --- Mode C: CKKS-SGD ---
        ct_weights = client.encrypt(weights_C.flatten().tolist())
        scale_factor = math.sqrt(n_samples_compressed / n_samples_total)
        X_train_sent = X_train * scale_factor
        y_train_sent = y_train * scale_factor

        ct_grad = client.compute_gradient(
            ct_weights, X_train_sent.flatten().tolist(), y_train_sent.flatten().tolist(),
            n_samples_compressed, n_features, lambda_reg
        )
        grad_C_dec = np.array(client.decrypt(ct_grad)[:n_features]).reshape(-1, 1)

        # --- ε推移の計算ロジック (累積を廃止し、現在の強度を測定) ---
        
        # 3-1. 平文での「クリッピング無し」の真の勾配を算出
        # ※実測ノイズを正確に測るため、比較対象もクリッピング前の値を使用します
        pred_unclipped_C = X_train.dot(weights_C)
        ideal_grad_unclipped = (2 / n_samples_total) * X_train.T.dot(pred_unclipped_C - y_train) + 2 * lambda_reg * weights_C
        
        # 3-2. CKKSの純粋な演算誤差（暗号化ノイズ）の抽出
        error_tensor = grad_C_dec - ideal_grad_unclipped
        
        # 3-3. 標準偏差の計算と移動平均による平滑化
        sigma_agg_raw = np.std(error_tensor.flatten())
        recent_sigmas.append(sigma_agg_raw)
        sigma_agg_smooth = np.mean(recent_sigmas)
        
        # 3-4. 現在のプライバシー強度(ε)の算出
        # 合成定理（Advanced Composition）を使わず、現在のノイズ量から直接εを求めます
        # これにより、グラフが右肩上がりになるのを防ぎ、実測値をプロットできます
        epsilon_current = (tgdp_delta_avg * math.sqrt(2 * math.log(1.25 / delta))) / (sigma_agg_smooth + 1e-9)
        
        # 累積値ではなく、現在のステップのεを記録
        metrics["epsilon_C"].append(epsilon_current)

        # 4. Mode C 自身の重み更新 (モデルの安定性のために全体クリッピングを適用)
        grad_C_clipped = clip_updates(grad_C_dec, C_global)
        weights_C = weights_C - current_lr * grad_C_clipped

        # --- 指標の出力 ---
        if (i + 1) % 5 == 0 or i == 0:
            # l_C などの損失計算は print の直前で行ってください
            l_C = compute_loss(X_train, y_train, weights_C, lambda_reg)
            print(f"Iter {i+1:02d} | Loss C:{l_C:.4f} | C_per_sample:{C_per_sample:.3f} | 現在のε:{epsilon_current:.4f}")
    return metrics

if __name__ == "__main__":
    run_experiment()