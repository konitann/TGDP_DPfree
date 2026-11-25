import os
import math
import numpy as np
import pandas as pd
import collections
import networkx as nx
import pulp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from grpc_client import HEClient 

def compute_smape(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    return 100/len(true) * np.sum(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-9))

# MAPEの計算関数を追加
def compute_mape(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    # 0割りを防ぐために分母に微小値を加算
    return 100 * np.mean(np.abs((true - pred) / (true + 1e-9)))

# --- データ読み込みと集約 (純粋な平均＆データリーク対策版) ---
# --- データ読み込みと集約 (純粋な平均＆データリーク対策版) ---
def load_and_aggregate_data_tgdp(target_nodes=442):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_path = os.path.join(current_dir, "../dataset/diabetes.csv")
    alpha_path = os.path.join(current_dir, "../dataset/soc-sign-bitcoinalpha.csv")
    
    df_diabetes = pd.read_csv(diabetes_path).head(target_nodes)
    X = df_diabetes.drop(columns=['target']).values
    y = df_diabetes['target'].values.reshape(-1, 1)

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

        # ▼▼▼ 修正ポイント: sqrt(weight) を掛けずに純粋な平均のみにする ▼▼▼
        X_mean = np.mean(X[indices], axis=0)
        y_mean = np.mean(y[indices], axis=0)
        
        X_agg_list.append(X_mean)
        y_agg_list.append(y_mean)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    X_agg = np.array(X_agg_list)
    y_agg = np.array(y_agg_list)

    # 1. 学習用(Train)として「MDSで集約された安全なデータ(CKKS/TGDP用)」の80%
    X_train_agg, _, y_train_agg, _ = train_test_split(X_agg, y_agg, test_size=0.2, random_state=42)

    # 2. テスト用(Test)として「集約する前の個人の生データ」の20%
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Trainデータ(生データ)の基準でスケーラーを学習させる
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_base = scaler_X.fit_transform(X_train_raw)
    y_train_base = scaler_y.fit_transform(y_train_raw)
    
    # 4. 集約データとTestデータを変換する
    X_train_C = scaler_X.transform(X_train_agg)
    y_train_C = scaler_y.transform(y_train_agg)
    
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw)
    
    # 5. 回帰モデルが正しく空間をフィットできるようにバイアス項(定数1)を追加
    X_train_base = np.hstack([np.ones((X_train_base.shape[0], 1)), X_train_base])
    X_train_C = np.hstack([np.ones((X_train_C.shape[0], 1)), X_train_C])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # 平文用の生データ(Base)と、提案手法用の集約データ(C)を分けて返す
    return X_train_C, X_train_base, X_test, y_train_C, y_train_base, y_test, mds_size, total_users, scaler_y

def compute_loss(X, y, weights, lambda_reg):
    predictions = X.dot(weights)
    mse = np.mean((predictions - y) ** 2)
    # バイアス項（インデックス0）をL2正則化から除外
    l2_penalty = lambda_reg * np.sum(weights[1:] ** 2)
    return mse + l2_penalty

def compute_dp_gradient(X_batch, y_batch, weights, lambda_reg, C_global, noise_scale):
    batch_size = len(X_batch)
    batch_grad_sum = np.zeros_like(weights)
    
    # 1. サンプルごとに勾配を計算して個別にクリッピング
    for i in range(batch_size):
        x_i = X_batch[i:i+1]
        y_i = y_batch[i:i+1]
        
        # 正則化を含めない純粋なサンプル勾配
        grad_i = 2 * x_i.T.dot(x_i.dot(weights) - y_i)
        
        # サンプル単位でクリッピング
        grad_i_clipped = clip_updates(grad_i, C_global)
        batch_grad_sum += grad_i_clipped
        
    # 2. Cに比例したDPノイズの生成 (感度 = 2 * C_global)
    if noise_scale > 0:
        # ラプラスノイズの例（必要に応じてnp.random.normalに変更してください）
        dp_noise = np.random.laplace(loc=0.0, scale=noise_scale * (2 * C_global), size=weights.shape)
    else:
        dp_noise = np.zeros_like(weights)
        
    # 3. ノイズを加算して平均化
    grad_final = (batch_grad_sum + dp_noise) / batch_size
    
    # 4. 最後に正則化項を足す（バイアス項は正則化しない）
    penalty_grad = 2 * lambda_reg * weights
    penalty_grad[0] = 0.0
    
    return grad_final + penalty_grad

def clip_updates(updates, max_norm_C):
    total_norm = np.linalg.norm(updates, ord=2)
    clip_coef = max_norm_C / (total_norm + 1e-9)
    if clip_coef < 1.0:
        return updates * clip_coef
    return updates

def compute_standard_gradient(X_batch, y_batch, weights, lambda_reg):
    pred = X_batch.dot(weights)
    grad = (2 / len(X_batch)) * X_batch.T.dot(pred - y_batch)
    penalty_grad = 2 * lambda_reg * weights
    penalty_grad[0] = 0.0
    return grad + penalty_grad

# ▼▼▼ 引数から n_samples_total を削除し、n_samples で割るように修正したバージョン ▼▼▼
def compute_median_clip_threshold(X_train, y_train, lambda_reg, pre_run_iters=15, lr=0.1):
    n_samples, n_features = X_train.shape
    weights_pre = np.zeros((n_features, 1))
    
    per_sample_norms = []
    global_norms = []

    print(f"--- プレラン（事前学習）を開始: {pre_run_iters} イテレーション ---")
    for i in range(pre_run_iters):
        pred = X_train.dot(weights_pre)
        per_sample_grads = X_train * (pred - y_train)
        norms = np.linalg.norm(per_sample_grads, axis=1)
        per_sample_norms.extend(norms.tolist())
        
        # 修正: n_samples_total ではなく len(X_train) である n_samples で割る
        penalty_grad = 2 * lambda_reg * weights_pre
        penalty_grad[0] = 0.0
        grad = (2 / n_samples) * X_train.T.dot(pred - y_train) + penalty_grad
        global_norms.append(np.linalg.norm(grad, ord=2))
        weights_pre = weights_pre - lr * grad
    
    C_per_sample = float(np.median(per_sample_norms))
    C_global = float(np.median(global_norms))
    print(f"プレラン完了。")
    print(f"  パーサンプル閾値 C_per_sample = {C_per_sample:.6f}")
    print(f"  グローバル安全閾値 C_global = {C_global:.6f}\n")
    return C_per_sample, C_global

def run_experiment(save_plot=True):
    # 受け取る変数を変更
    X_train_C, X_train_B, X_test, y_train_C, y_train_B, y_test, mds_size_T, n_samples_total, scaler_y = load_and_aggregate_data_tgdp(target_nodes=442)
    n_samples_C, n_features = X_train_C.shape
    n_samples_B = X_train_B.shape[0]
    print(f"元データ数: {n_samples_total} -> 圧縮後(MDSサイズ): {mds_size_T}")

    lambda_reg = 0.1       # 0.5から0.1に戻して過学習対策を適度にする
    initial_lr = 0.1       
    num_iterations = 30
    epsilon_dp = 1.0
    batch_size = 16        # バッチサイズを少し大きめにして安定させる

    # 閾値は平文(Base)のデータで計算
    #C_per_sample, C_global = compute_median_clip_threshold(X_train_B, y_train_B, lambda_reg, pre_run_iters=15, lr=initial_lr)
    C_global = 0.01
    sensitivity_input = 2.0 
    noise_scale_input = sensitivity_input / epsilon_dp
    sensitivity_gradient = 2.0 * C_global / batch_size

    weights_A = np.zeros((n_features, 1))
    weights_B = np.zeros((n_features, 1))
    weights_C = np.zeros((n_features, 1))

    cumulative_noise_var = 0.0
    noise_X = np.random.laplace(loc=0.0, scale=noise_scale_input, size=X_train_C.shape)
    noise_X[:, 0] = 0.0 
    X_train_A = X_train_C + noise_X

    client = HEClient()
    
    metrics = {
        "loss_A": [], "loss_B": [], "loss_C": [],
        "rmse_A": [], "rmse_B": [], "rmse_C": [],
        "mae_A": [], "mae_B": [], "mae_C": [],
        "r2_A": [], "r2_B": [], "r2_C": [],
        "smape_A": [], "smape_B": [], "smape_C": [], 
        "mape_A": [], "mape_B": [], "mape_C": [], 
        "l2_dist_AB": [], "l2_dist_CB": [],
        "epsilon_C": []
    }
    epsilon_dp = 1.0
    noise_scale_grad = 1.0 / epsilon_dp  # 勾配注入用

    for i in range(num_iterations):
        current_lr = initial_lr / (1 + 0.01 * i)
        
        indices_B = np.random.choice(n_samples_B, batch_size, replace=False)
        X_batch_B = X_train_B[indices_B]
        y_batch_B = y_train_B[indices_B]
        
        actual_batch_C = min(batch_size, n_samples_C)
        indices_C = np.random.choice(n_samples_C, actual_batch_C, replace=False)
        X_batch_C = X_train_C[indices_C]
        y_batch_C = y_train_C[indices_C]
        X_batch_A = X_train_A[indices_C]

        # Mode A (入力データにノイズが乗ったTGDP)
        # 比較用として、勾配ノイズなし (noise_scale=0) でサンプル単位クリッピングのみ適用
        grad_A_clipped = compute_dp_gradient(X_batch_A, y_batch_C, weights_A, lambda_reg, C_global, noise_scale=0.0)
        weights_A = weights_A - current_lr * grad_A_clipped

        # Mode B (真のDP-SGDのシミュレーション)
        # ここでノイズスケールを指定して、クリッピングと勾配ノイズ注入を同時に行う
        grad_B_dp = compute_dp_gradient(X_batch_B, y_batch_B, weights_B, lambda_reg, C_global, noise_scale=noise_scale_grad)
        weights_B = weights_B - current_lr * grad_B_dp

        # Mode C (HEを用いた暗号化計算)
        # ※ HEClient側の処理が変更できない場合は現状のままとし、
        # Mode B（ローカルシミュレーション）の結果と比較して挙動を確認してください
        ct_weights = client.encrypt(weights_C.flatten().tolist())
        ct_grad = client.compute_gradient(
            ct_weights, X_batch_C.flatten().tolist(), y_batch_C.flatten().tolist(),
            actual_batch_C, n_features, lambda_reg
        )
        grad_C_dec = np.array(client.decrypt(ct_grad)[:n_features]).reshape(-1, 1)
        
        # weights_Cを更新する「前」にideal_gradを計算して誤差を算出する
        ideal_grad_unclipped = compute_standard_gradient(X_batch_C, y_batch_C, weights_C, lambda_reg)
        error_tensor = grad_C_dec - ideal_grad_unclipped
        current_noise_var = np.var(error_tensor.flatten())
        cumulative_noise_var += current_noise_var
        total_sigma = math.sqrt(cumulative_noise_var)
        epsilon_current = (math.sqrt(2) * sensitivity_gradient) / (total_sigma + 1e-9)

        # 誤差計算後にクリッピングと重み更新を実行
        grad_C_clipped = clip_updates(grad_C_dec, C_global)
        weights_C = weights_C - current_lr * grad_C_clipped

        loss_A = compute_loss(X_train_C, y_train_C, weights_A, lambda_reg)
        loss_C = compute_loss(X_train_C, y_train_C, weights_C, lambda_reg)
        # Mode B (平文) は生データ (X_train_B) に対するLossを計算
        loss_B = compute_loss(X_train_B, y_train_B, weights_B, lambda_reg)
        
        pred_test_A = X_test.dot(weights_A)
        pred_test_B = X_test.dot(weights_B)
        pred_test_C = X_test.dot(weights_C)

        # ★ 逆変換 (Inverse Transform) で元のスケールに戻す
        y_test_inv = scaler_y.inverse_transform(y_test)
        pred_test_A_inv = scaler_y.inverse_transform(pred_test_A)
        pred_test_B_inv = scaler_y.inverse_transform(pred_test_B)
        pred_test_C_inv = scaler_y.inverse_transform(pred_test_C)

        # 指標の算出 (元のスケールで計算)
        rmse_A = math.sqrt(mean_squared_error(y_test_inv, pred_test_A_inv))
        rmse_B = math.sqrt(mean_squared_error(y_test_inv, pred_test_B_inv))
        rmse_C = math.sqrt(mean_squared_error(y_test_inv, pred_test_C_inv))
        
        mae_A = mean_absolute_error(y_test_inv, pred_test_A_inv)
        mae_B = mean_absolute_error(y_test_inv, pred_test_B_inv)
        mae_C = mean_absolute_error(y_test_inv, pred_test_C_inv)
        
        r2_A = r2_score(y_test_inv, pred_test_A_inv)
        r2_B = r2_score(y_test_inv, pred_test_B_inv)
        r2_C = r2_score(y_test_inv, pred_test_C_inv)
        
        smape_A = compute_smape(y_test_inv, pred_test_A_inv)
        smape_B = compute_smape(y_test_inv, pred_test_B_inv)
        smape_C = compute_smape(y_test_inv, pred_test_C_inv)
        
        mape_A = compute_mape(y_test_inv, pred_test_A_inv)
        mape_B = compute_mape(y_test_inv, pred_test_B_inv)
        mape_C = compute_mape(y_test_inv, pred_test_C_inv)

        l2_dist_AB = np.linalg.norm(weights_A - weights_B)
        l2_dist_CB = np.linalg.norm(weights_C - weights_B)
        
        # metrics への保存
        metrics["loss_A"].append(loss_A); metrics["loss_B"].append(loss_B); metrics["loss_C"].append(loss_C)
        metrics["rmse_A"].append(rmse_A); metrics["rmse_B"].append(rmse_B); metrics["rmse_C"].append(rmse_C)
        metrics["mae_A"].append(mae_A); metrics["mae_B"].append(mae_B); metrics["mae_C"].append(mae_C)
        metrics["r2_A"].append(r2_A); metrics["r2_B"].append(r2_B); metrics["r2_C"].append(r2_C)
        metrics["smape_A"].append(smape_A); metrics["smape_B"].append(smape_B); metrics["smape_C"].append(smape_C)
        metrics["mape_A"].append(mape_A); metrics["mape_B"].append(mape_B); metrics["mape_C"].append(mape_C)
        metrics["l2_dist_AB"].append(l2_dist_AB); metrics["l2_dist_CB"].append(l2_dist_CB)
        metrics["epsilon_C"].append(epsilon_current)

        # 最後のイテレーションの結果を保持 (y-yプロット用)
        if i == num_iterations - 1:
            metrics["y_test_final"] = y_test_inv.flatten()
            metrics["pred_A_final"] = pred_test_A_inv.flatten()
            metrics["pred_B_final"] = pred_test_B_inv.flatten()
            metrics["pred_C_final"] = pred_test_C_inv.flatten()

        print(f"Iter {i+1:02d} | Loss C: {loss_C:.4f} | RMSE C: {rmse_C:.4f} | L2 Dist CB: {l2_dist_CB:.4f} | ε: {epsilon_current:.6f}")

    return metrics

if __name__ == "__main__":
    run_experiment(save_plot=True)