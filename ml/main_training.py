# ml/main_training.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 修正したクライアントをインポート
from grpc_client import HEClient 

def load_and_preprocess_data():
    """Diabetesデータの読み込みと正規化"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../dataset/diabetes.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=['target']).values
    y = df['target'].values.reshape(-1, 1)

    # 特徴量とラベルを [-1, 1] に正規化
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def compute_loss(X, y, weights, lambda_reg):
    """L2正則化付きMSE損失"""
    predictions = X.dot(weights)
    mse = np.mean((predictions - y) ** 2)
    l2_penalty = lambda_reg * np.sum(weights ** 2)
    return mse + l2_penalty

def run_experiment():
    # 1. データ準備
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    n_samples, n_features = X_train.shape
    
    # dominating_add_dataset.py で計算した MDS サイズをここに反映
    mds_size_T = 100 
    
    # 2. パラメータ設定
    lambda_reg = 0.5       
    initial_lr = 0.1       
    num_iterations = 20    # Bootstrappingなしのため深さを抑えた設定
    epsilon = 1.0          
    
    # 理論的感度 Δ の計算
    max_beta_norm = 1.0 / lambda_reg
    sensitivity = 2.0 * (1.0 * max_beta_norm + 1.0) + 2.0 * lambda_reg * max_beta_norm
    noise_scale = (sensitivity * mds_size_T) / (epsilon * n_samples)

    # 重みの初期化
    weights_A = np.zeros((n_features, 1)) # Mode A (DP)
    weights_B = np.zeros((n_features, 1)) # Mode B (Plain)
    weights_C = np.zeros((n_features, 1)) # Mode C (CKKS)

    client = HEClient()
    metrics = {
        "loss_A": [], "loss_B": [], "loss_C": [],
        "rmse_A": [], "rmse_C": [],
        "l2_dist_AC": [], "l2_dist_BC": []
    }

    print(f"=== 実験開始 (A:DP, B:Baseline, C:CKKS) ===")
    
    for i in range(num_iterations):
        current_lr = initial_lr / (1 + 0.05 * i)

        # --- Mode A: 平文 DP-SGD (理論ノイズ) ---
        pred_A = X_train.dot(weights_A)
        grad_A = (2 / n_samples) * X_train.T.dot(pred_A - y_train) + 2 * lambda_reg * weights_A
        noise_A = np.random.laplace(loc=0.0, scale=noise_scale, size=grad_A.shape)
        weights_A = weights_A - current_lr * (grad_A + noise_A)

        # --- Mode B: 平文 Baseline (ノイズなし) ---
        pred_B = X_train.dot(weights_B)
        grad_B = (2 / n_samples) * X_train.T.dot(pred_B - y_train) + 2 * lambda_reg * weights_B
        weights_B = weights_B - current_lr * grad_B

        # --- Mode C: CKKS-SGD (近似誤差ノイズ) ---
        # 1. 重みを暗号化
        ct_weights = client.encrypt(weights_C.flatten().tolist())
        # 2. サーバー側で勾配を計算 (Go側のCKKS演算で近似誤差が発生)
        ct_grad = client.compute_gradient(
            ct_weights, X_train.flatten().tolist(), y_train.flatten().tolist(),
            n_samples, n_features, lambda_reg
        )
        # 3. 復号して Python側で更新
        grad_C_dec = np.array(client.decrypt(ct_grad)[:n_features]).reshape(-1, 1)
        weights_C = weights_C - current_lr * grad_C_dec

        # --- 指標の記録 ---
        l_A, l_B, l_C = [compute_loss(X_train, y_train, w, lambda_reg) for w in [weights_A, weights_B, weights_C]]
        rmse_A = np.sqrt(mean_squared_error(y_test, X_test.dot(weights_A)))
        rmse_C = np.sqrt(mean_squared_error(y_test, X_test.dot(weights_C)))
        
        metrics["loss_A"].append(l_A)
        metrics["loss_B"].append(l_B)
        metrics["loss_C"].append(l_C)
        metrics["rmse_A"].append(rmse_A)
        metrics["rmse_C"].append(rmse_C)
        metrics["l2_dist_AC"].append(np.linalg.norm(weights_A - weights_C))
        metrics["l2_dist_BC"].append(np.linalg.norm(weights_B - weights_C))

        if (i + 1) % 5 == 0 or i == 0:
            print(f"Iter {i+1:02d} | Loss A:{l_A:.4f} B:{l_B:.4f} C:{l_C:.4f} | Dist(BC):{metrics['l2_dist_BC'][-1]:.4f}")

    # --- 評価グラフの出力 ---
    plt.figure(figsize=(16, 5))
    
    # 1. Loss推移 (A, B, C)
    plt.subplot(1, 3, 1)
    plt.plot(metrics["loss_A"], label="Mode A (DP)")
    plt.plot(metrics["loss_B"], label="Mode B (Plain)", linestyle="--")
    plt.plot(metrics["loss_C"], label="Mode C (CKKS)")
    plt.title("Training Loss")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend()

    # 2. 予測精度の比較
    plt.subplot(1, 3, 2)
    plt.plot(metrics["rmse_A"], label="RMSE Mode A")
    plt.plot(metrics["rmse_C"], label="RMSE Mode C")
    plt.title("Test RMSE")
    plt.xlabel("Iteration"); plt.ylabel("RMSE"); plt.legend()

    # 3. ノイズの乗り方の比較 (L2誤差)
    plt.subplot(1, 3, 3)
    plt.plot(metrics["l2_dist_BC"], label="Dist(Plain vs CKKS)", color='green')
    plt.plot(metrics["l2_dist_AC"], label="Dist(DP vs CKKS)", color='orange')
    plt.title("Weight L2 Distance")
    plt.xlabel("Iteration"); plt.ylabel("L2 Norm"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "experiment_results.png"))
    print("✅ 指標グラフを保存しました: experiment_results.png")

if __name__ == "__main__":
    run_experiment()