import os
import math
import numpy as np
import pandas as pd
import collections
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

def clip_updates(updates, max_norm_C):
    """L2ノルムによる勾配クリッピング（感度の保証）"""
    total_norm = np.linalg.norm(updates, ord=2)
    clip_coef = max_norm_C / (total_norm + 1e-6)
    if clip_coef < 1.0:
        return updates * clip_coef
    return updates

# --- 追加: プレランによる C の事前計算関数 ---
def compute_median_clip_threshold(X_train, y_train, lambda_reg, pre_run_iters=15, lr=0.1):
    """
    本番前に少数のイテレーションを平文で回し、クリッピング前の勾配ノルムの中央値を計算する。
    この値が本番の固定定数 C となる。
    """
    n_samples, n_features = X_train.shape
    weights_pre = np.zeros((n_features, 1))
    grad_norms = []

    print(f"--- プレラン（事前学習）を開始: {pre_run_iters} イテレーション ---")
    for i in range(pre_run_iters):
        pred = X_train.dot(weights_pre)
        # クリッピング前の勾配（unclipped gradient）
        grad = (2 / n_samples) * X_train.T.dot(pred - y_train) + 2 * lambda_reg * weights_pre
        
        # L2ノルムを記録
        norm = np.linalg.norm(grad, ord=2)
        grad_norms.append(norm)
        
        # 重みを更新
        weights_pre = weights_pre - lr * grad
    
    # 記録したノルムの中央値を固定閾値 C とする
    median_C = float(np.median(grad_norms))
    print(f"プレラン完了。計算された固定クリッピング閾値 C = {median_C:.6f}\n")
    return median_C

def run_experiment(save_plot=True):
    # 1. データ準備
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    n_samples, n_features = X_train.shape
    
    # dominating_add_dataset.py で計算した MDS サイズをここに反映
    mds_size_T = 100 
    
    # 2. パラメータ設定
    lambda_reg = 0.5       
    initial_lr = 0.1       
    num_iterations = 100    # Bootstrappingなしのため深さを抑えた設定
    epsilon_dp = 10.0       # Mode A (理論DP) 用の目標ε

    delta = 1e-5            # 許容される失敗確率

    # ★ 追加: 事前計算により C を決定する（変動しない定数として扱う）
    C = compute_median_clip_threshold(X_train, y_train, lambda_reg, pre_run_iters=15, lr=initial_lr)

    # 理論的感度 Δ の計算 (Mode A 用)
    max_beta_norm = 1.0 / lambda_reg
    sensitivity = 2.0 * (1.0 * max_beta_norm + 1.0) + 2.0 * lambda_reg * max_beta_norm
    noise_scale = (sensitivity * mds_size_T) / (epsilon_dp * n_samples)

    # 重みの初期化
    weights_A = np.zeros((n_features, 1)) # Mode A (DP)
    weights_B = np.zeros((n_features, 1)) # Mode B (Plain)
    weights_C = np.zeros((n_features, 1)) # Mode C (CKKS)

    client = HEClient()
    metrics = {
        "loss_A": [], "loss_B": [], "loss_C": [],
        "rmse_A": [], "rmse_C": [],
        "l2_dist_AC": [], "l2_dist_BC": [],
        "epsilon_C": []  # CKKSノイズによる実効プライバシー予算(ε)の推移
    }

    # ★ 追加: σ_agg の乱高下を防ぐための移動平均用キュー (過去10イテレーション分)
    recent_sigmas = collections.deque(maxlen=10)

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
        # 3. 復号してPython側で取得 (これが実際の勾配)
        grad_C_dec = np.array(client.decrypt(ct_grad)[:n_features]).reshape(-1, 1)

        # --- ε推移の計算ロジック（パターンBの実装） ---
        # 3-1. 平文での理想的な勾配（現在の weights_C に対する真の勾配）を算出
        pred_ideal_C = X_train.dot(weights_C)
        ideal_grad_C = (2 / n_samples) * X_train.T.dot(pred_ideal_C - y_train) + 2 * lambda_reg * weights_C
        
        # 3-2. 演算誤差（ノイズ）の抽出
        error_tensor = grad_C_dec - ideal_grad_C
        
        # 3-3. 標準偏差の計算と移動平均による平滑化
        sigma_agg_raw = np.std(error_tensor.flatten())
        recent_sigmas.append(sigma_agg_raw)
        sigma_agg_smooth = np.mean(recent_sigmas)
        
        # 3-4. プライバシー予算(ε)の計算 (事前計算した固定のCを使用)
        epsilon_val = (C * math.sqrt(2 * math.log(1.25 / delta))) / (sigma_agg_smooth + 1e-9)
        metrics["epsilon_C"].append(epsilon_val)

        # 4. 感度を保証するため、固定の C でクリッピングを適用して重みを更新
        grad_C_clipped = clip_updates(grad_C_dec, C)
        weights_C = weights_C - current_lr * grad_C_clipped

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
            print(f"Iter {i+1:02d} | Loss A:{l_A:.4f} B:{l_B:.4f} C:{l_C:.4f} | ε:{epsilon_val:.4f} | Dist(BC):{metrics['l2_dist_BC'][-1]:.4f}")

    # --- 評価グラフの出力 ---
    if save_plot:
        plt.figure(figsize=(14, 10))
        
        # 1. Loss推移 (A, B, C)
        plt.subplot(2, 2, 1)
        plt.plot(metrics["loss_A"], label="Mode A (DP)")
        plt.plot(metrics["loss_B"], label="Mode B (Plain)", linestyle="--")
        plt.plot(metrics["loss_C"], label="Mode C (CKKS)")
        plt.title("Training Loss")
        plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend()

        # 2. 予測精度の比較
        plt.subplot(2, 2, 2)
        plt.plot(metrics["rmse_A"], label="RMSE Mode A")
        plt.plot(metrics["rmse_C"], label="RMSE Mode C")
        plt.title("Test RMSE")
        plt.xlabel("Iteration"); plt.ylabel("RMSE"); plt.legend()

        # 3. ノイズの乗り方の比較 (L2誤差)
        plt.subplot(2, 2, 3)
        plt.plot(metrics["l2_dist_BC"], label="Dist(Plain vs CKKS)", color='green')
        plt.plot(metrics["l2_dist_AC"], label="Dist(DP vs CKKS)", color='orange')
        plt.title("Weight L2 Distance")
        plt.xlabel("Iteration"); plt.ylabel("L2 Norm"); plt.legend()

        # 4. プライバシー予算(ε)の推移
        plt.subplot(2, 2, 4)
        plt.plot(metrics["epsilon_C"], label="Privacy Budget (ε)", color='purple', marker='.')
        plt.title(f"Transition of Privacy Strength (C={C:.4f}, δ={delta})")
        plt.xlabel("Iteration"); plt.ylabel("ε")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "experiment_results.png"))
        print("実験が完了し、結果を 'experiment_results.png' に保存しました。")
        plt.close()

    return metrics

if __name__ == "__main__":
    run_experiment()