import os
import numpy as np
import matplotlib.pyplot as plt
from main_training import run_experiment

def main():
    num_runs = 1
    all_epsilons_C = []

    print(f"=== {num_runs}回のプライバシ強度の平均計測")

    for run_idx in range(num_runs):
        print(f"\n--- 実行 {run_idx + 1} / {num_runs} ---")
        metrics = run_experiment(save_plot=False)
        all_epsilons_C.append(metrics["epsilon_C"])
        
        # モード A の目標ε（設定値）を一度だけ取得
        if run_idx == 0:
            # main_training.py で設定した epsilon_dp の値を全イテレーション分用意
            # (もしモードAも動的に計算しているなら metrics から取得)
            target_epsilon_A = [10.0] * len(metrics["epsilon_C"]) 

    avg_epsilons = np.mean(all_epsilons_C, axis=0)

    # --- 描画ロジックの修正 ---
    import japanize_matplotlib # 日本語化

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 平均エプシロンのプロット
    ax.plot(range(1, len(avg_epsilons) + 1), avg_epsilons, color='blue', marker='.', markersize=4)
    
    # 軸ラベルの日本語化
    ax.set_xlabel("イテレーション")
    ax.set_ylabel("平均プライバシー強度 (ε)")

    # 横軸を 0-100 に固定
    ax.set_xlim(0, 100)

    # 縦線（x軸の補助線）を消し、横線のみを表示
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 10^0 などの指数表記（対数スケール含む）を無効化し、通常の数値で表示
    # ※もし yscale('log') を継続する場合は plain 指定が無視されることがあるため、
    #   通常のスケールに戻すか、Formatterを明示的に設定します。
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')

    # タイトル（題名）を削除
    # plt.title(...) は記述しません

    # 結果を保存
    output_path = os.path.join(os.path.dirname(__file__), "average_epsilon_transition.png")
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"\n=== 全計測完了 ===")
    print(f"10回の平均化された推移グラフを '{output_path}' に保存しました。")

if __name__ == "__main__":
    main()