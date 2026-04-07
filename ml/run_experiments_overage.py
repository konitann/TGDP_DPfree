import os
import numpy as np
import matplotlib.pyplot as plt
from main_training import run_experiment

def main():
    num_runs = 10
    all_epsilons = []

    print(f"=== {num_runs}回のプライバシ強度の平均計測")

    for run_idx in range(num_runs):
        print(f"\n--- 実行 {run_idx + 1} / {num_runs} ---")
        metrics = run_experiment(save_plot=False)
        all_epsilons.append(metrics["epsilon_C"])

    avg_epsilons = np.mean(all_epsilons, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_epsilons) + 1), avg_epsilons, 
             label="Average Privacy Budget (ε)", color='purple', marker='.')
    
    # 対数で表示
    plt.yscale('log')

    plt.title(f"Average Transition of Privacy Strength over {num_runs} Runs")
    plt.xlabel("Iteration")
    plt.ylabel("Average Privacy Budget (ε) - Log Scale")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 結果を保存
    output_path = os.path.join(os.path.dirname(__file__), "average_epsilon_transition.png")
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"\n=== 全計測完了 ===")
    print(f"10回の平均化された推移グラフを '{output_path}' に保存しました。")

if __name__ == "__main__":
    main()