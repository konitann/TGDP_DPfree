import pulp
import networkx as nx
import os
import argparse

def calculate_exact_data_collectors(filepath):
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return

    # 1. グラフの構築 (無向グラフ)
    G = nx.Graph()
    all_nodes = set()
    
    # 1回目のスキャン: システムに存在するすべてのユーザー(ノード)を特定
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split(',') if ',' in line else line.split()
            if len(parts) < 2:
                continue
                
            try:
                u, v = int(parts[0]), int(parts[1])
                # すべてのユーザーを記録
                all_nodes.add(u)
                all_nodes.add(v)
            except ValueError:
                continue

    # すべてのノードを明示的にグラフに追加（これにより孤立ノードが保持されます）
    G.add_nodes_from(all_nodes)
    
    # 2回目のスキャン: 正の評価(RATING > 0)のみを信頼エッジとして追加
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split(',') if ',' in line else line.split()
            if len(parts) < 2:
                continue
                
            try:
                u, v = int(parts[0]), int(parts[1])
                
                # RATING（第3列）が存在する場合は、値が正(>0)の場合のみ信頼エッジとして扱う
                if len(parts) >= 3:
                    rating = float(parts[2])
                    if rating > 0:
                        G.add_edge(u, v)
                else:
                    # RATINGがないデータセットの場合はそのままエッジを追加
                    G.add_edge(u, v)
            except ValueError:
                continue

    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    
    print(f"構築されたグラフ: ノード数 {n}, エッジ数 {G.number_of_edges()}")

    # 2. 整数計画問題の定義 (正確な最小支配集合 = データ収集者)
    print("整数計画法(ILP)を用いてデータ収集者を計算中...")
    prob = pulp.LpProblem("MinimumDominatingSet", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)

    # 目的関数: 選択するデータ収集者の数を最小化
    prob += pulp.lpSum([x[i] for i in nodes])

    # 制約条件: 全てのユーザー自身、または信頼できる隣接ユーザーの少なくとも1人がデータ収集者になる
    for v in nodes:
        neighbors = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum([x[u] for u in neighbors]) >= 1

    # 3. ソルバー実行
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 結果表示
    if prob.status == pulp.LpStatusOptimal:
        mds_size = int(pulp.value(prob.objective))
        
        print(f"\n--- 結果 ---")
        print(f"データセット: {os.path.basename(filepath)}")
        print(f"データ収集者の数 (Number of Data Collectors, |S|): {mds_size}")
        return mds_size
    else:
        print("\n最適解が見つかりませんでした。ステータス:", pulp.LpStatus[prob.status])
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate exact Data Collectors using ILP.")
    parser.add_argument("--filepath", type=str, required=True, help="データセットのファイルパス")
    
    args = parser.parse_args()
    calculate_exact_data_collectors(args.filepath)