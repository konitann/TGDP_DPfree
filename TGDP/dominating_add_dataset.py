import pulp
import networkx as nx
import os

def solve_exact_dominating_set_fixed(filepath, total_expected_nodes=None):
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return

    # 1. グラフの構築
    G = nx.Graph()
    
    # エッジリストから読み込み
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            G.add_edge(u, v)
            
    # 【修正点1】 エッジリストにあるノードIDを特定
    existing_nodes = set(G.nodes())
    min_id = min(existing_nodes) if existing_nodes else 0
    max_id = max(existing_nodes) if existing_nodes else 0
    
    # 【修正点2】 孤立ノードの補完
    # データセットの仕様上、ノードIDが連続している(0～3782等)と仮定、
    # もしくはユーザー指定の総数に合わせて孤立ノードを追加する
    if total_expected_nodes:
        print(f"-> Adjusting to match expected total nodes: {total_expected_nodes}")
        
        # 現在のノード数が足りない場合、足りない分を孤立ノードとしてカウント
        # (IDマッピングが複雑な場合もあるため、ここでは集合として扱う)
        missing_count = max(0, total_expected_nodes - len(existing_nodes))
        
        if missing_count > 0:
            print(f"-> Found {missing_count} isolated nodes (missing from edge list).")
            # 便宜上、既存の最大IDの次から連番で追加して処理する（構造上は影響なし）
            start_dummy_id = max_id + 1
            for i in range(missing_count):
                G.add_node(start_dummy_id + i)
        else:
            print("-> Node count matches expected value.")

    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    print(f"Final Graph Nodes: {n}, Edges: {G.number_of_edges()}")

    # 2. 整数計画問題の定義
    prob = pulp.LpProblem("MinimumDominatingSet", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)

    # 目的関数
    prob += pulp.lpSum([x[i] for i in nodes])

    # 制約条件
    for v in nodes:
        neighbors = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum([x[u] for u in neighbors]) >= 1

    # 3. ソルバー実行
    print("Solving Integer Programming...")
    # ログ出力を抑制したい場合は msg=False にしてください
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # 結果表示
    status = pulp.LpStatus[prob.status]
    mds_size = int(pulp.value(prob.objective))
    
    print(f"\n--- 結果 ---")
    dataset_name = os.path.basename(filepath)
    print(f"Dataset File: {dataset_name}")
    print(f"Status: {status}")
    print(f"Calculated MDS Size (|T|): {mds_size}")

    return mds_size

# 実行
# Bitcoin Alphaの正しいノード数 3783 を指定して実行
file_path = "soc-sign-bitcoinotc.txt"
solve_exact_dominating_set_fixed(file_path, total_expected_nodes=5881)