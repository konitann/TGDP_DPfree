import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt
import os

def solve_and_visualize_trust_graph(filepath):
    print(f"--- 処理開始: {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return

    # 1. データの読み込みとグラフ構築
    edges = set()
    nodes = set()
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            
            nodes.add(u)
            nodes.add(v)
            
            # 無向エッジとして保存
            if u != v:
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
    
    # ノードIDのマッピング (0 ~ N-1)
    sorted_nodes = sorted(list(nodes))
    node_map = {node_id: i for i, node_id in enumerate(sorted_nodes)}
    n = len(nodes)
    
    print(f"ノード数: {n}, エッジ数: {len(edges)}")
    
    # NetworkXグラフオブジェクトの作成（可視化用）
    G = nx.Graph()
    G.add_nodes_from(sorted_nodes)
    G.add_edges_from(edges)

    # 2. LPの準備と実行 (Highsソルバー)
    # 隣接行列の作成
    row_ind = []
    col_ind = []
    
    for u, v in edges:
        i, j = node_map[u], node_map[v]
        row_ind.append(i); col_ind.append(j)
        row_ind.append(j); col_ind.append(i)
        
    # 自己ループ (v は N[v] に含まれる)
    for i in range(n):
        row_ind.append(i)
        col_ind.append(i)
        
    # 重みを1に正規化
    data = np.ones(len(row_ind))
    adj_matrix = sp.coo_matrix((data, (row_ind, col_ind)), shape=(n, n))
    adj_matrix = adj_matrix.tocsr()
    adj_matrix.data[:] = 1

    print("LPを解いています...")
    c = np.ones(n)
    A_ub = -adj_matrix
    b_ub = -np.ones(n)
    bounds = [(0, 1) for _ in range(n)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        print(f"最適化失敗: {res.message}")
        return

    y_values = res.x
    print(f"OPT_LP: {res.fun:.4f}")

    # 3. グラフの可視化
    print("グラフを描画中... (ノード数が多い場合、時間がかかります)")
    plt.figure(figsize=(12, 10))
    
    # レイアウト決定 (バネモデル)
    # k: ノード間の反発力（大きいほど広がる）, seed: 乱数固定
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # 色の決定
    # y_values は node_map の順序 (sorted_nodes順) に並んでいる
    # NetworkXの描画順序に合わせて並べ替える必要がある
    draw_nodes = sorted_nodes # 今回はソート順で渡すのでそのまま使える
    node_colors = [y_values[node_map[u]] for u in draw_nodes]

    # ノードの描画
    # cmap: カラーマップ (Reds, Blues, Viridis, Plasmaなど)
    # vmin=0, vmax=1: yは0~1の範囲なので固定
    nodes_plot = nx.draw_networkx_nodes(
        G, pos, 
        nodelist=draw_nodes,
        node_color=node_colors, 
        cmap=plt.cm.Reds,  # 薄い赤(0) -> 濃い赤(1)
        node_size=30,      # ノードサイズ
        vmin=0.0, vmax=1.0,
        alpha=0.9
    )
    
    # エッジの描画 (薄く表示)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    
    # カラーバーの追加
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Noise Level (y_u)')
    
    plt.title(f"Trust Graph DP Solution (OPT_LP = {res.fun:.2f})")
    plt.axis('off')
    
    # 保存または表示
    output_file = "trust_graph_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_file}")
    plt.show()

# 実行
file_path = "../dataset/email-Eu-core.txt"
solve_and_visualize_trust_graph(file_path)