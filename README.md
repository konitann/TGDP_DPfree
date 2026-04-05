# 信頼グラフ上の差分プライバシーとCKKS近似誤差の評価環境

Go言語の準同型暗号ライブラリlattigoのCKKS方式を用いて、データを暗号化したまま機械学習を行う。Bootstrappingは使用しない。

将来的にラプラシアンノイズ付与などの内部処理を変更するため、Lattigoは標準のGoパッケージ管理(`go get`)でそのまま利用するのではなく、ルートディレクトリにリポジトリをクローンしてローカルモジュールとして参照する。ベースとなるバージョンは `github.com/tuneinsight/lattigo/v6` (v6.2.0以降) を利用する。

Pythonで鍵管理と機械学習、Goで暗号演算を行う。言語間通信にはgRPCを用いる。

## 2. アーキテクチャ
- Python: データ処理、機械学習ロジックの管理、gRPCクライアント
- Go (Lattigo v6): CKKS暗号演算（加算、乗算、回転）、gRPCサーバ
- 通信: gRPCによる、暗号化ベクトル（重み）と平文データ（訓練データ）のやり取り

## 3. アルゴリズム

### クリッピングフリーのDP-SGD
暗号文上での複雑な条件分岐（クリッピング）を回避するため、以下の数理的保証に基づいたDP-SGDを実装しています。
- **入力の事前正規化**: すべてのデータエントリにおいてラベルと特徴量を $|y_i|, |x_{ij}| \le 1$ を満たすように事前に正規化。
- **学習率のヒューリスティック**: コスト関数 $J(\beta, D)$ が必ず減少するように学習率を設定。
- **重みベクトルの上限証明**: コスト関数の減少に基づき、更新される重みベクトル $\beta$ のL2ノルムが常に $||\beta||_2 \le 1/\lambda$ （$\lambda$ は正則化パラメータ）に収まる性質を利用し、勾配の感度（Sensitivity）を理論的に算出。

### CKKS上での正確な勾配計算
Goサーバー側では、Lattigo v6を用いてリッジ回帰の勾配 $\nabla J$ を数学的定義に基づいて正確に計算します。
$$\nabla J = \frac{2}{n} X^T (X\beta - y) + 2\lambda\beta$$
- **事前計算**: サーバー側で $A = \frac{2}{n} X^T X + 2\lambda I$ および $b = \frac{2}{n} X^T y$ を平文で計算。
- **行列・ベクトル積**: 暗号化された重み $\beta$ に対して $A\beta - b$ を実行。
- **回転（Rotation）演算**: ベクトルの内積計算のため、ガロア鍵（Galois Keys）を用いたスロットの回転操作を実装。

## 4. セットアップと実行手順

### ステップ1: データセットの準備
UCI Diabetesデータセットをダウンロードし、`dataset/` ディレクトリに保存します。
```bash
mkdir -p dataset
python -c "import pandas as pd; from sklearn.datasets import load_diabetes; d = load_diabetes(); pd.DataFrame(d.data, columns=d.feature_names).assign(target=d.target).to_csv('dataset/diabetes.csv', index=False); print('passed')"

### ステップ2: プロトコルバッファの生成
# Python用
python -m grpc_tools.protoc -I=./proto --python_out=./ml --grpc_python_out=./ml ./proto/he_service.proto

# Go用
protoc --go_out=./go_crypto --go-grpc_out=./go_crypto ./proto/he_service.proto

### ステップ3: Goサーバーの起動
```bash
cd go_crypto
go run main.go
```

### ステップ4: Pythonクライアントの実行
```bash
cd ml
python main_training.py
```

## 5. 評価・計測ロジック
各イテレーション終了時に、以下の指標を計算しグラフ（`experiment_results.png`）として出力します。

- **① モデルの予測精度**: 
  - **RMSE (Root Mean Squared Error)** および **MAE (Mean Absolute Error)** を算出。テストデータに対する予測性能の推移を評価します。
- **② ノイズ（誤差）の乗り方**:
  - **Dist(DP vs CKKS)**: Mode A と Mode C の重みベクトル間のL2距離。差分プライバシーノイズと暗号近似誤差の乖離を測定します。
  - **Dist(Plain vs CKKS)**: Mode B と Mode C の重みベクトル間のL2距離。純粋なCKKSの近似誤差が学習に与える影響を測定します。
- **③ 損失推移 (Loss)**: 
  - イテレーションごとの目的関数の損失。学習が正常に収束しているか、あるいはノイズによって発散していないかを確認します。

## 6. CKKSパラメータの調整ガイド (context.go)
近似誤差（ノイズ）の大きさを変化させて評価を行う場合、`go_crypto/lattigo_wrapper/context.go` 内の `LogDefaultScale`（$\Delta$）に合わせて `LogQ`（各レベルの素数のビット数）を調整してください。

| 評価目標 | LogDefaultScale | LogQ (素数ビット数) の構成案 | 備考 |
| :--- | :--- | :--- | :--- |
| 特大 | 15 | `[]int{30, 15, 15, 15, 15, 15, 15, 15}` | 論文の最小評価値。ノイズの影響が顕著。 |
| 大 | 20 | `[]int{35, 20, 20, 20, 20, 20, 20, 20}` | 近似誤差が明確に観測可能。 |
| 中 | 25 | `[]int{40, 25, 25, 25, 25, 25, 25, 25}` | バランスの取れた設定。 |
| 小 | 30 | `[]int{45, 30, 30, 30, 30, 30, 30, 30}` | 実用的な精度。 |
| 高精度 | 40 | `[]int{55, 40, 40, 40, 40, 40, 40, 40}` | デフォルト設定。平文計算に極めて近い。 |

### 設定変更の重要ルール
1. **整合性の維持**: リスケール時の数値爆発（発散）を防ぐため、`LogDefaultScale` の値は `LogQ` の2番目以降の素数ビット数と必ず一致させてください。
2. **サーバー再起動**: `context.go` を書き換えた後は、必ず Go サーバー（`main.go`）を停止し、`go run main.go` で再起動して設定を反映させてください。
3. **LogN の上限**: モジュラスの総ビット数（LogQの合計）が大きくなる場合は、`LogN` を `15` や `16` に上げる必要があります。

## ディレクトリ構造
.
├── dataset/                     # グラフネットワークのデータセット
│   ├── email-Eu-core.txt
│   ├── email-EuAll.txt
│   ├── facebook/                # facebookのエッジ・特徴量データ群
│   ├── soc-sign-bitcoinalpha.csv
│   └── soc-sign-bitcoinotc.csv
│
├── TGDP/                        # データ前処理・鍵保有ノード決定プログラム
│   ├── LP_add_dataset.py        # 線形計画法等のアプローチを用いたデータ処理
│   └── dominating_add_dataset.py# 支配集合等をベースにしたノード抽出・処理 (現在使用可能)
│
├── lattigo/                     # カスタマイズ用にCloneしたLattigo v6リポジトリ
│
├── proto/                       # gRPCのインターフェース定義
│   └── he_service.proto         # Go(サーバー)とPython(クライアント)の通信ルール定義
│
├── go_crypto/                   # Go言語による暗号計算バックエンド (Lattigo)
│   ├── go.mod                   # ローカルのlattigoを参照するようにreplaceを設定済み
│   ├── go.sum
│   ├── main.go                  # gRPCサーバーのエントリーポイント
│   ├── he_service/              # protocによって自動生成されたGo用gRPCコード群
│   ├── server/
│   │   └── grpc_server.go       # gRPCエンドポイント(Encrypt, Decrypt等)の実装
│   └── lattigo_wrapper/
│       ├── context.go           # 一般的CKKSパラメータ（Bootstrappingなし）の管理
│       ├── keygen.go            # 鍵生成（SK, PK, RelinKey等）
│       ├── encrypt_decrypt.go   # 暗号化と復号の処理
│       └── evaluator.go         # 暗号文同士、暗号文と平文の加算・乗算処理
│
└── ml/                          # Python側の機械学習ディレクトリ
    ├── requirements.txt
    ├── main_training.py         # gRPCクライアントを利用し、暗号化されたままMLを実行 (テスト実装済)
    ├── grpc_client.py           # Goサーバーと通信するためのPython用gRPCクライアント
    ├── model.py                 # 機械学習モデルの構造定義
    ├── he_service_pb2.py        # protocによって自動生成されたPython用gRPCコード
    └── he_service_pb2_grpc.py   # 同上

    ## 実装状況
    - gRPC経由で暗号化、復号、加算、乗算のテスト確認が官僚
    - ckksのスケール管理は乗算内において、暗号文の乗算後に必要なrelinearizeとrescaleを実装


## 環境構築と重要な注意事項（トラブルシューティング）

### 1. Lattigo v6 ローカルモジュールの参照設定
Go側 (`go_crypto`) は、インターネット上のLattigoではなく、カスタマイズ可能なローカルの `../lattigo` フォルダを参照するように `go.mod` に `replace` を設定している。
依存関係がおかしくなった場合は、`go_crypto` フォルダで以下を実行すること。
```bash
# モジュールのクリーンアップと同期
go clean -modcache
go get ./...
go mod tidy
```
### TGDPの実行
python3 dominating_add_dataset.py --filepath <データセットのパス>
