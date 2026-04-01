# 差分プライバシのラプラシアンメカニズムのノイズを準同型暗号CKKS方式で満たす

Go言語の準同型暗号ライブラリlattigoのCKKS方式を用いて、データを暗号化したまま機械学習を行う。Bootstrappingは使用しない。

将来的にラプラシアンノイズ付与などの内部処理を変更するため、Lattigoは標準のGoパッケージ管理(`go get`)でそのまま利用するのではなく、ルートディレクトリにリポジトリをクローンしてローカルモジュールとして参照する。ベースとなるバージョンは `github.com/tuneinsight/lattigo/v6` (v6.2.0以降) を利用する。

Pythonで鍵管理と機械学習、Goで暗号演算を行う。言語間通信にはgRPCを用いる。

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

# 起動
go serverを立てるときはgo run main.goを実行する。