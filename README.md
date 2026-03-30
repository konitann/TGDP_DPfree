# 差分プライバシのラプラシアンメカニズムのノイズを準同型暗号CKKS方式で満たす

Go言語の準同型暗号ライブラリlattigoのCKKS方式を用いて、データを暗号化したまま機械学習を行う。Bootstrappingは使用しない。

pythonで鍵管理と機械学習、goで暗号演算を行う。言語感はgRPCを用いる。

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
│   └── dominating_add_dataset.py# 支配集合等をベースにしたノード抽出・処理こちらが現在使用可能
│
├── proto/                       # gRPCのインターフェース定義（未実装）
│   └── he_service.proto         # Go(サーバー)とPython(クライアント)の通信ルール定義
│
├── go_crypto/                   # Go言語による暗号計算バックエンド (Lattigo)（未実装）
│   ├── go.mod
│   ├── go.sum
│   ├── main.go                  # gRPCサーバーのエントリーポイント
│   ├── server/
│   │   └── grpc_server.go       # gRPCエンドポイントの実装
│   └── lattigo_wrapper/
│       ├── context.go           # 一般的CKKSパラメータ（Bootstrappingなし）の管理
│       ├── keygen.go            # 鍵生成（SK, PK, RelinKey等）
│       ├── encrypt_decrypt.go   # 暗号化と復号の処理
│       └── evaluator.go         # 暗号文同士、暗号文と平文の加算・乗算処理
│
└── ml/                          # Python側の機械学習(未実装)
    ├── requirements.txt
    ├── main_training.py         # gRPCクライアントを利用し、暗号化されたままMLを実行
    ├── grpc_client.py           # Goサーバーと通信するためのPython用gRPCクライアント
    └── model.py                 # 機械学習モデルの構造定義