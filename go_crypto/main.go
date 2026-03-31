package main

import (
	"fmt"
	"log"
	"net"

	pb "go_crypto/he_service" // protocで自動生成されたパッケージ
	"go_crypto/lattigo_wrapper"
	"go_crypto/server"

	"google.golang.org/grpc"
)

func main() {
	fmt.Println("=== 準同型暗号(CKKS) gRPC サーバーを起動します ===")

	// 1. ポート50051で通信を待ち受ける設定
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("ポート 50051 のリッスンに失敗しました: %v", err)
	}

	// 2. Lattigoの暗号コンテキストを初期化
	cryptoCtx, err := lattigo_wrapper.NewCryptoContext()
	if err != nil {
		log.Fatalf("暗号コンテキストの初期化に失敗しました: %v", err)
	}

	// 3. gRPCサーバーのインスタンスを作成
	grpcServer := grpc.NewServer()

	// 4. 作成したHEServerをgRPCに登録
	pb.RegisterHEServiceServer(grpcServer, &server.HEServer{CryptoCtx: cryptoCtx})

	// 5. サーバー起動
	fmt.Println("サーバーがポート 50051 で待機中です。クライアントからの接続を待っています...")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("gRPCサーバーの起動に失敗しました: %v", err)
	}
}