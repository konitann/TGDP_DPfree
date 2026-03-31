package server

import (
	"context"
	"fmt"

	pb "go_crypto/he_service"
	"go_crypto/lattigo_wrapper"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type HEServer struct {
	pb.UnimplementedHEServiceServer
	CryptoCtx *lattigo_wrapper.CryptoContext
}

func (s *HEServer) Encrypt(ctx context.Context, req *pb.EncryptRequest) (*pb.EncryptResponse, error) {
	fmt.Println("リクエスト受信: Encrypt")
	values := req.GetPlainValues()

	plaintext := ckks.NewPlaintext(s.CryptoCtx.Params, s.CryptoCtx.Params.MaxLevel())

	if err := s.CryptoCtx.Encoder.Encode(values, plaintext); err != nil {
		return nil, fmt.Errorf("エンコードエラー: %v", err)
	}

	ciphertext, err := s.CryptoCtx.Encryptor.EncryptNew(plaintext)
	if err != nil {
		return nil, fmt.Errorf("暗号化エラー: %v", err)
	}

	bytesData, err := ciphertext.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("暗号文のシリアライズに失敗: %v", err)
	}

	return &pb.EncryptResponse{Ciphertext: bytesData}, nil
}

func (s *HEServer) Decrypt(ctx context.Context, req *pb.DecryptRequest) (*pb.DecryptResponse, error) {
	fmt.Println("リクエスト受信: Decrypt")
	cipherBytes := req.GetCiphertext()

	ciphertext := new(rlwe.Ciphertext)
	if err := ciphertext.UnmarshalBinary(cipherBytes); err != nil {
		return nil, fmt.Errorf("暗号文の復元に失敗: %v", err)
	}

	// 修正1: DecryptNew はエラーを返さず、Plaintext のみを返します
	plaintext := s.CryptoCtx.Decryptor.DecryptNew(ciphertext)

	// 修正2: デコード結果を受け取るための配列をあらかじめ作成します
	// サイズは暗号パラメータの MaxSlots() を指定します
	valuesComplex := make([]complex128, s.CryptoCtx.Params.MaxSlots())
	
	// 修正3: 第2引数に配列を渡し、そこに結果を書き込みます
	if err := s.CryptoCtx.Encoder.Decode(plaintext, valuesComplex); err != nil {
		return nil, fmt.Errorf("デコードエラー: %v", err)
	}

	// 複素数から実数部分(float64)のみを抽出
	valuesFloat := make([]float64, len(valuesComplex))
	for i, v := range valuesComplex {
		valuesFloat[i] = real(v)
	}

	return &pb.DecryptResponse{PlainValues: valuesFloat}, nil
}
// grpc_server.go の末尾などに追記

func (s *HEServer) Add(ctx context.Context, req *pb.AddRequest) (*pb.AddResponse, error) {
	fmt.Println("リクエスト受信: Add")

	ct1 := new(rlwe.Ciphertext)
	if err := ct1.UnmarshalBinary(req.GetCiphertext1()); err != nil {
		return nil, fmt.Errorf("ct1の復元に失敗: %v", err)
	}

	ct2 := new(rlwe.Ciphertext)
	if len(req.GetCiphertext2()) > 0 {
		if err := ct2.UnmarshalBinary(req.GetCiphertext2()); err != nil {
			return nil, fmt.Errorf("ct2の復元に失敗: %v", err)
		}
	} else {
		return nil, fmt.Errorf("ciphertext2が空です")
	}

	// 加算演算
	ctOut, err := s.CryptoCtx.Evaluator.AddNew(ct1, ct2)
	if err != nil {
		return nil, fmt.Errorf("加算エラー: %v", err)
	}

	bytesData, err := ctOut.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("加算結果のシリアライズに失敗: %v", err)
	}

	return &pb.AddResponse{ResultCiphertext: bytesData}, nil
}

func (s *HEServer) Multiply(ctx context.Context, req *pb.MultiplyRequest) (*pb.MultiplyResponse, error) {
	fmt.Println("リクエスト受信: Multiply")

	ct1 := new(rlwe.Ciphertext)
	if err := ct1.UnmarshalBinary(req.GetCiphertext1()); err != nil {
		return nil, fmt.Errorf("ct1の復元に失敗: %v", err)
	}

	ct2 := new(rlwe.Ciphertext)
	if len(req.GetCiphertext2()) > 0 {
		if err := ct2.UnmarshalBinary(req.GetCiphertext2()); err != nil {
			return nil, fmt.Errorf("ct2の復元に失敗: %v", err)
		}
	} else {
		return nil, fmt.Errorf("ciphertext2が空です")
	}

	// 乗算とリニアライゼーション（※サーバー側のCryptoCtxにRelinearization Keyが設定されている必要があります）
	ctOut, err := s.CryptoCtx.Evaluator.MulRelinNew(ct1, ct2)
	if err != nil {
		return nil, fmt.Errorf("乗算(MulRelin)エラー: %v", err)
	}

	// CKKSでは乗算後にスケールを戻すためのリスケーリングが必要
	if err := s.CryptoCtx.Evaluator.Rescale(ctOut, ctOut); err != nil {
		return nil, fmt.Errorf("リスケールエラー: %v", err)
	}

	bytesData, err := ctOut.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("乗算結果のシリアライズに失敗: %v", err)
	}

	return &pb.MultiplyResponse{ResultCiphertext: bytesData}, nil
}