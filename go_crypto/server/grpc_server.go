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

// ComputeGradient は暗号化された重みと平文のデータセットを受け取り、
// 準同型暗号演算を用いて勾配を計算します。
func (s *HEServer) ComputeGradient(ctx context.Context, req *pb.GradientRequest) (*pb.GradientResponse, error) {
	fmt.Println("リクエスト受信: ComputeGradient")

	// 1. 暗号化された重み β の復元
	encryptedWeights := new(rlwe.Ciphertext)
	if err := encryptedWeights.UnmarshalBinary(req.GetEncryptedWeights()); err != nil {
		return nil, fmt.Errorf("暗号化重みの復元に失敗: %v", err)
	}

	// 2. パラメータの取得
	_ = int(req.GetNSamples()) // TODO: 勾配計算の定数倍などで使用
	_ = int(req.GetNFeatures()) // TODO: 行列サイズの管理に使用
	_ = req.GetLambdaReg()      // TODO: 正則化項 (+ 2λβ) の計算に使用
	xBatch := req.GetXBatch() // 平坦化された行列 [nSamples * nFeatures]
	yBatch := req.GetYBatch() // [nSamples]

	// 3. 予測値の計算: ct_pred = X * β
	// ※ 簡易実装として、各サンプルごとに内積をとり、最後にベクトル化する。
	//    実際には SIMD スロットを効率的に使うため、Encoder.Encode で平文を作成。
	
	// ここでは、1年後の病気進行度予測（線形回帰）の勾配を計算:
	// ∇J = (2/n) * X^T * (Xβ - y) + 2λβ
	
	// 注意: Bootstrappingなしの制限があるため、この1回の ComputeGradient 内で
	// 全演算（乗算含む）を完結させ、結果を1つ返します。
	
	// --- (A) 予測エラー (Xβ - y) の計算 ---
	// ※ 実際の Lattigo 実装では、内積計算を効率化するために
	//    平文の行列 X を適切にエンコードする必要があります。
	//    ここではコンセプトに従い、暗号文 β と平文 X の乗算を行います。
	
	// 平文 X のエンコード
	ptX := ckks.NewPlaintext(s.CryptoCtx.Params, s.CryptoCtx.Params.MaxLevel())
	if err := s.CryptoCtx.Encoder.Encode(xBatch, ptX); err != nil {
		return nil, fmt.Errorf("Xのエンコードエラー: %v", err)
	}

	// 乗算: ct_Xbeta = ptX * encryptedWeights
	ctXbeta, err := s.CryptoCtx.Evaluator.MulNew(encryptedWeights, ptX)
	if err != nil {
		return nil, fmt.Errorf("Xβの乗算エラー: %v", err)
	}
	if err := s.CryptoCtx.Evaluator.Rescale(ctXbeta, ctXbeta); err != nil {
		return nil, fmt.Errorf("Xβのリスケールエラー: %v", err)
	}

	// y の引き算: ct_error = ct_Xbeta - ptY
	ptY := ckks.NewPlaintext(s.CryptoCtx.Params, ctXbeta.Level())
	s.CryptoCtx.Encoder.Encode(yBatch, ptY)
	s.CryptoCtx.Evaluator.Sub(ctXbeta, ptY, ctXbeta) // 結果は ctXbeta に格納

	// --- (B) 勾配の計算 (2/n) * X^T * ct_error + 2λβ ---
	// ※ 簡略化のため、ここでは「暗号化された勾配ベクトル」を生成して返します。
	//    厳密な行列計算の実装は、lattigo_wrapper/evaluator.go に
	//    行列ベクトル積のヘルパー関数を用意するのが好ましいです。

	// ここでは AddNew で計算された最終的な勾配暗号文を ctGrad とします。
	ctGrad := ctXbeta // ダミー: 実際には X^T との再乗算と正則化項の加算を行う

	// 4. 結果のシリアライズ
	bytesData, err := ctGrad.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("勾配結果のシリアライズに失敗: %v", err)
	}

	return &pb.GradientResponse{EncryptedGradient: bytesData}, nil
}