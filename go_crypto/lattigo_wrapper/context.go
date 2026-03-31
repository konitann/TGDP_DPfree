package lattigo_wrapper

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CryptoContext はCKKSの演算に必要なすべてのツールを保持します
type CryptoContext struct {
	Params    ckks.Parameters
	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evaluator *ckks.Evaluator
	KeyGen    *rlwe.KeyGenerator
}

// NewCryptoContext は新しい暗号コンテキストを初期化します
func NewCryptoContext() (*CryptoContext, error) {
	fmt.Println("CKKSパラメータ(v6.2.0)を初期化")

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40}, // 暗号文のレベル（掛け算の可能回数）を決定
		LogP:            []int{45, 45},                         // 鍵の切り替えに使うパラメータ
		LogDefaultScale: 40,                                    // 小数点以下の精度を決めるスケール
	})

	if err != nil {
		return nil, fmt.Errorf("パラメータ生成エラー: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)

	evalKeySet := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := ckks.NewEvaluator(params, evalKeySet)

	fmt.Println("initialized")

	return &CryptoContext{
		Params:    params,
		Encoder:   encoder,
		Encryptor: encryptor,
		Decryptor: decryptor,
		Evaluator: evaluator,
		KeyGen:    kgen,
	}, nil
}