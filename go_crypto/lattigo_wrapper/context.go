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
		LogN:            12,
		LogQ:            []int{40, 25, 25, 25, 25, 25, 25, 25},
		LogP:            []int{25, 25},
		LogDefaultScale: 25,
	})

	if err != nil {
		return nil, fmt.Errorf("パラメータ生成エラー: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// 修正: 回転に必要な「ガロア要素」のリストを取得
	// 全スロットの回転をサポートするデフォルトのリストを使用します
	galEls := params.GaloisElementsForInnerSum(1, params.MaxSlots())
	// 追加: 回転用の鍵（Galois Keys）を生成 (2のべき乗のシフトをサポート)
    gks := kgen.GenGaloisKeysNew(galEls, sk)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)

	// 修正: EvaluationKeySet に GaloisKeys を追加
    evalKeySet := rlwe.NewMemEvaluationKeySet(rlk, gks...)
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