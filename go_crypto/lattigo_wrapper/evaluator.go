package lattigo_wrapper

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// AddNew は 2つの暗号文の和を新しい暗号文として返します
func (hc *CryptoContext) AddNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return hc.Evaluator.AddNew(ct1, ct2)
}

// MulNew は 暗号文と平文の乗算を新しい暗号文として返します
func (hc *CryptoContext) MulNew(ct *rlwe.Ciphertext, pt *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return hc.Evaluator.MulNew(ct, pt)
}

// Sub は 暗号文 ct1 から 平文 pt を引き、結果を res に格納します
func (hc *CryptoContext) Sub(ct1 *rlwe.Ciphertext, pt *rlwe.Ciphertext, res *rlwe.Ciphertext) {
	hc.Evaluator.Sub(ct1, pt, res)
}

// Rescale は 乗算後の暗号文のスケールを調整します（CKKSでは必須）
func (hc *CryptoContext) Rescale(ctIn, ctOut *rlwe.Ciphertext) error {
	return hc.Evaluator.Rescale(ctIn, ctOut)
}

// MulRelinNew は 暗号文同士の乗算を行い、リニアライゼーション（次数低減）を適用します
func (hc *CryptoContext) MulRelinNew(ct1, ct2 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	// リニアライゼーションキーが必要なため、hc.Evaluator にセットされている必要があります
	return hc.Evaluator.MulRelinNew(ct1, ct2)
}

// MultiplyByScalarNew は 暗号文に平文のスカラー（実数）を乗算します
func (hc *CryptoContext) MultiplyByScalarNew(ct *rlwe.Ciphertext, scalar float64) (*rlwe.Ciphertext, error) {
	// 全スロットに同じスカラー値をエンコードして乗算する簡易実装
	slots := hc.Params.MaxSlots()
	values := make([]complex128, slots)
	for i := range values {
		values[i] = complex(scalar, 0)
	}
	
	pt := ckks.NewPlaintext(hc.Params, ct.Level())
	if err := hc.Encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("scalar encoding error: %v", err)
	}
	
	res, err := hc.Evaluator.MulNew(ct, pt)
	if err != nil {
		return nil, err
	}
	
	if err := hc.Rescale(res, res); err != nil {
		return nil, err
	}
	
	return res, nil
}