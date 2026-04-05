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


// SumSlots はベクトル内の全要素を合計し、全スロットにその合計値を格納します
func (hc *CryptoContext) SumSlots(ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	res := ct.CopyNew()
	for i := 1; i < hc.Params.MaxSlots(); i <<= 1 {
		// 修正: RotateNew は (Ciphertext, error) を返すため、2つの変数で受け取る
		rotated, err := hc.Evaluator.RotateNew(res, i)
		if err != nil {
			continue // エラー時はスキップ、またはログ出力
		}
		// Add 自体は通常エラーを返しません
		hc.Evaluator.Add(res, rotated, res)
	}
	return res
}

// MatrixVectorMulPlain は A(平文行列) * ct(暗号化ベクトル) を正確に計算します
func (hc *CryptoContext) MatrixVectorMulPlain(matrix [][]float64, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	dim := len(matrix)
	var res *rlwe.Ciphertext

	for i := 0; i < dim; i++ {
		ptRow := ckks.NewPlaintext(hc.Params, ct.Level())
		hc.Encoder.Encode(matrix[i], ptRow)
		
		// 修正: MulNew は (Ciphertext, error) を返す
		prod, err := hc.Evaluator.MulNew(ct, ptRow)
		if err != nil {
			return nil, err
		}
		hc.Evaluator.Rescale(prod, prod)
		
		// 内積の合計を算出
		summed := hc.SumSlots(prod)
		
		mask := make([]float64, hc.Params.MaxSlots())
		mask[i] = 1.0
		ptMask := ckks.NewPlaintext(hc.Params, summed.Level())
		hc.Encoder.Encode(mask, ptMask)
		
		// 修正: ここも MulNew の戻り値を 2つにする
		component, err := hc.Evaluator.MulNew(summed, ptMask)
		if err != nil {
			return nil, err
		}
		hc.Evaluator.Rescale(component, component)
		
		if res == nil {
			res = component
		} else {
			hc.Evaluator.Add(res, component, res)
		}
	}
	return res, nil
}