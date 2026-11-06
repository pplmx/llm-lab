package model

import "math"

// matMulRows multiplies each row of x (rows×inDim) by w (inDim×outDim, row
// major) and adds bias b, returning rows×outDim.
func matMulRows(x [][]float32, w []float32, b []float32, outDim int) [][]float32 {
	if len(x) == 0 {
		return nil
	}
	inDim := len(x[0])
	out := make([][]float32, len(x))
	for i := range x {
		row := make([]float32, outDim)
		if b != nil {
			copy(row, b)
		}
		xi := x[i]
		for k := 0; k < inDim; k++ {
			v := xi[k]
			// Forward inputs are dense after LayerNorm; skipping zero values
			// costs a branch per element and is slower than a straight loop.
			// Re-slicing keeps both operands the same length so the bounds
			// check on wv is provably in range and the loop stays tight.
			wk := w[k*outDim : (k+1)*outDim]
			for j, wv := range wk {
				row[j] += v * wv
			}
		}
		out[i] = row
	}
	return out
}

// layerNorm normalizes each row to zero mean and unit variance, then scales
// and shifts it with g and b.
func layerNorm(x [][]float32, g, b []float32) [][]float32 {
	out := make([][]float32, len(x))
	const eps = 1e-5
	for i, row := range x {
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(len(row))
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(len(row))
		inv := float32(1 / math.Sqrt(float64(variance+eps)))
		res := make([]float32, len(row))
		for j, v := range row {
			res[j] = (v-mean)*inv*g[j] + b[j]
		}
		out[i] = res
	}
	return out
}

// gelu is the exact Gaussian error linear unit tanh-free form PyTorch uses.
func gelu(v float32) float32 {
	return 0.5 * v * (1 + float32(math.Erf(float64(v)/math.Sqrt2)))
}

func addRow(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// softmaxInPlace normalizes scores to a distribution, skipping -Inf entries.
func softmaxInPlace(scores []float32) {
	max := float32(math.Inf(-1))
	for _, v := range scores {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range scores {
		if math.IsInf(float64(v), -1) {
			scores[i] = 0
			continue
		}
		e := float32(math.Exp(float64(v - max)))
		scores[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	for i := range scores {
		scores[i] /= sum
	}
}

// softmaxCopy returns a temperature-scaled softmax of logits without touching
// its input. A non-positive temperature is the zero-temperature limit: put all
// mass on the highest logit.
func softmaxCopy(logits []float32, temperature float64) []float32 {
	out := make([]float32, len(logits))
	if len(logits) == 0 {
		return out
	}
	if !(temperature > 0) { // also catches NaN
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		out[best] = 1
		return out
	}
	max := float32(math.Inf(-1))
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v-max) / temperature)
		out[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		return out
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}
