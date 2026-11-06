//go:build !goexperiment.simd

package model

// matMulRowsRange computes rows [start, end) of out.
func matMulRowsRange(out, x [][]float32, w []float32, outDim, start, end int) {
	inDim := len(x[0])
	// Four rows share one pass over w: each weight row is loaded once for
	// four accumulators instead of once per row. The wide FFN matrices
	// (96×384) do not fit in L1, so this cuts the dominant memory traffic.
	i := start
	for ; i+4 <= end; i += 4 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		r0, r1, r2, r3 := out[i], out[i+1], out[i+2], out[i+3]
		for k := 0; k < inDim; k++ {
			wk := w[k*outDim : (k+1)*outDim]
			v0, v1, v2, v3 := x0[k], x1[k], x2[k], x3[k]
			for j, wv := range wk {
				r0[j] += v0 * wv
				r1[j] += v1 * wv
				r2[j] += v2 * wv
				r3[j] += v3 * wv
			}
		}
	}
	for ; i < end; i++ {
		xi := x[i]
		row := out[i]
		for k := 0; k < inDim; k++ {
			v := xi[k]
			wk := w[k*outDim : (k+1)*outDim]
			for j, wv := range wk {
				row[j] += v * wv
			}
		}
	}
}
