//go:build goexperiment.simd && !amd64

package model

import "simd"

// matMulRowsRange computes rows [start, end) of out with the portable SIMD
// package: one weight vector feeds four rows at a time.
func matMulRowsRange(out, x [][]float32, w []float32, outDim, start, end int) {
	inDim := len(x[0])
	width := (simd.Float32s{}).Len()
	i := start
	for ; i+4 <= end; i += 4 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		r0, r1, r2, r3 := out[i], out[i+1], out[i+2], out[i+3]
		j := 0
		for ; j+width <= outDim; j += width {
			var a0, a1, a2, a3 simd.Float32s
			for k := 0; k < inDim; k++ {
				wv := simd.LoadFloat32s(w[k*outDim+j : k*outDim+j+width])
				a0 = simd.BroadcastFloat32s(x0[k]).MulAdd(wv, a0)
				a1 = simd.BroadcastFloat32s(x1[k]).MulAdd(wv, a1)
				a2 = simd.BroadcastFloat32s(x2[k]).MulAdd(wv, a2)
				a3 = simd.BroadcastFloat32s(x3[k]).MulAdd(wv, a3)
			}
			a0.Store(r0[j:])
			a1.Store(r1[j:])
			a2.Store(r2[j:])
			a3.Store(r3[j:])
		}
		for ; j < outDim; j++ {
			for k := 0; k < inDim; k++ {
				v := w[k*outDim+j]
				r0[j] += x0[k] * v
				r1[j] += x1[k] * v
				r2[j] += x2[k] * v
				r3[j] += x3[k] * v
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
