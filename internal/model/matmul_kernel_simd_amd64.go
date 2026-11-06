//go:build goexperiment.simd && amd64

package model

import "simd/archsimd"

// matMulRowsRange computes rows [start, end) of out with AVX vectors: one
// weight pair feeds four rows at a time, so sixteen multiply-adds share a
// single load of each weight vector.
func matMulRowsRange(out, x [][]float32, w []float32, outDim, start, end int) {
	inDim := len(x[0])
	i := start
	for ; i+4 <= end; i += 4 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		r0, r1, r2, r3 := out[i], out[i+1], out[i+2], out[i+3]
		j := 0
		for ; j+16 <= outDim; j += 16 {
			var a0, a1, b0, b1, c0, c1, d0, d1 archsimd.Float32x8
			for k := 0; k < inDim; k++ {
				w0 := archsimd.LoadFloat32x8(w[k*outDim+j:])
				w1 := archsimd.LoadFloat32x8(w[k*outDim+j+8:])
				v0 := archsimd.BroadcastFloat32x8(x0[k])
				v1 := archsimd.BroadcastFloat32x8(x1[k])
				v2 := archsimd.BroadcastFloat32x8(x2[k])
				v3 := archsimd.BroadcastFloat32x8(x3[k])
				a0 = v0.MulAdd(w0, a0)
				a1 = v0.MulAdd(w1, a1)
				b0 = v1.MulAdd(w0, b0)
				b1 = v1.MulAdd(w1, b1)
				c0 = v2.MulAdd(w0, c0)
				c1 = v2.MulAdd(w1, c1)
				d0 = v3.MulAdd(w0, d0)
				d1 = v3.MulAdd(w1, d1)
			}
			a0.Store(r0[j:])
			a1.Store(r0[j+8:])
			b0.Store(r1[j:])
			b1.Store(r1[j+8:])
			c0.Store(r2[j:])
			c1.Store(r2[j+8:])
			d0.Store(r3[j:])
			d1.Store(r3[j+8:])
		}
		for ; j+8 <= outDim; j += 8 {
			var a0, b0, c0, d0 archsimd.Float32x8
			for k := 0; k < inDim; k++ {
				w0 := archsimd.LoadFloat32x8(w[k*outDim+j:])
				a0 = archsimd.BroadcastFloat32x8(x0[k]).MulAdd(w0, a0)
				b0 = archsimd.BroadcastFloat32x8(x1[k]).MulAdd(w0, b0)
				c0 = archsimd.BroadcastFloat32x8(x2[k]).MulAdd(w0, c0)
				d0 = archsimd.BroadcastFloat32x8(x3[k]).MulAdd(w0, d0)
			}
			a0.Store(r0[j:])
			b0.Store(r1[j:])
			c0.Store(r2[j:])
			d0.Store(r3[j:])
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
