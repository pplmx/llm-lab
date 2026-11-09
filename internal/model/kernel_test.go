package model

import (
	"math"
	"testing"
)

// TestMatMulRowsMatchesANaiveReference checks the active matrix-multiply kernel against a
// plain triple loop, over shapes that reach every branch of it.
//
// There was no such test. The kernel is chosen at build time — matmul_kernel_scalar.go or
// one of the two SIMD files — and the three implementations share nothing but a signature,
// so a defect in one of them is visible only when that build runs. CI runs two of the three:
// the default amd64 build and GOEXPERIMENT=simd on amd64. The portable SIMD kernel
// (matmul_kernel_simd_generic.go) is compiled only for non-amd64 under the experiment, which
// means arm64 SIMD is the one configuration where the arithmetic has no test that compares it
// to anything. This test runs in every configuration, so it covers whichever kernel is in
// play rather than the two the CI happens to build.
//
// The shapes are chosen for the kernel's control flow rather than for realism: rows that are
// a multiple of four (the blocked path), rows that leave a 1-3 row remainder (the scalar tail
// loop), and outDim values below, at, and above a vector width, which is 8 on amd64 and
// whatever the portable package reports elsewhere.
func TestMatMulRowsMatchesANaiveReference(t *testing.T) {
	naive := func(x [][]float32, w []float32, b []float32, outDim int) [][]float32 {
		inDim := len(x[0])
		out := make([][]float32, len(x))
		for i := range x {
			out[i] = make([]float32, outDim)
			for j := 0; j < outDim; j++ {
				var s float64
				for k := 0; k < inDim; k++ {
					s += float64(x[i][k]) * float64(w[k*outDim+j])
				}
				if b != nil {
					s += float64(b[j])
				}
				out[i][j] = float32(s)
			}
		}
		return out
	}

	type shape struct {
		name          string
		rows, in, out int
		bias          bool
	}
	shapes := []shape{
		{"one row, one column", 1, 1, 1, false},
		{"rows below the block size", 3, 5, 7, false},
		{"rows are exactly one block", 4, 5, 7, true},
		{"rows leave a 1-row remainder", 5, 6, 8, false},
		{"rows leave a 2-row remainder", 6, 6, 9, true},
		{"rows leave a 3-row remainder", 7, 7, 16, false},
		{"outDim is one vector width", 4, 3, 8, true},
		{"outDim just over a vector width", 4, 3, 9, false},
		{"outDim well over a vector width", 8, 5, 33, true},
		{"the FFN's wide matrix", 4, 96, 384, true},
		{"the model's inner dimension", 4, 96, 96, false},
	}
	// A deterministic pattern, not rand: a failure has to be reproducible from the test
	// alone, and a random seed that only fails sometimes is worse than no test.
	fill := func(n int, mul float32, add int) []float32 {
		out := make([]float32, n)
		for i := range out {
			v := float32((i*add)%17-8) * mul
			out[i] = v
		}
		return out
	}

	for _, s := range shapes {
		s := s
		t.Run(s.name, func(t *testing.T) {
			x := make([][]float32, s.rows)
			for i := range x {
				x[i] = fill(s.in, 0.25, i+1)
			}
			w := fill(s.in*s.out, 0.125, 3)
			var b []float32
			if s.bias {
				b = fill(s.out, 0.5, 5)
			}

			want := naive(x, w, b, s.out)
			got := matMulRows(x, w, b, s.out)
			if len(got) != len(want) {
				t.Fatalf("%d rows, want %d", len(got), len(want))
			}
			worst := 0.0
			for i := range want {
				for j := range want[i] {
					d := math.Abs(float64(got[i][j] - want[i][j]))
					if d > worst {
						worst = d
					}
				}
			}
			// The kernels accumulate in float32 in different orders — the vector ones use
			// fused multiply-add over lanes, the scalar one rounds twice per term — so exact
			// equality is not available. The tolerance is relative to the magnitude of the
			// terms so it does not have to be retuned when a shape gets bigger: with
			// inDim=384 and terms of order 2, a float32 dot has ~1e-4 of accumulated error.
			var scale float32
			for _, v := range w {
				if av := float32(math.Abs(float64(v))); av > scale {
					scale = av
				}
			}
			tol := float64(scale) * float64(s.in) * 1e-5
			if worst > tol {
				t.Fatalf("worst deviation %.3e exceeds %.3e over a %dx%dx%d product; the "+
					"active kernel disagrees with a naive dot", worst, tol, s.rows, s.in, s.out)
			}
		})
	}
}

// TestMatMulRowsCoversEveryRemainder is the part of the kernel a shape sweep is meant to
// reach and a single realistic product would not: rows beyond the last full block of four.
//
// The blocked loop handles rows in fours and a scalar loop finishes the rest. A product
// whose row count is a multiple of four never enters that loop, so a defect in it would
// only show up at other row counts — and every real forward pass in this model happens to
// use 2, 25, 36 or 64 rows, three of which leave a remainder. This asserts the remainder
// rows are computed at all, which a "did it produce a matrix of the right size" check
// cannot: an unwritten tail leaves rows of zeros, and zeros are the right shape.
func TestMatMulRowsCoversEveryRemainder(t *testing.T) {
	const inDim, outDim = 5, 8
	x := make([][]float32, 7)
	for i := range x {
		x[i] = make([]float32, inDim)
		for k := range x[i] {
			x[i][k] = float32(i+1) * 0.5
		}
	}
	w := make([]float32, inDim*outDim)
	for i := range w {
		w[i] = 0.25
	}
	got := matMulRows(x, w, nil, outDim)
	for i := range got {
		for j := range got[i] {
			if got[i][j] == 0 {
				t.Fatalf("row %d column %d is zero; a row past the last full block was "+
					"never computed", i, j)
			}
		}
	}
}
