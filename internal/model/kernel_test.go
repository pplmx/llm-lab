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

// TestMatMulRowsBiasShapes pins what the bias parameter accepts.
//
// The signature takes a plain slice, so a caller can pass one shorter or longer than outDim,
// and the contract is not obvious from the types. It is: the first min(len(b), outDim) entries
// are added, anything beyond is ignored. That is what the old implementation did too, by way of
// copy() into the destination rows, and the fix that moved the bias after the multiply had to
// preserve it — a shape mismatch that used to be a silent truncation must not become a panic or
// a misaligned output.
//
// Worth a test because nothing in this model passes a non-nil bias: every call site in gpt.go
// uses nil, so these branches are reachable only from outside, and the behaviour there is a
// promise the signature makes rather than one the app depends on.
func TestMatMulRowsBiasShapes(t *testing.T) {
	const in, out = 2, 2
	x := [][]float32{{1, 2}, {3, 4}}
	// Identity product: w is in×out row-major, so this doubles nothing and passes x through.
	w := []float32{1, 0, 0, 1}
	base := matMulRows(x, w, nil, out)
	if base[0][0] != 1 || base[0][1] != 2 || base[1][0] != 3 || base[1][1] != 4 {
		t.Fatalf("the identity product is wrong before any bias: %v", base)
	}

	for _, c := range []struct {
		name string
		bias []float32
		want [][]float32
	}{
		{"nil", nil, [][]float32{{1, 2}, {3, 4}}},
		{"empty", []float32{}, [][]float32{{1, 2}, {3, 4}}},
		{"exact length", []float32{10, 20}, [][]float32{{11, 22}, {13, 24}}},
		{"shorter than outDim", []float32{10}, [][]float32{{11, 2}, {13, 4}}},
		// Longer is truncated, not applied shifted and not a panic.
		{"longer than outDim", []float32{10, 20, 30}, [][]float32{{11, 22}, {13, 24}}},
	} {
		got := matMulRows(x, w, c.bias, out)
		for i := range c.want {
			for j := range c.want[i] {
				if got[i][j] != c.want[i][j] {
					t.Fatalf("%s: got %v, want %v", c.name, got, c.want)
				}
			}
		}
	}
}

// TestSoftmaxCausalMatchesTheDefinition pins the causal softmax against a direct computation
// from its definition.
//
// It is the one math helper in this package with no accuracy test of its own. TestCausalMask
// reaches it through Forward and checks the properties a reader would notice — the row sums to
// 1, the future entries are zero — but those hold for any correctly shaped distribution, so a
// defect that produced a plausible-looking row in the wrong order would pass. This compares
// the actual values.
//
// Three things the indirect test cannot see are checked here: that dst beyond n is zeroed
// rather than left holding whatever the caller's buffer contained, that a row of one entry
// comes out as exactly 1, and that the biggest score gets the biggest weight — the ordering,
// which is the whole point of a softmax and which a sum-to-1 check says nothing about.
func TestSoftmaxCausalMatchesTheDefinition(t *testing.T) {
	rows := [][]float32{
		{0},
		{2.5},
		{1, -1},
		{0, 0, 0},
		{5, 4, 3, 2, 1},
		{-3.25, -3.5, 7.75},
		{100, 100, 100}, // large enough that exp overflows without the max subtraction
	}
	for _, scores := range rows {
		n := len(scores)
		// A destination longer than n, pre-filled with a sentinel: the masked tail must be
		// overwritten with zero, not left alone.
		dst := make([]float32, n+3)
		for i := range dst {
			dst[i] = -12345
		}
		softmaxCausal(dst, scores, n)

		var max float64 = math.Inf(-1)
		for _, v := range scores {
			if f := float64(v); f > max {
				max = f
			}
		}
		var sum float64
		for _, v := range scores {
			sum += math.Exp(float64(v) - max)
		}
		var worst float64
		for i, v := range scores {
			want := math.Exp(float64(v)-max) / sum
			if d := math.Abs(float64(dst[i]) - want); d > worst {
				worst = d
			}
		}
		if worst > 1e-6 {
			t.Fatalf("scores %v: worst deviation from the definition is %.3e", scores, worst)
		}
		// The masked tail.
		for i := n; i < len(dst); i++ {
			if dst[i] != 0 {
				t.Fatalf("scores %v: dst[%d] = %v, want 0 — the masked tail must be cleared",
					scores, i, dst[i])
			}
		}
		// A single-entry row is a certainty.
		if n == 1 && dst[0] != 1 {
			t.Fatalf("a one-entry row gave %v, want exactly 1", dst[0])
		}
		// Ordering: the largest score must hold the largest weight. Distinct maximum only,
		// or the assertion would depend on tie-breaking.
		best, ties := 0, 0
		for i := range scores {
			if scores[i] > scores[best] {
				best = i
			}
			if i > 0 && scores[i] == scores[0] {
				ties++
			}
		}
		if ties == 0 {
			for i := range scores {
				if i != best && dst[i] > dst[best] {
					t.Fatalf("scores %v: entry %d outweighs the largest score's entry %d (%v < %v)",
						scores, i, best, dst[best], dst[i])
				}
			}
		}
	}
}
