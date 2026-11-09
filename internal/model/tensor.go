package model

import (
	"math"
	"runtime"
	"sync"
)

// negInf is the causal mask value stored above the diagonal of the score
// matrices. Keeping it as a variable avoids a float64 conversion per cell.
var negInf = float32(math.Inf(-1))

// newMatrix returns rows×cols float32 rows backed by a single allocation.
// A forward pass records thousands of small rows; one backing array per
// tensor keeps the allocator and the garbage collector out of the hot path.
func newMatrix(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	if rows == 0 || cols == 0 {
		return out
	}
	buf := make([]float32, rows*cols)
	for i := range out {
		out[i] = buf[i*cols : (i+1)*cols]
	}
	return out
}

// parallelWork is the multiply-add count below which splitting a matmul
// across goroutines costs more than it saves.
const parallelWork = 1 << 18

// matMulRows multiplies each row of x (rows×inDim) by w (inDim×outDim, row
// major) and adds bias b, returning rows×outDim. Large products are split
// across rows: every row is an independent dot product, so the split is
// exact and needs no synchronization beyond waiting for the workers.
//
// The bias is added after the multiply, not written into out beforehand, because the kernels
// overwrite their rows rather than accumulating into them: matMulRowsRange's vector paths end
// with Store, which replaces the destination, while only their scalar tail loops use +=.
// Seeding out with the bias therefore survived for the columns the scalar tail handled and
// vanished for every column a vector wrote — a bug that no caller could see, because every
// call site in this model passes a nil bias. TestMatMulRowsMatchesANaiveReference caught it
// on the first run of the amd64 SIMD build.
func matMulRows(x [][]float32, w []float32, b []float32, outDim int) [][]float32 {
	if len(x) == 0 {
		return nil
	}
	inDim := len(x[0])
	out := newMatrix(len(x), outDim)
	rows := len(x)
	workers := 1
	if rows*inDim*outDim >= parallelWork {
		workers = runtime.GOMAXPROCS(0)
		if max := rows / 4; workers > max { // keep at least one 4-row block each
			workers = max
		}
		// Beyond eight workers the per-matmul synchronization and the shared
		// weight traffic cost more than the extra cores save at this size.
		if workers > 8 {
			workers = 8
		}
	}
	if workers <= 1 {
		matMulRowsRange(out, x, w, outDim, 0, rows)
		addBias(out, b)
		return out
	}
	chunk := (rows + workers - 1) / workers
	chunk = (chunk + 3) / 4 * 4
	var wg sync.WaitGroup
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matMulRowsRange(out, x, w, outDim, start, end)
		}(start, end)
	}
	wg.Wait()
	addBias(out, b)
	return out
}

// addBias adds b to every row of out. Separate from matMulRows so the multiply stays a
// multiply: the kernels must be free to overwrite their destination, which is what lets them
// use vector stores, and a bias applied afterwards is the only arrangement that works for
// both the accumulating scalar loop and the overwriting vector ones.
func addBias(out [][]float32, b []float32) {
	if b == nil {
		return
	}
	for _, row := range out {
		for j := range b {
			if j < len(row) {
				row[j] += b[j]
			}
		}
	}
}

// layerNorm normalizes each row to zero mean and unit variance, then scales
// and shifts it with g and b.
func layerNorm(x [][]float32, g, b []float32) [][]float32 {
	out := newMatrix(len(x), len(g))
	const eps = 1e-5
	for i, row := range x {
		var sum float32
		for _, v := range row {
			sum += v
		}
		mean := sum / float32(len(row))
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(len(row))
		inv := float32(1 / math.Sqrt(float64(variance+eps)))
		res := out[i]
		for j, v := range row {
			res[j] = (v-mean)*inv*g[j] + b[j]
		}
	}
	return out
}

// The GELU table samples the exact erf-based function on [-8, 8]; outside
// that interval the result is 0 or the identity to within 1e-9. Linear
// interpolation between the samples is accurate to 4e-7, and looking the
// value up is roughly twenty times faster than math.Erf, which otherwise
// dominates the FFN block.
const (
	geluN     = 8192
	geluMin   = -8
	geluMax   = 8
	geluScale = float32(geluN) / float32(geluMax-geluMin)
)

var geluTable = func() [geluN + 1]float32 {
	var t [geluN + 1]float32
	step := float64(geluMax-geluMin) / geluN
	for i := range t {
		x := float64(geluMin) + step*float64(i)
		t[i] = float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2)))
	}
	return t
}()

// gelu is the Gaussian error linear unit PyTorch uses, looked up from the
// table above.
func gelu(v float32) float32 {
	if v != v { // NaN
		return v
	}
	if v <= geluMin {
		return 0
	}
	if v >= geluMax {
		return v
	}
	f := (v - geluMin) * geluScale
	i := int(f)
	frac := f - float32(i)
	return geluTable[i] + frac*(geluTable[i+1]-geluTable[i])
}

// addInto stores a+b into dst.
func addInto(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// The softmax table samples e^x on [-16, 0]. Below -16 the value is under
// 1.2e-7 and rounds to zero; above 0 it is clamped to one. Linear
// interpolation is accurate to 2.4e-6 relative, and the lookup is about eight
// times faster than math.Exp, which otherwise dominates the attention softmax.
const (
	expN  = 4096
	expLo = -16
	expHi = 0
	expSc = float32(expN) / float32(expHi-expLo)
)

var expTable = func() [expN + 1]float32 {
	var t [expN + 1]float32
	step := float64(expHi-expLo) / expN
	for i := range t {
		t[i] = float32(math.Exp(float64(expLo) + step*float64(i)))
	}
	return t
}()

// expNeg returns e^x for x <= 0.
func expNeg(x float32) float32 {
	if x != x { // NaN
		return x
	}
	if x <= expLo {
		return 0
	}
	if x >= expHi {
		return 1
	}
	f := (x - expLo) * expSc
	i := int(f)
	frac := f - float32(i)
	return expTable[i] + frac*(expTable[i+1]-expTable[i])
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

// softmaxCausal writes the softmax of the first n scores into dst and zeroes
// the rest of dst, which is the shape causal attention stores. Only the
// unmasked entries are visited; the -Inf tail is never exponentiated.
func softmaxCausal(dst, scores []float32, n int) {
	max := scores[0]
	for _, v := range scores[1:n] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		e := expNeg(scores[i] - max)
		dst[i] = e
		sum += e
	}
	if sum > 0 {
		inv := 1 / sum
		for i := 0; i < n; i++ {
			dst[i] *= inv
		}
	}
	for i := n; i < len(dst); i++ {
		dst[i] = 0
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
