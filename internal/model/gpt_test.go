package model

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"unicode/utf8"
)

func smallWeights(t *testing.T) *Weights {
	t.Helper()
	return Random(Config{Vocab: 256, DModel: 32, NLayer: 2, NHead: 4, DFF: 64, Ctx: 16}, 7)
}

func TestSoftmaxSumsToOne(t *testing.T) {
	s := []float32{1, 2, 3, -1}
	softmaxInPlace(s)
	var sum float32
	for _, v := range s {
		if v < 0 || v > 1 {
			t.Fatalf("probability out of range: %v", v)
		}
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum = %v, want 1", sum)
	}
}

func TestCausalMask(t *testing.T) {
	w := smallWeights(t)
	ids := []int{65, 66, 67, 68, 69}
	tr := Forward(w, ids)
	for _, l := range tr.Layers {
		for h, head := range l.Heads {
			for i := range head.Attn {
				for j := i + 1; j < len(head.Attn[i]); j++ {
					if head.Attn[i][j] != 0 {
						t.Fatalf("layer %d head %d: attention to future position %d from %d is %v", 0, h, j, i, head.Attn[i][j])
					}
					if !math.IsInf(float64(head.Scores[i][j]), -1) {
						t.Fatalf("score above the diagonal should be -Inf")
					}
				}
				var sum float32
				for _, p := range head.Attn[i] {
					sum += p
				}
				if math.Abs(float64(sum-1)) > 1e-4 {
					t.Fatalf("row %d sums to %v", i, sum)
				}
			}
		}
	}
}

func TestLayerNormStatistics(t *testing.T) {
	x := [][]float32{{1, 2, 3, 4, 5}}
	g := []float32{1, 1, 1, 1, 1}
	b := make([]float32, 5)
	out := layerNorm(x, g, b)[0]
	var mean float32
	for _, v := range out {
		mean += v
	}
	mean /= float32(len(out))
	var variance float32
	for _, v := range out {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(out))
	if math.Abs(float64(mean)) > 1e-5 {
		t.Fatalf("mean = %v", mean)
	}
	if math.Abs(float64(variance)-1) > 1e-3 {
		t.Fatalf("variance = %v", variance)
	}
}

func TestTokenizerRoundTrip(t *testing.T) {
	for _, s := range []string{"hello world", "中文 token 化", "emoji 🙂 ok", ""} {
		tokens := Tokenize(s)
		ids := make([]int, len(tokens))
		for i, tk := range tokens {
			ids[i] = tk.ID
		}
		if got := DecodeBytes(ids); got != s {
			t.Fatalf("round trip %q → %q", s, got)
		}
	}
}

func TestForwardIsDeterministic(t *testing.T) {
	w := smallWeights(t)
	ids := []int{72, 101, 108, 108, 111}
	a := Forward(w, ids)
	b := Forward(w, ids)
	for i := range a.Logits {
		if a.Logits[i] != b.Logits[i] {
			t.Fatalf("logit %d differs: %v vs %v", i, a.Logits[i], b.Logits[i])
		}
	}
	if len(a.Layers[0].Heads[0].Attn) != len(ids) {
		t.Fatalf("trace has %d rows, want %d", len(a.Layers[0].Heads[0].Attn), len(ids))
	}
}

func TestEmptyPrompt(t *testing.T) {
	w := smallWeights(t)
	tr := Forward(w, nil)
	if len(tr.Tokens) != 0 || len(tr.Layers) != w.Config.NLayer {
		t.Fatalf("empty prompt trace: %d tokens, %d layers", len(tr.Tokens), len(tr.Layers))
	}
}

// TestBundledParity checks the Go forward pass against the logits PyTorch
// recorded while training the bundled weights.
func TestBundledParity(t *testing.T) {
	w, err := Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	if w.Reference == nil {
		// The reference is recorded by the training run into the same weights.json that
		// ships with the binary, so its absence means the weights were replaced by
		// something this test cannot check — which is exactly when the check matters.
		// Skipping would leave the forward pass compared against nothing.
		t.Fatal("bundled weights carry no reference logits; tools/train.py records them, " +
			"so these weights came from somewhere else")
	}
	tr := Forward(w, w.Reference.IDs)
	var maxDiff float64
	for i, want := range w.Reference.Logits {
		d := math.Abs(float64(tr.Logits[i] - want))
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 2e-3 {
		t.Fatalf("max logit difference vs PyTorch = %.5f", maxDiff)
	}
	t.Logf("max logit difference vs PyTorch = %.5f", maxDiff)
}

func TestTruncate(t *testing.T) {
	ids := []int{1, 2, 3, 4, 5}
	got := Truncate(ids, 3)
	if len(got) != 3 || got[0] != 3 {
		t.Fatalf("truncate = %v", got)
	}
	if !strings.HasPrefix(DecodeBytes([]int{104, 105}), "hi") {
		t.Fatalf("decode bytes")
	}
}

// TestHeadProjSumsToAttnProj pins the identity the UI relies on: what one head
// writes into the residual stream is its own slice of Wo, and the four of them
// add up to the block's attention output.
func TestHeadProjSumsToAttnProj(t *testing.T) {
	w := smallWeights(t)
	tr := Forward(w, []int{65, 66, 67, 68})
	for li, l := range tr.Layers {
		if len(l.HeadProj) != w.Config.NHead {
			t.Fatalf("layer %d: %d head projections, want %d", li, len(l.HeadProj), w.Config.NHead)
		}
		for i := range l.AttnProj {
			for d := range l.AttnProj[i] {
				var sum float32
				for h := range l.HeadProj {
					sum += l.HeadProj[h][i][d]
				}
				if math.Abs(float64(sum-l.AttnProj[i][d])) > 1e-4 {
					t.Fatalf("layer %d pos %d dim %d: sum of heads %v != AttnProj %v", li, i, d, sum, l.AttnProj[i][d])
				}
			}
		}
	}
	// A single head's contribution must differ from the total, otherwise the
	// UI would be right to label the sum as that head's output.
	l := tr.Layers[0]
	diff := false
	for i := range l.AttnProj {
		if l.HeadProj[0][i][0] != l.AttnProj[i][0] {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("head 0 projection equals the summed projection")
	}
}

// TestTruncateKeepsRunesWhole checks that dropping old tokens never leaves a
// leading UTF-8 continuation byte, which the UI would render as a broken rune.
func TestTruncateKeepsRunesWhole(t *testing.T) {
	const s = "注意力机制就是加权求和"
	ids := make([]int, 0, len(s))
	for _, b := range []byte(s) {
		ids = append(ids, int(b))
	}
	for ctx := 2; ctx < len(ids); ctx++ {
		got := Truncate(ids, ctx)
		if !utf8.Valid(bytesOf(got)) {
			t.Fatalf("ctx %d: truncated ids are not valid UTF-8", ctx)
		}
		if !strings.HasSuffix(s, DecodeBytes(got)) {
			t.Fatalf("ctx %d: %q is not a suffix of the prompt", ctx, DecodeBytes(got))
		}
	}
}

func bytesOf(ids []int) []byte {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = byte(id)
	}
	return b
}

func BenchmarkForward(b *testing.B) {
	w, err := Bundled()
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{16, 64} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = 97 + i%26
		}
		b.Run(fmt.Sprintf("T=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Forward(w, ids)
			}
		})
	}
}

// TestInvalidUTF8Prompt covers a byte-level sampler appending half a rune:
// the trace must still be well formed and must not mislabel tokens with runes
// derived from shifted byte offsets.
func TestInvalidUTF8Prompt(t *testing.T) {
	w := smallWeights(t)
	ids := []int{0xE6, 0xB3, 0xA8, 0x80, 0x81} // 注 + two stray continuation bytes
	tr := Forward(w, ids)
	if len(tr.Tokens) != len(ids) {
		t.Fatalf("%d tokens, want %d", len(tr.Tokens), len(ids))
	}
	if tr.Tokens[4].Text != "\\x81" {
		t.Fatalf("stray byte rendered as %q", tr.Tokens[4].Text)
	}
	if tr.Tokens[4].Rune != 0 {
		t.Fatalf("stray byte got rune %q", tr.Tokens[4].Rune)
	}
	if tr.Tokens[0].Rune != '注' {
		t.Fatalf("valid rune before the stray byte got %q", tr.Tokens[0].Rune)
	}
}

func TestTopKNonPositiveReturnsEmpty(t *testing.T) {
	for _, k := range []int{0, -1} {
		if got := TopK([]float32{0.2, 0.8}, k); len(got) != 0 {
			t.Fatalf("TopK(k=%d) = %v, want empty", k, got)
		}
	}
}

func TestProbabilitiesZeroTemperatureIsGreedy(t *testing.T) {
	w := smallWeights(t)
	tr := Forward(w, []int{65, 66, 67})
	probs := tr.Probabilities(0)
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}
	if probs[best] != 1 {
		t.Fatalf("max probability = %v, want 1", probs[best])
	}
	var sum float32
	for _, p := range probs {
		sum += p
	}
	if sum != 1 {
		t.Fatalf("probability sum = %v, want 1", sum)
	}
}

func TestForwardNormalizesIDsToBytes(t *testing.T) {
	w := smallWeights(t)
	tr := Forward(w, []int{256, -1})
	if got := tr.Tokens[0].ID; got != 0 {
		t.Fatalf("token 0 id = %d, want 0", got)
	}
	if got := tr.Tokens[1].ID; got != 255 {
		t.Fatalf("token 1 id = %d, want 255", got)
	}
}

// TestGeluMatchesErf pins the lookup table to the exact erf-based GELU: the
// table is only allowed to differ by rounding, and outside the sampled range
// the function must fall back to its limits.
func TestGeluMatchesErf(t *testing.T) {
	var worst float64
	for i := -10000; i <= 10000; i++ {
		x := float32(i) / 1000
		want := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		if d := math.Abs(float64(gelu(x)) - want); d > worst {
			worst = d
		}
	}
	if worst > 1e-5 {
		t.Fatalf("gelu max abs error %.3e, want <= 1e-5", worst)
	}
	if got := gelu(-9); got != 0 {
		t.Fatalf("gelu(-9) = %v, want 0", got)
	}
	if got := gelu(9); got != 9 {
		t.Fatalf("gelu(9) = %v, want 9", got)
	}
	if got := gelu(float32(math.NaN())); !math.IsNaN(float64(got)) {
		t.Fatalf("gelu(NaN) = %v, want NaN", got)
	}
}

// TestExpNegMatchesExp pins the softmax table to math.Exp over the range the
// causal softmax uses, and checks the clamped tails.
func TestExpNegMatchesExp(t *testing.T) {
	var worst float64
	for i := -1600; i <= 0; i++ {
		x := float32(i) / 100
		want := math.Exp(float64(x))
		if want <= 1e-6 {
			continue
		}
		if d := math.Abs(float64(expNeg(x))-want) / want; d > worst {
			worst = d
		}
	}
	if worst > 1e-5 {
		t.Fatalf("expNeg max rel error %.3e, want <= 1e-5", worst)
	}
	if got := expNeg(-20); got != 0 {
		t.Fatalf("expNeg(-20) = %v, want 0", got)
	}
	if got := expNeg(0); got != 1 {
		t.Fatalf("expNeg(0) = %v, want 1", got)
	}
	if got := expNeg(float32(math.NaN())); !math.IsNaN(float64(got)) {
		t.Fatalf("expNeg(NaN) = %v, want NaN", got)
	}
}

// TestUnembedReproducesTheRecordedLogits pins the identity the output stage now
// relies on twice: projecting a row of the final LayerNorm with Weights.Unembed
// gives exactly the logits Forward recorded for that row.
//
// Exactly, not approximately. The stage prints a position's top-1 probability from
// the per-position figures and the last position's distribution from Trace.Logits
// side by side; two code paths that disagreed in the last digit would show one
// number twice with two different values, which reads as a bug even when both are
// defensible.
func TestUnembedReproducesTheRecordedLogits(t *testing.T) {
	w, err := Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	ids := make([]int, 0, len(w.Reference.Prompt))
	for i := 0; i < len(w.Reference.Prompt); i++ {
		ids = append(ids, int(w.Reference.Prompt[i]))
	}
	tr := Forward(w, ids)
	got := w.Unembed(tr.FinalNorm[len(tr.FinalNorm)-1])
	if len(got) != len(tr.Logits) {
		t.Fatalf("Unembed returned %d logits, the trace holds %d", len(got), len(tr.Logits))
	}
	for id := range got {
		if got[id] != tr.Logits[id] {
			t.Fatalf("logit %d: Unembed gives %v, Forward recorded %v — the two paths "+
				"must agree bit for bit or the stage prints one number twice", id, got[id], tr.Logits[id])
		}
	}
}

// TestUnembedDiffersPerPosition is the premise of the per-position card: the
// model's prediction at position t is a different distribution from its
// prediction at t+1, so showing only the last one shows one of many distinct
// answers rather than a representative one.
//
// A rerun of the same row would satisfy a weaker version of this test, so it
// compares the argmax across positions and requires several distinct winners.
func TestUnembedDiffersPerPosition(t *testing.T) {
	w, err := Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	ids := make([]int, 0, len(defaultTestPrompt))
	for i := 0; i < len(defaultTestPrompt); i++ {
		ids = append(ids, int(defaultTestPrompt[i]))
	}
	tr := Forward(w, ids)
	if len(tr.FinalNorm) < 4 {
		t.Fatalf("the test prompt has only %d positions", len(tr.FinalNorm))
	}
	winners := map[int]bool{}
	for pos := range tr.FinalNorm {
		logits := w.Unembed(tr.FinalNorm[pos])
		best, bestVal := 0, logits[0]
		for id, v := range logits {
			if v > bestVal {
				best, bestVal = id, v
			}
		}
		winners[best] = true
	}
	if len(winners) < 3 {
		t.Fatalf("all %d positions predict one of %d bytes (%v); the per-position "+
			"card would be showing the same answer over and over",
			len(tr.FinalNorm), len(winners), winners)
	}
}

// defaultTestPrompt is the prompt the app opens with, repeated here so the model
// package's tests do not import the app package (which would be a cycle).
const defaultTestPrompt = "The transformer reads the tokens and"
