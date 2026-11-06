package app

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// Each misconception callout makes claims about this model, and a callout is the
// most damaging place to be wrong: it states a correction in the imperative,
// so a reader who trusts it learns the wrong thing twice.
//
// These verify the arithmetic underneath each one. The numbers rendered are
// already covered where they come from live measurements; what is checked here
// is that the measurements support the sentence.

// TestSoftmaxCalloutShiftClaim: subtracting 5 from every score leaves the
// softmax unchanged — the callout promises exactly that.
func TestSoftmaxCalloutShiftClaim(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	i := a.SelToken
	row := h.Scores[i]

	shifted := make([]float64, i+1)
	max := math.Inf(-1)
	for k := 0; k <= i; k++ {
		shifted[k] = float64(row[k]) - 5
		if shifted[k] > max {
			max = shifted[k]
		}
	}
	var sum float64
	for k := range shifted {
		shifted[k] = math.Exp(shifted[k] - max)
		sum += shifted[k]
	}
	for k := 0; k <= i; k++ {
		shifted[k] /= sum
		if d := math.Abs(shifted[k] - float64(h.Attn[i][k])); d > 2e-6 {
			t.Fatalf("key %d: shifted weight %.8f, actual %.8f (diff %.2e); "+
				"the callout promises these are identical", k, shifted[k], h.Attn[i][k], d)
		}
	}
}

// calloutText returns the rendered correction whose text begins with prefix.
// Matching on a prefix rather than "the longest text on the stage" because the
// lesson body is longer than most corrections and sits on the same stage.
func calloutText(t *testing.T, step int, prefix string) string {
	t.Helper()
	a := newTestApp(t)
	a.Step = step
	tester := ui.NewTester(a.View, 1500, 940)
	tester.Frame()
	for _, s := range tester.Texts() {
		if strings.HasPrefix(s, prefix) {
			return s
		}
	}
	t.Fatalf("step %d has no correction starting with %q", step+1, prefix)
	return ""
}

// TestAttentionColumnSumClaim: the callout says the column sums add to the
// token count, because each row contributes 1 across its columns.
//
// This reads the number out of the rendered sentence, not out of the model: the
// model's total is 36 by construction, so a test that recomputed it would pass
// even if the callout printed something else.
func TestAttentionColumnSumClaim(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	n := len(a.Trace.Tokens)
	var total float64
	for c := 0; c < n; c++ {
		for r := c; r < n; r++ {
			total += float64(h.Attn[r][c])
		}
	}
	if math.Abs(total-float64(n)) > 1e-3 {
		t.Fatalf("column sums total %.6f, one row contributes 1 each so it must be %d", total, n)
	}

	// And the sentence must print that same number.
	text := calloutText(t, 3, "只有行有这个约束")
	want := fmt.Sprintf("总共是 %d", n)
	if !strings.Contains(text, want) {
		t.Fatalf("the callout does not print %q:\n%s", want, text)
	}
}

// TestGELUCalloutQuotesTheRealFloor checks the sentence itself: it must print the
// geluFloor constant, and that constant must be GELU's actual minimum.
func TestGELUCalloutQuotesTheRealFloor(t *testing.T) {
	probe := newTestApp(t)
	probe.Step = 5
	text := calloutText(t, 5, fmt.Sprintf("%d 个维度里", probe.W.Config.DFF))
	want := fmt.Sprintf("最小值 %.3f", geluFloor)
	if !strings.Contains(text, want) {
		t.Fatalf("the callout does not print %q:\n%s", want, text)
	}
}

// TestGELUCalloutFloorClaim: the callout quotes the minimum of GELU as -0.170 at
// x ≈ -0.75, and says the negative region dips into a pit before returning to 0.
func TestGELUCalloutFloorClaim(t *testing.T) {
	// The model's gelu is unexported, so this recomputes the definition.
	g := func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }
	minX, minV := 0.0, math.Inf(1)
	for x := -6.0; x < 0; x += 1e-4 {
		if v := g(x); v < minV {
			minX, minV = x, v
		}
	}
	if math.Abs(minV-geluFloor) > 0.002 {
		t.Fatalf("GELU minimum is %.4f but geluFloor is %.3f", minV, geluFloor)
	}
	if math.Abs(minX+0.75) > 0.01 {
		t.Fatalf("GELU minimum is at x=%.4f, the callout says about -0.75", minX)
	}
	// A pit, not a floor: the value is below zero throughout and comes back.
	if g(-3) > -1e-3 || g(-3) < -1e-2 {
		t.Fatalf("GELU(-3) = %.5f; the negative region should be back near zero", g(-3))
	}
	if v := g(minX + 1); v <= minV {
		t.Fatalf("GELU does not rise after its minimum (%.5f <= %.5f)", v, minV)
	}
}

// TestResidualCalloutGrowthClaim: the callout says the stream gets longer at
// every layer and ends about 5x its starting size.
func TestResidualCalloutGrowthClaim(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	n := len(tr.Tokens)
	for li := range tr.Layers {
		grew := 0
		for i := 0; i < n; i++ {
			if norm(tr.Layers[li].AfterFFN[i]) > norm(tr.Layers[li].In[i]) {
				grew++
			}
		}
		if grew != n {
			t.Fatalf("layer %d grew the stream on %d/%d tokens, not all of them", li, grew, n)
		}
	}
	in, out := residualNormMeans(tr)
	if out <= in {
		t.Fatalf("mean norm %.2f -> %.2f: the stream did not grow", in, out)
	}
	if ratio := out / in; ratio < 2 || ratio > 20 {
		t.Fatalf("mean norm grew %.1fx; the callout would be quoting a strange scale", ratio)
	}
}

// TestLNCalloutRangeClaim: the callout says every normalized row has length
// near sqrt(d_model). The point of LayerNorm is that this holds regardless of
// how large the input row is.
func TestLNCalloutRangeClaim(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	lo, hi := lnNormRange(tr.Layers, len(tr.Tokens), tr.FinalNorm)
	pin := math.Sqrt(float64(tr.Config.DModel))
	if lo <= 0 || hi <= 0 || lo > hi {
		t.Fatalf("range [%.2f, %.2f] is not a range", lo, hi)
	}
	// "almost constant" has to mean the two ends are close together.
	if hi/lo > 2 {
		t.Fatalf("normalized norms range %.2f to %.2f, a factor of %.2f", lo, hi, hi/lo)
	}
	// And the middle of that range should be near sqrt(d_model).
	mid := (lo + hi) / 2
	if math.Abs(mid-pin)/pin > 0.25 {
		t.Fatalf("normalized norms centre on %.2f, but sqrt(d_model) is %.2f", mid, pin)
	}
}

// TestResidualCalloutQuotesTheGrowth checks the sentence prints the ratio it
// describes, and that the ratio follows the measurement.
func TestResidualCalloutQuotesTheGrowth(t *testing.T) {
	text := calloutText(t, 4, "残差流确实会一路变长")
	a := newTestApp(t)
	in, out := residualNormMeans(a.Trace)
	if !strings.Contains(text, fmt.Sprintf("从块输入处的 %.0f", in)) {
		t.Fatalf("the callout does not quote the input mean %.0f:\n%s", in, text)
	}
	if !strings.Contains(text, fmt.Sprintf("长到最后一层输出的 %.0f", out)) {
		t.Fatalf("the callout does not quote the output mean %.0f:\n%s", out, text)
	}
	if !strings.Contains(text, fmt.Sprintf("约 %.1f 倍", out/in)) {
		t.Fatalf("the callout does not quote the %.1fx ratio:\n%s", out/in, text)
	}
}

// TestEveryCalloutIsOnScreen checks each callout actually renders at the width
// it is written for. A correction that lives below the fold teaches nobody.
func TestEveryCalloutIsOnScreen(t *testing.T) {
	// Callouts live on stages 3, 4, 5, 6 (zero-based), by inspection of the code.
	for _, step := range []int{2, 3, 4, 5, 6} {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "「") && strings.Contains(s, "」") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("step %d shows no callout", step+1)
		}
	}
}
