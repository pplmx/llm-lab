package app

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

// TestWorkingsStaysFreeOfTheInterface keeps workings.go what it claims to be:
// arithmetic over the trace, with no dependency on the UI or on App state. The
// file exists so those pure functions can be read and tested on their own, and
// that property is easy to erode — one helper that needs a ui.Context to format
// a line, and the split stops paying for itself.
func TestWorkingsStaysFreeOfTheInterface(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "workings.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse workings.go: %v", err)
	}

	// No UI import.
	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		if strings.HasPrefix(path, "github.com/egoist/mygo") {
			t.Fatalf("workings.go imports %s; it is supposed to be UI-free", path)
		}
	}

	// No method on App, and no parameter of a UI type.
	ast.Inspect(file, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			if decl.Recv != nil {
				t.Fatalf("workings.go declares a method %s; it should hold plain functions", decl.Name.Name)
			}
			if decl.Type.Params != nil {
				for _, field := range decl.Type.Params.List {
					if sel, ok := field.Type.(*ast.SelectorExpr); ok {
						if id, ok := sel.X.(*ast.Ident); ok && id.Name == "ui" {
							t.Fatalf("%s takes a ui.%s; rendering belongs in a stage file",
								decl.Name.Name, sel.Sel.Name)
						}
					}
				}
			}
		}
		return true
	})
}

// TestEveryWorkingsComputerHasATest ties the file's contents to its coverage:
// each compute* function should have a test that names it. This is a coarse
// check, not a coverage measurement, but it catches the case of adding a
// seventh block and forgetting to assert anything about its arithmetic.
func TestEveryWorkingsComputerHasATest(t *testing.T) {
	src, err := os.ReadFile("workings.go")
	if err != nil {
		t.Fatalf("read workings.go: %v", err)
	}
	// Both test files count: the arithmetic tests for the newer blocks live in
	// workings_test.go beside the assertions on their printed lines, and a helper
	// covered there is covered.
	testSrc, err := os.ReadFile("app_test.go")
	if err != nil {
		t.Fatalf("read app_test.go: %v", err)
	}
	more, err := os.ReadFile("workings_test.go")
	if err != nil {
		t.Fatalf("read workings_test.go: %v", err)
	}
	testSrc = append(testSrc, more...)

	var missing []string
	for _, line := range strings.Split(string(src), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "func compute") {
			continue
		}
		name := strings.SplitN(strings.TrimPrefix(line, "func "), "(", 2)[0]
		if !strings.Contains(string(testSrc), name) {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("these arithmetic helpers are never named in a test: %v", missing)
	}
}

// TestPrintedScoreLineRecomputes checks the arithmetic a reader is asked to
// verify by hand actually closes.
//
// The score workings print three lines: the top dot-product terms, the remainder,
// and "score = 点积 / √d × scale = result". Existing tests compare computeScore's
// return value against the trace, which only shows the panel repeats the model —
// it says nothing about whether the printed chain adds up. A reader who takes the
// printed terms and re-adds them must land on the printed dot, and dividing that
// must land on the printed score.
func TestPrintedScoreLineRecomputes(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	h := a.layer().Heads[a.Head]
	dHead := a.Trace.Config.DHead()
	q := a.SelToken
	k := q / 2

	p := computeScore(h.Q[q], h.K[k], dHead, h.Scores[q][k])
	if !p.OK {
		t.Fatal("computeScore declined")
	}

	// The printed terms plus the printed remainder must equal the printed dot.
	printedDot := 0.0
	for _, term := range p.Top {
		printedDot += term.V
	}
	printedDot += p.Rest
	if math.Abs(printedDot-p.Dot) > 1e-9 {
		t.Fatalf("the printed terms (%v) plus the remainder %v come to %.9f, but the "+
			"printed dot is %.9f", p.Top, p.Rest, printedDot, p.Dot)
	}

	// And the printed multiplication must give the printed score. Redo it the way
	// the forward pass does — float32 dot, float32 scale, float32 multiply — since
	// that is the only chain a reader can reproduce from the printed numbers.
	scale32 := float32(1 / math.Sqrt(float64(dHead)))
	if got := float64(float32(p.Dot) * scale32); math.Abs(got-p.Score) > 1e-9 {
		t.Fatalf("printed dot %.9f × printed scale %.9f = %.9f, but the printed "+
			"score is %.9f", p.Dot, float64(scale32), got, p.Score)
	}

	// Independently: the dot is the sum over every dimension, so the top terms
	// plus the remainder must equal a plain loop.
	var plain float64
	for d := 0; d < dHead; d++ {
		plain += float64(h.Q[q][d]) * float64(h.K[k][d])
	}
	if math.Abs(plain-p.Dot) > 1e-5 {
		t.Fatalf("an independent dot gives %.6f, the workings print %.6f", plain, p.Dot)
	}
}

// TestPrintedSoftmaxLineRecomputes does the same for the softmax block: the
// printed denominator and numerator must produce the printed weight.
func TestPrintedSoftmaxLineRecomputes(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	h := a.layer().Heads[a.Head]
	q := a.SelToken
	k := q / 2

	p := computeSoftmaxParts(h, q, k)
	if !p.ok {
		t.Fatal("computeSoftmaxParts declined")
	}
	// The block prints "分母 = Σ exp(score − rowMax) = denom" and
	// "α = exp(score − rowMax) ÷ denom".
	want := float64(p.num) / float64(p.denom)
	if math.Abs(want-float64(p.num/p.denom)) > 1e-9 {
		t.Fatalf("α from the printed numerator and denominator is %.9f, the printed "+
			"value is %.9f", want, p.num/p.denom)
	}
	// And the denominator really is the sum the line claims.
	var sum float64
	for j := 0; j <= q; j++ {
		sum += math.Exp(float64(h.Scores[q][j]) - float64(p.rowMax))
	}
	if math.Abs(sum-float64(p.denom)) > 1e-4 {
		t.Fatalf("Σ exp(s − max) over the row is %.6f, the printed denominator is %.6f",
			sum, p.denom)
	}
	// The result must match the trace's attention weight.
	if math.Abs(float64(p.num/p.denom)-float64(h.Attn[q][k])) > 1e-5 {
		t.Fatalf("the recomputed α is %.6f, the trace holds %.6f",
			p.num/p.denom, h.Attn[q][k])
	}
}

// TestPrintedLNLineStaysClose checks the LayerNorm line a reader is asked to verify
// by hand stays within the precision it advertises.
//
// The line prints μ, σ, 1/√(σ²+ε), γ and β, then the result. With γ and β at three
// decimals the round-off was 3e-4, and 90 of 96 dimensions disagreed with the
// printed result in the fourth decimal — a reader redoing the arithmetic would
// find the line wrong. Five decimals for γ/β cut that; the block marks the result
// "≈" and says the last digits may move, because rounding five factors cannot
// close exactly.
func TestPrintedLNLineStaysClose(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	l := a.layer()
	w := a.W.Layers[a.Layer]
	row := l.AfterAttn[a.SelToken]

	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}

	worst := 0.0
	for dim := 0; dim < len(row); dim++ {
		p := computeLN(row, w.LN2G, w.LN2B, dim)
		if !p.ok {
			t.Fatalf("dim %d: computeLN declined", dim)
		}
		// Substitute exactly what the panel prints.
		hand := (round(p.x, 4)-round(p.mean, 4))*round(p.invStd, 4)*
			round(p.gamma, 5) + round(p.beta, 5)
		if d := math.Abs(hand - p.out); d > worst {
			worst = d
		}
	}
	if worst > 5e-4 {
		t.Fatalf("rounding to the printed decimals moves the result by %.2e, which "+
			"changes the fourth decimal the panel shows", worst)
	}
	t.Logf("worst gap between the printed line and its own arithmetic: %.2e", worst)
}

// TestPrintedLNActuallyMatchesTheTrace is the other half: whatever the line shows
// must be the model's value, not a float64 re-derivation of it.
func TestPrintedLNActuallyMatchesTheTrace(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	l := a.layer()
	w := a.W.Layers[a.Layer]
	row := l.AfterAttn[a.SelToken]

	for dim := 0; dim < len(row); dim++ {
		p := computeLN(row, w.LN2G, w.LN2B, dim)
		if got, want := p.out, float64(l.Norm2[a.SelToken][dim]); math.Abs(got-want) > 1e-6 {
			t.Fatalf("dim %d: the panel would print %.9f, the trace holds %.9f", dim, got, want)
		}
	}
}

// TestPrintedWeightedSumStaysClose checks the Σα·V line stays within the precision
// it prints.
//
// The line lists the three largest terms, the remainder, and the total. At three
// decimals for the terms — one less than the remainder and the total — each term
// was rounded by up to 5e-4, and the reader's sum landed in the wrong fourth
// decimal on all 24 dimensions of the row measured, worst gap 9.8e-4. The line
// exists to be re-added, so the pieces are now printed at the same precision as
// the total, which brings the worst gap to 7.9e-5.
func TestPrintedWeightedSumStaysClose(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	h := a.layer().Heads[a.Head]
	q := a.SelToken

	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}

	worst := 0.0
	for dim := 0; dim < a.Trace.Config.DHead(); dim++ {
		p := computeWeighted(h.Attn[q], h.V, q, dim, len(a.Trace.Tokens))
		if !p.OK {
			t.Fatalf("dim %d: computeWeighted declined", dim)
		}
		sum := 0.0
		for _, tm := range p.Top {
			sum += round(tm.Val, 4)
		}
		sum += round(p.Rest, 4)
		if d := math.Abs(sum - p.Out); d > worst {
			worst = d
		}
	}
	if worst > 2e-4 {
		t.Fatalf("adding the printed pieces moves the total by %.2e, more than a "+
			"last-digit rounding", worst)
	}
	t.Logf("worst gap between the printed terms and the printed total: %.2e", worst)
}

// TestPrintedWeightedSumMatchesTheTrace keeps the printed total equal to the
// model's output.
func TestPrintedWeightedSumMatchesTheTrace(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	h := a.layer().Heads[a.Head]
	q := a.SelToken
	for dim := 0; dim < a.Trace.Config.DHead(); dim++ {
		p := computeWeighted(h.Attn[q], h.V, q, dim, len(a.Trace.Tokens))
		if got, want := p.Out, float64(h.Out[q][dim]); math.Abs(got-want) > 1e-5 {
			t.Fatalf("dim %d: the panel would print %.6f, the trace holds %.6f", dim, got, want)
		}
	}
}

// TestPrintedFFNSumStaysClose checks the FFN line stays within the precision it
// prints.
//
// The line lists the three largest contributions, the exact remainder, and the
// output. With the terms and the remainder at three decimals and the total at
// four, the reader's sum missed the printed result on 88 of 96 dimensions, worst
// 1.5e-3 — the widest of the arithmetic mismatches found so far. All four pieces
// now share four decimals, which brings it to 1.5e-4 with the remaining
// disagreements in the last digit only.
func TestPrintedFFNSumStaysClose(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	l := a.layer()
	w2 := a.W.Layers[a.Layer].W2
	dff, d := a.W.Config.DFF, a.W.Config.DModel

	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}

	worst := 0.0
	for j := 0; j < d; j++ {
		p := computeFFN(l.Hidden[a.SelToken], w2, dff, d, j)
		if !p.OK {
			t.Fatalf("dim %d: computeFFN declined", j)
		}
		sum := 0.0
		for _, tm := range p.Top {
			sum += round(tm.V, 4)
		}
		sum += round(p.Rest, 4)
		if gap := math.Abs(sum - p.Out); gap > worst {
			worst = gap
		}
	}
	if worst > 3e-4 {
		t.Fatalf("adding the printed pieces moves the output by %.2e, more than a "+
			"last-digit rounding", worst)
	}
	t.Logf("worst gap between the printed terms and the printed output: %.2e", worst)
}

// TestPrintedFFNSumMatchesTheTrace keeps the printed output equal to the model's.
func TestPrintedFFNSumMatchesTheTrace(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	l := a.layer()
	w2 := a.W.Layers[a.Layer].W2
	dff, d := a.W.Config.DFF, a.W.Config.DModel
	for j := 0; j < d; j++ {
		p := computeFFN(l.Hidden[a.SelToken], w2, dff, d, j)
		if got, want := p.Out, float64(l.FFNOut[a.SelToken][j]); math.Abs(got-want) > 1e-5 {
			t.Fatalf("dim %d: the panel would print %.6f, the trace holds %.6f", j, got, want)
		}
	}
}

// TestPrintedResidualChainCloses checks the residual chain's lines add up.
//
// Unlike the three blocks fixed this and the two previous rounds, this one already
// printed every term and every running total at four decimals, so the reader's
// running sum stays within a last-digit rounding (worst 1.2e-4 across 96
// dimensions). The test records that, so a future change to the printout cannot
// quietly reintroduce the mismatch the others had.
func TestPrintedResidualChainCloses(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	l := a.layer()

	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}

	worst := 0.0
	for dim := 0; dim < a.Trace.Config.DModel; dim++ {
		steps, ok := residualChain(*l, a.SelToken, dim)
		if !ok {
			t.Fatalf("dim %d: residualChain declined", dim)
		}
		// The first line prints the starting value as both term and sum.
		running := round(steps[0].Term, 4)
		for _, s := range steps[1:] {
			running += round(s.Term, 4)
			if gap := math.Abs(running - s.Sum); gap > worst {
				worst = gap
			}
		}
	}
	if worst > 2e-4 {
		t.Fatalf("adding the printed residual terms moves the running total by %.2e",
			worst)
	}
	t.Logf("worst gap in the printed residual chain: %.2e", worst)
}

// TestEveryPrintedArithmeticBlockStaysClose is the closure check for the class of
// defect found across rounds 81-84: a block whose numbers a reader is asked to
// re-add, printed at precisions that do not close.
func TestEveryPrintedArithmeticBlockStaysClose(t *testing.T) {
	// Each entry returns the worst gap between what is printed and what a reader
	// gets by substituting it.
	blocks := map[string]func(*testing.T) float64{
		"score":    worstPrintedScoreGap,
		"softmax":  worstPrintedSoftmaxGap,
		"weighted": worstPrintedWeightedGap,
		"ffn":      worstPrintedFFNGap,
		"ln":       worstPrintedLNGap,
	}
	for name, measure := range blocks {
		if gap := measure(t); gap > 1e-3 {
			t.Errorf("%s block: the printed arithmetic is off by %.2e, which changes "+
				"a printed digit", name, gap)
		}
	}
}

func worstPrintedScoreGap(t *testing.T) float64 {
	t.Helper()
	a := newTestApp(t)
	a.Step = 2
	h := a.layer().Heads[a.Head]
	q, k := a.SelToken, a.SelToken/2
	p := computeScore(h.Q[q], h.K[k], a.Trace.Config.DHead(), h.Scores[q][k])
	scale := float64(float32(1 / math.Sqrt(float64(a.Trace.Config.DHead()))))
	return math.Abs(p.Dot*scale - p.Score)
}

func worstPrintedSoftmaxGap(t *testing.T) float64 {
	t.Helper()
	a := newTestApp(t)
	a.Step = 3
	h := a.layer().Heads[a.Head]
	p := computeSoftmaxParts(h, a.SelToken, a.SelToken/2)
	if !p.ok {
		return 0
	}
	return math.Abs(float64(p.num/p.denom) - float64(h.Attn[a.SelToken][a.SelToken/2]))
}

func worstPrintedWeightedGap(t *testing.T) float64 {
	t.Helper()
	a := newTestApp(t)
	a.Step = 3
	h := a.layer().Heads[a.Head]
	round := func(v float64) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 4, 64), 64)
		return out
	}
	worst := 0.0
	for dim := 0; dim < a.Trace.Config.DHead(); dim++ {
		p := computeWeighted(h.Attn[a.SelToken], h.V, a.SelToken, dim, len(a.Trace.Tokens))
		sum := 0.0
		for _, tm := range p.Top {
			sum += round(tm.Val)
		}
		sum += round(p.Rest)
		if gap := math.Abs(sum - p.Out); gap > worst {
			worst = gap
		}
	}
	return worst
}

func worstPrintedFFNGap(t *testing.T) float64 {
	t.Helper()
	a := newTestApp(t)
	a.Step = 5
	l := a.layer()
	w2 := a.W.Layers[a.Layer].W2
	dff, d := a.W.Config.DFF, a.W.Config.DModel
	round := func(v float64) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 4, 64), 64)
		return out
	}
	worst := 0.0
	for j := 0; j < d; j++ {
		p := computeFFN(l.Hidden[a.SelToken], w2, dff, d, j)
		sum := 0.0
		for _, tm := range p.Top {
			sum += round(tm.V)
		}
		sum += round(p.Rest)
		if gap := math.Abs(sum - p.Out); gap > worst {
			worst = gap
		}
	}
	return worst
}

func worstPrintedLNGap(t *testing.T) float64 {
	t.Helper()
	a := newTestApp(t)
	a.Step = 4
	l := a.layer()
	w := a.W.Layers[a.Layer]
	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}
	worst := 0.0
	for dim := 0; dim < a.Trace.Config.DModel; dim++ {
		p := computeLN(l.AfterAttn[a.SelToken], w.LN2G, w.LN2B, dim)
		hand := (round(p.x, 4)-round(p.mean, 4))*round(p.invStd, 4)*
			round(p.gamma, 5) + round(p.beta, 5)
		if gap := math.Abs(hand - p.out); gap > worst {
			worst = gap
		}
	}
	return worst
}

// TestWorkingsComputersRejectOutOfRangeInput covers the guard clauses.
//
// Mutation testing removed each one in turn and every test still passed: the call
// sites clamp their indices, so nothing exercises the guards in normal use. They
// are still load-bearing — with the clamp in computeSoftmaxParts's `q >= len(...)`
// removed, computeSoftmaxParts(h, 999, 0) panics with "index out of range [999] with
// length 36" rather than declining.
func TestWorkingsComputersRejectOutOfRangeInput(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[0]
	row := a.layer().AfterAttn[a.SelToken]
	w := a.W.Layers[a.Layer]
	dHead := a.Trace.Config.DHead()
	n := len(a.Trace.Tokens)
	d, dff := a.W.Config.DModel, a.W.Config.DFF

	t.Run("softmax rejects q past the matrix", func(t *testing.T) {
		if p := computeSoftmaxParts(h, 999, 0); p.ok {
			t.Fatal("accepted q=999")
		}
		if p := computeSoftmaxParts(h, 0, 999); p.ok {
			t.Fatal("accepted k=999")
		}
		if p := computeSoftmaxParts(h, 0, 5); p.ok {
			t.Fatal("accepted k > q, which is behind the causal mask")
		}
		if p := computeSoftmaxParts(h, -1, 0); p.ok {
			t.Fatal("accepted q=-1")
		}
	})

	t.Run("score rejects short vectors and bad dHead", func(t *testing.T) {
		if p := computeScore(nil, h.K[0], dHead, 0); p.OK {
			t.Fatal("accepted a nil q")
		}
		if p := computeScore(h.Q[0], h.K[0], 0, 0); p.OK {
			t.Fatal("accepted dHead=0, which would divide by zero at scale time")
		}
		if p := computeScore(h.Q[0], h.K[0], 999, 0); p.OK {
			t.Fatal("accepted dHead beyond the vector length")
		}
	})

	t.Run("ffn rejects out-of-range dims", func(t *testing.T) {
		w2 := a.W.Layers[a.Layer].W2
		if p := computeFFN(a.layer().Hidden[a.SelToken], w2, dff, d, -1); p.OK {
			t.Fatal("accepted j=-1")
		}
		if p := computeFFN(a.layer().Hidden[a.SelToken], w2, dff, d, d); p.OK {
			t.Fatal("accepted j=d")
		}
		if p := computeFFN(nil, w2, dff, d, 0); p.OK {
			t.Fatal("accepted a nil hidden row")
		}
	})

	t.Run("weighted rejects out-of-range dims", func(t *testing.T) {
		if p := computeWeighted(h.Attn[0], h.V, 0, -1, n); p.OK {
			t.Fatal("accepted dim=-1")
		}
		if p := computeWeighted(h.Attn[0], h.V, 0, 99, n); p.OK {
			t.Fatal("accepted dim beyond d_head")
		}
		if p := computeWeighted(h.Attn[0], h.V, -1, 0, n); p.OK {
			t.Fatal("accepted q=-1")
		}
	})

	t.Run("ln rejects out-of-range dims", func(t *testing.T) {
		if p := computeLN(row, w.LN2G, w.LN2B, -1); p.ok {
			t.Fatal("accepted dim=-1")
		}
		if p := computeLN(row, w.LN2G, w.LN2B, len(row)); p.ok {
			t.Fatal("accepted dim == len(row)")
		}
		if p := computeLN(row, nil, w.LN2B, 0); p.ok {
			t.Fatal("accepted a nil gamma")
		}
	})

	t.Run("layerDivergence rejects a missing previous layer", func(t *testing.T) {
		if d := computeLayerDivergence(a.Trace.Layers, 0, 0, 0); d.Ok {
			t.Fatal("accepted layer 0, which has no previous layer")
		}
		if d := computeLayerDivergence(a.Trace.Layers, 99, 0, 0); d.Ok {
			t.Fatal("accepted layer 99")
		}
	})

	t.Run("activationConcentration handles empty input", func(t *testing.T) {
		if share, topN := activationConcentration(nil); share != 0 || topN != 0 {
			t.Fatalf("nil input gave (%v, %d)", share, topN)
		}
		if share, _ := activationConcentration([]float32{}); share != 0 {
			t.Fatalf("empty input gave share %v", share)
		}
		if share, _ := activationConcentration([]float32{0, 0, 0}); share != 0 {
			t.Fatalf("all-zero input gave share %v", share)
		}
	})
}

// TestWorkingsStateHowManyTermsTheyOmit checks the "其余 N" figures are the real
// counts.
//
// Mutation testing changed each by one and every test passed. Those numbers tell the
// reader what the printed sum leaves out — "三项 + 其余 21 维 = 点积" for a 24-wide head,
// "其余 381 项" for the 384-wide FFN — so an off-by-one makes the arithmetic look
// incomplete or double-counted while the printed total still closes.
func TestWorkingsStateHowManyTermsTheyOmit(t *testing.T) {
	t.Run("score omits dHead minus the terms shown", func(t *testing.T) {
		a := newTestApp(t)
		a.Step = 2
		a.SelToken = 10
		a.KeyQ, a.KeyK = 10, 5 // pin a cell so the workings render
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		p := computeScore(a.layer().Heads[a.Head].Q[10], a.layer().Heads[a.Head].K[5],
			a.Trace.Config.DHead(), a.layer().Heads[a.Head].Scores[10][5])
		if !p.OK {
			t.Fatal("computeScore declined")
		}
		want := fmt.Sprintf("+ 其余 %d 维", a.Trace.Config.DHead()-len(p.Top))
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("the score workings do not state %q", want)
		}
	})

	t.Run("ffn omits dff minus the terms shown", func(t *testing.T) {
		a := newTestApp(t)
		a.Step = 5
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		l := a.layer()
		p := computeFFN(l.Hidden[a.SelToken], a.W.Layers[a.Layer].W2, a.W.Config.DFF,
			a.W.Config.DModel, 0)
		if !p.OK {
			t.Fatal("computeFFN declined")
		}
		want := fmt.Sprintf("+ 其余 %d 项", a.W.Config.DFF-len(p.Top))
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("the FFN workings do not state %q", want)
		}
	})

	t.Run("weighted omits q+1 minus the terms shown", func(t *testing.T) {
		a := newTestApp(t)
		a.Step = 3
		a.SelToken = 10
		a.KeyQ, a.KeyK = 10, 5
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		h := a.layer().Heads[a.Head]
		p := computeWeighted(h.Attn[10], h.V, 10, 0, len(a.Trace.Tokens))
		if !p.OK {
			t.Fatal("computeWeighted declined")
		}
		want := fmt.Sprintf("+ 其余 %d 项", 11-len(p.Top))
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("the weighted workings do not state %q", want)
		}
	})
}

// TestWeightedShareUsesAbsoluteMass pins the denominator of the "三项占..." sentence.
//
// The obvious denominator, |out|, is wrong: the signed sum cancels heavily — the
// helper's own comment says dividing by it "would overstate them several times" —
// so the printed share has to be against Σ|term|. Mutation testing swapped the
// signed total in and nothing failed.
func TestWeightedShareUsesAbsoluteMass(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5
	h := a.layer().Heads[a.Head]
	p := computeWeighted(h.Attn[10], h.V, 10, 0, len(a.Trace.Tokens))
	if !p.OK {
		t.Fatal("computeWeighted declined")
	}
	if p.AbsTotal == 0 || p.Out == 0 {
		t.Skip("this row has no mass or no output, so the two denominators coincide")
	}
	var top float64
	for _, tm := range p.Top {
		top += math.Abs(tm.Val)
	}
	absoluteShare := 100 * top / p.AbsTotal
	signedShare := 100 * top / math.Abs(p.Out)
	if math.Abs(absoluteShare-signedShare) < 1 {
		t.Skip("the two denominators give the same percentage here, so the panel " +
			"cannot distinguish them")
	}

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("三项占绝对值总量的 %.0f%%", absoluteShare)
	wrong := fmt.Sprintf("三项占绝对值总量的 %.0f%%", signedShare)
	var sawWant, sawWrong bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			sawWant = true
		}
		if strings.Contains(s, wrong) {
			sawWrong = true
		}
	}
	if !sawWant {
		t.Fatalf("the panel does not print the absolute-mass share %q", want)
	}
	if sawWrong {
		t.Fatalf("the panel prints the signed share %q as well", wrong)
	}
}

// TestPrintedScoreLineUsesTheComputedNumbers reads the rendered line rather than
// calling computeScore again.
//
// TestPrintedScoreLineRecomputes above checks that computeScore's own fields close —
// terms plus remainder equal Dot, Dot times scale equal Score. That passes no matter
// what the panel prints, because it never looks at the panel. Mutation testing
// swapped the printed dot for p.Rest, and the printed result for p.Dot, and the suite
// stayed green: the line a reader is asked to verify could name any number.
func TestPrintedScoreLineUsesTheComputedNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5 // pin a cell so the workings render
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	h := a.layer().Heads[a.Head]
	dHead := a.Trace.Config.DHead()
	p := computeScore(h.Q[10], h.K[5], dHead, h.Scores[10][5])
	if !p.OK {
		t.Fatal("computeScore declined")
	}
	scale := float64(float32(1 / math.Sqrt(float64(dHead))))

	want := fmt.Sprintf("score = 点积 / √%d = %.4f × %.4f = %+.4f",
		dHead, p.Dot, scale, p.Score)
	found := false
	for _, s := range tr.Texts() {
		if s == want {
			found = true
		}
	}
	if !found {
		var seen string
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "score = 点积") {
				seen = s
			}
		}
		t.Fatalf("the panel prints %q, want %q", seen, want)
	}

	// The printed value a reader divides must be the dot the previous line closed on.
	wantDot := fmt.Sprintf("+ 其余 %d 维 %+.4f = 点积 %.4f", dHead-len(p.Top), p.Rest, p.Dot)
	dotFound := false
	for _, s := range tr.Texts() {
		if s == wantDot {
			dotFound = true
		}
	}
	if !dotFound {
		t.Fatalf("the panel does not show the dot %q that the score line divides", wantDot)
	}
}

// TestPrintedSoftmaxLinesUseTheComputedNumbers reads the rendered softmax workings.
//
// The same blind spot as the score line: TestPrintedSoftmaxLineRecomputes checks that
// computeSoftmaxParts's fields are consistent with each other, but never looks at the
// panel. Swapping the printed denominator for the row maximum, or for the numerator,
// left the suite green — the two lines a reader is told to divide could name anything.
func TestPrintedSoftmaxLinesUseTheComputedNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	h := a.layer().Heads[a.Head]
	p := computeSoftmaxParts(h, 10, 5)
	if !p.ok {
		t.Fatal("computeSoftmaxParts declined")
	}

	// The subtrahend is formatted the way the panel formats it: unsigned when positive,
	// bracketed when negative, so the line never runs two signs together. Rebuilding the
	// string with %+.4f here would pin the old double-sign form instead of what a reader
	// should see.
	sub := fmt.Sprintf("%.4f", p.rowMax)
	if p.rowMax < 0 {
		sub = fmt.Sprintf("(%+.4f)", p.rowMax)
	}
	wantDenom := fmt.Sprintf("分母 = Σ exp(score − %s) = %.4f", sub, p.denom)
	wantWeight := fmt.Sprintf("α = exp(%+.4f − %s) ÷ %.4f = %.4f",
		h.Scores[10][5], sub, p.denom, p.num/p.denom)

	var sawDenom, sawWeight bool
	for _, s := range tr.Texts() {
		if s == wantDenom {
			sawDenom = true
		}
		if s == wantWeight {
			sawWeight = true
		}
	}
	if !sawDenom {
		t.Fatalf("the panel does not print %q", wantDenom)
	}
	if !sawWeight {
		t.Fatalf("the panel does not print %q", wantWeight)
	}

	// And the division the second line shows must reproduce the printed weight.
	if got, want := p.num/p.denom, float64(h.Attn[10][5]); math.Abs(got-want) > 1e-4 {
		t.Fatalf("the printed division gives %.6f, the trace holds %.6f", got, want)
	}
}

// TestPrintedLNLineUsesTheComputedNumbers reads the rendered LayerNorm workings.
//
// The third instance of the same blind spot. The tests above check that
// computeLN's fields are consistent, but never read the panel, so the printed γ
// could be β, the printed μ could be 1/σ, and the printed result could be γ — all
// four mutations passed. This is the line with the most hand-verifiable parts of any
// in the app: five values a reader multiplies and adds.
//
// Uses the untrained weights' learned-ness only for setup, not for the assertion.
func TestPrintedLNLineUsesTheComputedNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4 // the LayerNorm stage
	a.SelToken = 10
	a.Layer = 1
	a.HoverDimIn = 7 // the workings follow the hovered dimension, defaulting to 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	l := a.layer()
	w := a.W.Layers[a.Layer]
	const dim = 7
	p := computeLN(l.AfterAttn[10], w.LN2G, w.LN2B, dim)
	if !p.ok {
		t.Fatal("computeLN declined")
	}

	wantHeader := fmt.Sprintf("LayerNorm d%d：μ = %.4f，σ = %.4f，1/√(σ²+ε) = %.4f",
		dim, p.mean, 1/p.invStd, p.invStd)
	// The mean is bracketed when negative, the same rule the softmax line follows:
	// "(x − -0.1499)" makes the reader decode two signs before checking the arithmetic.
	mu := fmt.Sprintf("%.4f", p.mean)
	if p.mean < 0 {
		mu = "(" + mu + ")"
	}
	wantBody := fmt.Sprintf("(%.4f − %s) × %.4f × γ %.5f + β %.5f ≈ %.4f",
		p.x, mu, p.invStd, p.gamma, p.beta, p.out)

	var sawHeader, sawBody bool
	for _, s := range tr.Texts() {
		if s == wantHeader {
			sawHeader = true
		}
		if s == wantBody {
			sawBody = true
		}
	}
	if !sawHeader {
		t.Fatalf("the panel does not print %q", wantHeader)
	}
	if !sawBody {
		t.Fatalf("the panel does not print %q", wantBody)
	}

	// The result the line ends on must be the trace's LayerNorm output for this
	// dimension, not merely some value computeLN happened to produce.
	if got, want := p.out, float64(l.Norm2[10][dim]); math.Abs(got-want) > 1e-4 {
		t.Fatalf("the printed result is %.6f, the trace holds %.6f", got, want)
	}
	// And substituting the printed five-decimal values must land near the printed
	// result — the property the "≈" and the note underneath are there to describe.
	hand := (p.x-p.mean)*p.invStd*p.gamma + p.beta
	if gap := math.Abs(hand - p.out); gap > 2e-4 {
		t.Fatalf("substituting the printed numbers gives %.6f, %.1e away from the "+
			"printed result %.6f", hand, gap, p.out)
	}
}

// TestPrintedResidualChainUsesTheComputedSteps reads the rendered residual chain.
//
// The same blind spot the last round found in three other blocks: the residual tests
// call residualChain and check the steps close with each other, but never read the
// panel. Three mutations passed — printing the first line's sum where the term
// belongs, printing each step's term where its sum belongs, and not carrying the
// running total forward at all.
func TestPrintedResidualChainUsesTheComputedSteps(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4 // the residual stage
	a.SelToken = 10
	a.Layer = 1
	a.HoverDimIn = 7
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	l := a.layer()
	const dim = 7
	steps, ok := residualChain(*l, 10, dim)
	if !ok {
		t.Fatal("residualChain declined")
	}

	// First line: "<label> = <term>".
	// The layer is named 0-based, matching the segmented control and every card heading.
	wantFirst := fmt.Sprintf("d%d 的残差流（层 %d）：%s = %+.4f",
		dim, a.Layer, steps[0].Label, steps[0].Term)
	var sawFirst bool
	for _, s := range tr.Texts() {
		if s == wantFirst {
			sawFirst = true
		}
	}
	if !sawFirst {
		t.Fatalf("the panel does not print the chain's first line %q", wantFirst)
	}

	// Every later line: "<running> <label> <term> = <sum>".
	running := steps[0].Sum
	for _, s := range steps[1:] {
		term := fmt.Sprintf("%+.4f", s.Term)
		if s.Term < 0 {
			term = fmt.Sprintf("(%+.4f)", s.Term)
		}
		want := fmt.Sprintf("%+.4f %s %s = %+.4f", running, s.Label, term, s.Sum)
		found := false
		for _, got := range tr.Texts() {
			if got == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("the panel does not print the chain step %q", want)
		}
		running = s.Sum
	}
}

// TestPrintedFFNTermsUseTheComputedValues reads the rendered FFN workings.
//
// Two mutations passed: the printed output replaced by the remainder, and each
// term's value replaced by its key index. The second is the more instructive — it
// prints a plausible-looking four-decimal number in the right column, so only a test
// that recomputes the term can tell.
func TestPrintedFFNTermsUseTheComputedValues(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	a.SelToken = 10
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	l := a.layer()
	w2 := a.W.Layers[a.Layer].W2
	const dim = 0
	p := computeFFN(l.Hidden[10], w2, a.W.Config.DFF, a.W.Config.DModel, dim)
	if !p.OK {
		t.Fatal("computeFFN declined")
	}

	var terms []string
	for _, tm := range p.Top {
		terms = append(terms, fmt.Sprintf("k%d %+.4f", tm.K, tm.V))
	}
	wantHead := fmt.Sprintf("out[%d] 的三项最大贡献：%s", dim, strings.Join(terms, "  "))
	wantFoot := fmt.Sprintf("+ 其余 %d 项 %+.4f ≈ %+.4f",
		a.W.Config.DFF-len(p.Top), p.Rest, p.Out)

	var sawHead, sawFoot bool
	for _, s := range tr.Texts() {
		if s == wantHead {
			sawHead = true
		}
		if s == wantFoot {
			sawFoot = true
		}
	}
	if !sawHead {
		t.Fatalf("the panel does not print %q", wantHead)
	}
	if !sawFoot {
		t.Fatalf("the panel does not print %q", wantFoot)
	}

	// The printed output must be the trace's, not merely computeFFN's own field.
	if got, want := p.Out, float64(l.FFNOut[10][dim]); math.Abs(got-want) > 1e-4 {
		t.Fatalf("the printed output is %.6f, the trace holds %.6f", got, want)
	}
}

// TestPrintedHeadContributionsUseTheComputedValues reads the head-contribution card.
//
// The sixth block with the same blind spot, and the widest so far: six mutations
// passed. Each head's bar could show the sum of all four norms, or the merged
// vector's norm, instead of its own. The cancellation sentence could print the sum
// where the total belongs or the total where the sum belongs — either way the ratio
// it then states no longer divides the two numbers printed beside it. The ratio could
// be inverted, and the row count could be replaced by the row total.
//
// The ratio one is the worst: "四个头的长度之和是 17.05，但它们相加后的长度只有
// 9.70，比值 1.76" reads as a coherent sentence about reinforcement when the truth is
// the opposite.
func TestPrintedHeadContributionsUseTheComputedValues(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3 // the softmax stage, whose card computes the per-head norms
	a.SelToken = 10
	a.Layer = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	l := a.layer()
	hc := computeHeadContributions(*l, 10)
	if !hc.OK {
		t.Fatal("computeHeadContributions declined")
	}

	// Each head's own length must appear as a bar value.
	for _, hh := range hc.Heads {
		want := fmt.Sprintf("%.2f", hh.Norm)
		var found bool
		for _, s := range tr.Texts() {
			if s == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("head %d's length %.2f does not appear on the panel", hh.Head, hh.Norm)
		}
	}

	// The sentence must quote the sum, the total, and their ratio in that order.
	st := a.cancellation()
	want := fmt.Sprintf("范数不能直接相加。四个头的长度之和是 %.2f，但它们相加后的长度只有 %.2f，"+
		"比值 %.2f", hc.SumNorm, hc.TotalNorm, hc.Cancellation)
	var sawSentence bool
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, want) {
			sawSentence = true
		}
	}
	if !sawSentence {
		t.Fatalf("the panel does not print %q", want)
	}

	// The ratio must be the total over the sum — reinforcement would be above 1.
	if hc.SumNorm > 0 {
		if got, wantRatio := hc.Cancellation, hc.TotalNorm/hc.SumNorm; math.Abs(got-wantRatio) > 1e-9 {
			t.Fatalf("the printed ratio is %v, but total/sum is %v", got, wantRatio)
		}
	}

	// And the row counts it cites must be the ones measured.
	wantRows := fmt.Sprintf("%d 行里 %d 行都是这样", st.Rows, st.Below)
	foundRows := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantRows) {
			foundRows = true
		}
	}
	if !foundRows {
		t.Fatalf("the panel does not cite %q", wantRows)
	}
}

// TestAttentionInspectorRowsUseTheComputedValues reads the attention stage's panel.
//
// Four mutations passed. The layer-divergence figure could be replaced by its
// complement or by a constant, the entropy by a sum over one element fewer, and the
// output norm by V's norm instead of the head output's.
//
// The divergence one deserves a note: the line's own legend says "0 = 相同，1 =
// 不重叠", so printing 1-TV inverts what the number means while the legend still
// claims to describe it.
func TestAttentionInspectorRowsUseTheComputedValues(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3 // the softmax stage carries the per-head inspector rows
	a.SelToken = 10
	a.Layer = 1
	a.Head = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	l := a.layer()
	h := l.Heads[a.Head]
	const i = 10

	// Layer divergence: the label names the previous layer, and the number is TV.
	d := computeLayerDivergence(a.Trace.Layers, a.Layer, a.Head, i)
	if !d.Ok {
		t.Fatal("computeLayerDivergence declined")
	}
	wantLabel := fmt.Sprintf("与第 %d 层之差", a.Layer-1)
	wantValue := fmt.Sprintf("%.2f（0 = 相同，1 = 不重叠）", d.TV)
	var sawLabel, sawValue bool
	for _, s := range tr.Texts() {
		if s == wantLabel {
			sawLabel = true
		}
		if s == wantValue {
			sawValue = true
		}
	}
	if !sawLabel {
		t.Fatalf("the panel does not label the comparison %q", wantLabel)
	}
	if !sawValue {
		t.Fatalf("the panel does not show the divergence %q", wantValue)
	}
	// The complement must not be what is shown: computing it proves the test can
	// tell the two apart rather than passing on a coincidental match.
	if comp := fmt.Sprintf("%.2f（0 = 相同，1 = 不重叠）", 1-d.TV); comp == wantValue {
		t.Skip("the divergence is 0.5, so it and its complement print the same")
	}

	// Entropy: over the causal prefix of the row.
	wantEnt := fmt.Sprintf("%.2f nats", model.Entropy(h.Attn[i][:i+1]))
	var sawEnt bool
	for _, s := range tr.Texts() {
		if s == wantEnt {
			sawEnt = true
		}
	}
	if !sawEnt {
		t.Fatalf("the panel does not show the entropy %q", wantEnt)
	}

	// Output norm: the head's own output, not V.
	wantOut := fmt.Sprintf("%.3f", norm(h.Out[i]))
	if wrong := fmt.Sprintf("%.3f", norm(h.V[i])); wantOut != wrong {
		var saw bool
		for _, s := range tr.Texts() {
			if s == wantOut {
				saw = true
			}
		}
		if !saw {
			t.Fatalf("the panel does not show the output norm %q", wantOut)
		}
	}
}

// TestPrintedLayerNormCeilingUsesSqrtDModel reads the sentence that states what
// LayerNorm pins a row's length to.
//
// The claim is "√d_model = 9.80，实测所有行落在 9.25–11.33，也就是 95%–116%". Replacing
// math.Sqrt(d_model) with d_model left the suite green: the sentence would then read
// "√d_model = 96.00 … 也就是 10%–12%", where the stated ceiling contradicts the
// numbers beside it and the percentages describe nothing.
func TestPrintedLayerNormCeilingUsesSqrtDModel(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4 // the residual stage carries this comparison
	a.SelToken = 10
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	pin := math.Sqrt(float64(a.Trace.Config.DModel))
	lnLo, lnHi := lnNormRange(a.Trace.Layers, len(a.Trace.Tokens), a.Trace.FinalNorm)
	want := fmt.Sprintf("√d_model = %.2f，实测所有行落在 %.2f–%.2f，也就是 %.0f%%–%.0f%%",
		pin, lnLo, lnHi, 100*lnLo/pin, 100*lnHi/pin)
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			found = true
		}
	}
	if !found {
		t.Fatalf("the panel does not print %q", want)
	}

	// The ceiling must be the square root, which is the whole point: a row's length
	// stays near √d_model rather than anywhere near d_model.
	if pin >= float64(a.Trace.Config.DModel) {
		t.Fatalf("√d_model (%.2f) is not below d_model (%d), so the claim is odd",
			pin, a.Trace.Config.DModel)
	}
}

// TestLogitsRowsShowTheirOwnProbability reads the Top-12 rows' percentages.
//
// Each row prints its own probability, and the rank can be printed in its place, or
// zero. Both passed: the rows are drawn as bars whose width already varies, so a
// wrong number in the text column is the only visible symptom and nothing checked it.
func TestLogitsRowsShowTheirOwnProbability(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	probs := a.Trace.Probabilities(a.Temp)
	top := model.TopK(probs, 12)
	for rank, id := range top {
		// The formatter has three tiers: scientific below 0.0001, two decimals
		// below 0.01, one decimal otherwise. Reproduce it rather than assuming one.
		var want string
		switch {
		case probs[id] == 0:
			want = "0%"
		case probs[id] < 0.0001:
			want = fmt.Sprintf("%.1e%%", probs[id]*100)
		case probs[id] < 0.01:
			want = fmt.Sprintf("%.2f%%", probs[id]*100)
		default:
			want = fmt.Sprintf("%.1f%%", probs[id]*100)
		}
		found := false
		for _, s := range tr.Texts() {
			if s == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("rank %d (id %d) has probability %v; %q does not appear",
				rank, id, probs[id], want)
		}
	}
}

// TestScoreStageLegendStatesTheRealScale reads the two figures the score stage's
// legend quotes about itself.
//
// The legend says "缩放 1/√24 = 0.204" and "可见 N 个 key". Two mutations passed: the
// scale printed as 1/d_head, and the visible count off by one. Both are the sort of
// number a reader checks against the formula directly above them.
func TestScoreStageLegendStatesTheRealScale(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	a.Layer = 1
	a.Head = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// The legend sits below the first screen: content outside the scroll viewport
	// is not in Texts(), so the panel has to be scrolled to before reading it.
	tr.Scroll(700, 500, 0, -120)
	tr.Frame()

	dHead := a.Trace.Config.DHead()
	wantScale := fmt.Sprintf("1/√%d = %.3f", dHead, 1/math.Sqrt(float64(dHead)))
	wrongScale := fmt.Sprintf("1/√%d = %.3f", dHead, 1/float64(dHead))
	var sawScale, sawWrong bool
	for _, s := range tr.Texts() {
		if s == wantScale {
			sawScale = true
		}
		if s == wrongScale {
			sawWrong = true
		}
	}
	if !sawScale {
		t.Fatalf("the legend does not state %q", wantScale)
	}
	// The score workings print the same string from the same quantity, so its
	// presence alone does not prove the legend is right — the linear version must be
	// absent too.
	if sawWrong {
		t.Fatalf("the linear 1/d_head value %q also appears", wrongScale)
	}

	// The causal row has SelToken+1 visible keys.
	wantVisible := fmt.Sprintf("%d 个 key", a.SelToken+1)
	var sawVisible bool
	for _, s := range tr.Texts() {
		if s == wantVisible {
			sawVisible = true
		}
	}
	if !sawVisible {
		t.Fatalf("the legend does not state %q", wantVisible)
	}
}

// TestPrintedArithmeticClosesAcrossTheControls sweeps the gap the meta-test fixes at 1e-3
// over the inputs a reader can actually vary.
//
// TestEveryPrintedArithmeticBlockStaysClose checks each block once, on the default prompt,
// the default token, layer 0 and head 0. Everything a reader can change — the prompt, the
// selected token, the layer, the head — was outside it, and those are exactly the controls
// the interface invites them to drag.
//
// Measured across 4 prompts × 2 stages × 4 layers × 4 heads × 4 tokens: the worst gap between
// the printed arithmetic and a re-substitution of its own numbers is 2.6e-07, four thousand
// times inside the threshold. That is the property the whole workings feature rests on — a
// reader who redoes the line by hand must land on the printed result.
func TestPrintedArithmeticClosesAcrossTheControls(t *testing.T) {
	worst, where := 0.0, ""
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"注意力就是加权求和",
		"hi",
		"a",
	} {
		for _, layer := range []int{0, 1} {
			for _, head := range []int{0, 1, 2, 3} {
				a := newTestApp(t)
				a.Prompt = prompt
				a.recompute()
				a.Step = 2
				a.Layer = layer
				a.Head = head
				n := len(a.Trace.Tokens)
				if n == 0 {
					continue
				}
				scale := float64(float32(1 / math.Sqrt(float64(a.Trace.Config.DHead()))))
				for token := 0; token < n && token < 4; token++ {
					a.SelToken = token
					h := a.layer().Heads[a.Head]
					q, k := token, token/2
					if q >= len(h.Q) || k >= len(h.K) {
						continue
					}
					p := computeScore(h.Q[q], h.K[k], a.Trace.Config.DHead(), h.Scores[q][k])
					if g := math.Abs(p.Dot*scale - p.Score); g > worst {
						worst = g
						where = fmt.Sprintf("prompt %q, layer %d, head %d, token %d",
							prompt, layer, head, token)
					}
				}
			}
		}
	}
	if worst > 1e-3 {
		t.Fatalf("the printed score line is off by %.2e at %s; a reader redoing it would "+
			"not land on the printed result", worst, where)
	}
	t.Logf("worst gap %.2e at %s", worst, where)
}

// TestWorkingsLinesDoNotRunTwoSignsTogether checks the two lines that subtract a quantity
// which can be negative.
//
// The softmax line subtracts the row maximum; the LayerNorm line subtracts the mean, and
// about 43 of 96 means are negative on the default prompt. Both formatted the subtrahend
// with a sign, so a positive value produced "score − +2.9311" and a negative one produced
// "(x − -0.1499)" — two signs in a row that the reader has to decode before the arithmetic
// in the line makes sense.
//
// Positive subtrahends now print unsigned and negative ones print in brackets. The test walks
// both stages over several tokens and rejects any run of two signs.
func TestWorkingsLinesDoNotRunTwoSignsTogether(t *testing.T) {
	bad := []string{"− +", "− -", "- +", "- -"}
	for _, stage := range []struct {
		step   int
		marker string
	}{
		{3, "exp("}, // softmax: subtracts the row maximum
		{4, "× γ"},  // LayerNorm: subtracts the mean
	} {
		for _, prompt := range []string{"The transformer reads the tokens and", "hi", "a"} {
			for _, token := range []int{0, 5, 10} {
				a := newTestApp(t)
				a.Prompt = prompt
				a.recompute()
				a.Step = stage.step
				if token >= len(a.Trace.Tokens) {
					continue
				}
				a.SelToken = token
				a.HoverDimIn = 7
				tr := ui.NewTester(a.View, 1500, 940)
				tr.Frame()
				for _, s := range tr.Texts() {
					if !strings.Contains(s, stage.marker) {
						continue
					}
					for _, b := range bad {
						if strings.Contains(s, b) {
							t.Fatalf("stage %d prompt %q token %d: the line runs two "+
								"signs together: %q", stage.step, prompt, token, s)
						}
					}
				}
			}
		}
	}
}

// TestResidualChainClosesAcrossTokensAndLayers widens the chain check to the inputs a reader
// can vary.
//
// TestPrintedResidualChainCloses walks all 96 dimensions but only on the selected token of
// the selected layer — one row. The chain is the same three-step structure everywhere, but the
// numbers are not, and a rounding problem that shows on one row need not show on another.
//
// Measured over four prompts, both layers and the first twelve rows: adding the printed terms
// keeps the running total within 1.0e-4, which is one digit at four decimals — exactly what
// rounding three numbers separately can cost.
func TestResidualChainClosesAcrossTokensAndLayers(t *testing.T) {
	round := func(v float64, dp int) float64 {
		out, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', dp, 64), 64)
		return out
	}

	worst := 0.0
	for _, prompt := range []string{
		"The transformer reads the tokens and", "注意力就是加权求和", "hi", "a",
	} {
		for _, layer := range []int{0, 1} {
			a := newTestApp(t)
			a.Prompt = prompt
			a.recompute()
			a.Layer = layer
			l := a.layer()
			if l == nil || len(l.In) == 0 {
				continue
			}
			for row := 0; row < len(l.In) && row < 12; row++ {
				for dim := 0; dim < a.Trace.Config.DModel; dim++ {
					steps, ok := residualChain(*l, row, dim)
					if !ok {
						continue
					}
					running := round(steps[0].Term, 4)
					for _, s := range steps[1:] {
						running += round(s.Term, 4)
						if gap := math.Abs(running - s.Sum); gap > worst {
							worst = gap
						}
					}
				}
			}
		}
	}
	if worst > 2e-4 {
		t.Fatalf("adding the printed residual terms moves the running total by %.2e", worst)
	}
	t.Logf("worst gap across prompts, layers and rows: %.2e", worst)
}

// TestEveryWorkingsLineNamesItsOwnSubject checks that a workings line can be read on its own.
//
// A tall card's arithmetic sits hundreds of DIP below the card's heading — measured at 1180×720,
// the LayerNorm line is 243 DIP under its title, and a reader scrolling down loses the heading
// before reaching it. So each block's first line has to say which quantity it is about.
//
// The residual chain already did ("d7 的残差流（层 0）"). The LayerNorm header started at
// "d7：μ = …", naming the dimension but not the operation, on a card that also discusses the
// residual stream — the reader had to infer which of the card's two subjects they had arrived at.
func TestEveryWorkingsLineNamesItsOwnSubject(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	a.SelToken = 10
	a.HoverDimIn = 7
	tr := ui.NewTester(a.View, 1180, 1400)
	tr.Frame()

	var lnLine, residualLine string
	for _, s := range tr.Texts() {
		switch {
		case strings.HasPrefix(s, "LayerNorm d7："):
			lnLine = s
		case strings.Contains(s, "的残差流（层 "):
			residualLine = s
		}
	}
	if residualLine == "" {
		t.Fatal("the residual chain no longer names its own subject")
	}
	if lnLine == "" {
		t.Fatal("the LayerNorm header no longer names its own subject; it must start with " +
			"the operation, not just the dimension")
	}
	// And it must still carry the dimension, which is what ties it to the hovered strip.
	if !strings.Contains(lnLine, "d7") {
		t.Fatalf("the LayerNorm header lost its dimension: %q", lnLine)
	}
}

// TestLayerNumberingIsTheSameOnScreenAndInTheArithmetic checks that "which layer" has one answer.
//
// The segmented control above the stage reads "层 0 | 1" and every card heading quotes a.Layer, so the
// reader's layer numbers are 0-based. The residual chain's arithmetic said "第 2 层" for layer 1 —
// a second numbering scheme on the same screen, which makes comparing the control to the arithmetic
// impossible without translating.
//
// Both lines now name the layer the way the control does.
func TestLayerNumberingIsTheSameOnScreenAndInTheArithmetic(t *testing.T) {
	for _, layer := range []int{0, 1} {
		a := newTestApp(t)
		a.Layer = layer
		a.Step = 4
		a.SelToken = 10
		a.HoverDimIn = 7
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()

		// The heading quotes the control's number.
		wantHeading := fmt.Sprintf("层 %d · 两次加法", layer)
		if !tr.HasText(wantHeading) {
			t.Fatalf("layer %d: the card heading does not read %q", layer, wantHeading)
		}

		// And the arithmetic names the same layer, not the next integer.
		var chainLine, closing string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "的残差流（层 ") {
				chainLine = s
			}
			if strings.Contains(s, "的输出完整") {
				closing = s
			}
		}
		if chainLine == "" || closing == "" {
			t.Fatalf("layer %d: the residual chain lost a line (chain %q, closing %q)",
				layer, chainLine, closing)
		}
		want := fmt.Sprintf("（层 %d）", layer)
		if !strings.Contains(chainLine, want) {
			t.Fatalf("layer %d: the chain says %q, not %q", layer, chainLine, want)
		}
		if !strings.Contains(closing, fmt.Sprintf("所以层 %d 的输出", layer)) {
			t.Fatalf("layer %d: the closing line is %q", layer, closing)
		}
		// The old 1-based form would put layer+1 here; make sure it is gone.
		if strings.Contains(chainLine, fmt.Sprintf("（层 %d）", layer+1)) && layer == 0 {
			t.Fatalf("layer 0 is still called layer 1 in the arithmetic: %q", chainLine)
		}
	}
}

// TestNextByteSkillMatchesTheModel checks the per-position figures against the
// trace, position by position: the guess must be the argmax of that row's own
// distribution, the actual byte must be the token after it, and the loss must be
// −ln of the probability the model gave the truth.
//
// Asserting on the rendered text is the point of the project's rule, but this
// block's numbers come from a computation the stage would otherwise be trusted to
// do correctly in private. Both sides are checked: this pins computeNextByteSkill
// to the trace, and TestNextByteSkillCardQuotesItsOwnNumbers pins the card to
// computeNextByteSkill.
func TestNextByteSkillMatchesTheModel(t *testing.T) {
	a := newTestApp(t)
	skill := computeNextByteSkill(a.W, a.Trace, a.Temp)
	n := len(a.Trace.Tokens)
	if skill.Scored != n-1 {
		t.Fatalf("scored %d positions for a %d-token prompt; every position except the "+
			"last has a successor, so it should be %d", skill.Scored, n, n-1)
	}
	if len(skill.Rows) != skill.Scored {
		t.Fatalf("%d rows for %d scored positions", len(skill.Rows), skill.Scored)
	}
	for i, row := range skill.Rows {
		if row.Pos != i {
			t.Fatalf("row %d describes position %d", i, row.Pos)
		}
		if row.Actual != a.Trace.Tokens[i+1].ID {
			t.Fatalf("position %d compares against byte %d, the prompt's next byte is %d",
				i, row.Actual, a.Trace.Tokens[i+1].ID)
		}
		if len(row.Probs) != a.W.Config.Vocab {
			t.Fatalf("position %d carries %d probabilities, want %d", i, len(row.Probs), a.W.Config.Vocab)
		}
		// The guess is this row's argmax, taken from the same distribution the
		// row keeps — not recomputed from the model a second way.
		best, bestP := 0, row.Probs[0]
		var sum float64
		for id, p := range row.Probs {
			sum += float64(p)
			if p > bestP {
				best, bestP = id, p
			}
		}
		if row.Guess != best {
			t.Fatalf("position %d says the model guessed byte %d; its argmax is %d", i, row.Guess, best)
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Fatalf("position %d's distribution sums to %v", i, sum)
		}
		wantLoss := -math.Log(float64(row.Probs[row.Actual]))
		if math.Abs(row.Loss-wantLoss) > 1e-9 {
			t.Fatalf("position %d: loss %.6f, −ln p(actual) is %.6f", i, row.Loss, wantLoss)
		}
		if row.Hit != (row.Guess == row.Actual) {
			t.Fatalf("position %d: hit = %v with guess %d and actual %d",
				i, row.Hit, row.Guess, row.Actual)
		}
	}
	// The last position has no successor and must not be scored: the card claims
	// T−1 checks, and scoring the last row would mean inventing an answer.
	if last := skill.Rows[len(skill.Rows)-1]; last.Pos != n-2 {
		t.Fatalf("the last scored position is %d, want %d (the last row has no successor)",
			last.Pos, n-2)
	}
}

// TestNextByteSkillLossIsReal pins the claim the summary makes: the model's mean
// loss over the prompt is meaningfully below the uniform reference, and the
// reference is ln(vocab). If the weights were random this would fail — which is
// correct, since the card's sentence is written for a trained model.
func TestNextByteSkillLossIsReal(t *testing.T) {
	a := newTestApp(t)
	skill := computeNextByteSkill(a.W, a.Trace, a.Temp)
	if math.Abs(skill.UniformReference-math.Log(256)) > 1e-9 {
		t.Fatalf("uniform reference %.6f, ln(256) is %.6f", skill.UniformReference, math.Log(256))
	}
	if skill.MeanLoss >= skill.UniformReference {
		t.Fatalf("mean loss %.3f is not below the uniform reference %.3f; the summary "+
			"claims the weights lower it", skill.MeanLoss, skill.UniformReference)
	}
	// A trained model is wrong on some positions but not most of them. Both
	// bounds matter: all-right would make the colour legend's warm case unused,
	// all-wrong would mean the tokenizer or the comparison is broken.
	if skill.Hits == 0 {
		t.Fatal("the model got no position right; the comparison is probably misaligned")
	}
	if skill.Hits == skill.Scored {
		t.Fatalf("the model got all %d positions right, so the strip is one colour", skill.Scored)
	}
	t.Logf("top-1 %d/%d, mean loss %.3f nats, uniform %.3f", skill.Hits, skill.Scored,
		skill.MeanLoss, skill.UniformReference)
}

// TestNextByteSkillIsCalibratedAtTOne is the measurement that made this test worth
// writing: the mean loss is *not* monotone in temperature, and assuming it was
// produced a failing assertion before this test was rewritten.
//
// Measured on the bundled weights with the default prompt: T=0.1 → 6.392 nats,
// T=0.5 → 1.510, T=0.8 → 1.184, T=1.0 → 1.121, T=2.0 → 1.246, T=3.0 → 1.524. The
// minimum sits at T = 1, which is the temperature the loss was trained at — the
// weights are calibrated for exactly that scaling. Sharpening past it makes the
// model confidently wrong about the positions it misses (T=0.1 turns an 11/35 miss
// rate into 6.4 nats), and flattening it hands mass to 244 wrong bytes.
//
// That U shape is the reason the card names the temperature beside its figures:
// a reader who drags the slider will watch a number the app presents as the
// model's score move, and the honest explanation is that the score is defined at
// T=1 and only the reader's choice of T changes it.
func TestNextByteSkillIsCalibratedAtTOne(t *testing.T) {
	a := newTestApp(t)
	loss := func(temp float64) float64 {
		return computeNextByteSkill(a.W, a.Trace, temp).MeanLoss
	}
	one := loss(1.0)
	// Both directions cost more, which is the U shape rather than a slope.
	for _, temp := range []float64{0.5, 2.0} {
		if got := loss(temp); got <= one {
			t.Fatalf("mean loss at T=%.1f is %.3f, not above T=1.0's %.3f; the loss is "+
				"trained for T=1 and should be lowest there", temp, got, one)
		}
	}
	// And the fall from sharpening is the steep one, so the ordering on each side
	// is pinned too — a flat plateau would pass the check above.
	if loss(0.1) <= loss(0.5) {
		t.Fatalf("T=0.1 costs %.3f, T=0.5 costs %.3f; overconfidence must be the expensive end",
			loss(0.1), loss(0.5))
	}
	// Temperature rescales a row's logits, so it cannot reorder them: the guesses
	// and the hit count are the same at every temperature.
	cold, hot := computeNextByteSkill(a.W, a.Trace, 0.5), computeNextByteSkill(a.W, a.Trace, 2.0)
	if cold.Hits != hot.Hits || cold.Hits != computeNextByteSkill(a.W, a.Trace, 1.0).Hits {
		t.Fatalf("the top-1 hit count moved with temperature: %d, %d, %d",
			cold.Hits, hot.Hits, computeNextByteSkill(a.W, a.Trace, 1.0).Hits)
	}
	for i := range cold.Rows {
		if cold.Rows[i].Guess != hot.Rows[i].Guess {
			t.Fatalf("position %d: T=0.5 guesses byte %d, T=2.0 guesses %d; temperature "+
				"rescales the distribution but cannot reorder it", i, cold.Rows[i].Guess, hot.Rows[i].Guess)
		}
	}
}

// TestNextByteSkillCardQuotesItsOwnNumbers reads the card's figures out of the
// rendered text and checks them against the model — the rule this project follows
// because a test that only recomputes would pass with the wrong number on screen.
func TestNextByteSkillCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	a.SelToken = len(a.Trace.Tokens) - 1
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()

	skill := computeNextByteSkill(a.W, a.Trace, a.Temp)
	if skill.Scored < 2 {
		t.Fatalf("the default prompt scores only %d positions", skill.Scored)
	}

	// The header states how many positions there are to check.
	header := ""
	for _, s := range tr.Texts() {
		if strings.Contains(s, "个可核对的位置") {
			header = s
		}
	}
	if header == "" {
		t.Fatal("the per-position card is missing")
	}
	if want := fmt.Sprintf("%d 个可核对的位置", skill.Scored); !strings.Contains(header, want) {
		t.Fatalf("the card's header says %q, the model has %q", header, want)
	}

	// The mean loss and the hit rate are both printed and both must match.
	wantLoss := fmt.Sprintf("%.3f nats", skill.MeanLoss)
	wantHits := fmt.Sprintf("猜对 %d / %d", skill.Hits, skill.Scored)
	var sawLoss, sawHits bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantLoss) {
			sawLoss = true
		}
		if strings.Contains(s, wantHits) {
			sawHits = true
		}
	}
	if !sawLoss {
		t.Fatalf("the card does not print the mean loss %q", wantLoss)
	}
	if !sawHits {
		t.Fatalf("the card does not print the hit count %q", wantHits)
	}

	// The reference line has to name ln(vocab), or 1.12 nats means nothing.
	if !tr.HasText("均匀猜测") {
		t.Fatal("the card does not give the uniform reference the loss is measured against")
	}
}

// TestNextByteSkillCardOpensOnTheWorstPosition: the detail block defaults to the
// position the model did worst at, not to position 0, because the card's point is
// that this model is often wrong.
func TestNextByteSkillCardOpensOnTheWorstPosition(t *testing.T) {
	a := newTestApp(t)
	skill := computeNextByteSkill(a.W, a.Trace, a.Temp)
	worst, worstLoss := 0, -1.0
	for i, row := range skill.Rows {
		if row.Loss > worstLoss {
			worst, worstLoss = i, row.Loss
		}
	}
	if worst == 0 {
		t.Skip("the worst position happens to be 0, so this test cannot tell the default from it")
	}
	if got := a.predictionCursor(skill); got != worst {
		t.Fatalf("the detail block opens on position %d; the largest loss is at %d", got, worst)
	}
	// Pinning wins over the default, and hovering wins over nothing.
	a.PredSel = 3
	if got := a.predictionCursor(skill); got != 3 {
		t.Fatalf("a pinned position 3 was ignored (cursor %d)", got)
	}
	a.PredSel = -1
	a.PredHover = 5
	if got := a.predictionCursor(skill); got != 5 {
		t.Fatalf("a hovered position 5 was ignored (cursor %d)", got)
	}
}

// TestPinnedPredictionSurvivesThePointerButNotThePrompt is the state-lifecycle
// test this app's CONTRIBUTING demands for anything clickable: say what a pin
// survives and what invalidates it.
func TestPinnedPredictionSurvivesThePointerButNotThePrompt(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()
	a.PredSel = 4

	// A pointer elsewhere must not drop the pin: that is what pinning is for.
	tr.Move(1200, 40)
	tr.Frame()
	tr.Frame()
	if a.PredSel != 4 {
		t.Fatalf("moving the pointer cleared the pinned position (now %d)", a.PredSel)
	}
	// Esc is the documented way out, and it has to work here too.
	tr.Key(0, ui.KeyEscape)
	tr.Frame()
	if a.PredSel != -1 {
		t.Fatalf("Esc left the pinned position at %d", a.PredSel)
	}

	// A new prompt has different positions, and a stale pin would describe a
	// position the new trace may not have — recompute must clear it.
	a.PredSel = 4
	a.Prompt = "hi"
	a.recompute()
	if a.PredSel != -1 {
		t.Fatalf("the pinned position survived a prompt change: %d", a.PredSel)
	}
}

// TestPredictionSentenceUsesTheAppsIndexing pins the fix for a bug this card had
// in its first draft: the sentence mixed one-based and zero-based numbering, so a
// panel headed "t=8" said "没有任何来自位置 9 及之后的信息 … 拿提示词的第 9 个字节"
// when the byte it compared was token 9 (the ninth byte of a zero-based sequence
// is token 8).
//
// The app numbers tokens from 0 everywhere — the strip, the heatmap axes, the
// inspector's "token #35 / 35" — so the sentence has to as well.
func TestPredictionSentenceUsesTheAppsIndexing(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	a.SelToken = len(a.Trace.Tokens) - 1
	skill := computeNextByteSkill(a.W, a.Trace, a.Temp)
	// Pick a position whose successor is a distinctive byte, then pin it.
	pos := 8
	if pos >= len(skill.Rows) {
		pos = len(skill.Rows) - 1
	}
	a.PredSel = pos
	// The per-position detail is the inspector's card, so the frame has to be the
	// output stage with a tall enough window for the panel to have been laid out.
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()

	want := fmt.Sprintf("所以拿 token %d（%q）与它比对",
		pos+1, byteText(a.Trace.Tokens[pos+1].ID))
	var found string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "与它比对") && strings.Contains(s, "因果掩码") {
			found = s
		}
	}
	if found == "" {
		t.Fatal("the honesty sentence is missing from the pinned prediction's panel")
	}
	if !strings.Contains(found, want) {
		t.Fatalf("the sentence does not name the byte it compares (%q):\n%s", want, found)
	}
	if !strings.Contains(found, fmt.Sprintf("位置 %d 的计算", pos)) ||
		!strings.Contains(found, fmt.Sprintf("位置 %d 及之后", pos+1)) {
		t.Fatalf("the sentence's cut and its comparison are not both stated zero-based:\n%s", found)
	}
}

// TestFFNLessonSparsityIsMeasured pins the numbers in the FFN lesson's "sparse but
// continuous" sentence.
//
// The sentence used to be an unqualified claim — "所以激活会呈现稀疏但连续的分布" —
// carried from the GELU literature. It is true of this checkpoint, but the lesson's
// job is to describe what the reader's screen shows, and the card right below it
// already reports a concentration figure for the selected row. Two sentences making
// the same claim with different numbers is the defect this project keeps finding.
//
// The replacement quotes three measured figures plus the exact-zero count, and this
// test checks each of them against the trace.
func TestFFNLessonSparsityIsMeasured(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 1400)
	tr.Frame()

	var lesson string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "GELU 是固定的数学函数") {
			lesson = s
		}
	}
	if lesson == "" {
		t.Fatal("the FFN lesson paragraph is missing")
	}

	// The four figures, each derived here from the trace independently of the
	// helper under test — so a change to the helper cannot silently agree with
	// itself.
	rows, dims := len(a.Trace.Layers[0].Hidden), len(a.Trace.Layers[0].Hidden[0])
	all := make([]float32, 0, rows*dims)
	var peak float64
	for _, row := range a.Trace.Layers[0].Hidden {
		all = append(all, row...)
		for _, v := range row {
			if x := math.Abs(float64(v)); x > peak {
				peak = x
			}
		}
	}
	abs := make([]float64, len(all))
	for i, v := range all {
		abs[i] = math.Abs(float64(v))
	}
	sort.Float64s(abs)
	median := abs[len(abs)/2]
	below, zeros := 0, 0
	for i, v := range abs {
		if v < 0.1*peak {
			below++
		}
		if all[i] == 0 {
			zeros++
		}
	}
	var total float64
	for _, v := range abs {
		total += v
	}
	var topSum float64
	for _, v := range abs[len(abs)-len(abs)/10:] {
		topSum += v
	}

	for _, want := range []string{
		fmt.Sprintf("%d 个位置 × %d 维", rows, dims),
		fmt.Sprintf("%.0f%% 的激活绝对值", 100*topSum/total),
		fmt.Sprintf("中位数只有峰值的 %.0f%%", 100*median/peak),
		fmt.Sprintf("%.0f%% 的维度低于峰值的十分之一", 100*float64(below)/float64(len(abs))),
		fmt.Sprintf("恰好为 0 的有 %d 个", zeros),
	} {
		if !strings.Contains(lesson, want) {
			t.Fatalf("the lesson does not state %q:\n%s", want, lesson)
		}
	}

	// And the premise the sentence depends on: this checkpoint's activations really
	// are concentrated and really are never exactly zero. If a future checkpoint
	// broke either, the wording would need to change rather than the numbers.
	if zeros != 0 {
		t.Fatalf("this trace has %d exact zeros; the lesson says GELU never truncates "+
			"to zero and quotes that count", zeros)
	}
	if topSum/total < 0.3 {
		t.Fatalf("the largest tenth carries only %.0f%% of the absolute activation; "+
			"calling that distribution sparse is no longer supportable", 100*topSum/total)
	}
}

// TestLogitLensFinalRungIsTheOutputAtThatPosition pins the rung that made the ladder
// contradict itself.
//
// The final rung must be the model's real answer *at the position the ladder was built
// for*. Trace.Logits holds one row — the last position's — because that is the only
// row Forward projects. Using it for every position was a bug this round found by
// cross-checking the ladder against the per-position prediction card: at pos 0 the
// ladder's last rung said "␣" (the last position's answer) while the real output there
// is "h", and 30 of 36 positions were wrong. Every earlier rung was computed at pos
// correctly, so the ladder's own last step disagreed with the four above it.
//
// The previous version of this test checked only the last position, which is exactly
// the one position where the bug is invisible.
func TestLogitLensFinalRungIsTheOutputAtThatPosition(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	if n < 4 {
		t.Fatalf("the test prompt has only %d positions", n)
	}
	for pos := 0; pos < n; pos++ {
		lens := computeLogitLens(a.W, a.Trace, pos, 1.0)
		if !lens.OK {
			t.Fatalf("pos %d: the ladder did not build", pos)
		}
		last := lens.Rungs[len(lens.Rungs)-1]
		if !last.IsFinal {
			t.Fatalf("pos %d: the last rung is not marked as the model's own output", pos)
		}
		// The real projection at this position: FinalNorm[pos] is already
		// normalised, so this is Unembed, not LogitLens.
		want := model.Softmax(a.W.Unembed(a.Trace.FinalNorm[pos]), 1.0)
		for id := range last.Probs {
			if math.Abs(float64(last.Probs[id]-want[id])) > 1e-9 {
				t.Fatalf("pos %d: the final rung's probability for byte %d is %v, the model's "+
					"real output there is %v", pos, id, last.Probs[id], want[id])
			}
		}
	}
	// At the last position it must also agree with Trace.Logits exactly, since that
	// is the row Forward actually projected.
	tr := a.Trace.Probabilities(1.0)
	lastRung := computeLogitLens(a.W, a.Trace, n-1, 1.0).Rungs
	final := lastRung[len(lastRung)-1]
	for id := range final.Probs {
		if math.Abs(float64(final.Probs[id]-tr[id])) > 1e-9 {
			t.Fatalf("the last position's final rung differs from Trace.Logits at byte %d: %v vs %v",
				id, final.Probs[id], tr[id])
		}
	}
}

// TestLogitLensAgreesWithThePerPositionCard is the cross-check that found the bug
// above, kept as a standing test.
//
// The output stage's per-position card and the residual stage's ladder are built from
// the same unembedding and must name the same next byte for the same position. They
// disagreed at 30 of 36 positions, which is what a bug in one of them looks like when
// two figures on two stages describe one quantity.
func TestLogitLensAgreesWithThePerPositionCard(t *testing.T) {
	a := newTestApp(t)
	skill := computeNextByteSkill(a.W, a.Trace, 1.0)
	for _, row := range skill.Rows {
		lens := computeLogitLens(a.W, a.Trace, row.Pos, 1.0)
		if !lens.OK {
			t.Fatalf("pos %d: no ladder", row.Pos)
		}
		if lens.Final != row.Guess {
			t.Fatalf("position %d: the lens ladder says the model's answer is %q, the "+
				"per-position card says %q; both are the same unembedding and must agree",
				row.Pos, byteText(lens.Final), byteText(row.Guess))
		}
	}
}

// TestLogitLensCalloutNumbersAreMeasured is the guard on the card's whole-prompt
// claim: 153 of 180 intermediate rungs disagree with the final answer, 105 of those
// confidently. Those are the numbers the sentence quotes, so they are what the model
// must still produce.
func TestLogitLensCalloutNumbersAreMeasured(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	disagree, confidentWrong := 0, 0
	positionsWithConfidentWrong := 0
	rungs := 0
	for pos := 0; pos < n; pos++ {
		lens := computeLogitLens(a.W, a.Trace, pos, a.Temp)
		posHas := false
		for _, r := range lens.Rungs {
			if r.IsFinal {
				continue
			}
			rungs++
			if !r.AgreesWithFinal {
				disagree++
				if r.Probs[r.Top] > lensConfident {
					confidentWrong++
					posHas = true
				}
			}
		}
		if posHas {
			positionsWithConfidentWrong++
		}
	}
	if rungs != 5*n {
		t.Fatalf("the ladder has %d intermediate rungs for %d positions, want 5 per position", rungs, n)
	}
	// Measured on the bundled weights and the default prompt: 92 of 180 intermediate
	// rungs disagree with the final answer, 76 of those are over 90% sure of
	// themselves, and all 36 positions have at least one.
	//
	// These replaced 153/105/31 after the final-rung bug was fixed. The old figures
	// were not merely stale — they were measured through a defect that made every
	// non-final position's ladder end on the last position's answer, inflating every
	// disagreement. A callout whose numbers came from a broken computation is worse
	// than no callout, and this is why the test recomputes rather than trusting the
	// prose.
	//
	// Checked as bands rather than exact values, because the figures move with the
	// prompt; the prompt is fixed here, so the failure message quotes what was seen.
	if disagree < rungs/3 {
		t.Fatalf("only %d of %d intermediate rungs disagree with the final answer; the callout "+
			"says a large share do", disagree, rungs)
	}
	if confidentWrong < disagree/2 {
		t.Fatalf("%d of %d disagreeing rungs are over %.0f%% sure of themselves; the callout says "+
			"most of them are", confidentWrong, disagree, 100*lensConfident)
	}
	if positionsWithConfidentWrong < n-1 {
		t.Fatalf("%d of %d positions have a confidently wrong intermediate rung; the callout says "+
			"nearly all do", positionsWithConfidentWrong, n)
	}
	t.Logf("measured: %d rungs, %d disagree, %d confident-wrong, %d/%d positions affected",
		rungs, disagree, confidentWrong, positionsWithConfidentWrong, n)
}

// TestLogitLensCardQuotesItsOwnNumbers reads the card's figures out of the rendered
// text, per this project's rule that an assertion must not be satisfied by
// recomputation alone.
func TestLogitLensCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	a.SelToken = 10 // a position whose intermediate rungs are confidently wrong
	tr := ui.NewTester(a.View, 1500, 1600)
	tr.Frame()

	// T=1, matching the card: the ladder is a read-out diagnostic and the
	// temperature slider is scoped to the output stage, so it must not move these
	// figures. (Reading it at a.Temp here is what made this test fail against the
	// view after the card was scoped — the two were computing different numbers.)
	lens := computeLogitLens(a.W, a.Trace, 10, 1.0)
	if !lens.OK {
		t.Fatal("the ladder did not build")
	}
	// Every rung's label is rendered, and the bar column prints that rung's
	// probability of the final answer — the quantity the chart draws.
	//
	// The first version of this asserted each rung's value cell equalled its own
	// top-1 probability, which failed on the final rung: it prints p(final), not
	// p(top-1), and those differ whenever a rung's own answer is not the final one.
	// On the position used here they differ by a lot — every intermediate rung gives
	// the final byte 0.0% while the real output gives it 93.0% — which is the card's
	// whole point.
	labels := map[string]bool{}
	for _, s := range tr.Texts() {
		labels[s] = true
	}
	var values []string
	for _, rung := range lens.Rungs {
		if !labels[rung.Name] {
			t.Fatalf("the ladder does not render the rung %q", rung.Name)
		}
		values = append(values, fmt.Sprintf("%.1f%%", 100*rung.Probs[lens.Final]))
	}
	for _, want := range values {
		if !labels[want] {
			t.Fatalf("the ladder does not print %q as a rung's probability of the final answer; "+
				"it prints %v", want, values)
		}
	}
	// The scoped count must say it is scoped, since the callout quotes the whole
	// prompt and the two would otherwise read as disagreeing.
	var scoped string
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "只看这个位置：") {
			scoped = s
		}
	}
	if scoped == "" {
		t.Fatal("the per-position count is missing or is not labelled as per-position")
	}
	if lens.ConfidentWrong > 0 {
		want := fmt.Sprintf("%d / %d 个中间层「自信地错」", lens.ConfidentWrong, lens.RungsCompared)
		if !strings.Contains(scoped, want) {
			t.Fatalf("the card says %q, the model has %q", scoped, want)
		}
	}
	// And the whole-prompt warning is on the stage.
	if !tr.HasText("常见误解") {
		t.Fatal("the logit lens card has no misconception callout")
	}
}
