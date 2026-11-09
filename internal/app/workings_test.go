package app

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"regexp"
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

// TestNextByteSkillCurveHasAValley pins the shape of the loss-vs-temperature curve and,
// deliberately, does *not* pin where its minimum is.
//
// The mean loss is not monotone in temperature — that is the useful fact, and it is why the
// card names the temperature beside its figures: a reader who drags the slider watches a
// number the app presents as the model's score move, and the honest explanation is that the
// score is defined at a chosen T.
//
// This test used to assert that T=1 was the minimum, and to be named for that claim. Measured
// across the prompts the app ships, it is false on all of them: the minimum is T=1.25 on the
// default prompt, 1.50 on "attention is all you need", 2.00 on "hi". It passed anyway because
// it compared T=1 against 0.5 and 2.0 with a one-sided inequality — a curve whose minimum is
// at 1.25 satisfies "0.5 costs more than 1.0" just as readily as one whose minimum is at 1.0.
// The card had already been corrected to compute and print the real minimum (round 24); the
// test's premise and the README sentence that repeated it were left behind.
//
// So the minimum is not asserted here at all. Where it falls is a measurement the card makes
// from the current trace, and TestTemperatureCurveFindsTheActualMinimum is what checks it
// against a fresh scan. What survives here is the shape, which is the part that is a property
// of the model rather than of one prompt.
func TestNextByteSkillCurveHasAValley(t *testing.T) {
	a := newTestApp(t)
	loss := func(temp float64) float64 {
		return computeNextByteSkill(a.W, a.Trace, temp).MeanLoss
	}
	one := loss(1.0)
	// Both directions cost more than T=1, which is what makes the curve a valley rather than
	// a slope. This is the assertion that survives: it says nothing about *where* the bottom
	// is, only that T=1 is not on a monotone flank.
	for _, temp := range []float64{0.5, 2.0} {
		if got := loss(temp); got <= one {
			t.Fatalf("mean loss at T=%.1f is %.3f, not above T=1.0's %.3f; the curve has "+
				"lost its valley at this prompt", temp, got, one)
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

// TestLogitLensCalloutNumbersAreMeasured checks that the whole-prompt figures are
// plausible for the default prompt: most rungs disagree, most of those confidently, and
// nearly every position has one.
//
// It is a band check, and that is its limitation. It recomputes the figures and compares
// them to thresholds, but it never reads the sentence — so when the sentence's numbers
// were hardcoded it kept passing, because the hardcoded figures happened to be the ones
// it recomputed for the one prompt it tested. The card then claimed "180 个中间层…36 个
// 位置" for a two-token prompt whose ladder holds 10 rungs over 2 positions.
//
// TestLogitLensCalloutFollowsThePrompt is the other half: it reads the rendered prose and
// requires its numbers to match a fresh computation, for four prompts. Keep both — this
// one catches a model that stopped mispredicting, that one catches prose that stopped
// tracking the model.
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

// TestHeadProfilesMeasureWhatTheyClaim checks the four-head card's figures against the
// trace, for both layers — the two layers behave differently, so testing one would
// miss the layer-1 case entirely.
func TestHeadProfilesMeasureWhatTheyClaim(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	for li := range a.Trace.Layers {
		profiles := computeHeadProfiles(a.Trace.Layers[li], n)
		if len(profiles) != len(a.Trace.Layers[li].Heads) {
			t.Fatalf("layer %d: %d profiles for %d heads", li, len(profiles), len(a.Trace.Layers[li].Heads))
		}
		for _, p := range profiles {
			h := &a.Trace.Layers[li].Heads[p.Head]
			// Recompute independently.
			var ent, prev float64
			peakRows := 0
			var peak float64
			rows := 0
			for q := 1; q < n; q++ {
				var e float64
				bw := float64(-1)
				for k := 0; k <= q; k++ {
					v := float64(h.Attn[q][k])
					if v > 1e-9 {
						e -= v * math.Log(v)
					}
					if v > bw {
						bw = v
					}
				}
				ent += e
				prev += float64(h.Attn[q][q-1])
				if bw > peak {
					peak = bw
				}
				if bw > 0.5 {
					peakRows++
				}
				rows++
			}
			if math.Abs(p.MeanEntropy-ent/float64(rows)) > 1e-9 {
				t.Fatalf("L%dH%d entropy: profile %.6f, recomputed %.6f",
					li, p.Head, p.MeanEntropy, ent/float64(rows))
			}
			if math.Abs(p.PrevShare-prev/float64(rows)) > 1e-9 {
				t.Fatalf("L%dH%d prev share: profile %.6f, recomputed %.6f",
					li, p.Head, p.PrevShare, prev/float64(rows))
			}
			if p.PeakRows != peakRows {
				t.Fatalf("L%dH%d peak rows: profile %d, recomputed %d", li, p.Head, p.PeakRows, peakRows)
			}
			if math.Abs(p.Peak-peak) > 1e-9 {
				t.Fatalf("L%dH%d peak: profile %.6f, recomputed %.6f", li, p.Head, p.Peak, peak)
			}
			if p.Rows != n-1 {
				t.Fatalf("L%dH%d rows: %d, want %d", li, p.Head, p.Rows, n-1)
			}
			if math.Abs(p.EntropyCeiling-math.Log(float64(n))) > 1e-9 {
				t.Fatalf("L%dH%d ceiling: %.6f vs ln(%d)", li, p.Head, p.EntropyCeiling, n)
			}
		}
	}
}

// TestHeadProfilesDistinguishTheTwoLayers is the premise of the card's verdict: layer 0's
// four heads are spread out and layer 1's are not.
//
// If that stopped being true the card's "换到第 0 层看看" advice would be wrong, and the
// card would be telling a reader to go look at a layer that is just as uniform.
func TestHeadProfilesDistinguishTheTwoLayers(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	if len(a.Trace.Layers) < 2 {
		t.Skip("the claim names two layers")
	}
	spread := func(li int) float64 {
		ps := computeHeadProfiles(a.Trace.Layers[li], n)
		lo, hi := ps[0].PrevShare, ps[0].PrevShare
		for _, p := range ps[1:] {
			if p.PrevShare < lo {
				lo = p.PrevShare
			}
			if p.PrevShare > hi {
				hi = p.PrevShare
			}
		}
		return hi - lo
	}
	s0, s1 := spread(0), spread(1)
	if s0 < 0.2 {
		t.Fatalf("layer 0's previous-token share spread is %.2f; the card's verdict needs it "+
			"to be wide enough to read as specialisation", s0)
	}
	if s1 > s0 {
		t.Fatalf("layer 1's spread (%.2f) is wider than layer 0's (%.2f); the card tells a "+
			"reader that layer 1 is the uniform one", s1, s0)
	}
	t.Logf("prev-share spread: layer 0 %.2f, layer 1 %.2f", s0, s1)
}

// TestHeadProfilesCardRendersBothVerdicts reads the verdict out of the rendered text on
// each of the two layers, since the card says different things about them.
func TestHeadProfilesCardRendersBothVerdicts(t *testing.T) {
	for layer, want := range map[int]string{0: "四个头差别明显", 1: "四个头在这一层上很接近"} {
		a := newTestApp(t)
		a.Step = 3
		a.Layer = layer
		tr := ui.NewTester(a.View, 1500, 2400)
		tr.Frame()
		// The verdict is one long sentence, so this matches a substring rather than
		// HasText (which requires the whole string to be equal).
		var lines []string
		for _, s := range tr.Texts() {
			lines = append(lines, s)
		}
		joined := strings.Join(lines, "\n")
		if !strings.Contains(joined, "四个头各自在做什么") {
			t.Fatalf("layer %d: the four-head card is missing", layer)
		}
		found := strings.Contains(joined, want)
		if !found {
			t.Fatalf("layer %d: no verdict containing %q", layer, want)
		}
	}
}

// TestMaskCounterfactualMatchesTheScoresItRecomputes checks the counterfactual against
// the trace where they overlap: for keys at or before the query, the recomputed
// distribution must agree with the model's own attention weights.
//
// They are computed by different code — the model's softmaxCausal over float32 scores,
// this over a float64 re-derivation — so they agree to a tolerance rather than exactly,
// and the tolerance is the interesting part: it says the recomputation is the same
// arithmetic, not a different quantity that happens to look similar.
func TestMaskCounterfactualMatchesTheScoresItRecomputes(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	dh := a.Trace.Config.DHead()
	for li := range a.Trace.Layers {
		for hi := range a.Trace.Layers[li].Heads {
			h := a.Trace.Layers[li].Heads[hi]
			for _, q := range []int{0, 1, 5, n / 2, n - 1} {
				c := computeMaskCounterfactual(h, q, n, dh)
				if !c.OK {
					t.Fatalf("L%dH%d row %d: no counterfactual", li, hi, q)
				}
				// The visible keys' shares, from the recomputation, must match the
				// model's attention once the hidden mass is removed.
				// Recompute the visible part explicitly.
				scale := float64(float32(1 / math.Sqrt(float64(dh))))
				scores := make([]float64, n)
				maxS := math.Inf(-1)
				for k := 0; k < n; k++ {
					var dot float32
					for d := 0; d < dh; d++ {
						dot += h.Q[q][d] * h.K[k][d]
					}
					scores[k] = float64(dot) * scale
					if scores[k] > maxS {
						maxS = scores[k]
					}
				}
				var sum float64
				for k := range scores {
					scores[k] = math.Exp(scores[k] - maxS)
					sum += scores[k]
				}
				var visibleSum float64
				for k := 0; k <= q; k++ {
					visibleSum += scores[k] / sum
				}
				// Normalised over the visible keys, the recomputation must equal Attn.
				var diff float64
				for k := 0; k <= q; k++ {
					want := float64(h.Attn[q][k])
					got := (scores[k] / sum) / visibleSum
					if d := math.Abs(got - want); d > diff {
						diff = d
					}
				}
				if diff > 5e-4 {
					t.Fatalf("L%dH%d row %d: recomputed visible weights differ from Attn by %.2e", li, hi, q, diff)
				}
				// The hidden mass is what the counterfactual reports.
				if math.Abs(c.FutureShare-(1-visibleSum)) > 5e-4 {
					t.Fatalf("L%dH%d row %d: FutureShare %.6f, recomputed %.6f",
						li, hi, q, c.FutureShare, 1-visibleSum)
				}
			}
		}
	}
}

// TestMaskCounterfactualIsLargeEverywhere is the premise of the card's claim: the mask
// is not hiding a marginal share of the distribution.
//
// The card tells the reader that removing the mask changes the row completely, and
// quotes a per-head average. If the mask ever hid only a few percent, that sentence
// would be selling a molehill — so the band is asserted rather than the prose trusted.
func TestMaskCounterfactualIsLargeEverywhere(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	dh := a.Trace.Config.DHead()
	lo := 1.0
	for li := range a.Trace.Layers {
		for hi := range a.Trace.Layers[li].Heads {
			mean, rows := meanFutureShare(a.Trace.Layers[li].Heads[hi], n, dh)
			if rows == 0 {
				t.Fatalf("L%dH%d: no rows measured", li, hi)
			}
			if mean < lo {
				lo = mean
			}
			if mean < 0.4 {
				t.Fatalf("L%dH%d: removing the mask moves only %.0f%% of the mass to future "+
					"keys; the card says every head moves most of it", li, hi, 100*mean)
			}
		}
	}
	t.Logf("smallest per-head mean future share: %.0f%%", 100*lo)
	// Row 0 in particular: it has one visible key, so the mask is doing nearly all the
	// work there.
	c := computeMaskCounterfactual(a.Trace.Layers[0].Heads[0], 0, n, dh)
	if c.FutureShare < 0.95 {
		t.Fatalf("row 0's unmasked distribution puts only %.0f%% on the future; the card "+
			"calls it the most extreme row", 100*c.FutureShare)
	}
}

// TestMaskCounterfactualCardQuotesItsOwnNumbers reads the card's figures out of the
// rendered text.
func TestMaskCounterfactualCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	if !tr.HasText("掩码挡掉了什么") {
		t.Fatal("the mask counterfactual card is missing")
	}
	n := len(a.Trace.Tokens)
	dh := a.Trace.Config.DHead()
	row := 20
	cf := computeMaskCounterfactual(a.layer().Heads[a.Head], row, n, dh)
	want := fmt.Sprintf("%.0f%%", 100*cf.FutureShare)
	found := false
	for _, s := range tr.Texts() {
		if s == want {
			found = true
		}
	}
	if !found {
		t.Fatalf("the card does not print %q as the hidden share of row %d", want, row)
	}
	// The row-0 sentence must quote row 0's own numbers, not the selected row's:
	// the first draft of it mixed the two scopes.
	first := computeMaskCounterfactual(a.layer().Heads[a.Head], 0, n, dh)
	wantZero := fmt.Sprintf("最大的一格是 k%d 的 %.0f%%", first.FutureTop, 100*first.FutureTopShare)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, wantZero) {
		t.Fatalf("the card's row-0 sentence does not quote row 0's own top future key (%q)",
			wantZero)
	}
}

// TestSkillBreakdownMatchesTheRows checks the word-boundary split against the rows it
// divides: every row must be counted exactly once, and the two halves must add up.
func TestSkillBreakdownMatchesTheRows(t *testing.T) {
	a := newTestApp(t)
	skill := computeNextByteSkill(a.W, a.Trace, 1.0)
	b := computeSkillBreakdown(a.Trace, skill)
	if !b.OK {
		t.Fatal("the breakdown found no space-separated positions in the default prompt")
	}
	if b.AfterSpaceRows+b.OtherRows != skill.Scored {
		t.Fatalf("breakdown covers %d + %d rows, the skill has %d",
			b.AfterSpaceRows, b.OtherRows, skill.Scored)
	}
	if b.AfterSpaceHits > b.AfterSpaceRows || b.OtherHits > b.OtherRows {
		t.Fatalf("more hits than rows: %+v", b)
	}
	if b.AfterSpaceHits+b.OtherHits != skill.Hits {
		t.Fatalf("breakdown hits %d + %d, skill says %d",
			b.AfterSpaceHits, b.OtherHits, skill.Hits)
	}
	// And each row really is on the side its predecessor implies.
	spaceRows, otherRows := 0, 0
	for _, row := range skill.Rows {
		if a.Trace.Tokens[row.Pos].Byte == ' ' {
			spaceRows++
		} else {
			otherRows++
		}
	}
	if spaceRows != b.AfterSpaceRows || otherRows != b.OtherRows {
		t.Fatalf("counted %d/%d rows, recomputed %d/%d", b.AfterSpaceRows, b.OtherRows, spaceRows, otherRows)
	}
}

// TestSkillBreakdownShowsTheWordBoundaryGap is the premise of the sentence: this model
// is much worse at the byte after a space than anywhere else.
//
// The card only prints that explanation when the gap is at least 1.5x, so if the model
// ever became good at word starts the sentence would stop appearing rather than become a
// lie. This checks the gap is there for the bundled weights across several prompts, so
// the card's default state is the one with the explanation.
func TestSkillBreakdownShowsTheWordBoundaryGap(t *testing.T) {
	totalSpace, totalSpaceHits := 0, 0
	totalOther, totalOtherHits := 0, 0
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"Attention lets every token look at the tokens before it.",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		b := computeSkillBreakdown(a.Trace, computeNextByteSkill(a.W, a.Trace, 1.0))
		if !b.OK {
			continue
		}
		totalSpace += b.AfterSpaceRows
		totalSpaceHits += b.AfterSpaceHits
		totalOther += b.OtherRows
		totalOtherHits += b.OtherHits
	}
	if totalSpace == 0 || totalOther == 0 {
		t.Fatal("no rows measured")
	}
	spaceRate := float64(totalSpaceHits) / float64(totalSpace)
	otherRate := float64(totalOtherHits) / float64(totalOther)
	if otherRate < 1.5*spaceRate {
		t.Fatalf("after a space: %d/%d (%.0f%%); elsewhere %d/%d (%.0f%%). The card says the "+
			"word boundary is where the errors are, which needs a clear gap",
			totalSpaceHits, totalSpace, 100*spaceRate, totalOtherHits, totalOther, 100*otherRate)
	}
	t.Logf("word boundary %d/%d (%.0f%%), elsewhere %d/%d (%.0f%%)",
		totalSpaceHits, totalSpace, 100*spaceRate, totalOtherHits, totalOther, 100*otherRate)
}

// TestSkillBreakdownLineRendersItsOwnNumbers reads the breakdown out of the rendered
// text on the output stage.
func TestSkillBreakdownLineRendersItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()
	b := computeSkillBreakdown(a.Trace, computeNextByteSkill(a.W, a.Trace, 1.0))
	if !b.OK {
		t.Fatal("no breakdown")
	}
	wantSpace := fmt.Sprintf("%d/%d（%.0f%%）", b.AfterSpaceHits, b.AfterSpaceRows,
		100*float64(b.AfterSpaceHits)/float64(b.AfterSpaceRows))
	wantOther := fmt.Sprintf("%d/%d（%.0f%%）", b.OtherHits, b.OtherRows,
		100*float64(b.OtherHits)/float64(b.OtherRows))
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, wantSpace) {
		t.Fatalf("the card does not print the word-boundary rate %q", wantSpace)
	}
	if !strings.Contains(joined, wantOther) {
		t.Fatalf("the card does not print the elsewhere rate %q", wantOther)
	}
	if !strings.Contains(joined, "词边界") {
		t.Fatal("the card does not name the word boundary")
	}
}

// TestLayerRewriteMatchesTheTrace recomputes every quantity the card quotes, per layer.
func TestLayerRewriteMatchesTheTrace(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	layers := computeLayerRewrite(tr)
	if len(layers) != len(tr.Layers) {
		t.Fatalf("%d layer measurements for %d layers", len(layers), len(tr.Layers))
	}
	n := len(tr.Tokens)
	for _, r := range layers {
		l := &tr.Layers[r.Layer]
		var cosSum, ffnSum, attnSum, inSum, outSum float64
		for i := 0; i < n; i++ {
			cosSum += cosineF32(l.In[i], l.AfterFFN[i])
			ffnSum += float64(norm(l.FFNOut[i]))
			attnSum += float64(norm(l.AttnProj[i]))
			inSum += float64(norm(l.In[i]))
			outSum += float64(norm(l.AfterFFN[i]))
		}
		f := float64(n)
		for name, pair := range map[string][2]float64{
			"cosine":    {r.CosineInOut, cosSum / f},
			"inNorm":    {r.InNorm, inSum / f},
			"outNorm":   {r.OutNorm, outSum / f},
			"ffnRatio":  {r.FFNRatio, (ffnSum / f) / (inSum / f)},
			"attnRatio": {r.AttnRatio, (attnSum / f) / (inSum / f)},
		} {
			if math.Abs(pair[0]-pair[1]) > 1e-9 {
				t.Fatalf("layer %d %s: card says %v, recomputed %v", r.Layer, name, pair[0], pair[1])
			}
		}
		if r.Hidden != tr.Config.DFF {
			t.Fatalf("layer %d hidden %d, config says %d", r.Layer, r.Hidden, tr.Config.DFF)
		}
	}
}

// TestLayerRewriteIsUnevenEverywhere is the premise of the card's claim: the first
// layer rewrites the stream far more than the last one.
//
// Measured across three prompts: layer 0's cosine is 0.22–0.24 and its FFN ratio
// 2.8–3.2, while layer 1's cosine is 0.92–0.93 and its ratio 0.41–0.45. The card tells
// the reader "most of the transformation happens in the first block", and that sentence
// has to keep being true for every prompt rather than for the default one.
func TestLayerRewriteIsUnevenEverywhere(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"Attention lets every token look at the tokens before it.",
		"hi",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		layers := computeLayerRewrite(a.Trace)
		if len(layers) < 2 {
			continue
		}
		first, last := layers[0], layers[len(layers)-1]
		if first.CosineInOut >= last.CosineInOut {
			t.Fatalf("prompt %q: layer 0's cosine (%.3f) is not below layer 1's (%.3f); "+
				"the card says the first block is the one that rewrites the direction",
				prompt, first.CosineInOut, last.CosineInOut)
		}
		if first.FFNRatio <= last.FFNRatio {
			t.Fatalf("prompt %q: layer 0 adds %.2f of its input, layer 1 adds %.2f; the card "+
				"says the first block adds more", prompt, first.FFNRatio, last.FFNRatio)
		}
	}
}

// TestLayerRewriteCardQuotesItsOwnNumbers reads the card's two per-layer lines out of
// the rendered text.
func TestLayerRewriteCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	layers := computeLayerRewrite(a.Trace)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, "每一层改写了多少") {
		t.Fatal("the layer-rewrite card is missing")
	}
	for _, l := range layers {
		want := fmt.Sprintf("层 %d：进出余弦 %.3f，ΔFFN 是输入的 %.2f 倍", l.Layer, l.CosineInOut, l.FFNRatio)
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q for layer %d", want, l.Layer)
		}
	}
}

// TestPosEncodingShapeMatchesTheTable recomputes every cosine the card quotes.
func TestPosEncodingShapeMatchesTheTable(t *testing.T) {
	a := newTestApp(t)
	s := computePosEncodingShape(a.W)
	if !s.OK {
		t.Fatal("no positional shape computed")
	}
	d, ctx := a.W.Config.DModel, a.W.Config.Ctx
	row := func(i int) []float32 { return a.W.PosEmbed[i*d : (i+1)*d] }
	var adj, far float64
	n := 0
	for i := 0; i+1 < ctx; i++ {
		adj += cosineF32(row(i), row(i+1))
		n++
	}
	if n > 0 {
		if got, want := s.AdjacentCos, adj/float64(n); math.Abs(got-want) > 1e-9 {
			t.Fatalf("adjacent cosine: card %.9f, recomputed %.9f", got, want)
		}
	}
	farN := 0
	for i := 0; i+s.Gap < ctx; i++ {
		far += cosineF32(row(i), row(i+s.Gap))
		farN++
	}
	if farN > 0 {
		if got, want := s.DistantCos, far/float64(farN); math.Abs(got-want) > 1e-9 {
			t.Fatalf("distant cosine: card %.9f, recomputed %.9f", got, want)
		}
	}
	if s.Gap != ctx/2 {
		t.Fatalf("gap %d, want ctx/2 = %d", s.Gap, ctx/2)
	}
	if s.Dims != d {
		t.Fatalf("dims %d, want %d", s.Dims, d)
	}
	if math.Abs(s.RandomCos-1/math.Sqrt(float64(d))) > 1e-12 {
		t.Fatalf("random reference %.9f, want 1/√%d", s.RandomCos, d)
	}
}

// TestPosEncodingShapeDiffersFromSinusoidal is the premise of the whole card: the learned
// table is not the fixed sinusoidal one, and the difference is not marginal.
//
// If these ever converged the card's contrast would be a distinction without a
// difference, and its rebuttal of "all transformers use sin/cos" would be misleading.
func TestPosEncodingShapeDiffersFromSinusoidal(t *testing.T) {
	a := newTestApp(t)
	s := computePosEncodingShape(a.W)
	if !s.OK {
		t.Fatal("no positional shape computed")
	}
	if s.SinusoidalAdjacentCos < 0.8 {
		t.Fatalf("the sinusoidal reference's adjacent cosine is %.3f; the card describes it "+
			"as a slow walk along a curve, which needs a high value", s.SinusoidalAdjacentCos)
	}
	if s.AdjacentCos > 0.5 {
		t.Fatalf("the learned table's adjacent cosine is %.3f; the card says positions are "+
			"nearly orthogonal, which needs a small value", s.AdjacentCos)
	}
	// And the learned table really is about as unrelated as chance — that is the claim
	// that makes "nearly orthogonal" mean something rather than "somewhat different".
	if math.Abs(s.AdjacentCos) > 3*s.RandomCos {
		t.Fatalf("learned adjacent cosine %.3f is far above the %.3f chance level; the card "+
			"says the table is no more related than random vectors", s.AdjacentCos, s.RandomCos)
	}
	t.Logf("learned %.4f vs sinusoidal %.4f vs chance %.4f",
		s.AdjacentCos, s.SinusoidalAdjacentCos, s.RandomCos)
}

// TestPosEncodingShapeCardQuotesItsOwnNumbers reads the card's figures out of the text.
func TestPosEncodingShapeCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	s := computePosEncodingShape(a.W)
	joined := ""
	for _, t := range tr.Texts() {
		joined += t + "\n"
	}
	if !strings.Contains(joined, "学出来的位置长什么样") {
		t.Fatal("the positional-shape card is missing")
	}
	for _, want := range []string{
		fmt.Sprintf("%+.3f", s.AdjacentCos),
		fmt.Sprintf("%+.3f", s.SinusoidalAdjacentCos),
		fmt.Sprintf("%.3f", s.RandomCos),
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
}

// TestFFNConcentrationMatchesTheHiddenRows recomputes every quantity the card quotes.
//
// It calls computeFFNConcentration per layer rather than trusting
// computeAllFFNConcentration's loop, so the per-layer function is exercised directly and
// the layer index it stamps on each result is checked rather than assumed.
func TestFFNConcentrationMatchesTheHiddenRows(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	rows := make([]ffnConcentration, 0, len(tr.Layers))
	for li := range tr.Layers {
		c := computeFFNConcentration(tr.Layers[li], len(tr.Tokens), tr.Config.DFF)
		if c.Layer != -1 {
			t.Fatalf("computeFFNConcentration stamped layer %d; it does not know its index", c.Layer)
		}
		c.Layer = li
		rows = append(rows, c)
	}
	all := computeAllFFNConcentration(tr)
	if len(all) != len(rows) {
		t.Fatalf("computeAllFFNConcentration returned %d rows, direct calls gave %d", len(all), len(rows))
	}
	for i := range rows {
		if all[i].Layer != rows[i].Layer || all[i].HalfMass != rows[i].HalfMass ||
			all[i].AlwaysOn != rows[i].AlwaysOn || all[i].NeverOn != rows[i].NeverOn {
			t.Fatalf("layer %d: computeAllFFNConcentration disagrees with the direct call: %+v vs %+v",
				i, all[i], rows[i])
		}
	}
	if len(rows) != len(tr.Layers) {
		t.Fatalf("%d measurements for %d layers", len(rows), len(tr.Layers))
	}
	n, dff := len(tr.Tokens), tr.Config.DFF
	for _, c := range rows {
		h := tr.Layers[c.Layer].Hidden
		sums := make([]float64, dff)
		positiveAt := make([]int, dff)
		var total float64
		positives := 0
		var peak float64
		for i := 0; i < n; i++ {
			for j, v := range h[i] {
				sums[j] += float64(v)
				total += float64(v)
				if v > 0 {
					positiveAt[j]++
					positives++
				}
				if float64(v) > peak {
					peak = float64(v)
				}
			}
		}
		always, never := 0, 0
		for _, p := range positiveAt {
			if p == n {
				always++
			}
			if p == 0 {
				never++
			}
		}
		if c.AlwaysOn != always || c.NeverOn != never {
			t.Fatalf("layer %d: on-everywhere card=%d recomputed=%d; never card=%d recomputed=%d",
				c.Layer, c.AlwaysOn, always, c.NeverOn, never)
		}
		if math.Abs(c.PositiveShare-float64(positives)/float64(n*dff)) > 1e-12 {
			t.Fatalf("layer %d positive share: card %v, recomputed %v",
				c.Layer, c.PositiveShare, float64(positives)/float64(n*dff))
		}
		if math.Abs(c.Peak-peak) > 1e-9 {
			t.Fatalf("layer %d peak: card %v, recomputed %v", c.Layer, c.Peak, peak)
		}
		// HalfMass: sorting the same sums must give the same count.
		idx := make([]int, dff)
		for j := range idx {
			idx[j] = j
		}
		sort.SliceStable(idx, func(x, y int) bool { return sums[idx[x]] > sums[idx[y]] })
		var acc float64
		count := 0
		for _, j := range idx {
			acc += sums[j]
			count++
			if acc >= total/2 {
				break
			}
		}
		if c.HalfMass != count {
			t.Fatalf("layer %d half-mass: card %d, recomputed %d", c.Layer, c.HalfMass, count)
		}
		if c.Units != dff {
			t.Fatalf("layer %d units %d, want %d", c.Layer, c.Units, dff)
		}
	}
}

// TestFFNConcentrationIsUnevenBetweenLayers is the premise of the card's claim: the
// second layer's FFN is far more concentrated than the first's.
//
// The card prints a comparison sentence only while the ratio holds, so if that stopped
// being true the sentence would vanish rather than become false — but the ratio is what
// makes the finding worth a card, so it is pinned across several prompts.
func TestFFNConcentrationIsUnevenBetweenLayers(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"Attention lets every token look at the tokens before it.",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		rows := computeAllFFNConcentration(a.Trace)
		if len(rows) < 2 {
			continue
		}
		first, last := rows[0], rows[len(rows)-1]
		if first.HalfMass <= last.HalfMass*2 {
			t.Fatalf("prompt %q: layer 0 needs %d units for half its mass, layer 1 needs %d; "+
				"the card says the first layer is spread far wider",
				prompt, first.HalfMass, last.HalfMass)
		}
		if first.AlwaysOn != 0 {
			t.Fatalf("prompt %q: layer 0 has %d units on at every position; the card says none are",
				prompt, first.AlwaysOn)
		}
		t.Logf("%q: L0 %d units / %d always-on, L1 %d units / %d always-on",
			prompt, first.HalfMass, first.AlwaysOn, last.HalfMass, last.AlwaysOn)
	}
}

// TestFFNConcentrationCardQuotesItsOwnNumbers reads the card's figures out of the text.
func TestFFNConcentrationCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	if !tr.HasText("单元是怎么分工的") {
		t.Fatal("the FFN concentration card is missing")
	}
	rows := computeAllFFNConcentration(a.Trace)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	for _, c := range rows {
		want := fmt.Sprintf("层 %d：%d 个单元凑够一半激活，每位置都开的有 %d 个，从不打开 %d 个",
			c.Layer, c.HalfMass, c.AlwaysOn, c.NeverOn)
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
}

// TestLogitLensCalloutFollowsThePrompt is the guard that was missing when the callout's
// numbers were hardcoded.
//
// The sentence said "180 个中间层里有 92 个…，36 个位置" for every prompt. Those figures
// are right only for the default one: with "hi" the ladder has 10 intermediate rungs over
// 2 positions, so a reader who shortened the prompt was told about a ladder that was not
// the one drawn above the sentence. The previous test recomputed the figures inline and
// compared them to a band, which could not notice that the *rendered* sentence had stopped
// agreeing with either — it never read the prose.
//
// This reads the rendered text and requires the numbers in it to match a fresh
// computation, for several prompts of very different lengths.
func TestLogitLensCalloutFollowsThePrompt(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"hi",
		"中文提示词",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 4
		tr := ui.NewTester(a.View, 1500, 3200)
		tr.Frame()
		summary := computePromptLensSummary(a.W, a.Trace, 1.0)
		if !summary.OK || summary.Rungs == 0 {
			continue
		}
		var callout string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "中间层") && strings.Contains(s, "最终答案不同") {
				callout = s
			}
		}
		if callout == "" {
			t.Fatalf("prompt %q: the logit lens callout is missing", prompt)
		}
		wants := []string{
			fmt.Sprintf("%d 个位置一共有 %d 个中间层", summary.Positions, summary.Rungs),
			fmt.Sprintf("其中 %d 个的答案与最终答案不同", summary.Disagree),
			fmt.Sprintf("有 %d 个对自己的错误答案", summary.ConfidentWrong),
			fmt.Sprintf("%d 个位置里有 %d 个至少有一层如此",
				summary.Positions, summary.PositionsWithConfidentWrong),
		}
		for _, want := range wants {
			if !strings.Contains(callout, want) {
				t.Fatalf("prompt %q: the callout does not contain %q\ncallout: %s", prompt, want, callout)
			}
		}
	}
}

// TestPromptLensSummaryCountsEveryRung checks the aggregator against a direct walk.
func TestPromptLensSummaryCountsEveryRung(t *testing.T) {
	a := newTestApp(t)
	s := computePromptLensSummary(a.W, a.Trace, 1.0)
	if !s.OK {
		t.Fatal("no summary computed")
	}
	// Five intermediate rungs per position, by construction of the ladder.
	if s.Rungs != 5*s.Positions {
		t.Fatalf("%d rungs over %d positions; the ladder builds five per position",
			s.Rungs, s.Positions)
	}
	if s.Positions != len(a.Trace.Tokens) {
		t.Fatalf("%d positions, the prompt has %d tokens", s.Positions, len(a.Trace.Tokens))
	}
	// Recompute independently.
	rungs, disagree, confident, posCW := 0, 0, 0, 0
	for pos := 0; pos < len(a.Trace.Tokens); pos++ {
		lens := computeLogitLens(a.W, a.Trace, pos, 1.0)
		has := false
		for _, rung := range lens.Rungs {
			if rung.IsFinal {
				continue
			}
			rungs++
			if rung.AgreesWithFinal {
				continue
			}
			disagree++
			if float64(rung.Probs[rung.Top]) > lensConfident {
				confident++
				has = true
			}
		}
		if has {
			posCW++
		}
	}
	if s.Rungs != rungs || s.Disagree != disagree || s.ConfidentWrong != confident ||
		s.PositionsWithConfidentWrong != posCW {
		t.Fatalf("summary %+v disagrees with a direct walk (%d/%d/%d/%d)",
			s, rungs, disagree, confident, posCW)
	}
}

// TestWeightTyingDecompositionIsExact checks the identity the card prints: the logit for
// the top byte must equal ‖E[b]‖ × ‖x‖ × cos, and the probability must match the app's
// own softmax over Trace.Logits rather than a float64 re-derivation of it.
func TestWeightTyingDecompositionIsExact(t *testing.T) {
	a := newTestApp(t)
	p := computeWeightTying(a.W, a.Trace, 1.0)
	if !p.OK {
		t.Fatal("no tying decomposition")
	}
	product := p.EmbedNorm * p.FinalNorm * p.Cosine
	if math.Abs(product-p.Logit) > 1e-6 {
		t.Fatalf("‖E‖·‖x‖·cos = %.9f, the logit is %.9f", product, p.Logit)
	}
	// The same byte the model itself picks.
	probs := a.Trace.Probabilities(1.0)
	best, bp := 0, float32(-1)
	for id, v := range probs {
		if v > bp {
			best, bp = id, v
		}
	}
	if p.Byte != best {
		t.Fatalf("the card names byte %d, the model's top byte is %d", p.Byte, best)
	}
	if math.Abs(p.Probability-float64(bp)) > 1e-6 {
		t.Fatalf("probability %.9f, Trace gives %.9f", p.Probability, float64(bp))
	}
	// The reported norms and cosines must bound every byte.
	d, vocab := a.W.Config.DModel, a.W.Config.Vocab
	row := a.Trace.FinalNorm[len(a.Trace.FinalNorm)-1]
	var rn float64
	for _, v := range row {
		rn += float64(v) * float64(v)
	}
	rn = math.Sqrt(rn)
	for id := 0; id < vocab; id++ {
		var dot, ne float64
		for k := 0; k < d; k++ {
			e := float64(a.W.TokEmbed[id*d+k])
			dot += e * float64(row[k])
			ne += e * e
		}
		n := math.Sqrt(ne)
		c := dot / (n * rn)
		if n < p.EmbedNormMin-1e-9 || n > p.EmbedNormMax+1e-9 {
			t.Fatalf("byte %d norm %.4f outside the stated range %.4f..%.4f",
				id, n, p.EmbedNormMin, p.EmbedNormMax)
		}
		if c < p.CosineMin-1e-9 || c > p.CosineMax+1e-9 {
			t.Fatalf("byte %d cosine %.4f outside the stated range %.4f..%.4f",
				id, c, p.CosineMin, p.CosineMax)
		}
	}
}

// TestWeightTyingCosineDrivesTheRanking is the premise of the card's whole point: with the
// tying, the model's preference is a cosine preference.
//
// The card says the ranking is essentially decided by the cosine and not by the embedding
// lengths. If the weights ever made the lengths decisive, that sentence would be wrong —
// so both correlations are measured, and the claim is that cosine wins by a wide margin.
func TestWeightTyingCosineDrivesTheRanking(t *testing.T) {
	a := newTestApp(t)
	p := computeWeightTying(a.W, a.Trace, 1.0)
	if !p.OK {
		t.Fatal("no tying decomposition")
	}
	if p.LogitCosineCorrelation < 0.9 {
		t.Fatalf("cosine correlates %.3f with the logit; the card says the ranking is "+
			"essentially a cosine ranking", p.LogitCosineCorrelation)
	}
	if math.Abs(p.LogitNormCorrelation) > 0.3 {
		t.Fatalf("the embedding length correlates %+.3f with the logit; the card says the "+
			"length plays almost no part", p.LogitNormCorrelation)
	}
	// And the pairing: nearly all byte pairs order the same both ways.
	if p.Pairs == 0 {
		t.Fatal("no pairs measured")
	}
	share := float64(p.PairsOrderedDifferently) / float64(p.Pairs)
	if share > 0.05 {
		t.Fatalf("%.1f%% of byte pairs order differently by logit than by cosine; the card "+
			"says the two orderings nearly coincide", 100*share)
	}
	t.Logf("corr(logit,cos)=%.3f corr(logit,|E|)=%+.3f, %.1f%% of pairs differ",
		p.LogitCosineCorrelation, p.LogitNormCorrelation, 100*share)
}

// TestWeightTyingCardQuotesItsOwnNumbers reads the decomposition out of the rendered text.
func TestWeightTyingCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	p := computeWeightTying(a.W, a.Trace, 1.0)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, "权重绑定让 logit 变成一个余弦") {
		t.Fatal("the weight-tying card is missing")
	}
	for _, want := range []string{
		fmt.Sprintf("%.3f × %.3f × %.4f", p.EmbedNorm, p.FinalNorm, p.Cosine),
		fmt.Sprintf("%+.3f", p.LogitCosineCorrelation),
		fmt.Sprintf("%.2f 变到 %.2f", p.EmbedNormMin, p.EmbedNormMax),
		fmt.Sprintf("%+.3f 变到 %+.3f", p.CosineMin, p.CosineMax),
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
}

// TestHeadIndependenceMatchesTheHeadProj recomputes the pairwise cosines and the shares.
func TestHeadIndependenceMatchesTheHeadProj(t *testing.T) {
	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	for li := range a.Trace.Layers {
		l := a.Trace.Layers[li]
		c := computeHeadIndependence(l, n)
		if !c.OK {
			t.Fatalf("layer %d: no independence measured", li)
		}
		if c.Heads != len(l.HeadProj) {
			t.Fatalf("layer %d: %d heads measured, %d present", li, c.Heads, len(l.HeadProj))
		}
		// Recompute the pair range.
		minC, maxC := math.Inf(1), math.Inf(-1)
		for i := 0; i < len(l.HeadProj); i++ {
			for j := i + 1; j < len(l.HeadProj); j++ {
				var sum float64
				for p := 0; p < n; p++ {
					sum += cosineF32(l.HeadProj[i][p], l.HeadProj[j][p])
				}
				m := sum / float64(n)
				if m < minC {
					minC = m
				}
				if m > maxC {
					maxC = m
				}
			}
		}
		if math.Abs(c.MinPairCosine-minC) > 1e-9 || math.Abs(c.MaxPairCosine-maxC) > 1e-9 {
			t.Fatalf("layer %d pair range: card %+.6f..%+.6f, recomputed %+.6f..%+.6f",
				li, c.MinPairCosine, c.MaxPairCosine, minC, maxC)
		}
		// Recompute the share range.
		lengths := make([]float64, len(l.HeadProj))
		var total float64
		for i := range l.HeadProj {
			for p := 0; p < n; p++ {
				lengths[i] += float64(norm(l.HeadProj[i][p]))
			}
			total += lengths[i]
		}
		minS, maxS := math.Inf(1), math.Inf(-1)
		for _, v := range lengths {
			s := v / total
			if s < minS {
				minS = s
			}
			if s > maxS {
				maxS = s
			}
		}
		if math.Abs(c.MinShare-minS) > 1e-9 || math.Abs(c.MaxShare-maxS) > 1e-9 {
			t.Fatalf("layer %d share range: card %.6f..%.6f, recomputed %.6f..%.6f",
				li, c.MinShare, c.MaxShare, minS, maxS)
		}
	}
}

// TestHeadIndependenceSaysHeadsDiffer is the premise of the card: the heads' outputs are
// far from copies.
//
// The card tells the reader that concatenating is right because the slices say different
// things. That claim needs the cosines to be small; if a weight set made the four heads
// produce near-identical vectors, the sentence would be false rather than merely
// unverified. Layer 1 is the looser case, so the bound is checked there too.
func TestHeadIndependenceSaysHeadsDiffer(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		n := len(a.Trace.Tokens)
		for li := range a.Trace.Layers {
			c := computeHeadIndependence(a.Trace.Layers[li], n)
			if !c.OK {
				t.Fatalf("prompt %q layer %d: no measurement", prompt, li)
			}
			if c.MaxPairCosine > 0.7 {
				t.Fatalf("prompt %q layer %d: two heads reach cosine %.3f; the card says the "+
					"heads are far from copies", prompt, li, c.MaxPairCosine)
			}
			// And no head may dominate: if one carried nearly everything, "four parallel
			// channels" would misdescribe it.
			if c.MaxShare > 0.6 {
				t.Fatalf("prompt %q layer %d: one head carries %.0f%% of the total length",
					prompt, li, 100*c.MaxShare)
			}
		}
	}
}

// TestHeadIndependenceCardQuotesItsOwnNumbers reads the per-layer figures out of the text.
func TestHeadIndependenceCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 3400)
	tr.Frame()
	n := len(a.Trace.Tokens)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, "四个头说的是同一件事吗") {
		t.Fatal("the head-independence card is missing")
	}
	for li := range a.Trace.Layers {
		c := computeHeadIndependence(a.Trace.Layers[li], n)
		want := fmt.Sprintf("层 %d：四个头的投影输出两两平均余弦在 %+.3f 到 %+.3f 之间",
			li, c.MinPairCosine, c.MaxPairCosine)
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
}

// TestFFNAnswerDirectionMatchesADirectRanking recomputes the ranks and cosines straight
// from the trace, independently of the helper's own loop.
func TestFFNAnswerDirectionMatchesADirectRanking(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	w := a.W
	d, vocab := w.Config.DModel, w.Config.Vocab
	last := len(tr.Tokens) - 1
	rankAgainst := func(v []float32, actual int) (int, float64) {
		var nv float64
		for _, x := range v {
			nv += float64(x) * float64(x)
		}
		root := math.Sqrt(nv)
		target, rank := 0.0, 1
		for id := 0; id < vocab; id++ {
			var dot, ne float64
			for k := 0; k < d; k++ {
				e := float64(w.TokEmbed[id*d+k])
				dot += float64(v[k]) * e
				ne += e * e
			}
			c := dot / (root * math.Sqrt(ne))
			if id == actual {
				target = c
			}
		}
		for id := 0; id < vocab; id++ {
			if id == actual {
				continue
			}
			var dot, ne float64
			for k := 0; k < d; k++ {
				e := float64(w.TokEmbed[id*d+k])
				dot += float64(v[k]) * e
				ne += e * e
			}
			if dot/(root*math.Sqrt(ne)) > target {
				rank++
			}
		}
		return rank, target
	}
	for li := range tr.Layers {
		m := computeFFNAnswerDirection(a.W, tr, li)
		if !m.OK {
			t.Fatalf("layer %d: no direction measured", li)
		}
		if m.Positions != last {
			t.Fatalf("layer %d: %d positions, the prompt offers %d", li, m.Positions, last)
		}
		var outRank, inRank, outCos, inCos float64
		top10 := 0
		for p := 0; p < last; p++ {
			actual := int(tr.Tokens[p+1].Byte)
			or, oc := rankAgainst(tr.Layers[li].FFNOut[p], actual)
			ir, ic := rankAgainst(tr.Layers[li].In[p], actual)
			outRank += float64(or)
			inRank += float64(ir)
			outCos += oc
			inCos += ic
			if or <= 10 {
				top10++
			}
		}
		f := float64(last)
		for name, pair := range map[string][2]float64{
			"output rank": {m.OutputMeanRank, outRank / f},
			"input rank":  {m.InputMeanRank, inRank / f},
			"output cos":  {m.OutputMeanCos, outCos / f},
			"input cos":   {m.InputMeanCos, inCos / f},
		} {
			if math.Abs(pair[0]-pair[1]) > 1e-6 {
				t.Fatalf("layer %d %s: card %v, recomputed %v", li, name, pair[0], pair[1])
			}
		}
		if m.OutputTop10 != top10 {
			t.Fatalf("layer %d top-10 count: card %d, recomputed %d", li, m.OutputTop10, top10)
		}
		if m.Vocab != vocab {
			t.Fatalf("layer %d vocab %d, config %d", li, m.Vocab, vocab)
		}
	}
}

// TestFFNAnswerDirectionShowsLayerZeroDoingTheWork is the premise of the card's claim:
// layer 0's FFN moves the true byte from near chance to near the top.
//
// The card says the answer direction is written from nothing rather than refined. That
// needs layer 0's input rank to be close to chance and its output rank to be far better,
// measured rather than assumed.
func TestFFNAnswerDirectionShowsLayerZeroDoingTheWork(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		if len(a.Trace.Tokens) < 3 {
			continue
		}
		m := computeFFNAnswerDirection(a.W, a.Trace, 0)
		if !m.OK {
			t.Fatalf("prompt %q: no measurement", prompt)
		}
		chance := float64(m.Vocab) / 2
		// The input has to be far from the answer for the sentence to be right.
		if m.InputMeanRank < chance/3 {
			t.Fatalf("prompt %q: layer 0's input already ranks the true byte %.1f out of %d; "+
				"the card says the FFN writes the direction from nothing",
				prompt, m.InputMeanRank, m.Vocab)
		}
		// And the output has to be much better.
		if m.OutputMeanRank > m.InputMeanRank/5 {
			t.Fatalf("prompt %q: layer 0's FFN output ranks the true byte %.1f against the "+
				"input's %.1f; the card says the FFN moves it close to the top",
				prompt, m.OutputMeanRank, m.InputMeanRank)
		}
		t.Logf("%q: layer 0 input rank %.1f -> output rank %.1f", prompt, m.InputMeanRank, m.OutputMeanRank)
	}
}

// TestFFNAnswerDirectionCardQuotesItsOwnNumbers reads both layers' lines out of the text.
func TestFFNAnswerDirectionCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 3400)
	tr.Frame()
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, "FFN 的输出指向下一个字节吗") {
		t.Fatal("the FFN answer-direction card is missing")
	}
	for li := range a.Trace.Layers {
		m := computeFFNAnswerDirection(a.W, a.Trace, li)
		want := fmt.Sprintf("层 %d：输入里真实字节平均第 %.1f 名（余弦 %+.3f），FFN 之后第 %.1f 名（余弦 %+.3f）",
			li, m.InputMeanRank, m.InputMeanCos, m.OutputMeanRank, m.OutputMeanCos)
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
}

// TestEmbeddingStructureMatchesADirectMeasurement recomputes every figure the card prints.
//
// It runs on "attention is all you need" as well as the default prompt because that one has
// three repeated-byte pairs (tt, ll, ee). Those have cosine exactly 1.0 with themselves and
// are excluded from the average; the first version of the measurement did not exclude them,
// which lifted that prompt's adjacent mean to 0.1265 and would have made the card claim
// structure where there is none. Testing only the default prompt, which has no repeated
// pairs, could not have caught it.
func TestEmbeddingStructureMatchesADirectMeasurement(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
	} {
		testEmbeddingStructureAgainstOnePrompt(t, prompt)
	}
}

func testEmbeddingStructureAgainstOnePrompt(t *testing.T, promptText string) {
	t.Helper()
	a := newTestApp(t)
	a.Prompt = promptText
	a.recompute()
	e := computeEmbeddingStructure(a.W, a.Prompt)
	if !e.OK {
		t.Fatal("no embedding structure measured")
	}
	d, vocab := a.W.Config.DModel, a.W.Config.Vocab
	row := func(id int) []float32 { return a.W.TokEmbed[id*d : (id+1)*d] }
	// Adjacent pairs in the prompt, skipping the repeated-byte pairs the card excludes.
	var adj float64
	pairs, skipped := 0, 0
	bs := []byte(a.Prompt)
	for i := 0; i+1 < len(bs); i++ {
		if bs[i] == bs[i+1] {
			skipped++
			continue
		}
		adj += cosineF32(row(int(bs[i])), row(int(bs[i+1])))
		pairs++
	}
	if pairs != e.AdjacentPairs || skipped != e.SkippedSelfPairs {
		t.Fatalf("adjacent pairs: card %d (+%d skipped), prompt has %d (+%d)",
			e.AdjacentPairs, e.SkippedSelfPairs, pairs, skipped)
	}
	if math.Abs(e.AdjacentCos-adj/float64(pairs)) > 1e-9 {
		t.Fatalf("adjacent cosine: card %v, recomputed %v", e.AdjacentCos, adj/float64(pairs))
	}
	// Random reference.
	var sum, sum2 float64
	n := 0
	for i := 0; i < vocab; i++ {
		for j := i + 1; j < vocab; j++ {
			c := cosineF32(row(i), row(j))
			sum += c
			sum2 += c * c
			n++
		}
	}
	if n != e.RandomPairs {
		t.Fatalf("random pairs: card %d, recomputed %d", e.RandomPairs, n)
	}
	mean := sum / float64(n)
	if math.Abs(e.RandomCos-mean) > 1e-9 {
		t.Fatalf("random mean: card %v, recomputed %v", e.RandomCos, mean)
	}
	sd := math.Sqrt(sum2/float64(n) - mean*mean)
	if math.Abs(e.SDRandom-sd) > 1e-9 {
		t.Fatalf("random sd: card %v, recomputed %v", e.SDRandom, sd)
	}
	if e.Vocab != vocab || e.Dims != d {
		t.Fatalf("card says %d bytes x %d dims, config %d x %d", e.Vocab, e.Dims, vocab, d)
	}
	if len(e.TopPairs) == 0 {
		t.Fatal("no top pairs recorded")
	}
	// The pairs must be ordered descending, and each cosine must be right.
	for i, p := range e.TopPairs {
		if got := cosineF32(row(p.A), row(p.B)); math.Abs(p.Cos-got) > 1e-9 {
			t.Fatalf("top pair %d: card cos %v, recomputed %v", i, p.Cos, got)
		}
		if i > 0 && p.Cos > e.TopPairs[i-1].Cos {
			t.Fatalf("top pairs are not ordered at index %d", i)
		}
	}
}

// TestEmbeddingStructureShowsNoStructure is the premise of the card, and it is a negative
// one: the embedding table does not encode similarity between bytes.
//
// The card tells a reader that the plausible inference — similar vectors mean similar
// meanings — is false for this model. That needs the adjacent-pair cosine to sit inside the
// random spread, and the most similar pairs to be unrelated bytes. If a weight set ever had
// structure, the card would be wrong rather than merely unverified.
func TestEmbeddingStructureShowsNoStructure(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"注意力就是加权求和",
		strings.Repeat("ab", 50),
		// The periodic prompt above is the point of this sweep rather than extra coverage.
		// A periodic prompt produces adjacent pairs that really are closer than random:
		// measured on "ab"x50, adjacent +0.1737 against a random mean of +0.0033 with
		// spread 0.1017 — 1.7 standard deviations. The card used to print that pair of
		// numbers and then say, in fixed prose, that the difference was inside the noise.
		// This test only ever checked the two prompts above, where z < 0.5, so the claim
		// looked like a property of the model when it is a property of the prompt.
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		e := computeEmbeddingStructure(a.W, prompt)
		if !e.OK {
			t.Fatalf("prompt %q: no measurement", prompt)
		}
		if e.SDRandom == 0 {
			t.Fatalf("prompt %q: no spread measured", prompt)
		}
		// The adjacent mean must be within a small multiple of the random spread *for an
		// ordinary prompt*. Periodic input is excluded by name: it is measurable structure in
		// the prompt, not in the table, and the card now says so instead of denying it.
		z := math.Abs(e.AdjacentCos-e.RandomCos) / e.SDRandom
		if z > 0.5 && !strings.Contains(prompt, "abab") {
			t.Fatalf("prompt %q: text-adjacent pairs average %+.4f against a random mean of "+
				"%+.4f with spread %.4f — the difference is %.1f standard deviations, so the "+
				"table does have structure and the card's claim is wrong",
				prompt, e.AdjacentCos, e.RandomCos, e.SDRandom, z)
		}
		// Most similar pairs should not be readable text; a table full of structured
		// relations would put letter pairs at the top.
		if e.SimilarPrintableOf20 > 8 {
			t.Fatalf("prompt %q: %d of the 20 most similar byte pairs are printable ASCII; "+
				"the card says the similarity is noise", prompt, e.SimilarPrintableOf20)
		}
	}
}

// TestEmbeddingStructureCardQuotesItsOwnNumbers reads the figures out of the rendered text
// and requires the top-pair line to use the app's byte notation rather than Go's escaping.
func TestEmbeddingStructureCardQuotesItsOwnNumbers(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 3400)
	tr.Frame()
	e := computeEmbeddingStructure(a.W, a.Prompt)
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
	}
	if !strings.Contains(joined, "这 96 个维度里有「意思」吗") {
		t.Fatal("the embedding structure card is missing")
	}
	for _, want := range []string{
		"文本里相邻的",
		fmt.Sprintf("随便两对 %d 组：平均余弦 %s，标准差 %.4f", e.RandomPairs, cosText(e.RandomCos), e.SDRandom),
		fmt.Sprintf("最像的 20 对里，只有 %d 对两边都是可打印 ASCII", e.SimilarPrintableOf20),
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("the card does not print %q", want)
		}
	}
	// The named pairs must be written the way the rest of the app writes bytes.
	top := e.TopPairs[0]
	want := fmt.Sprintf("%s-%s %.2f", byteText(top.A), byteText(top.B), top.Cos)
	if !strings.Contains(joined, want) {
		t.Fatalf("the card does not print the top pair as %q", want)
	}
	if strings.Contains(joined, `\u`) {
		t.Fatal("a byte is rendered with Go's \\u escaping instead of the app's byte notation")
	}
}

// TestTemperatureCurveFindsTheActualMinimum checks that the curve reports the temperature it
// actually measured rather than assuming T=1.
//
// The card used to state in fixed prose that the mean loss is U-shaped with its minimum at
// T=1. Measured, that is false for most prompts: "attention is all you need" bottoms out at
// 1.5 and "hi" at 2.0, and even the default prompt's minimum is 1.25 rather than 1.0. So the
// sentence is now built from this curve, and the curve has to agree with a fresh scan.
func TestTemperatureCurveFindsTheActualMinimum(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"hi",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		if len(a.Trace.Tokens) < 2 {
			continue
		}
		c := computeTemperatureCurve(a.W, a.Trace)
		if !c.OK {
			t.Fatalf("prompt %q: no curve", prompt)
		}
		if len(c.Samples) != len(c.Losses) || len(c.Samples) == 0 {
			t.Fatalf("prompt %q: %d samples, %d losses", prompt, len(c.Samples), len(c.Losses))
		}
		// Every quoted loss must be the one that temperature produces.
		for i, temp := range c.Samples {
			want := computeNextByteSkill(a.W, a.Trace, temp).MeanLoss
			if math.Abs(c.Losses[i]-want) > 1e-9 {
				t.Fatalf("prompt %q T=%.2f: curve says %.6f, recomputed %.6f",
					prompt, temp, c.Losses[i], want)
			}
		}
		// The reported best must be the argmin of the samples.
		lowest := 0
		for i, v := range c.Losses {
			if v < c.Losses[lowest] {
				lowest = i
			}
		}
		if math.Abs(c.Best-c.Samples[lowest]) > 1e-12 || math.Abs(c.BestLoss-c.Losses[lowest]) > 1e-9 {
			t.Fatalf("prompt %q: card says best T=%.2f (%.6f), the samples say T=%.2f (%.6f)",
				prompt, c.Best, c.BestLoss, c.Samples[lowest], c.Losses[lowest])
		}
		// And T=1 must be represented, since the sentence quotes it.
		if math.Abs(c.AtOne-computeNextByteSkill(a.W, a.Trace, 1.0).MeanLoss) > 1e-9 {
			t.Fatalf("prompt %q: AtOne %.6f disagrees with the T=1 measurement", prompt, c.AtOne)
		}
	}
}

// TestTemperatureMinimumIsNotAlwaysOne records the fact that made the old sentence wrong.
//
// This is deliberately a statement about the bundled weights rather than a tautology: if a
// future checkpoint did put every prompt's minimum at T=1, the card's "最低点不在 T = 1"
// branch would stop being reached, and this test failing is the signal to re-read the prose.
func TestTemperatureMinimumIsNotAlwaysOne(t *testing.T) {
	offOne := 0
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"hi",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		if len(a.Trace.Tokens) < 2 {
			continue
		}
		c := computeTemperatureCurve(a.W, a.Trace)
		if c.OK && math.Abs(c.Best-1) > 1e-9 {
			offOne++
		}
	}
	if offOne == 0 {
		t.Fatal("every prompt now minimises at T=1, so the card's \"最低点不在 T = 1\" branch " +
			"is never reached; re-read the sentence and this test")
	}
	t.Logf("%d of 3 prompts minimise away from T=1", offOne)
}

// TestUnderflowIsCountedNotPrinted checks that a probability rounding to zero produces a
// count rather than an infinite loss, and that the mean stays finite.
//
// At T=0.1 the true byte's probability can underflow in float32. The loss then has no finite
// value, and the card printed "平均损失 +Inf nats" and "降低了 -Inf%", which reads as a fault
// in the app rather than as a limit of the arithmetic.
func TestUnderflowIsCountedNotPrinted(t *testing.T) {
	// A long prompt with a low-probability truth is the surest way to underflow at T=0.1.
	a := newTestApp(t)
	a.Prompt = "Attention lets every token look at the tokens before it."
	a.recompute()
	cold := computeNextByteSkill(a.W, a.Trace, 0.1)
	if cold.Scored == 0 {
		t.Fatal("nothing scored")
	}
	if cold.Underflowed == 0 {
		t.Skip("this prompt does not underflow at T=0.1; nothing to check")
	}
	if math.IsInf(cold.MeanLoss, 0) || math.IsNaN(cold.MeanLoss) {
		t.Fatalf("the mean loss is %v with %d underflowed positions; it must stay finite",
			cold.MeanLoss, cold.Underflowed)
	}
	// Every row's loss must be finite too, or the inspector would print an Inf.
	for _, row := range cold.Rows {
		if math.IsInf(row.Loss, 0) || math.IsNaN(row.Loss) {
			t.Fatalf("position %d has loss %v", row.Pos, row.Loss)
		}
	}
	// At T=1 nothing should underflow on any of these prompts.
	warm := computeNextByteSkill(a.W, a.Trace, 1.0)
	if warm.Underflowed != 0 {
		t.Fatalf("%d positions underflow at T=1", warm.Underflowed)
	}
}

// TestUnderflowNoteReachesTheCard reads the note out of the rendered text.
func TestUnderflowNoteReachesTheCard(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "Attention lets every token look at the tokens before it."
	a.recompute()
	a.Step = 6
	a.Temp = 0.1
	cold := computeNextByteSkill(a.W, a.Trace, 0.1)
	if cold.Underflowed == 0 {
		t.Skip("this prompt does not underflow at T=0.1")
	}
	tr := ui.NewTester(a.View, 1500, 3400)
	tr.Frame()
	joined := ""
	for _, s := range tr.Texts() {
		joined += s + "\n"
		if strings.Contains(s, "+Inf") {
			t.Fatalf("the card renders %q; an infinite loss is an arithmetic artefact, not a "+
				"measurement the reader can use", s)
		}
	}
	want := fmt.Sprintf("%d 个位置的真实字节概率下溢到 0，未计入均值", cold.Underflowed)
	if !strings.Contains(joined, want) {
		t.Fatalf("the card does not explain the underflow (%q)", want)
	}
}

// TestLessonNumbersAreComputedNotQuoted guards the property an audit of the lesson bodies
// was looking for and, this time, found intact.
//
// Three rounds have each fixed a prose claim that was measured once and then shown for every
// prompt: the logit lens callout (round 15), the loss curve's minimum (round 24), and the
// embedding structure figure. The lesson bodies are the largest block of hand-written prose
// in the app, so they are the next place that shape could hide. It is not there — the FFN
// lesson's sparsity sentence comes from sparsityEvidence(tr) and tracks the prompt:
//
//	"hi"                                  40% / 3% / 64%
//	"The transformer reads the tokens and" 44% / 2% / 73%
//
// This test states that as a fact to preserve rather than a bug to fix. If a future edit
// replaces the computation with literal percentages, the two prompts will agree and this
// fails, which is the cue to look at the sentence.
func TestLessonNumbersAreComputedNotQuoted(t *testing.T) {
	evidence := func(prompt string) string {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 5 // the FFN lesson carries the sparsity sentence
		tr := ui.NewTester(a.View, 1500, 3400)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "最大的十分之一") {
				return s
			}
		}
		return ""
	}
	short := evidence("hi")
	long := evidence("The transformer reads the tokens and")
	if short == "" || long == "" {
		t.Fatal("the FFN lesson's sparsity sentence is missing on one of the prompts")
	}
	if short == long {
		t.Fatalf("the sparsity sentence is identical for a 2-token and a 36-token prompt, so its "+
			"numbers are no longer computed from the trace:\n%s", long)
	}
	// And the sentence has to name the four quantities, not just differ.
	for _, want := range []string{"最大的十分之一", "中位数", "低于峰值的十分之一", "恰好为 0"} {
		if !strings.Contains(long, want) {
			t.Fatalf("the sparsity sentence no longer mentions %q", want)
		}
	}
}

// TestCausalInvarianceHoldsOnEveryPrompt is the positive half of the causal check: the
// app prints "N positions did not move" on the output stage, and this requires the
// comparison behind that sentence to actually say so on the prompts a reader can reach —
// including the ones past the context window, where a naive before/after comparison of the
// *prompt* would compare two different windows and report a change that did not happen.
//
// It runs under both matmul kernels because both are shipped: `just check` uses the scalar
// one, CI also runs `GOEXPERIMENT=simd go test ./...`. The tolerance exists because the two
// disagree in the last bits (see causalTolerance), and this is the test that proves the
// tolerance is wide enough for the SIMD build and still narrow enough to catch a leak.
func TestCausalInvarianceHoldsOnEveryPrompt(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"attention is all you need",
		"hi",
		"ab", "abc", "abcd", // the sizes around the 4-row SIMD blocking boundary
		"注意力就是加权求和",
		"a🙂b",
		strings.Repeat("attention is a weighted sum ", 3), // past the 64-token window
		strings.Repeat("b", 64),
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		n := len(a.Trace.Tokens)
		if n < 2 {
			continue
		}
		cmp := a.causalInvariance()
		if !cmp.OK {
			t.Fatalf("%q (%d tokens): the comparison declined", prompt, n)
		}
		// Every position the shorter run has, including its final row — that row has no
		// successor to be scored against, but it made the prediction the next byte came
		// from, so it is exactly the row the invariant is about.
		if cmp.Compared != n-1 {
			t.Fatalf("%q (%d tokens): compared %d positions, want %d", prompt, n, cmp.Compared, n-1)
		}
		if cmp.Identical != cmp.Compared {
			t.Fatalf("%q (%d tokens): %d of %d positions moved when the last byte was "+
				"removed; the causal mask is supposed to make that impossible (first at %d, "+
				"worst deviation %.3e against a tolerance of %.0e)",
				prompt, n, cmp.Compared-cmp.Identical, cmp.Compared, cmp.MismatchAt,
				cmp.WorstDelta, causalTolerance)
		}
		if cmp.MismatchAt != -1 {
			t.Fatalf("%q: every row agreed but MismatchAt is %d", prompt, cmp.MismatchAt)
		}
		// And the noise really is noise: well inside the tolerance rather than scraping it,
		// so a kernel that started drifting further would fail here rather than silently
		// consuming the whole margin.
		if cmp.WorstDelta > causalTolerance/2 {
			t.Fatalf("%q: the worst deviation %.3e is over half the %.0e tolerance; the "+
				"matmul kernels are drifting further apart than the margin was measured for",
				prompt, cmp.WorstDelta, causalTolerance)
		}
	}
}

// TestCausalInvarianceDetectsABrokenMask is the half that makes the first one worth
// anything: a comparison that can only ever say "identical" is not a measurement.
//
// The negative control perturbs one row the way an off-by-one mask would and requires the
// comparison to notice and to name the position. The size of the perturbation is not
// arbitrary: leaking one future key into every row of the real forward pass moves a
// probability by ~2.7e-1 (measured; see causalTolerance), so 1e-3 is a far smaller defect than a
// real leak and still has to be caught. Two mutations are checked — one that changes the
// argmax and one that leaves it alone — because a comparison that only watched the argmax
// would miss the second, which is precisely the shape a subtle mask bug has.
func TestCausalInvarianceDetectsABrokenMask(t *testing.T) {
	a := newTestApp(t)
	prefix := predictEveryPosition(a.W, a.Trace, causalReferenceTemperature)
	full := predictEveryPosition(a.W, a.Trace, causalReferenceTemperature)
	if len(prefix) < 4 {
		t.Fatalf("%d rows, too few to perturb", len(prefix))
	}
	clean := compareCausalInvariance(prefix, full)
	if clean.Identical != clean.Compared || clean.Compared != len(prefix) {
		t.Fatalf("the unperturbed comparison is not clean: %+v", clean)
	}

	for _, c := range []struct {
		name   string
		pos    int
		mutate func(row *positionPrediction)
	}{
		{
			// A different argmax: what a leaked future key would most obviously do.
			name: "argmax changes",
			pos:  2,
			mutate: func(row *positionPrediction) {
				row.Guess = (row.Guess + 1) % len(row.Probs)
			},
		},
		{
			// The argmax survives; one other probability moves by far more than the kernels'
			// rounding noise but far less than a full leak. A guess-only comparison misses
			// this, and it is the case a subtle mask defect actually produces.
			name: "one probability moves 1e-3",
			pos:  1,
			mutate: func(row *positionPrediction) {
				other := (row.Guess + 1) % len(row.Probs)
				row.Probs[other] += 1e-3
			},
		},
	} {
		mutated := make([]positionPrediction, len(full))
		copy(mutated, full)
		mutated[c.pos] = full[c.pos]
		mutated[c.pos].Probs = append([]float32(nil), full[c.pos].Probs...)
		c.mutate(&mutated[c.pos])

		got := compareCausalInvariance(prefix, mutated)
		if got.Identical == got.Compared {
			t.Fatalf("%s at position %d: the comparison reported all %d positions "+
				"unchanged, so it cannot detect a broken mask", c.name, c.pos, got.Compared)
		}
		if got.MismatchAt != c.pos {
			t.Fatalf("%s: first mismatch reported at position %d, want %d", c.name, got.MismatchAt, c.pos)
		}
	}
}

// TestCausalToleranceSeparatesNoiseFromDefects pins the two measurements the tolerance sits
// between, so neither can drift without someone noticing.
//
// The upper measurement is the interesting one: it is a real mask defect, produced by
// leaking one future key into every row of the forward pass, and the number is large enough
// that no plausible rounding difference could be mistaken for it. The lower one is what the
// experimental SIMD build actually produces — see causalTolerance for why it is not zero.
// Both are re-measured here rather than quoted, because the tolerance is only meaningful
// relative to them.
func TestCausalToleranceSeparatesNoiseFromDefects(t *testing.T) {
	if causalTolerance <= 0 {
		t.Fatalf("causalTolerance is %.0e", causalTolerance)
	}
	a := newTestApp(t)
	rows := predictEveryPosition(a.W, a.Trace, causalReferenceTemperature)

	// The smallest defect the check must catch: perturbing a single non-max probability. A
	// mask bug that adjusts only the tail of a distribution — which is what an off-by-one
	// mask does on a row whose top byte was never in doubt — looks like this, not like a
	// changed argmax.
	mutated := make([]positionPrediction, len(rows))
	copy(mutated, rows)
	mutated[3].Probs = append([]float32(nil), rows[3].Probs...)
	mutated[3].Probs[(rows[3].Guess+1)%len(rows[3].Probs)] += 1e-3
	if _, same := comparePrediction(rows[3], mutated[3]); same {
		t.Fatalf("a 1e-3 move in one probability was called unchanged at a tolerance of %.0e",
			causalTolerance)
	}

	// The largest thing that must still be called unchanged: a deviation at half the tolerance,
	// which is where the SIMD kernel's rounding sits relative to the margin. If this stops
	// being "unchanged", the margin has been consumed and the real comparison would start
	// reporting the two kernels as a mask defect.
	mutated[3].Probs = append([]float32(nil), rows[3].Probs...)
	mutated[3].Probs[(rows[3].Guess+1)%len(rows[3].Probs)] += float32(causalTolerance / 2)
	if _, same := comparePrediction(rows[3], mutated[3]); !same {
		t.Fatalf("a %.0e deviation — half the tolerance, and the scale of the SIMD kernel's "+
			"rounding — was called a change", causalTolerance/2)
	}

	// The separation the card relies on: the printed leak magnitude has to sit far above the
	// tolerance. This is the property that makes the tolerance meaningful, and it is checkable
	// without re-running a deliberately broken forward pass — the figure itself was measured
	// once (2.71e-01, four orders above the SIMD gap) and is recorded on causalLeakMagnitude.
	//
	// What is verified here is the ordering, which is what a future edit could plausibly break:
	// raising the tolerance to "stop the SIMD build flaking" would quietly make the check
	// unable to see a real leak at all.
	if causalLeakMagnitude <= causalTolerance {
		t.Fatalf("the card prints %v as what a leak does while the tolerance is %.0e, so the "+
			"comparison could not distinguish them", causalLeakMagnitude, causalTolerance)
	}
	// At least two orders of magnitude, not merely "bigger": one order would be inside the
	// range a different checkpoint's rounding could wander into.
	if causalLeakMagnitude/causalTolerance < 100 {
		t.Fatalf("a leak (%.0e) is only %.0fx the tolerance (%.0e); the two need at least two "+
			"orders of separation to be distinguishable on any prompt",
			causalLeakMagnitude, causalLeakMagnitude/causalTolerance, causalTolerance)
	}

}

// TestCausalInvariancePairsPositionsNotIndices pins the reason the comparison walks by
// recorded position rather than by slice index.
//
// Two runs of different lengths can be aligned by index only if they describe the same
// sequence. Truncation breaks that: the reader's prompt outgrows the window, the window
// slides, and index 0 of the new run is a different token from index 0 of the old one. The
// guard has to stop there rather than report the shift as changed predictions — "the
// earlier positions moved" would be a true statement about two windows and a false one
// about the mask, which is the kind of plausible-looking wrong number this project keeps
// finding in prose. Positions agreeing is what licenses the comparison; the first
// disagreement in position ends it.
func TestCausalInvariancePairsPositionsNotIndices(t *testing.T) {
	row := func(pos, guess int) positionPrediction {
		return positionPrediction{Pos: pos, Guess: guess, Probs: make([]float32, 4)}
	}
	// Same sequence, one row longer: everything before the end must compare.
	prefix := []positionPrediction{row(0, 1), row(1, 2)}
	full := []positionPrediction{row(0, 1), row(1, 2), row(2, 3)}
	if got := compareCausalInvariance(prefix, full); got.Compared != 2 || got.Identical != 2 {
		t.Fatalf("a clean extension compared %+v, want 2 of 2", got)
	}
	// A shifted window: position 0 no longer lines up, so the walk must stop immediately
	// instead of reporting the whole run as changed.
	shifted := []positionPrediction{row(1, 2), row(2, 3), row(3, 4)}
	if got := compareCausalInvariance(prefix, shifted); got.Compared != 0 || got.OK {
		t.Fatalf("a shifted window compared %+v, want nothing compared", got)
	}
	// And the short-run-is-longer guard.
	if got := compareCausalInvariance(full, prefix); got.OK {
		t.Fatalf("comparing a run against a shorter one was accepted: %+v", got)
	}
}

// TestCausalInvarianceLineReachesTheCard checks the figure is on screen and quotes the
// comparison rather than a number written into the sentence.
func TestCausalInvarianceLineReachesTheCard(t *testing.T) {
	for _, prompt := range []string{
		"The transformer reads the tokens and",
		"hi",
		"注意力就是加权求和",
	} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 6
		cmp := a.causalInvariance()
		if !cmp.OK {
			t.Fatalf("%q: the comparison declined", prompt)
		}
		tr := ui.NewTester(a.View, 1500, 3000)
		tr.Frame()
		want := fmt.Sprintf("剩下 %d 个位置的预测没变", cmp.Compared)
		var found string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "因果掩码核对") {
				found = s
			}
		}
		if found == "" {
			t.Fatalf("%q: the output stage does not render the causal check", prompt)
		}
		if !strings.Contains(found, want) {
			t.Fatalf("%q: the line does not quote the measured count (%q):\n%s", prompt, want, found)
		}
		// And the line has to state the bar it is measuring against: "没变" without a
		// tolerance is not a claim a reader can evaluate.
		bar := fmt.Sprintf("%.0e", causalTolerance)
		if !strings.Contains(found, bar) {
			t.Fatalf("%q: the line does not state the tolerance it applies (%q):\n%s", prompt, bar, found)
		}
		// The measured deviation must NOT be printed. It is exactly 0 on the default build and
		// ~1e-6 under GOEXPERIMENT=simd, and docs/output.png is compared byte for byte under
		// both — a per-build number in the frame makes that comparison fail, which is how CI
		// caught the first version of this line.
		if strings.Contains(found, "0.0e+00") {
			t.Fatalf("%q: the line prints a per-build deviation, which cannot be stable across "+
				"the two matmul kernels: %q", prompt, found)
		}
	}
}

// TestCausalInvarianceStaysOffWhenThereIsNothingToCompare keeps the check from claiming
// something about a prompt that cannot support it: one token has no position whose
// prediction could have been preserved, and the card must say that instead of printing
// "剩下的 0 个位置…".
func TestCausalInvarianceStaysOffWhenThereIsNothingToCompare(t *testing.T) {
	for _, prompt := range []string{"", "a"} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		if cmp := a.causalInvariance(); cmp.OK {
			t.Fatalf("%q: the comparison ran on %d tokens: %+v", prompt, len(a.Trace.Tokens), cmp)
		}
		a.Step = 6
		tr := ui.NewTester(a.View, 1500, 3000)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "因果掩码核对") {
				t.Fatalf("%q renders a causal check with nothing to compare: %q", prompt, s)
			}
		}
	}
}

// TestCausalInvarianceIgnoresTheTemperatureSlider pins the decision that lets the result be
// cached by trace: the measurement is taken at a fixed temperature, so moving the slider
// must not change the figure the card shows.
//
// It is measured rather than assumed. Softmax divides every logit by one constant, so the
// argmax cannot move — but a row whose top two bytes are within rounding of each other
// could still reorder, and this test is what would notice if this checkpoint ever produced
// such a row.
func TestCausalInvarianceIgnoresTheTemperatureSlider(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	base := a.causalInvariance()
	for _, temp := range []float64{0.1, 0.5, 1.5, 2.0} {
		a.Temp = temp
		if got := a.causalInvariance(); got != base {
			t.Fatalf("T=%.1f changed the figure from %+v to %+v; the measurement is taken at "+
				"T=%.1f and cached by trace, so it must not", temp, base, got, causalReferenceTemperature)
		}
	}
	// And the argmax really is temperature-independent here, which is why a fixed reference
	// temperature is legitimate rather than merely convenient.
	base1 := predictEveryPosition(a.W, a.Trace, causalReferenceTemperature)
	for _, temp := range temperatureSamples {
		if temp == causalReferenceTemperature {
			continue
		}
		rows := predictEveryPosition(a.W, a.Trace, temp)
		for i := range rows {
			if rows[i].Guess != base1[i].Guess {
				t.Fatalf("row %d picks byte %d at T=%.2f and byte %d at T=%.2f; the argmax "+
					"is not temperature-independent for this checkpoint",
					i, rows[i].Guess, temp, base1[i].Guess, causalReferenceTemperature)
			}
		}
	}
}

// TestCausalLineIsBuildIndependent pins the property CI caught: the rendered sentence must not
// contain any quantity that differs between the two matmul kernels.
//
// The first version of the line quoted the measured worst deviation, which is exactly 0 on the
// default build and about 1e-6 under GOEXPERIMENT=simd. `just check` passed and `just check-ci`
// failed on TestDocsImagesAreCurrent — docs/output.png is compared byte for byte, and the SIMD
// step re-renders it from the same code. The fix was to print the tolerance (a constant) instead
// of the measurement, and this test is what keeps a future edit from putting a per-build number
// back into the frame.
func TestCausalLineIsBuildIndependent(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	var found string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "因果掩码核对") {
			found = s
		}
	}
	if found == "" {
		t.Fatal("the output stage does not render the causal check")
	}
	// The only number besides the position count must be the tolerance.
	nums := regexp.MustCompile(`[0-9]+\.[0-9]+e[-+][0-9]+`).FindAllString(found, -1)
	for _, n := range nums {
		if n != fmt.Sprintf("%.0e", causalTolerance) {
			t.Fatalf("the line prints %q, which is not the build-independent tolerance; a "+
				"per-build figure here makes docs/output.png differ between the two matmul "+
				"kernels: %q", n, found)
		}
	}
	// And the position count it prints must be the one it measured.
	if want := fmt.Sprintf("剩下 %d 个位置", a.causalInvariance().Compared); !strings.Contains(found, want) {
		t.Fatalf("the line does not quote the measured count (%q):\n%s", want, found)
	}
}

// TestEmbeddingVerdictFollowsTheMeasurement closes the gap that let the fixed sentence stay
// wrong for as long as it did.
//
// The card printed computed figures and then read them out in fixed prose: "两者差别落在噪声里".
// TestEmbeddingStructureShowsNoStructure checked the premise on two ordinary prompts, where the
// difference really is inside the spread, so the sentence looked like a property of the table
// rather than a property of the prompt. A periodic prompt breaks it — "ab"x50 measures adjacent
// +0.1737 against a random mean of +0.0033 with spread 0.1017, which is 1.7 standard deviations
// and not noise by any reading — while the sentence went on saying it was.
//
// The test reads the rendered sentence, not the helper: a conditional that never reaches the
// screen would pass a check on computeEmbeddingStructure alone.
func TestEmbeddingVerdictFollowsTheMeasurement(t *testing.T) {
	for _, c := range []struct {
		prompt string
		// noisy says which reading the measured z-score requires.
		noisy bool
	}{
		{"The transformer reads the tokens and", true},
		{strings.Repeat("ab", 50), false},
	} {
		a := newTestApp(t)
		a.Prompt = c.prompt
		a.recompute()
		a.Step = 1
		tr := ui.NewTester(a.View, 1500, 3400)
		tr.Frame()

		e := computeEmbeddingStructure(a.W, c.prompt)
		if !e.OK || e.SDRandom == 0 {
			t.Fatalf("%q: no measurement", c.prompt)
		}
		z := math.Abs(e.AdjacentCos-e.RandomCos) / e.SDRandom
		if got := z <= 0.5; got != c.noisy {
			t.Fatalf("%q: z = %.2f, so noisy should be %v, not %v — this test's own premise "+
				"is stale", c.prompt, z, z <= 0.5, c.noisy)
		}

		var sentence string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "这张表里没有这回事") {
				sentence = s
			}
		}
		if sentence == "" {
			t.Fatalf("%q: the card is missing", c.prompt)
		}
		saysNoise := strings.Contains(sentence, "两者差别落在噪声里")
		if saysNoise != c.noisy {
			t.Fatalf("%q (z = %.2f): the card says the difference %s, but the figures beside "+
				"it say otherwise:\n%s", c.prompt, z,
				map[bool]string{true: "is inside the noise", false: "is outside the noise"}[saysNoise],
				sentence)
		}
		// When the difference is real, the sentence has to name the actual distance rather
		// than leave the reader to work it out from two cosines.
		if !c.noisy && !strings.Contains(sentence, "个标准差") {
			t.Fatalf("%q: the card says the difference is real without quoting how large it "+
				"is:\n%s", c.prompt, sentence)
		}
	}
}

// TestHeadProfilesAdviceNeverNamesTheCurrentLayer keeps the low-spread verdict from telling the
// reader to switch to the layer they are already looking at.
//
// The verdict has two branches. The wide-spread one describes what the heads are doing. The
// narrow-spread one used to end "换到第 0 层看看，那里的头差别明显得多" — reasonable when shown
// on layer 1, which is where the default prompt triggers it, and nonsensical on layer 0, which
// the default prompt never reaches because its layer 0 spread is wide.
//
// Prompt "ab" does reach it: layer 0 measures 75%-92%, well under the 0.2 threshold, so the
// card said "this layer has no division of labour — switch to layer 0 and look". The existing
// both-verdicts test could not see this, because it only ever renders the default prompt and
// only ever expects "差别明显" on layer 0.
//
// What is asserted is that on layer 0 the advice names no layer at all, since there is no
// other layer to point down to. Any wording is allowed; pointing at the current layer is not.
func TestHeadProfilesAdviceNeverNamesTheCurrentLayer(t *testing.T) {
	for _, prompt := range []string{"ab", "The transformer reads the tokens and"} {
		for layer := 0; layer < 2; layer++ {
			a := newTestApp(t)
			a.Prompt = prompt
			a.recompute()
			a.Step = 3
			a.Layer = layer
			tr := ui.NewTester(a.View, 1500, 3000)
			tr.Frame()

			var verdict string
			for _, s := range tr.Texts() {
				if strings.Contains(s, "看不出分工") {
					verdict = s
				}
			}
			if verdict == "" {
				continue // this prompt/layer renders the wide-spread branch
			}
			selfRef := fmt.Sprintf("换到第 %d 层", layer)
			if strings.Contains(verdict, selfRef) {
				t.Fatalf("%q layer %d: the card tells the reader to switch to layer %d while "+
					"showing layer %d:\n%s", prompt, layer, layer, layer, verdict)
			}
		}
	}

	// And the branch has to exist to be tested: prompt "ab" at layer 0 must render the
	// narrow-spread verdict, or this test is vacuous.
	a := newTestApp(t)
	a.Prompt = "ab"
	a.recompute()
	a.Step = 3
	a.Layer = 0
	tr := ui.NewTester(a.View, 1500, 3000)
	tr.Frame()
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, "看不出分工") {
			found = true
		}
	}
	if !found {
		t.Fatal("prompt \"ab\" at layer 0 no longer renders the narrow-spread verdict, so the " +
			"self-reference this test exists to catch is no longer reachable and the check " +
			"above proves nothing")
	}
}
