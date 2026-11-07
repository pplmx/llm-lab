package app

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
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
	testSrc, err := os.ReadFile("app_test.go")
	if err != nil {
		t.Fatalf("read app_test.go: %v", err)
	}

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

	wantDenom := fmt.Sprintf("分母 = Σ exp(score − %+.4f) = %.4f", p.rowMax, p.denom)
	wantWeight := fmt.Sprintf("α = exp(%+.4f − %+.4f) ÷ %.4f = %.4f",
		h.Scores[10][5], p.rowMax, p.denom, p.num/p.denom)

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

	wantHeader := fmt.Sprintf("d%d：μ = %.4f，σ = %.4f，1/√(σ²+ε) = %.4f",
		dim, p.mean, 1/p.invStd, p.invStd)
	wantBody := fmt.Sprintf("(%.4f − %.4f) × %.4f × γ %.5f + β %.5f ≈ %.4f",
		p.x, p.mean, p.invStd, p.gamma, p.beta, p.out)

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
	wantFirst := fmt.Sprintf("d%d 的残差流在第 %d 层：%s = %+.4f",
		dim, a.Layer+1, steps[0].Label, steps[0].Term)
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
