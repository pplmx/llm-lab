package app

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
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
