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
