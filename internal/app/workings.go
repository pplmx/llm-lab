package app

// The arithmetic behind every "how this number was computed" block in the lab.
//
// These are pure functions over the model's trace: they take slices and indices
// and return the terms, remainder, and total that a stage prints. They live
// together, apart from app.go, because they are the one part of the interface
// that has nothing to do with the interface — no ui.Context, no layout, no
// state — and because there are now six of them with the same shape:
//
//	compute<Thing> -> <thing>Parts -> <thing>Workings (in a stage file)
//
// Reading them side by side makes the shared decisions visible: which terms to
// show (largest by magnitude, never by signed value), what the remainder means
// (exact, not an estimate), and what denominator a "share" uses (Σ|term|, not
// the signed total). Those choices were each arrived at by measurement, and
// each has a test.

import (
	"math"
	"unicode/utf8"

	"llmlab/internal/model"
)

// softmaxParts is the arithmetic behind one attention weight, as the inspector
// prints it: the row maximum subtracted for stability, the denominator summed
// over the visible keys, the cell's numerator, and their quotient.
type softmaxParts struct {
	rowMax, denom, num, weight float64
	ok                         bool
}

// computeSoftmaxParts derives softmaxParts for cell (q, k) from the scores the
// model actually holds. It is the only place that arithmetic lives, so the
// inspector cannot print a formula that differs from what it computed, and a
// test can check the printed numbers against the trace without re-deriving them.
func computeSoftmaxParts(h model.HeadTrace, q, k int) softmaxParts {
	n := q + 1
	if q < 0 || k < 0 || k > q || q >= len(h.Scores) || n > len(h.Scores[q]) {
		return softmaxParts{}
	}
	p := softmaxParts{rowMax: math.Inf(-1)}
	for j := 0; j < n; j++ {
		if s := float64(h.Scores[q][j]); s > p.rowMax {
			p.rowMax = s
		}
	}
	if math.IsInf(p.rowMax, -1) {
		return softmaxParts{}
	}
	for j := 0; j < n; j++ {
		p.denom += math.Exp(float64(h.Scores[q][j]) - p.rowMax)
	}
	p.num = math.Exp(float64(h.Scores[q][k]) - p.rowMax)
	p.weight = p.num / p.denom
	p.ok = true
	return p
}

// dotTerm is one contribution q[d]·k[d] to a score.
type dotTerm struct {
	D int
	V float64
}

// scoreParts is the arithmetic behind one attention score: the largest few
// products, the exact remainder, the dot product they sum to, and the scaled
// score.
type scoreParts struct {
	Top  []dotTerm
	Rest float64
	Dot  float64
	// Score is the trace's own value, so the printed line ends on the number
	// the heatmap shows rather than on a recomputed near-miss.
	Score float64
	OK    bool
}

// computeScore derives score[q][k] from the head's q and k slices. The products
// are accumulated in float32 order, exactly as model.dot does, so the result is
// bit-identical to the trace rather than merely close: a reader checking the
// arithmetic gets the same number the matrix shows.
func computeScore(q, k []float32, dHead int, traceScore float32) scoreParts {
	if dHead <= 0 || len(q) < dHead || len(k) < dHead {
		return scoreParts{}
	}
	terms := make([]dotTerm, dHead)
	for d := 0; d < dHead; d++ {
		terms[d] = dotTerm{D: d, V: float64(q[d]) * float64(k[d])}
	}
	const top = 3
	if dHead < top {
		return scoreParts{}
	}
	for x := 0; x < top; x++ {
		best := x
		for y := x + 1; y < dHead; y++ {
			if math.Abs(terms[y].V) > math.Abs(terms[best].V) {
				best = y
			}
		}
		terms[x], terms[best] = terms[best], terms[x]
	}
	// The dot product as the model computes it: float32 accumulation.
	var dot32 float32
	for d := 0; d < dHead; d++ {
		dot32 += q[d] * k[d]
	}
	p := scoreParts{Top: terms[:top], Dot: float64(dot32), Score: float64(traceScore), OK: true}
	p.Rest = p.Dot
	for _, t := range p.Top {
		p.Rest -= t.V
	}
	return p
}

// ffnTerm is one contribution to an FFN output dimension: hidden[k] times the
// W2 row k, column j.
type ffnTerm struct {
	K int
	V float64
}

// ffnParts is the arithmetic behind one FFN output dimension: the largest few
// contributions, the exact remainder that closes the sum, the result, and the
// absolute mass of every term.
type ffnParts struct {
	Top  []ffnTerm
	Rest float64
	Out  float64
	// AbsTotal is Σ|hidden[k]·W2[k][j]| over all k. It is the honest
	// denominator for "how much of this sum do the shown terms carry": the
	// signed total cancels heavily, so dividing by it would overstate them.
	AbsTotal float64
	OK       bool
}

// computeFFN derives out[j] = Σ_k hidden[k]·W2[k][j] and reports its largest
// terms by absolute value plus the exact remainder.
//
// The remainder is not optional. This model's FFN sum is not dominated by a few
// terms — measured, the top three carry only 14–16% of the absolute mass — so
// showing three terms alone would read as if they made the output. They do not;
// they are only the largest single contributions, and the remainder closes the
// sum exactly.
func computeFFN(hidden []float32, w2 []float32, dff, d, j int) ffnParts {
	if j < 0 || j >= d || len(hidden) < dff || len(w2) < dff*d {
		return ffnParts{}
	}
	terms := make([]ffnTerm, dff)
	for k := 0; k < dff; k++ {
		terms[k] = ffnTerm{K: k, V: float64(hidden[k]) * float64(w2[k*d+j])}
	}
	const top = 3
	if dff < top {
		return ffnParts{}
	}
	for x := 0; x < top; x++ {
		best := x
		for y := x + 1; y < dff; y++ {
			if math.Abs(terms[y].V) > math.Abs(terms[best].V) {
				best = y
			}
		}
		terms[x], terms[best] = terms[best], terms[x]
	}
	p := ffnParts{Top: terms[:top], OK: true}
	var total, absTotal float64
	for _, t := range terms {
		total += t.V
		absTotal += math.Abs(t.V)
	}
	p.Out = total
	p.AbsTotal = absTotal
	p.Rest = total
	for _, t := range p.Top {
		p.Rest -= t.V
	}
	return p
}

// runeBytes is one character's path through the tokenizer: the rune, its
// codepoint, and the bytes it became.
type runeBytes struct {
	Rune      rune
	Codepoint rune
	Bytes     []byte
	// Index is the byte offset of the character in the prompt.
	Index int
	OK    bool
}

// runeAt returns the character whose bytes cover byte offset i in the trace, or
// the zero value if i is past the end. A byte-level tokenizer has no character
// boundaries of its own — Tokens[i].Start counts runes, not bytes — so the
// answer comes from decoding the prompt text and walking it.
func runeAt(tr *model.Trace, i int) runeBytes {
	if tr == nil || i < 0 || i >= len(tr.Text) {
		return runeBytes{}
	}
	off := 0
	for _, r := range tr.Text {
		n := utf8.RuneLen(r)
		if n < 0 {
			n = 1
		}
		if i >= off && i < off+n {
			return runeBytes{
				Rune: r, Codepoint: r,
				Bytes: []byte(tr.Text[off : off+n]),
				Index: off, OK: true,
			}
		}
		off += n
	}
	return runeBytes{}
}

// utf8Prefix describes the fixed high bits of a UTF-8 lead and continuation
// byte, which is what makes the encoding checkable by hand.
func utf8Prefix(b byte) string {
	switch {
	case b < 0x80:
		return "0"
	case b&0xE0 == 0xC0:
		return "110"
	case b&0xF0 == 0xE0:
		return "1110"
	case b&0xF8 == 0xF0:
		return "11110"
	default:
		return "10"
	}
}

// utf8Payload returns the data bits a byte carries, after its length prefix.
func utf8Payload(b byte) byte {
	switch {
	case b < 0x80:
		return b
	case b&0xE0 == 0xC0:
		return b & 0x1F
	case b&0xF0 == 0xE0:
		return b & 0x0F
	case b&0xF8 == 0xF0:
		return b & 0x07
	default:
		return b & 0x3F
	}
}

// weightedTerm is one α·V product contributing to a head output dimension.
type weightedTerm struct {
	Key int
	Val float64
}

// weightedParts is the arithmetic behind one dimension of a head's output: the
// largest contributions, the exact remainder, and the total.
type weightedParts struct {
	Top  []weightedTerm
	Rest float64
	Out  float64
	// AbsTotal is Σ|α·V| over the visible keys, the honest denominator for
	// "how much of this sum do the shown terms carry".
	AbsTotal float64
	OK       bool
}

// computeWeighted derives Out[q][dim] = Σ_k α[q][k]·V[k][dim] and reports its
// largest terms by absolute value plus the exact remainder, the way the FFN
// stage does for its own sum.
func computeWeighted(attn []float32, v [][]float32, q, dim, n int) weightedParts {
	if q < 0 || dim < 0 || q >= n || len(attn) < n {
		return weightedParts{}
	}
	terms := make([]weightedTerm, q+1)
	for k := 0; k <= q; k++ {
		if k >= len(v) || dim >= len(v[k]) {
			return weightedParts{}
		}
		terms[k] = weightedTerm{Key: k, Val: float64(attn[k]) * float64(v[k][dim])}
	}
	// Row 0 has a single key, so "the largest three" has to shrink to fit
	// rather than refusing to describe a real row.
	top := 3
	if len(terms) < top {
		top = len(terms)
	}
	for x := 0; x < top; x++ {
		best := x
		for y := x + 1; y < len(terms); y++ {
			if math.Abs(terms[y].Val) > math.Abs(terms[best].Val) {
				best = y
			}
		}
		terms[x], terms[best] = terms[best], terms[x]
	}
	p := weightedParts{Top: terms[:top], OK: true}
	var total, absTotal float64
	for _, tm := range terms {
		total += tm.Val
		absTotal += math.Abs(tm.Val)
	}
	p.Out = total
	p.AbsTotal = absTotal
	p.Rest = total
	for _, tm := range p.Top {
		p.Rest -= tm.Val
	}
	return p
}

// residualStep is one addition in the residual chain, at one dimension.
type residualStep struct {
	// Label names the term: "进入" for the block input, or the sublayer that
	// contributed.
	Label string
	Term  float64
	Sum   float64
}

// residualChain returns the additions that build one dimension of the residual
// stream through one block: the block input, plus the attention projection,
// plus the FFN output. Each Sum is the running total after that addition, and
// the model builds them this way with addInto, so the printed sums are the
// trace's values rather than a recomputation.
func residualChain(l model.LayerTrace, i, dim int) ([]residualStep, bool) {
	if i < 0 || i >= len(l.In) || dim < 0 || dim >= len(l.In[i]) ||
		dim >= len(l.AttnProj[i]) || dim >= len(l.AfterAttn[i]) ||
		dim >= len(l.FFNOut[i]) || dim >= len(l.AfterFFN[i]) {
		return nil, false
	}
	return []residualStep{
		{Label: "进入本层的 x", Term: float64(l.In[i][dim]), Sum: float64(l.In[i][dim])},
		{Label: "+ 注意力子层 Δ", Term: float64(l.AttnProj[i][dim]), Sum: float64(l.AfterAttn[i][dim])},
		{Label: "+ FFN 子层 Δ", Term: float64(l.FFNOut[i][dim]), Sum: float64(l.AfterFFN[i][dim])},
	}, true
}

// lnParts is the arithmetic behind one LayerNorm output entry: the row mean,
// the inverse standard deviation that scales it, and the per-dimension γ and β
// the model learned.
type lnParts struct {
	mean, invStd, gamma, beta, x, out float64
	ok                                bool
}

// lnEps mirrors the constant inside model.layerNorm. It is duplicated rather
// than exported because this is a presentation detail of the lab, not a knob:
// if the model ever changes it, TestLNWorkingsAddUp fails and both move.
const lnEps = 1e-5

// computeLN derives one output entry of layerNorm(x) from the input row and the
// learned γ/β, the same way model.layerNorm does it. The inspector prints these
// numbers, so a reader can reproduce the value the histogram is built from.
func computeLN(x, g, b []float32, dim int) lnParts {
	if dim < 0 || dim >= len(x) || dim >= len(g) || dim >= len(b) {
		return lnParts{}
	}
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var varsum float64
	for _, v := range x {
		d := float64(v) - mean
		varsum += d * d
	}
	variance := varsum / float64(len(x))
	invStd := 1 / math.Sqrt(variance+lnEps)
	return lnParts{
		mean: mean, invStd: invStd,
		gamma: float64(g[dim]), beta: float64(b[dim]),
		x:   float64(x[dim]),
		out: (float64(x[dim])-mean)*invStd*float64(g[dim]) + float64(b[dim]),
		ok:  true,
	}
}
