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
	"sort"
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
	// Byte is set when the character is a byte that does not decode to any
	// rune, in which case Rune is left zero and the card shows the byte.
	Byte byte
	// Index is the rune index of the character in the prompt.
	Index int
	OK    bool
}

// runeAt returns the character that the token at byte offset i belongs to, or
// the zero value if i is past the end.
//
// The index is Tokens[i].Start, which counts runes rather than bytes. Two
// sources disagree about invalid UTF-8 and this function has to prefer the
// tokens: Trace.Text is built by DecodeBytes, which replaces an invalid byte
// with U+FFFD, while the tokens keep the real byte. Reading the character back
// out of Trace.Text therefore describes the replacement rather than what was
// typed, and the replacement is one rune where the original may be several
// bytes — which shifts every later index too.
//
// So the token's own rune wins when it has one. Only a byte that decodes to no
// character at all (Rune == 0, set by alignTokens) falls back to the text.
func runeAt(tr *model.Trace, runeIdx int) runeBytes {
	if tr == nil || runeIdx < 0 {
		return runeBytes{}
	}
	// runeIdx is a rune index (Tokens[i].Start counts runes, not bytes). Find the
	// first token of that character and take its rune plus the run of bytes that
	// share it.
	first := -1
	for k := range tr.Tokens {
		if tr.Tokens[k].Start == runeIdx {
			first = k
			break
		}
	}
	if first >= 0 && tr.Tokens[first].Rune == 0 {
		// An invalid byte: alignTokens leaves Rune zero because it decodes to no
		// character. The honest answer is the byte itself, not the U+FFFD that
		// Trace.Text shows in its place.
		return runeBytes{Byte: tr.Tokens[first].Byte, Bytes: []byte{tr.Tokens[first].Byte},
			Index: runeIdx, OK: true}
	}
	if first >= 0 {
		r := tr.Tokens[first].Rune
		var bs []byte
		for k := first; k < len(tr.Tokens) && tr.Tokens[k].Start == runeIdx; k++ {
			bs = append(bs, tr.Tokens[k].Byte)
		}
		// The bytes must actually encode this rune, or the tokens are describing
		// something the text cannot express; fall through to the text then.
		if len(bs) > 0 && utf8.RuneLen(r) == len(bs) {
			return runeBytes{Rune: r, Codepoint: r, Bytes: bs, Index: runeIdx, OK: true}
		}
	}
	// Fall back to walking the decoded text by rune, which is right whenever the
	// text and the tokens agree.
	byteOff, ri := 0, 0
	for _, r := range tr.Text {
		n := utf8.RuneLen(r)
		if n < 0 {
			n = 1
		}
		if ri == runeIdx {
			return runeBytes{Rune: r, Codepoint: r, Bytes: []byte(tr.Text[byteOff : byteOff+n]), Index: runeIdx, OK: true}
		}
		byteOff += n
		ri++
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

// scoreColorScale is the |score| that saturates the heatmap's colour ramp: the
// largest absolute value anywhere in the matrix. The cell colour is |v|/scale,
// so without this number on screen a reader can see that one cell is warmer than
// another but not what either is worth — and the scale moves with the prompt
// (5.42 for "hi", 8.25 for a longer one), so it cannot be guessed from the
// weights either.
func scoreColorScale(h model.HeadTrace) float64 {
	var scale float64
	for i := range h.Scores {
		for j := 0; j <= i && j < len(h.Scores[i]); j++ {
			if v := math.Abs(float64(h.Scores[i][j])); v > scale {
				scale = v
			}
		}
	}
	return scale
}

// layerDivergence measures how differently two layers attend from the same
// position: the total variation distance between their attention distributions,
// from 0 (identical) to 1 (no overlap).
//
// This is the number the layer control has never shown. Switching between layer 0
// and layer 1 redraws the heatmap, but a reader cannot tell from two successive
// pictures whether the layers are doing the same thing at slightly different
// strengths or genuinely looking elsewhere. Measured on this model the distance
// averages about 0.75 across heads, so the answer is "genuinely elsewhere".
type layerDivergence struct {
	TV float64
	// Ok is false when there is nothing to compare: fewer than two layers, or a
	// row with a single visible key (where both layers are forced to put all
	// their weight on that key, so any difference would be an artefact of the
	// mask rather than of what the layers learned).
	Ok bool
}

// computeLayerDivergence derives the total variation distance between layer
// `layer` and layer `layer-1` for one row and head.
func computeLayerDivergence(layers []model.LayerTrace, layer, head, row int) layerDivergence {
	if layer < 0 {
		return layerDivergence{}
	}
	if row < 1 {
		// With one visible key both distributions are [1.0]; the distance is
		// zero for a reason that says nothing about the layers.
		return layerDivergence{}
	}
	if len(layers) < 2 || layer < 1 || layer >= len(layers) {
		return layerDivergence{}
	}
	if head < 0 || head >= len(layers[layer].Heads) || head >= len(layers[layer-1].Heads) {
		return layerDivergence{}
	}
	cur := layers[layer].Heads[head].Attn
	prev := layers[layer-1].Heads[head].Attn
	if row >= len(cur) || row >= len(prev) {
		return layerDivergence{}
	}
	var tv float64
	for k := 0; k <= row && k < len(cur[row]) && k < len(prev[row]); k++ {
		d := float64(cur[row][k] - prev[row][k])
		if d < 0 {
			d = -d
		}
		tv += d
	}
	return layerDivergence{TV: tv / 2, Ok: true}
}

// residualStep is one addition in the residual chain, at one dimension.
type residualStep struct {
	// Label names the term: "进入" for the block input, or the sublayer that
	// contributed.
	Label string
	Term  float64
	Sum   float64
}

// headContribution is one head's share of a block's attention output.
type headContribution struct {
	Head int
	// Norm is ||HeadProj[h]|| for this token: how much this head writes into
	// the residual stream.
	Norm float64
	// Share is Norm as a percentage of the sum of all heads' norms.
	Share float64
}

// headContributions measures what each head writes into the residual stream at
// one token, plus how much the heads cancel when summed.
//
// The cancellation number is the finding worth teaching. The heads' norms sum
// to about 17 here while the norm of their sum is about 9.7 - a ratio of 0.62 -
// and that holds for every row of this model's trace (72/72), not as an
// occasional coincidence. So "AttnProj = Sum HeadProj" adds vectors that partly
// point in opposite directions, and treating the four norms as contributions to
// be added would overstate the result by about a factor of 1.6.
type headContributions struct {
	Heads []headContribution
	// SumNorm is the sum of the heads' individual norms.
	SumNorm float64
	// TotalNorm is the norm of their sum, which is ||AttnProj||.
	TotalNorm float64
	// Cancellation is TotalNorm / SumNorm: below 1 the heads partly cancel,
	// above 1 they reinforce.
	Cancellation float64
	OK           bool
}

// computeHeadContributions derives the per-head norms for one token of one
// layer, and the cancellation ratio between them.
func computeHeadContributions(l model.LayerTrace, i int) headContributions {
	if i < 0 || i >= len(l.AttnProj) || len(l.HeadProj) == 0 || i >= len(l.HeadProj[0]) {
		return headContributions{}
	}
	n := len(l.HeadProj)
	c := headContributions{Heads: make([]headContribution, 0, n), OK: true}
	for h := 0; h < n; h++ {
		hn := float64(norm(l.HeadProj[h][i]))
		c.Heads = append(c.Heads, headContribution{Head: h, Norm: hn})
		c.SumNorm += hn
	}
	c.TotalNorm = float64(norm(l.AttnProj[i]))
	if c.SumNorm > 0 {
		c.Cancellation = c.TotalNorm / c.SumNorm
		for k := range c.Heads {
			c.Heads[k].Share = 100 * c.Heads[k].Norm / c.SumNorm
		}
	}
	return c
}

// headCancellation measures one row's cancellation without building the head
// slice. measureCancellation walks every row of the trace once per frame, and
// allocating the four-entry slice 72 times dominated its cost (216 allocations
// per frame) for numbers the summary never reads.
func headCancellation(l model.LayerTrace, i int) (cancellation float64, ok bool) {
	if i < 0 || i >= len(l.AttnProj) || len(l.HeadProj) == 0 || i >= len(l.HeadProj[0]) {
		return 0, false
	}
	var sum float64
	for h := range l.HeadProj {
		sum += float64(norm(l.HeadProj[h][i]))
	}
	if sum <= 0 {
		return 0, false
	}
	return float64(norm(l.AttnProj[i])) / sum, true
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

// activationConcentration reports how lopsided a GELU row is: the share of the
// total absolute activation held by the largest tenth of the dimensions, and how
// many dimensions that tenth is.
//
// This is what "稀疏" should mean for GELU. A count of values above a threshold
// does not measure it: on this model none of the 384 activations is anywhere near
// zero, so "38% are above 0.01" describes the cut, not the distribution. What is
// actually true is that half the mass sits in a tenth of the dimensions.
func activationConcentration(x []float32) (share float64, topN int) {
	if len(x) == 0 {
		return 0, 0
	}
	abs := make([]float64, len(x))
	var total float64
	for i, v := range x {
		a := math.Abs(float64(v))
		abs[i] = a
		total += a
	}
	if total == 0 {
		return 0, 0
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(abs)))
	topN = len(abs) / 10
	if topN < 1 {
		topN = 1
	}
	var top float64
	for i := 0; i < topN; i++ {
		top += abs[i]
	}
	return top / total, topN
}
