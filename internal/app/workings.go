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
	"fmt"
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
	// The printed chain has to close in the reader's hands, so the score shown
	// beside it is recomputed the way the model computes it — float32 dot times a
	// float32 scale — not by dividing the float64 dot by sqrt(d). Those two differ
	// in the last digit (4.569528103 × 0.204124145 = 0.932751018 by hand, while
	// the trace holds 0.932751060), which is exactly the gap a reader would hit
	// after carefully redoing the arithmetic.
	scale32 := float32(1 / math.Sqrt(float64(len(q[:dHead]))))
	printed := float64(dot32 * scale32)
	if math.IsInf(float64(traceScore), -1) {
		printed = math.Inf(-1)
	}
	p := scoreParts{Top: terms[:top], Dot: float64(dot32), Score: printed, OK: true}
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
	// Recompute in float32, exactly as model.layerNorm does. The printed line asks
	// the reader to substitute μ, σ and γ into "(x − μ) × 1/√(σ²+ε) × γ + β" and
	// check the result; doing the statistics and the multiply in float64 gives a
	// value that differs from the shown output in the eighth digit (5.6e-8 here),
	// so the hand calculation would appear wrong. Same chain as the model means the
	// same answer the reader gets.
	var sum float32
	for _, v := range x {
		sum += v
	}
	mean32 := sum / float32(len(x))
	var varsum float32
	for _, v := range x {
		d := v - mean32
		varsum += d * d
	}
	variance := varsum / float32(len(x))
	invStd32 := float32(1 / math.Sqrt(float64(variance+lnEps)))
	out32 := (x[dim]-mean32)*invStd32*g[dim] + b[dim]
	return lnParts{
		mean: float64(mean32), invStd: float64(invStd32),
		gamma: float64(g[dim]), beta: float64(b[dim]),
		x:   float64(x[dim]),
		out: float64(out32),
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

// lnParamsAreLearned reports whether this layer's LayerNorm scale and shift are
// anything other than the identity (γ = 1, β = 0).
//
// The stages say "γ and β are learned, not 1 and 0", which is true of the bundled
// trained weights and false of the random fallback — model.Random sets γ to ones
// and β to zeros. Reading the weights is exact; recovering the same answer from
// the trace is not, because a trained γ can sit close enough to 1 that the
// normalised row is indistinguishable from the plain one at float32 precision.
func lnParamsAreLearned(w *model.Weights) bool {
	if w == nil {
		return false
	}
	for _, g := range w.LNFG {
		if g != 1 {
			return true
		}
	}
	for _, b := range w.LNFB {
		if b != 0 {
			return true
		}
	}
	return false
}

// positionPrediction is one position's next-byte guess, checked against the byte
// the prompt actually has there.
//
// The model is trained to predict a next token at every position, not only at the
// end of the prompt, and the trace already holds every such row: Trace.FinalNorm
// is T × d_model and Forward only projects the last one. Teaching "this is a
// next-token predictor" while showing a single row asks the reader to take the
// other T-1 on faith — and, worse, hides the fact that the model is *wrong* at
// most positions, which is the interesting part.
type positionPrediction struct {
	// Pos is the row that predicted; its prediction is the byte at Pos+1.
	Pos int
	// Guess and Actual are the predicted and the true next byte id.
	Guess, Actual int
	// Probs is the temperature-scaled distribution that Guess came from, kept so
	// the caller can quote the probability without re-deriving it.
	Probs []float32
	// Loss is -ln p(actual) in nats: how surprised the model was by the byte the
	// prompt actually contains.
	Loss float64
	// Hit reports whether Guess is the byte that came next.
	Hit bool
}

// nextByteSkill is what the model got right over the prompt it was given, with
// the reference points that make the figures mean something.
type nextByteSkill struct {
	Rows []positionPrediction
	// Scored is len(Rows): every position with a successor, so T-1.
	Scored int
	Hits   int
	// MeanLoss is the mean of Loss in nats. UniformReference is ln(vocab), the
	// loss of a model that spreads its probability evenly over every byte, so the
	// reader can tell 1.12 from "no better than guessing".
	MeanLoss         float64
	UniformReference float64
	Temperature      float64
	// Underflowed counts positions whose true byte's probability rounded to exactly zero at
	// this temperature, so their loss was not finite. When it is non-zero MeanLoss is a lower
	// bound over the positions that did produce a number, not a mean over all of them —
	// which the card has to say rather than print "+Inf".
	Underflowed int
}

// computeNextByteSkill runs the model's own unembedding over every row of the
// final LayerNorm and compares each prediction with the byte that followed.
//
// This is teacher forcing made visible: at position t the model sees the prompt
// up to t and nothing after it (that is what the causal mask guarantees), so its
// guess for t+1 is an honest prediction and not a lookup. The last row has no
// successor in the prompt, which is why Scored is T-1 rather than T.
func computeNextByteSkill(w *model.Weights, tr *model.Trace, temperature float64) nextByteSkill {
	s := nextByteSkill{Temperature: temperature}
	if w == nil || tr == nil || len(tr.Tokens) == 0 {
		return s
	}
	s.UniformReference = math.Log(float64(w.Config.Vocab))
	s.Scored = len(tr.FinalNorm) - 1
	if s.Scored <= 0 {
		return s
	}
	s.Rows = make([]positionPrediction, 0, s.Scored)
	var total float64
	for t := 0; t < s.Scored; t++ {
		probs := model.Softmax(w.Unembed(tr.FinalNorm[t]), temperature)
		guess, best := 0, float32(-1)
		for id, p := range probs {
			if p > best {
				best, guess = p, id
			}
		}
		actual := tr.Tokens[t+1].ID
		pActual := float64(probs[actual])
		// At a low enough temperature the true byte's probability can underflow to exactly
		// zero in float32, which makes -ln(p) infinite. That is a limit of the arithmetic, not
		// a property of the model worth printing: the card said "平均损失 +Inf nats" and
		// "降低了 -Inf%", which reads as a bug in the app. Count them instead and let the
		// summary say the mean is a lower bound.
		loss := 0.0
		if pActual > 0 {
			loss = -math.Log(pActual)
		} else {
			s.Underflowed++
		}
		hit := guess == actual
		if hit {
			s.Hits++
		}
		total += loss
		s.Rows = append(s.Rows, positionPrediction{
			Pos: t, Guess: guess, Actual: actual, Probs: probs, Loss: loss, Hit: hit,
		})
	}
	s.MeanLoss = total / float64(s.Scored)
	return s
}

// sparsityEvidence states, in one clause, what makes this model's activations
// "sparse but continuous" — the claim the FFN lesson makes.
//
// It aggregates over every row of the trace rather than quoting the selected
// token's row, and that is deliberate: the FFN stage's card already prints the
// selected row's concentration figure ("最大的 38 / 384 个维度承担 49%…"). A
// lesson paragraph repeating those words with a different number would read as
// two disagreeing measurements of one quantity, so the two figures answer
// different questions and say which: the card describes the row the reader has
// selected, this describes the whole prompt.
//
// Every number is measured, because the lesson's job is to describe this
// checkpoint rather than repeat the GELU literature. Measured on the bundled
// weights and the default prompt over all 36 rows × 384 dimensions.
func sparsityEvidence(tr *model.Trace) string {
	if tr == nil || len(tr.Layers) == 0 || len(tr.Layers[0].Hidden) == 0 {
		return ""
	}
	rows, dims := len(tr.Layers[0].Hidden), len(tr.Layers[0].Hidden[0])
	all := make([]float32, 0, rows*dims)
	var peak float64
	for _, row := range tr.Layers[0].Hidden {
		all = append(all, row...)
		for _, v := range row {
			if a := math.Abs(float64(v)); a > peak {
				peak = a
			}
		}
	}
	if peak == 0 {
		return ""
	}
	share, _ := activationConcentration(all)
	abs := make([]float64, len(all))
	for i, v := range all {
		abs[i] = math.Abs(float64(v))
	}
	sort.Float64s(abs)
	median := abs[len(abs)/2]
	cut := 0.1 * peak
	below, zeros := 0, 0
	for i, v := range abs {
		if v < cut {
			below++
		}
		if all[i] == 0 {
			zeros++
		}
	}
	return fmt.Sprintf("全部 %d 个位置 × %d 维里，最大的十分之一承担 %.0f%% 的激活绝对值、"+
		"中位数只有峰值的 %.0f%%、%.0f%% 的维度低于峰值的十分之一，恰好为 0 的有 %d 个",
		rows, dims, 100*share, 100*median/peak,
		100*float64(below)/float64(len(abs)), zeros)
}

// lensRung is one depth's answer, read out of the residual stream at that depth.
//
// The ladder walks the same row down the network: the input vector, then each
// layer's attention output and FFN output, and finally the model's real answer.
type lensRung struct {
	// Name labels the depth the way the rest of the app names tensors.
	Name string
	// Step is the lesson step whose figure draws that tensor, or -1 when the
	// tensor has no stage of its own. It lets the ladder point at where the
	// reader can see the vector this row was read from.
	Step int
	// Probs is the lens's next-byte distribution at this depth, temperature
	// applied. For the final rung it comes from the real projection rather than
	// from the lens.
	Probs []float32
	// Top is the argmax of Probs.
	Top int
	// Entropy is the distribution's Shannon entropy in nats.
	Entropy float64
	// AgreesWithFinal reports whether this depth's top-1 is the final answer's.
	AgreesWithFinal bool
	// IsFinal marks the rung that is the model's own output, so the view can say
	// which rung is a read-out trick and which is the real thing.
	IsFinal bool
}

// logitLens is the ladder for one position.
type logitLens struct {
	Rungs []lensRung
	// Pos is the position the ladder was read at.
	Pos int
	// Final is the model's own top-1 at that position.
	Final int
	// ConfidentWrong counts intermediate rungs that disagree with the final
	// answer while giving their own answer more than lensConfident probability.
	// It is the number that justifies the card's warning: on the bundled weights
	// and the default prompt it is 105 of 180 rungs, and 31 of 36 positions have
	// at least one of them.
	ConfidentWrong int
	// RungsCompared is how many intermediate rungs there were to disagree.
	RungsCompared int
	OK            bool
}

// lensConfident is where a rung counts as sure of itself. 0.9 rather than a
// softer threshold because the card says "自信地错" and has to mean it: a rung at
// 60% is hedging, one at 90% has stopped hedging.
const lensConfident = 0.9

// computeLogitLens builds the ladder for position pos.
//
// The idea it makes visible: the residual stream holds a usable next-byte
// distribution long before the network is done, so a reader can ask what the model
// "would have said" at each depth. Measured on the bundled weights and the default
// prompt at the last position, the answer is that attention alone does not decide
// anything — after layer 0's attention the argmax is still whatever the input
// vector already pointed at — and the FFN is what turns it into the final byte.
// Across all 36 positions: the input and post-attention rows agree with the final
// argmax in 0 of 36, after layer 0's FFN 25 of 36, after layer 1's FFN 35 of 36.
//
// The last rung is Trace.Logits, not a lens call on Trace.FinalNorm. That row has
// already been normalised by the final LayerNorm, and running the lens on it would
// normalise a second time — measured, every one of the 256 logits then differs from
// what the model computed.
func computeLogitLens(w *model.Weights, tr *model.Trace, pos int, temperature float64) logitLens {
	lens := logitLens{Pos: pos}
	if w == nil || tr == nil || pos < 0 || pos >= len(tr.Tokens) || len(tr.Layers) == 0 {
		return lens
	}
	add := func(name string, step int, logits []float32, isFinal bool) {
		probs := model.Softmax(logits, temperature)
		top, best := 0, float32(-1)
		for id, p := range probs {
			if p > best {
				top, best = id, p
			}
		}
		lens.Rungs = append(lens.Rungs, lensRung{
			Name: name, Step: step, Probs: probs, Top: top, IsFinal: isFinal,
			Entropy: float64(model.Entropy(probs)),
		})
	}

	add("输入向量 x", 1, w.LogitLens(tr.Input[pos]), false)
	for li := range tr.Layers {
		l := &tr.Layers[li]
		// Only the last layer's FFN output is what the model actually projects;
		// naming the step is still right for earlier layers because the residual
		// stage draws the same tensor per layer.
		// Both tensors are drawn by the residual stage, so the ladder can point a
		// reader at where to see the vector each rung was read from.
		add(fmt.Sprintf("层 %d 注意力后", li), 4, w.LogitLens(l.AfterAttn[pos]), false)
		add(fmt.Sprintf("层 %d FFN 后", li), 4, w.LogitLens(l.AfterFFN[pos]), false)
	}
	// The final rung is the model's real output at THIS position, which is only
	// Trace.Logits when pos is the last one: that field holds a single row, the
	// last position's, because that is the only row Forward projects. Using it
	// unconditionally was a bug — measured, at pos 0 the ladder's last rung said
	// "␣" (the last position's answer) while the real output there is "h", and 30 of
	// the 36 positions were wrong. Every earlier rung was computed at pos correctly,
	// so the ladder contradicted itself: a "final output" that matched no rung above
	// it and no other figure in the app.
	//
	// FinalNorm[pos] has already been through the final LayerNorm, so this calls
	// Unembed directly rather than LogitLens (which would normalise a second time).
	final := tr.Logits
	if pos != len(tr.FinalNorm)-1 {
		final = w.Unembed(tr.FinalNorm[pos])
	}
	add("最终输出", 6, final, true)

	lens.Final = lens.Rungs[len(lens.Rungs)-1].Top
	for i := range lens.Rungs {
		lens.Rungs[i].AgreesWithFinal = lens.Rungs[i].Top == lens.Final
		if lens.Rungs[i].IsFinal {
			continue
		}
		lens.RungsCompared++
		if !lens.Rungs[i].AgreesWithFinal && lens.Rungs[i].Probs[lens.Rungs[i].Top] > lensConfident {
			lens.ConfidentWrong++
		}
	}
	lens.OK = true
	return lens
}

// headProfile summarises what one attention head does over the whole prompt, so the
// four heads can be compared as behaviours rather than as one row's numbers.
//
// The lesson says "each head learns a different pattern". The inspector could show
// one row's entropy per head, which is four numbers about a single query and does not
// answer the question — two heads can have the same entropy on row 20 and attend to
// entirely different places. These are the two behaviours that are actually visible in
// this model's matrices:
type headProfile struct {
	// Head is the head index.
	Head int
	// MeanEntropy is the mean over rows of the row's entropy in nats, and
	// EntropyCeiling is ln(n), the entropy of attending evenly to everything visible.
	MeanEntropy    float64
	EntropyCeiling float64
	// PrevShare is the mean probability a row puts on the token immediately before
	// it — the "watch the previous token" pattern. Rows with no predecessor are not
	// counted, since row 0 has nothing before it.
	PrevShare float64
	// Peak is the largest single weight any row puts on one key, and PeakRows is how
	// many rows concentrate more than half their weight on one key: the "points at one
	// place" behaviour, as opposed to spreading.
	Peak     float64
	PeakRows int
	// Rows is how many rows were measured (n-1, or 0 for a one-token prompt).
	Rows int
	OK   bool
}

// computeHeadProfiles measures every head of one layer over every row.
//
// Measured on the bundled weights and the default prompt, the result is that layer 0's
// heads differ sharply and layer 1's do not: layer 0's previous-token share runs
// 0.66 / 0.22 / 0.37 / 0.61 across the four heads, while layer 1's sits at
// 0.12–0.14 for all four. That is a fact about this checkpoint worth showing — it is
// the concrete form of "attention heads specialise", and it also shows how unevenly
// they specialise: only the first layer's four heads are doing visibly different jobs.
func computeHeadProfiles(l model.LayerTrace, tokens int) (profiles []headProfile) {
	if tokens < 2 {
		return nil
	}
	rows := tokens - 1
	for hi := range l.Heads {
		h := &l.Heads[hi]
		p := headProfile{Head: hi, Rows: rows, EntropyCeiling: math.Log(float64(tokens))}
		var ent, prev float64
		for q := 1; q < tokens && q < len(h.Attn); q++ {
			row := h.Attn[q]
			var e float64
			best, bw := 0, float64(-1)
			for k := 0; k <= q && k < len(row); k++ {
				v := float64(row[k])
				if v > 1e-9 {
					e -= v * math.Log(v)
				}
				if v > bw {
					bw, best = v, k
				}
			}
			_ = best
			ent += e
			prev += float64(row[q-1])
			if bw > p.Peak {
				p.Peak = bw
			}
			if bw > 0.5 {
				p.PeakRows++
			}
		}
		p.MeanEntropy = ent / float64(rows)
		p.PrevShare = prev / float64(rows)
		p.OK = true
		profiles = append(profiles, p)
	}
	return profiles
}

// maskCounterfactual is what one row of an attention head would attend to if the
// causal mask were not there.
//
// The mask is architecture, not a learned weight, and the stage says so — but its
// effect is only ever shown negatively ("the upper triangle is empty"). This asks the
// question the empty triangle invites: what would row q look at if it could see
// everything? The scores above the diagonal are -Inf in the trace, so they are not
// recoverable from Scores; they are recomputed here from the same Q and K the model
// stored, with the same 1/√d_head scale.
type maskCounterfactual struct {
	// Row is the query row the counterfactual was computed for.
	Row int
	// FutureShare is the total attention the row would put on keys after it — the
	// positions the mask hides. This is the number that makes the mask's purpose
	// concrete: measured over this model, every head puts 51–84% of its unmasked
	// attention there.
	FutureShare float64
	// FutureTop is the future key with the largest share, and FutureTopShare is that
	// share. At row 0 the answer is striking: the model would give the rest of the
	// prompt the whole distribution.
	FutureTop      int
	FutureTopShare float64
	// MeanEntropy is the row's entropy without the mask, against the ceiling ln(n).
	MeanEntropy float64
	// VisibleEntropy is the same row's entropy as the model actually computes it,
	// over the q+1 keys it can see.
	VisibleEntropy float64
	// Total is how many keys the unmasked row spreads over.
	Total int
	OK    bool
}

// computeMaskCounterfactual recomputes row q of one head without the causal mask.
func computeMaskCounterfactual(h model.HeadTrace, q, tokens, dHead int) maskCounterfactual {
	c := maskCounterfactual{Row: q, Total: tokens}
	if q < 0 || q >= tokens || dHead <= 0 || len(h.Q) <= q || len(h.K) < tokens {
		return c
	}
	scale := float64(float32(1 / math.Sqrt(float64(dHead))))
	scores := make([]float64, tokens)
	maxScore := math.Inf(-1)
	for k := 0; k < tokens; k++ {
		// Same float32 dot the model takes, so the numbers below are the ones the
		// mask actually removed rather than a float64 re-derivation of them.
		var dot float32
		for d := 0; d < dHead; d++ {
			dot += h.Q[q][d] * h.K[k][d]
		}
		scores[k] = float64(dot) * scale
		if scores[k] > maxScore {
			maxScore = scores[k]
		}
	}
	var sum float64
	for k := range scores {
		scores[k] = math.Exp(scores[k] - maxScore)
		sum += scores[k]
	}
	if sum == 0 {
		return c
	}
	for k := range scores {
		scores[k] /= sum
	}
	// The row as the model computes it, for the entropy comparison.
	var visibleEntropy float64
	for k := 0; k <= q && k < len(h.Attn[q]); k++ {
		if p := float64(h.Attn[q][k]); p > 1e-9 {
			visibleEntropy -= p * math.Log(p)
		}
	}
	var entropy float64
	c.FutureTop = -1
	for k, p := range scores {
		if p > 1e-9 {
			entropy -= p * math.Log(p)
		}
		if k > q {
			c.FutureShare += p
			if p > c.FutureTopShare {
				c.FutureTopShare = p
				c.FutureTop = k
			}
		}
	}
	c.MeanEntropy = entropy
	c.VisibleEntropy = visibleEntropy
	c.OK = true
	return c
}

// meanFutureShare is the average, over every row that has a future, of how much of the
// unmasked distribution lands on keys the mask hides. It is the per-head figure the
// stage quotes, and it is what turns "the upper triangle is empty" into a number.
func meanFutureShare(h model.HeadTrace, tokens, dHead int) (share float64, rows int) {
	if tokens < 2 {
		return 0, 0
	}
	for q := 0; q < tokens-1; q++ {
		if c := computeMaskCounterfactual(h, q, tokens, dHead); c.OK {
			share += c.FutureShare
			rows++
		}
	}
	if rows == 0 {
		return 0, 0
	}
	return share / float64(rows), rows
}

// skillBreakdown splits the per-position predictions by what came immediately before
// them, which is where this model's errors actually concentrate.
//
// Measured on the bundled weights: positions whose predecessor is a space are guessed
// correctly 4 times in 18 across three prompts, against 73 in 96 elsewhere, and on the
// default prompt the mean loss there is 2.87 nats against 0.83 — a factor of 3.5. The
// reason is structural rather than statistical: a space is a word boundary, so the next
// byte is the first letter of a word, and this model's whole training set is 61
// sentences. It knows which letters continue a word it is already inside; it does not
// know which word comes next.
//
// Stated as a breakdown rather than a single average because the average hides it: 69%
// correct overall reads as "mostly fine", while 0/5 after spaces reads as what it is.
type skillBreakdown struct {
	// AfterSpaceHits/AfterSpaceRows are the positions whose predecessor is a space.
	AfterSpaceHits, AfterSpaceRows int
	// AfterSpaceLoss is the mean loss over those rows, and OtherLoss over the rest.
	AfterSpaceLoss, OtherLoss float64
	// OtherHits/OtherRows are everything else.
	OtherHits, OtherRows int
	OK                   bool
}

// computeSkillBreakdown splits a skill summary's rows by word boundary.
func computeSkillBreakdown(tr *model.Trace, skill nextByteSkill) skillBreakdown {
	b := skillBreakdown{}
	if tr == nil || len(skill.Rows) == 0 {
		return b
	}
	var spaceLoss, otherLoss float64
	for _, row := range skill.Rows {
		if row.Pos < 0 || row.Pos >= len(tr.Tokens) {
			continue
		}
		if tr.Tokens[row.Pos].Byte == ' ' {
			b.AfterSpaceRows++
			if row.Hit {
				b.AfterSpaceHits++
			}
			spaceLoss += row.Loss
			continue
		}
		b.OtherRows++
		if row.Hit {
			b.OtherHits++
		}
		otherLoss += row.Loss
	}
	if b.AfterSpaceRows > 0 {
		b.AfterSpaceLoss = spaceLoss / float64(b.AfterSpaceRows)
	}
	if b.OtherRows > 0 {
		b.OtherLoss = otherLoss / float64(b.OtherRows)
	}
	b.OK = b.AfterSpaceRows > 0 && b.OtherRows > 0
	return b
}

// layerRewrite measures how much one layer changes the residual stream passing
// through it, which is what distinguishes this model's two layers.
//
// The app compares heads (computeHeadProfiles) and reads predictions out of every
// depth (computeLogitLens), but nothing said that the layers are not doing the same
// amount of work. Measured on the bundled weights, the difference is large and stable
// across prompts: layer 0 leaves the stream almost unrelated to what entered it
// (cosine 0.22–0.24) and pushes in 2.8–3.2 times the length of its own input, while
// layer 1 barely moves the direction (cosine 0.92–0.93) and adds 0.41–0.45 of it.
type layerRewrite struct {
	// Layer is the block index.
	Layer int
	// CosineInOut is the mean cosine between the stream entering the block and the
	// stream leaving it. 1.0 would mean the block changed nothing.
	CosineInOut float64
	// FFFNRatio is the mean ‖FFNOut‖ divided by the mean ‖In‖ — how much new
	// material the block adds relative to what it was given.
	FFNRatio float64
	// AttnRatio is the same for the attention sublayer.
	AttnRatio float64
	// InNorm and OutNorm are the mean L2 lengths entering and leaving.
	InNorm, OutNorm float64
	// Hidden is the layer's FFN width, so a caller can compare against d_ff.
	Hidden int
	OK     bool
}

// computeLayerRewrite measures every layer's effect on the residual stream.
func computeLayerRewrite(tr *model.Trace) (out []layerRewrite) {
	if tr == nil || len(tr.Layers) == 0 || len(tr.Tokens) == 0 {
		return nil
	}
	n := len(tr.Tokens)
	for li := range tr.Layers {
		l := &tr.Layers[li]
		r := layerRewrite{Layer: li, Hidden: tr.Config.DFF}
		var cosSum, ffnSum, attnSum, inSum, outSum float64
		rows := 0
		for i := 0; i < n && i < len(l.In) && i < len(l.AfterFFN); i++ {
			cosSum += cosineF32(l.In[i], l.AfterFFN[i])
			ffnSum += float64(norm(l.FFNOut[i]))
			attnSum += float64(norm(l.AttnProj[i]))
			inSum += float64(norm(l.In[i]))
			outSum += float64(norm(l.AfterFFN[i]))
			rows++
		}
		if rows == 0 {
			continue
		}
		f := float64(rows)
		r.CosineInOut = cosSum / f
		r.InNorm = inSum / f
		r.OutNorm = outSum / f
		if r.InNorm > 0 {
			r.FFNRatio = (ffnSum / f) / r.InNorm
			r.AttnRatio = (attnSum / f) / r.InNorm
		}
		r.OK = true
		out = append(out, r)
	}
	return out
}

// cosineF32 is the cosine similarity of two float32 rows, 0 when either is zero.
func cosineF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := 0; i < len(a) && i < len(b); i++ {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// posEncodingShape measures what a positional table actually looks like, which is the
// question stage 2 raises and never answered.
//
// The stage says the position vector is learned rather than fixed, and draws P[t] for a
// few positions — but it never says what "learned" produced. The answer is surprising
// and worth a card: measured on the bundled weights, adjacent positions have a cosine of
// 0.040, so the table encodes position as a set of nearly orthogonal directions. The
// classic sinusoidal encoding does the opposite — adjacent positions there have cosine
// 0.985, because each row is a small step along a smooth curve.
//
// Neither is wrong; they are different designs, and the model found the one nothing in
// its architecture forced it to choose.
type posEncodingShape struct {
	// AdjacentCos is the mean cosine between P[t] and P[t+1].
	AdjacentCos float64
	// DistantCos is the same at a gap of half the context window.
	DistantCos float64
	// SinusoidalAdjacentCos is what the fixed sinusoidal encoding would give for the
	// same model shape, computed here so the comparison cannot be misremembered.
	SinusoidalAdjacentCos float64
	SinusoidalDistantCos  float64
	// RandomCos is the |cos| two unrelated vectors in this many dimensions would
	// have on average, 1/√d. It tells a reader whether 0.04 means "deliberately
	// orthogonal" or "no more structured than chance".
	RandomCos float64
	// Gap used for the distant measurement.
	Gap int
	// Dims is d_model, so the card can name the space the 1/√d reference refers to.
	Dims int
	OK   bool
}

// computePosEncodingShape measures the positional table of a model against the
// sinusoidal alternative and against chance.
func computePosEncodingShape(w *model.Weights) posEncodingShape {
	s := posEncodingShape{}
	if w == nil || w.Config.DModel <= 0 || w.Config.Ctx < 3 {
		return s
	}
	d, ctx := w.Config.DModel, w.Config.Ctx
	if len(w.PosEmbed) < ctx*d {
		return s
	}
	s.Gap = ctx / 2
	if s.Gap < 1 {
		s.Gap = 1
	}
	s.Dims = d
	s.RandomCos = 1 / math.Sqrt(float64(d))
	row := func(i int) []float32 { return w.PosEmbed[i*d : (i+1)*d] }
	// sinusoidal is the fixed encoding from the original transformer, indexed the way
	// this model's embedding width allows: even dimensions sine, odd cosine, with
	// frequencies spread over half the width.
	sinusoidal := func(pos, k int) float64 {
		half := d / 2
		if half == 0 {
			return 0
		}
		x := float64(pos) / math.Pow(10000, float64(k/2)/float64(half))
		if k%2 == 0 {
			return math.Sin(x)
		}
		return math.Cos(x)
	}
	cosOf := func(a, b []float32) float64 { return cosineF32(a, b) }
	var adj, far, sAdj, sFar float64
	adjN, farN := 0, 0
	for i := 0; i+1 < ctx; i++ {
		adj += cosOf(row(i), row(i+1))
		adjN++
	}
	for i := 0; i+s.Gap < ctx; i++ {
		far += cosOf(row(i), row(i+s.Gap))
		farN++
	}
	// The sinusoidal reference as vectors, so the same cosine function is used.
	sinVec := func(pos int) []float64 {
		v := make([]float64, d)
		for k := range v {
			v[k] = sinusoidal(pos, k)
		}
		return v
	}
	cosF := func(a, b []float64) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += a[i] * b[i]
			na += a[i] * a[i]
			nb += b[i] * b[i]
		}
		if na == 0 || nb == 0 {
			return 0
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb))
	}
	if adjN > 0 {
		s.AdjacentCos = adj / float64(adjN)
	}
	if farN > 0 {
		s.DistantCos = far / float64(farN)
	}
	for i := 0; i+1 < ctx; i++ {
		sAdj += cosF(sinVec(i), sinVec(i+1))
	}
	for i := 0; i+s.Gap < ctx; i++ {
		sFar += cosF(sinVec(i), sinVec(i+s.Gap))
	}
	if adjN > 0 {
		s.SinusoidalAdjacentCos = sAdj / float64(adjN)
	}
	if farN > 0 {
		s.SinusoidalDistantCos = sFar / float64(farN)
	}
	s.OK = true
	return s
}

// ffnConcentration measures how the FFN's hidden units divide the work, which is the
// layer-level counterpart to the head comparison and the missing piece on the FFN stage.
//
// The stage draws one row's 384 hidden units and the top contributors to that row. It
// never says whether the same units matter everywhere or a different few matter per
// position — and the two layers answer that differently. Measured on the bundled
// weights, layer 0 needs 118–130 of its 384 units to account for half the total
// activation across the prompt and has no unit that is positive at every position,
// while layer 1 needs only 28–35 and has 5–18 units on at every position.
//
// So layer 1's FFN is doing focused, repeated work — a few units firing everywhere —
// while layer 0's is spread thin across many. That is the same asymmetry
// computeLayerRewrite reports from the geometry side: the first block transforms, the
// second applies a narrow correction.
type ffnConcentration struct {
	// Layer is the block index.
	Layer int
	// Units is d_ff.
	Units int
	// HalfMass is how many units, largest first, together carry half the total
	// activation summed over every position. Small means a few units dominate.
	HalfMass int
	// AlwaysOn is how many units are positive at every position of the prompt.
	AlwaysOn int
	// NeverOn is how many are never positive.
	NeverOn int
	// PositiveShare is the fraction of the n×d_ff hidden values that are above zero,
	// which is what GELU's one-sidedness makes meaningful.
	PositiveShare float64
	// Peak is the largest single hidden value anywhere.
	Peak float64
	OK   bool
}

// computeFFNConcentration measures one layer's hidden-unit usage over the prompt.
func computeFFNConcentration(l model.LayerTrace, tokens, dff int) ffnConcentration {
	c := ffnConcentration{Layer: -1, Units: dff}
	if tokens < 1 || dff < 1 || len(l.Hidden) < tokens {
		return c
	}
	sums := make([]float64, dff)
	positiveAt := make([]int, dff)
	var total float64
	positives := 0
	for i := 0; i < tokens; i++ {
		row := l.Hidden[i]
		for j := 0; j < dff && j < len(row); j++ {
			v := float64(row[j])
			sums[j] += v
			total += v
			if row[j] > 0 {
				positiveAt[j]++
				positives++
			}
			if v > c.Peak {
				c.Peak = v
			}
		}
	}
	for _, n := range positiveAt {
		if n == tokens {
			c.AlwaysOn++
		}
		if n == 0 {
			c.NeverOn++
		}
	}
	// Count units, largest first, until half the mass is covered. Sorted by a
	// selection over indices so the sums stay aligned with their unit numbers.
	order := make([]int, dff)
	for j := range order {
		order[j] = j
	}
	sort.SliceStable(order, func(a, b int) bool { return sums[order[a]] > sums[order[b]] })
	var acc float64
	for _, j := range order {
		acc += sums[j]
		c.HalfMass++
		if acc >= total/2 {
			break
		}
	}
	c.PositiveShare = float64(positives) / float64(tokens*dff)
	c.OK = true
	return c
}

// computeAllFFNConcentration measures every layer.
func computeAllFFNConcentration(tr *model.Trace) (out []ffnConcentration) {
	if tr == nil || len(tr.Tokens) == 0 {
		return nil
	}
	for li := range tr.Layers {
		c := computeFFNConcentration(tr.Layers[li], len(tr.Tokens), tr.Config.DFF)
		if !c.OK {
			continue
		}
		c.Layer = li
		out = append(out, c)
	}
	return out
}

// promptLensSummary aggregates the logit lens over every position of the prompt, which
// is the figure the residual stage's callout quotes.
//
// It exists because the sentence it backs was hardcoded. The card said "180 个中间层里有
// 92 个…，36 个位置里每一个都至少有一层如此" for every prompt, and those numbers are
// right only for the default one: with "hi" the ladder has 10 intermediate rungs across
// 2 positions, and the card still claimed 180 and 36. A reader who typed a short prompt
// then read a sentence contradicting the ladder directly above it.
//
// The test that guarded the sentence recomputed the figures inline, so the prose could
// drift from both the test and the card without anything noticing. This is the single
// definition now: the card prints it and the test checks it.
type promptLensSummary struct {
	// Positions is how many rows the ladder was built for, and Rungs is how many
	// intermediate rungs they hold altogether (five per position: the input vector and
	// the four sublayer outputs, with the final projection counted separately).
	Positions int
	Rungs     int
	// Disagree is how many rungs name a different byte from the final answer.
	Disagree int
	// ConfidentWrong is how many of those disagreeing rungs are more than confidentCut
	// sure of themselves.
	ConfidentWrong int
	// PositionsWithConfidentWrong is how many positions have at least one such rung.
	PositionsWithConfidentWrong int
	OK                          bool
}

// computePromptLensSummary measures the ladder across the whole prompt.
func computePromptLensSummary(w *model.Weights, tr *model.Trace, temp float64) promptLensSummary {
	var s promptLensSummary
	if w == nil || tr == nil {
		return s
	}
	for pos := 0; pos < len(tr.Tokens); pos++ {
		lens := computeLogitLens(w, tr, pos, temp)
		if !lens.OK {
			continue
		}
		s.Positions++
		has := false
		for _, rung := range lens.Rungs {
			if rung.IsFinal {
				continue
			}
			s.Rungs++
			if rung.AgreesWithFinal {
				continue
			}
			s.Disagree++
			if float64(rung.Probs[rung.Top]) > lensConfident {
				s.ConfidentWrong++
				has = true
			}
		}
		if has {
			s.PositionsWithConfidentWrong++
		}
	}
	s.OK = s.Positions > 0
	return s
}

// weightTyingParts decomposes one byte's logit, which is what weight tying makes possible
// to state exactly.
//
// The output stage says the unembedding is the token embedding transposed ("权重绑定") and
// draws the top-12 probabilities. It never says what that structure *means* for a
// prediction, and the answer is exact rather than approximate: with the tying, byte b's
// logit is <E[b], x>, so it factors into |E[b]| · |x| · cos(E[b], x). The model literally
// prefers the bytes whose input embedding points the same way as the final vector.
//
// Measured on the bundled weights at the default prompt's last position: the cosine runs
// from −0.324 to +0.276 across the 256 bytes and correlates +0.997 with the logit, while
// the embedding lengths vary only 8.29–11.71 (35% spread) and correlate −0.000. So the
// ranking is essentially a cosine ranking: 563 of the 32640 byte pairs order differently
// by logit than by cosine, 1.7%.
type weightTyingParts struct {
	// Byte is the vocabulary id the decomposition is for.
	Byte int
	// Logit is the score the model produced for it, Dot is <E[b], x>, and the three
	// factors multiply to it: EmbedNorm · FinalNorm · Cosine.
	Logit       float64
	Dot         float64
	EmbedNorm   float64
	FinalNorm   float64
	Cosine      float64
	Probability float64
	// Rank is the byte's place by logit, 1 for the top. Total is the vocabulary size.
	Rank, Total int
	// EmbedNormMin and EmbedNormMax bound every byte's embedding length, and
	// CosineMin and CosineMax bound the cosine the final vector makes with each of them.
	EmbedNormMin, EmbedNormMax float64
	CosineMin, CosineMax       float64
	// LogitCosineCorrelation is how much of the ranking the cosine explains, and
	// LogitNormCorrelation how much the embedding's length does — the two together are
	// what lets the card say which factor decides the order instead of asserting it.
	LogitCosineCorrelation float64
	LogitNormCorrelation   float64
	// PairsOrderedDifferently is how many byte pairs rank differently by logit than by
	// cosine, out of Pairs.
	PairsOrderedDifferently, Pairs int
	OK                             bool
}

// computeWeightTying decomposes the top byte's logit and measures how much of the
// vocabulary's ranking the cosine cosine accounts for.
func computeWeightTying(w *model.Weights, tr *model.Trace, temp float64) weightTyingParts {
	var p weightTyingParts
	if w == nil || tr == nil || len(tr.FinalNorm) == 0 {
		return p
	}
	d := w.Config.DModel
	vocab := w.Config.Vocab
	if d <= 0 || vocab <= 0 || len(w.TokEmbed) < vocab*d {
		return p
	}
	row := tr.FinalNorm[len(tr.FinalNorm)-1]
	var rn float64
	for _, v := range row {
		rn += float64(v) * float64(v)
	}
	rn = math.Sqrt(rn)
	if rn == 0 {
		return p
	}
	logits := make([]float64, vocab)
	cosines := make([]float64, vocab)
	norms := make([]float64, vocab)
	p.EmbedNormMin = math.Inf(1)
	p.CosineMin = math.Inf(1)
	p.CosineMax = math.Inf(-1)
	for id := 0; id < vocab; id++ {
		var dot, ne float64
		for k := 0; k < d; k++ {
			e := float64(w.TokEmbed[id*d+k])
			dot += e * float64(row[k])
			ne += e * e
		}
		n := math.Sqrt(ne)
		logits[id] = dot
		norms[id] = n
		if n > 0 {
			cosines[id] = dot / (n * rn)
		}
		if n < p.EmbedNormMin {
			p.EmbedNormMin = n
		}
		if n > p.EmbedNormMax {
			p.EmbedNormMax = n
		}
		if c := cosines[id]; c < p.CosineMin {
			p.CosineMin = c
		} else if c > p.CosineMax {
			p.CosineMax = c
		}
	}
	// The reported byte is the model's own top choice.
	top := 0
	for id := 1; id < vocab; id++ {
		if logits[id] > logits[top] {
			top = id
		}
	}
	p.Byte = top
	p.Logit = logits[top]
	p.Dot = logits[top]
	p.EmbedNorm = norms[top]
	p.FinalNorm = rn
	p.Cosine = cosines[top]
	p.Rank = 1
	p.Total = vocab
	probs := model.Softmax(toF32(logits), temp)
	p.Probability = float64(probs[top])
	// How much of the logit's ordering the cosine accounts for.
	p.LogitCosineCorrelation = pearson(logits, cosines)
	p.LogitNormCorrelation = pearson(logits, norms)
	for i := 0; i < vocab; i++ {
		for j := i + 1; j < vocab; j++ {
			p.Pairs++
			if (logits[i] > logits[j]) != (cosines[i] > cosines[j]) {
				p.PairsOrderedDifferently++
			}
		}
	}
	p.OK = true
	return p
}

// toF32 converts the logits once, so the probability shown agrees with the one the app
// computes from Trace.Logits rather than being a float64 re-derivation of it.
func toF32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// pearson is the correlation of two equal-length series, 0 when either is constant.
func pearson(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	n := float64(len(a))
	ma /= n
	mb /= n
	var num, da, db float64
	for i := range a {
		da += (a[i] - ma) * (a[i] - ma)
		db += (b[i] - mb) * (b[i] - mb)
		num += (a[i] - ma) * (b[i] - mb)
	}
	if da == 0 || db == 0 {
		return 0
	}
	return num / (math.Sqrt(da) * math.Sqrt(db))
}

// headIndependence measures whether the heads' outputs are complements or near-copies,
// which is the justification for concatenating them instead of summing.
//
// The Softmax stage draws each head's contribution and then the sum, and says the heads
// are "相加" — but never says why they are worth having separately. Measured on the
// bundled weights at layer 0, the four heads' projected outputs are close to orthogonal
// (pairwise mean cosine 0.027 to 0.144) and carry comparable magnitudes (19% to 32% of
// the total), so summing them before their own projections would be destructive: the
// whole point of the four-slice concatenation is that the slices say different things.
type headIndependence struct {
	// Layer is the block the measurement is for, and Heads is how many it has.
	Layer int
	Heads int
	// MaxPairCosine is the largest mean cosine between any two heads' projected outputs,
	// so a small value means every pair is close to orthogonal.
	MaxPairCosine float64
	// MinPairCosine is the smallest, so the pair range is readable.
	MinPairCosine float64
	// MinShare and MaxShare bound the share of the total ‖Δ_h‖ each head contributes.
	MinShare, MaxShare float64
	OK                 bool
}

// computeHeadIndependence measures one layer's head outputs against each other.
func computeHeadIndependence(l model.LayerTrace, tokens int) headIndependence {
	c := headIndependence{Layer: -1, Heads: len(l.HeadProj), MinPairCosine: math.Inf(1), MaxPairCosine: math.Inf(-1)}
	if tokens < 1 || len(l.HeadProj) < 2 {
		return c
	}
	// Pairwise cosine, averaged over positions.
	first := true
	for i := 0; i < len(l.HeadProj); i++ {
		for j := i + 1; j < len(l.HeadProj); j++ {
			var sum float64
			rows := 0
			for p := 0; p < tokens && p < len(l.HeadProj[i]) && p < len(l.HeadProj[j]); p++ {
				sum += cosineF32(l.HeadProj[i][p], l.HeadProj[j][p])
				rows++
			}
			if rows == 0 {
				continue
			}
			mean := sum / float64(rows)
			if mean < c.MinPairCosine {
				c.MinPairCosine = mean
			}
			if mean > c.MaxPairCosine {
				c.MaxPairCosine = mean
			}
			first = false
		}
	}
	if first {
		return c
	}
	// Each head's share of the total length.
	lengths := make([]float64, len(l.HeadProj))
	var total float64
	for i := range l.HeadProj {
		for p := 0; p < tokens && p < len(l.HeadProj[i]); p++ {
			lengths[i] += float64(norm(l.HeadProj[i][p]))
		}
		total += lengths[i]
	}
	if total > 0 {
		c.MinShare, c.MaxShare = math.Inf(1), math.Inf(-1)
		for _, v := range lengths {
			share := v / total
			if share < c.MinShare {
				c.MinShare = share
			}
			if share > c.MaxShare {
				c.MaxShare = share
			}
		}
	}
	c.OK = true
	return c
}

// ffnWritesTheAnswer measures whether the FFN's output points at the byte that actually
// comes next — which is the sharpest available statement of what the FFN is for, and the
// mechanism behind the residual stage's logit-lens observation.
//
// The FFN stage draws one position's hidden units and its output sum. It never says
// whether that output has anything to do with the answer. Measured on the bundled weights
// with the default prompt, it very much does: rank every one of the 256 bytes by the
// cosine between its input embedding and the vector in question, and at each position ask
// where the byte that actually follows lands.
//
//	layer 0's FFN output   mean rank 4    the true byte is in the top 10 at 30 of 35 positions
//	layer 0's input        mean rank 103  the true byte is indistinguishable from chance
//
// So the FFN is not refining a hint that was already there — the hint is not there before
// it, and is close to the answer after it. That is the same fact the logit lens shows from
// the other direction ("the prediction is formed after layer 0's FFN"), stated as a rank
// rather than as a sampled byte.
type ffnAnswerDirection struct {
	// Layer is the block measured. Positions is how many next-byte predictions the prompt
	// offers, t = 0 .. T-2.
	Layer     int
	Positions int
	// Vocab is how many bytes the rank is out of.
	Vocab int
	// OutputMeanRank and InputMeanRank are the mean rank of the true next byte, ranked by
	// cosine against the FFN output and against the block's input.
	OutputMeanRank, InputMeanRank float64
	// OutputTop10 is how many positions put the true byte in the top ten by output.
	OutputTop10 int
	// OutputMeanCos and InputMeanCos are the mean cosine to the true byte.
	OutputMeanCos, InputMeanCos float64
	OK                          bool
}

// computeFFNAnswerDirection ranks the true next byte at every position, against the FFN
// output and against the block's input, for one layer.
func computeFFNAnswerDirection(w *model.Weights, tr *model.Trace, layer int) ffnAnswerDirection {
	var m ffnAnswerDirection
	if w == nil || tr == nil || layer < 0 || layer >= len(tr.Layers) {
		return m
	}
	d, vocab := w.Config.DModel, w.Config.Vocab
	last := len(tr.Tokens) - 1
	if d <= 0 || vocab <= 0 || last < 1 || len(w.TokEmbed) < vocab*d {
		return m
	}
	m.Layer = layer
	m.Vocab = vocab
	l := &tr.Layers[layer]
	// cosineToAll returns the cosine of v against every byte's embedding.
	cosineToAll := func(v []float32) []float64 {
		out := make([]float64, vocab)
		var nv float64
		for _, x := range v {
			nv += float64(x) * float64(x)
		}
		if nv == 0 {
			return out
		}
		root := math.Sqrt(nv)
		for id := 0; id < vocab; id++ {
			var dot, ne float64
			for k := 0; k < d; k++ {
				e := float64(w.TokEmbed[id*d+k])
				dot += float64(v[k]) * e
				ne += e * e
			}
			if ne > 0 {
				out[id] = dot / (root * math.Sqrt(ne))
			}
		}
		return out
	}
	rankOf := func(cosines []float64, actual int) int {
		target := cosines[actual]
		rank := 1
		for id := 0; id < vocab; id++ {
			if id != actual && cosines[id] > target {
				rank++
			}
		}
		return rank
	}
	var outRank, inRank, outCos, inCos float64
	for p := 0; p < last && p < len(l.FFNOut) && p < len(l.In); p++ {
		actual := int(tr.Tokens[p+1].Byte)
		oc := cosineToAll(l.FFNOut[p])
		ic := cosineToAll(l.In[p])
		outRank += float64(rankOf(oc, actual))
		inRank += float64(rankOf(ic, actual))
		outCos += oc[actual]
		inCos += ic[actual]
		if rankOf(oc, actual) <= 10 {
			m.OutputTop10++
		}
		m.Positions++
	}
	if m.Positions == 0 {
		return m
	}
	f := float64(m.Positions)
	m.OutputMeanRank = outRank / f
	m.InputMeanRank = inRank / f
	m.OutputMeanCos = outCos / f
	m.InputMeanCos = inCos / f
	m.OK = true
	return m
}

// embeddingStructure tests whether the token embedding table encodes anything a reader
// might assume it does, and the answer for these weights is that it does not.
//
// The Embedding stage draws a 96-wide coloured strip per token. Colour in this app means
// a value, and the strips invite the reading that similar bytes sit in similar places.
// Measured on the bundled weights, that reading is wrong at every scale tested: bytes that
// follow each other in the prompt (t-h, h-e, e-space) have a mean cosine of 0.0007 against
// a random-pair mean of 0.0033 and standard deviation 0.102, vowels cluster 0.03 above
// consonants, and only one of the twenty most-similar byte pairs is printable ASCII on both
// sides.
//
// This is worth stating rather than leaving to assumption: a 96-dimension table trained on
// 61 sentences is a lookup, not a geometry, and a reader who took the picture as evidence of
// semantic structure would be learning something false about the model.
type embeddingStructure struct {
	// Vocab and Dims describe the table measured.
	Vocab, Dims int
	// AdjacentCos is the mean cosine of byte pairs that follow each other in the prompt,
	// and RandomCos/SDRandom are the mean and spread over byte pairs chosen without regard
	// to text — the reference the adjacent figure has to be judged against.
	AdjacentCos, RandomCos, SDRandom float64
	// AdjacentPairs and RandomPairs are how many pairs each figure averages, and
	// SkippedSelfPairs is how many adjacent pairs were left out because both bytes were
	// the same — their cosine is 1.0 by definition and would swamp the average.
	AdjacentPairs, RandomPairs, SkippedSelfPairs int
	// VowelCos, ConsonantCos are the mean cosines within vowels and between vowels and
	// consonants; their difference is the largest hint of structure found.
	VowelCos, ConsonantCos float64
	// SimilarPrintableOf20 is how many of the twenty most similar byte pairs are both
	// printable ASCII — a pair of unrelated control bytes at the top would be the clearest
	// sign that the similarity is noise.
	SimilarPrintableOf20 int
	// TopPairs are the most similar byte pairs, for the card to name a few.
	TopPairs []embeddingPair
	OK       bool
}

// embeddingPair is two bytes and the cosine between their embeddings.
type embeddingPair struct {
	A, B int
	Cos  float64
}

// computeEmbeddingStructure measures whether the embedding table has structure.
func computeEmbeddingStructure(w *model.Weights, prompt string) embeddingStructure {
	var e embeddingStructure
	if w == nil || w.Config.DModel <= 0 || w.Config.Vocab <= 0 {
		return e
	}
	d, vocab := w.Config.DModel, w.Config.Vocab
	if len(w.TokEmbed) < vocab*d {
		return e
	}
	e.Vocab, e.Dims = vocab, d
	row := func(id int) []float32 { return w.TokEmbed[id*d : (id+1)*d] }
	// Adjacent pairs in the prompt, excluding the ones where both bytes are the same.
	// A repeated byte ("tt" in attention, "ll" in all) has cosine exactly 1.0 with itself,
	// which says nothing about whether similar-sounding bytes sit near each other and
	// dominated the average when it was included: measured on "attention is all you need",
	// three such pairs lifted the mean from 0.02 to 0.13.
	var adjSum float64
	bs := []byte(prompt)
	for i := 0; i+1 < len(bs); i++ {
		if bs[i] == bs[i+1] {
			e.SkippedSelfPairs++
			continue
		}
		adjSum += cosineF32(row(int(bs[i])), row(int(bs[i+1])))
		e.AdjacentPairs++
	}
	if e.AdjacentPairs > 0 {
		e.AdjacentCos = adjSum / float64(e.AdjacentPairs)
	}
	// Every pair, for the random reference and the ranking. 256 choose 2 is small enough.
	var sum, sum2 float64
	var all []embeddingPair
	for a := 0; a < vocab; a++ {
		for b := a + 1; b < vocab; b++ {
			c := cosineF32(row(a), row(b))
			all = append(all, embeddingPair{A: a, B: b, Cos: c})
			sum += c
			sum2 += c * c
			e.RandomPairs++
		}
	}
	if e.RandomPairs > 0 {
		f := float64(e.RandomPairs)
		e.RandomCos = sum / f
		variance := sum2/f - e.RandomCos*e.RandomCos
		if variance > 0 {
			e.SDRandom = math.Sqrt(variance)
		}
	}
	// Vowel-versus-consonant, the most likely place for structure in English text.
	isVowel := func(id int) bool {
		switch byte(id) {
		case 'a', 'e', 'i', 'o', 'u':
			return true
		}
		return false
	}
	var vv, vc float64
	vvn, vcn := 0, 0
	for a := int('a'); a <= int('z'); a++ {
		for b := a + 1; b <= int('z'); b++ {
			c := cosineF32(row(a), row(b))
			if isVowel(a) && isVowel(b) {
				vv += c
				vvn++
			} else if isVowel(a) != isVowel(b) {
				vc += c
				vcn++
			}
		}
	}
	if vvn > 0 {
		e.VowelCos = vv / float64(vvn)
	}
	if vcn > 0 {
		e.ConsonantCos = vc / float64(vcn)
	}
	// The most similar pairs, and how many of them are readable text.
	sort.Slice(all, func(i, j int) bool { return all[i].Cos > all[j].Cos })
	if len(all) > 0 {
		n := 20
		if len(all) < n {
			n = len(all)
		}
		e.TopPairs = all[:n]
		for _, p := range e.TopPairs {
			if p.A >= 32 && p.A < 127 && p.B >= 32 && p.B < 127 {
				e.SimilarPrintableOf20++
			}
		}
	}
	e.OK = true
	return e
}

// temperatureMinimum finds the temperature where this prompt's mean loss is lowest, and
// reports enough of the curve for a reader to see the shape.
//
// The per-position card used to state, in fixed prose, that the mean loss is U-shaped with
// its minimum at T=1 — "the value the weights were trained at". That was measured once on
// the default prompt and then shown for every prompt. Measured across four prompts, T=1 is
// the minimum for only one of them: "attention is all you need" and a longer sentence bottom
// out at 1.5, and "hi" at 2.0. So the sentence was not merely unverified, it was false for
// most of the prompts a reader can type.
//
// The shape is still worth stating — but which temperature is lowest is a measurement, so
// the card makes it one.
type temperatureCurve struct {
	// Samples are the temperatures probed, ascending, and Losses the mean loss at each.
	Samples []float64
	Losses  []float64
	// Best is the sample with the lowest loss, and BestLoss that loss.
	Best     float64
	BestLoss float64
	// AtOne is the loss at T=1, which is where the weights were trained — the reference the
	// curve is judged against rather than the answer.
	AtOne float64
	// UnderflowedAt is how many positions underflowed at the coldest sample, so a caller can
	// say the low end is a limit of the arithmetic rather than a property of the model.
	UnderflowedAt int
	OK            bool
}

// temperatureSamples are the temperatures the curve is measured at. They span the slider's
// range (0.1 to 2.0) with a denser middle, where the minimum usually is.
var temperatureSamples = []float64{0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}

// computeTemperatureCurve measures the mean loss across the temperature samples.
func computeTemperatureCurve(w *model.Weights, tr *model.Trace) temperatureCurve {
	var c temperatureCurve
	if w == nil || tr == nil || len(tr.Tokens) < 2 {
		return c
	}
	for _, t := range temperatureSamples {
		s := computeNextByteSkill(w, tr, t)
		if s.Scored == 0 {
			continue
		}
		c.Samples = append(c.Samples, t)
		c.Losses = append(c.Losses, s.MeanLoss)
		if len(c.Samples) == 1 {
			c.UnderflowedAt = s.Underflowed
		}
		if math.Abs(t-1.0) < 1e-9 {
			c.AtOne = s.MeanLoss
		}
	}
	if len(c.Samples) == 0 {
		return c
	}
	c.Best, c.BestLoss = c.Samples[0], c.Losses[0]
	for i, v := range c.Losses {
		if v < c.BestLoss {
			c.Best, c.BestLoss = c.Samples[i], v
		}
	}
	c.OK = true
	return c
}
