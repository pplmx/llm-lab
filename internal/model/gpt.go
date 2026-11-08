package model

import (
	"math"
	"runtime"
	"sync"
	"unicode/utf8"
)

// Token is declared in tokenizer.go.

// HeadTrace is one attention head over the whole sequence.
type HeadTrace struct {
	Q      [][]float32 // T × d_head
	K      [][]float32
	V      [][]float32
	Scores [][]float32 // T × T, scaled dot products, -Inf above the diagonal
	Attn   [][]float32 // T × T, softmax of Scores
	Out    [][]float32 // T × d_head, Attn · V
}

// LayerTrace is everything one transformer block computed.
type LayerTrace struct {
	In        [][]float32 // residual stream entering the block
	Norm1     [][]float32
	Heads     []HeadTrace
	HeadProj  [][][]float32 // per head: Out_h · Wo[h*dh:(h+1)*dh], sums to AttnProj
	AttnProj  [][]float32   // concatenated heads · Wo
	AfterAttn [][]float32   // In + AttnProj
	Norm2     [][]float32
	Hidden    [][]float32 // GELU(x·W1), d_ff wide
	FFNOut    [][]float32 // Hidden · W2
	AfterFFN  [][]float32 // AfterAttn + FFNOut
}

// Trace is a recorded forward pass: every intermediate tensor the UI can show.
type Trace struct {
	Text      string
	Tokens    []Token
	Embed     [][]float32 // token embedding only
	Pos       [][]float32 // positional embedding only
	Input     [][]float32 // Embed + Pos
	Layers    []LayerTrace
	FinalNorm [][]float32
	Logits    []float32 // next-token logits, last position, vocab wide
	Config    Config
}

// Forward runs the model over ids and records every stage.
func Forward(w *Weights, ids []int) *Trace {
	c := w.Config
	d, dh := c.DModel, c.DHead()
	ids = Truncate(ids, c.Ctx)
	T := len(ids)

	tokens := make([]Token, T)
	for i, id := range ids {
		b := byte(id)
		tokens[i] = Token{ID: int(b), Byte: b, Text: renderByte(b)}
	}
	// Tokenize the prompt separately to recover rune grouping and text.
	if len(ids) > 0 {
		tokens = alignTokens(ids, tokens)
	}

	tr := &Trace{Text: DecodeBytes(ids), Tokens: tokens, Config: c}
	if T == 0 {
		tr.Embed, tr.Pos, tr.Input, tr.FinalNorm = [][]float32{}, [][]float32{}, [][]float32{}, [][]float32{}
		tr.Logits = make([]float32, c.Vocab)
		tr.Layers = make([]LayerTrace, c.NLayer)
		return tr
	}
	tr.Embed = newMatrix(T, d)
	tr.Pos = newMatrix(T, d)
	tr.Input = newMatrix(T, d)
	for i, id := range ids {
		b := int(byte(id))
		emb := tr.Embed[i]
		copy(emb, w.TokEmbed[b*d:(b+1)*d])
		pos := tr.Pos[i]
		copy(pos, w.PosEmbed[i*d:(i+1)*d])
		addInto(tr.Input[i], emb, pos)
	}

	x := tr.Input
	for li := 0; li < c.NLayer; li++ {
		lw := &w.Layers[li]
		lt := LayerTrace{In: x}
		lt.Norm1 = layerNorm(x, lw.LN1G, lw.LN1B)

		q := matMulRows(lt.Norm1, lw.WQ, nil, d)
		k := matMulRows(lt.Norm1, lw.WK, nil, d)
		v := matMulRows(lt.Norm1, lw.WV, nil, d)

		lt.Heads = make([]HeadTrace, c.NHead)
		lt.HeadProj = make([][][]float32, c.NHead)
		lt.AttnProj = newMatrix(T, d)
		scale := float32(1 / math.Sqrt(float64(dh)))
		runHead := func(h int) {
			ht := HeadTrace{
				Q:      sliceCols(q, h*dh, dh),
				K:      sliceCols(k, h*dh, dh),
				V:      sliceCols(v, h*dh, dh),
				Scores: newMatrix(T, T),
				Attn:   newMatrix(T, T),
				Out:    newMatrix(T, dh),
			}
			for i := 0; i < T; i++ {
				scores := ht.Scores[i]
				qi := ht.Q[i]
				for j := 0; j <= i; j++ {
					scores[j] = dot(qi, ht.K[j]) * scale
				}
				for j := i + 1; j < T; j++ {
					scores[j] = negInf
				}
				attn := ht.Attn[i]
				softmaxCausal(attn, scores, i+1)
				out := ht.Out[i]
				for j := 0; j <= i; j++ {
					p := attn[j]
					vj := ht.V[j]
					for x := 0; x < dh; x++ {
						out[x] += p * vj[x]
					}
				}
			}
			lt.Heads[h] = ht
			// Head h only touches the rows of Wo that line up with its own
			// slice of the concatenation, so AttnProj is exactly the sum of
			// the per-head projections.
			wo := lw.WO[h*dh*d : (h+1)*dh*d]
			lt.HeadProj[h] = matMulRows(ht.Out, wo, nil, d)
		}
		// Heads are independent, so they run concurrently when the attention
		// work is large enough to pay for the goroutines.
		workers := 1
		if work := T * T * dh * c.NHead * 2; work >= parallelWork {
			workers = runtime.GOMAXPROCS(0)
			if workers > c.NHead {
				workers = c.NHead
			}
		}
		if workers <= 1 {
			for h := 0; h < c.NHead; h++ {
				runHead(h)
			}
		} else {
			var wg sync.WaitGroup
			for h := 0; h < c.NHead; h++ {
				h := h
				wg.Add(1)
				go func() {
					defer wg.Done()
					runHead(h)
				}()
			}
			wg.Wait()
		}
		// Accumulating the per-head projections into AttnProj keeps the same
		// addition order as a full concat·Wo multiply; it is only T×d adds.
		for h := 0; h < c.NHead; h++ {
			for i := range lt.HeadProj[h] {
				row := lt.HeadProj[h][i]
				acc := lt.AttnProj[i]
				for j := range row {
					acc[j] += row[j]
				}
			}
		}
		lt.AfterAttn = newMatrix(T, d)
		for i := 0; i < T; i++ {
			addInto(lt.AfterAttn[i], lt.In[i], lt.AttnProj[i])
		}
		lt.Norm2 = layerNorm(lt.AfterAttn, lw.LN2G, lw.LN2B)
		lt.Hidden = matMulRows(lt.Norm2, lw.W1, nil, c.DFF)
		for i := range lt.Hidden {
			row := lt.Hidden[i]
			for j, v := range row {
				row[j] = gelu(v)
			}
		}
		lt.FFNOut = matMulRows(lt.Hidden, lw.W2, nil, d)
		lt.AfterFFN = newMatrix(T, d)
		for i := 0; i < T; i++ {
			addInto(lt.AfterFFN[i], lt.AfterAttn[i], lt.FFNOut[i])
		}

		tr.Layers = append(tr.Layers, lt)
		x = lt.AfterFFN
	}

	tr.FinalNorm = layerNorm(x, w.LNFG, w.LNFB)
	last := tr.FinalNorm[T-1]

	// Only the last position is projected here. Every earlier row of
	// tr.FinalNorm is a perfectly usable next-byte predictor — the model was
	// trained to predict from all of them — and the output stage now shows that
	// through Weights.Unembed. Forward still records one row because that is
	// what the app's own sampling path reads, and computing 64 × 256 dots on
	// every keystroke for a figure one stage uses would be work spent per
	// keystroke for a number nobody is looking at.
	tr.Logits = w.Unembed(last)
	return tr
}

// sliceCols returns copies of a column range in m. The copies keep each
// head's Q/K/V rows contiguous, which benchmarks noticeably faster than
// sharing the wider parent rows.
func sliceCols(m [][]float32, start, width int) [][]float32 {
	out := newMatrix(len(m), width)
	for i, row := range m {
		copy(out[i], row[start:start+width])
	}
	return out
}

// alignTokens recovers the printable text of each byte token by decoding the
// prompt and walking it byte by byte. Invalid bytes keep their raw byte token
// and Rune zero; valid runes around them still map to their own bytes, so one
// stray continuation byte from the sampler does not hide all the other runes.
func alignTokens(ids []int, tokens []Token) []Token {
	buf := make([]byte, len(ids))
	for i, id := range ids {
		buf[i] = byte(id)
	}
	runeIndex := 0
	for i := 0; i < len(buf); {
		r, n := utf8.DecodeRune(buf[i:])
		if r == utf8.RuneError && n == 1 {
			tokens[i].Rune = 0
			tokens[i].Start = runeIndex
			i++
			runeIndex++
			continue
		}
		for k := 0; k < n && i+k < len(tokens); k++ {
			tokens[i+k].Rune = r
			tokens[i+k].Start = runeIndex
		}
		i += n
		runeIndex++
	}
	return tokens
}
