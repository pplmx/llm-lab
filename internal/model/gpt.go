package model

import (
	"math"
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

	tr := &Trace{Text: DecodeBytes(ids), Tokens: tokens, Config: c, Logits: make([]float32, c.Vocab)}
	if T == 0 {
		tr.Embed, tr.Pos, tr.Input, tr.FinalNorm = [][]float32{}, [][]float32{}, [][]float32{}, [][]float32{}
		tr.Layers = make([]LayerTrace, c.NLayer)
		return tr
	}
	tr.Embed = make([][]float32, T)
	tr.Pos = make([][]float32, T)
	tr.Input = make([][]float32, T)
	for i, id := range ids {
		b := int(byte(id))
		emb := append([]float32(nil), w.TokEmbed[b*d:(b+1)*d]...)
		pos := append([]float32(nil), w.PosEmbed[i*d:(i+1)*d]...)
		tr.Embed[i] = emb
		tr.Pos[i] = pos
		tr.Input[i] = addRow(emb, pos)
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
		lt.AttnProj = make([][]float32, T)
		for i := range lt.AttnProj {
			lt.AttnProj[i] = make([]float32, d)
		}
		scale := float32(1 / math.Sqrt(float64(dh)))
		for h := 0; h < c.NHead; h++ {
			ht := HeadTrace{
				Q:      sliceCols(q, h*dh, dh),
				K:      sliceCols(k, h*dh, dh),
				V:      sliceCols(v, h*dh, dh),
				Scores: make([][]float32, T),
				Attn:   make([][]float32, T),
				Out:    make([][]float32, T),
			}
			for i := 0; i < T; i++ {
				scores := make([]float32, T)
				for j := 0; j < T; j++ {
					if j > i { // causal mask
						scores[j] = float32(math.Inf(-1))
						continue
					}
					scores[j] = dot(ht.Q[i], ht.K[j]) * scale
				}
				attn := append([]float32(nil), scores...)
				softmaxInPlace(attn)
				ht.Scores[i] = scores
				ht.Attn[i] = attn

				out := make([]float32, dh)
				for j := 0; j < T; j++ {
					w := attn[j]
					if w == 0 {
						continue
					}
					vj := ht.V[j]
					for x := 0; x < dh; x++ {
						out[x] += w * vj[x]
					}
				}
				ht.Out[i] = out
			}
			lt.Heads[h] = ht
			// Head h only touches the rows of Wo that line up with its own
			// slice of the concatenation, so AttnProj is exactly the sum of
			// the per-head projections. Accumulating those avoids computing
			// the same d×d product a second time while keeping the same
			// addition order as a full concat·Wo multiply.
			wo := lw.WO[h*dh*d : (h+1)*dh*d]
			lt.HeadProj[h] = matMulRows(ht.Out, wo, nil, d)
			for i := range lt.HeadProj[h] {
				row := lt.HeadProj[h][i]
				acc := lt.AttnProj[i]
				for j := range row {
					acc[j] += row[j]
				}
			}
		}
		lt.AfterAttn = make([][]float32, T)
		for i := 0; i < T; i++ {
			lt.AfterAttn[i] = addRow(lt.In[i], lt.AttnProj[i])
		}
		lt.Norm2 = layerNorm(lt.AfterAttn, lw.LN2G, lw.LN2B)
		lt.Hidden = matMulRows(lt.Norm2, lw.W1, nil, c.DFF)
		for i := range lt.Hidden {
			for j := range lt.Hidden[i] {
				lt.Hidden[i][j] = gelu(lt.Hidden[i][j])
			}
		}
		lt.FFNOut = matMulRows(lt.Hidden, lw.W2, nil, d)
		lt.AfterFFN = make([][]float32, T)
		for i := 0; i < T; i++ {
			lt.AfterFFN[i] = addRow(lt.AfterAttn[i], lt.FFNOut[i])
		}

		tr.Layers = append(tr.Layers, lt)
		x = lt.AfterFFN
	}

	tr.FinalNorm = layerNorm(x, w.LNFG, w.LNFB)
	last := tr.FinalNorm[T-1]

	// The unembedding is the token embedding matrix, tied as GPT-2 does.
	tr.Logits = make([]float32, c.Vocab)
	for id := 0; id < c.Vocab; id++ {
		tr.Logits[id] = dot(last, w.TokEmbed[id*d:(id+1)*d])
	}
	return tr
}

// sliceCols returns copies of a column range in m. The copies keep each
// head's Q/K/V rows contiguous, which benchmarks noticeably faster than
// sharing the wider parent rows.
func sliceCols(m [][]float32, start, width int) [][]float32 {
	out := make([][]float32, len(m))
	for i, row := range m {
		out[i] = append([]float32(nil), row[start:start+width]...)
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
	s := string(buf)
	if utf8.ValidString(s) {
		// Fast path for the overwhelmingly common valid prompt: range over
		// the string gives the rune and byte offset directly.
		runeAt := make([]rune, 0, len(ids))
		for _, r := range s {
			runeAt = append(runeAt, r)
		}
		byteToRune := map[int]int{}
		pos := 0
		runeIdx := 0
		for _, r := range s {
			n := len(string(r))
			for k := 0; k < n; k++ {
				byteToRune[pos+k] = runeIdx
			}
			pos += n
			runeIdx++
		}
		for i := range tokens {
			if r, ok := byteToRune[i]; ok && r < len(runeAt) {
				tokens[i].Rune = runeAt[r]
				tokens[i].Start = r
			}
		}
		return tokens
	}
	runeIndex := 0
	for i := 0; i < len(buf); {
		r, n := utf8.DecodeRune(buf[i:])
		if r == utf8.RuneError && n == 1 {
			if i < len(tokens) {
				tokens[i].Rune = 0
				tokens[i].Start = runeIndex
			}
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
