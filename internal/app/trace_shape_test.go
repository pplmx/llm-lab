package app

import (
	"testing"
)

// The README's "七个阶段与代码的对应关系" table names the exact field of
// model.Trace that each stage reads and writes. These tests check those fields
// hold what the table says, so the table cannot drift from the model without a
// failure here.
//
// They are shape and invariant checks rather than numeric ones: the parity of
// the numbers themselves is TestParityEveryTensor's job in internal/model.
func TestTraceShapesMatchTheStageTable(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	c := tr.Config
	n := len(tr.Tokens)
	if n == 0 {
		t.Fatal("no tokens")
	}

	// Stage 2: Input is Embed + Pos, all three T x d_model.
	for _, m := range [][][]float32{tr.Embed, tr.Pos, tr.Input} {
		if len(m) != n || len(m[0]) != c.DModel {
			t.Fatalf("embedding matrix is %dx%d, want %dx%d", len(m), len(m[0]), n, c.DModel)
		}
	}
	for i := 0; i < n; i++ {
		for d := 0; d < c.DModel; d++ {
			if tr.Input[i][d] != tr.Embed[i][d]+tr.Pos[i][d] {
				t.Fatalf("Input != Embed+Pos at [%d][%d]", i, d)
			}
		}
	}

	// Stage 3: Q/K/V are T x d_head; Scores is T x T with the causal mask at -Inf.
	l := tr.Layers[0]
	h := l.Heads[0]
	if len(h.Q) != n || len(h.Q[0]) != c.DHead() {
		t.Fatalf("Q is %dx%d, want %dx%d", len(h.Q), len(h.Q[0]), n, c.DHead())
	}
	if len(h.Scores) != n || len(h.Scores[0]) != n {
		t.Fatalf("Scores is %dx%d, want %dx%d", len(h.Scores), len(h.Scores[0]), n, n)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if h.Scores[i][j] > -1e30 {
				t.Fatalf("Scores[%d][%d] = %v, want -Inf above the diagonal", i, j, h.Scores[i][j])
			}
		}
	}

	// Stage 4: Attn rows sum to 1, Out is T x d_head.
	for i := 0; i < n; i++ {
		var s float32
		for j := 0; j <= i; j++ {
			s += h.Attn[i][j]
		}
		if s < 0.999 || s > 1.001 {
			t.Fatalf("Attn row %d sums to %v", i, s)
		}
	}
	if len(h.Out) != n || len(h.Out[0]) != c.DHead() {
		t.Fatalf("Out is %dx%d, want %dx%d", len(h.Out), len(h.Out[0]), n, c.DHead())
	}

	// Stage 5: the two residual adds, and AttnProj as the sum of the heads.
	for d := 0; d < c.DModel; d++ {
		if l.AfterAttn[0][d] != l.In[0][d]+l.AttnProj[0][d] {
			t.Fatalf("AfterAttn != In + AttnProj at %d", d)
		}
		if l.AfterFFN[0][d] != l.AfterAttn[0][d]+l.FFNOut[0][d] {
			t.Fatalf("AfterFFN != AfterAttn + FFNOut at %d", d)
		}
		var sum float32
		for hi := range l.HeadProj {
			sum += l.HeadProj[hi][0][d]
		}
		if sum != l.AttnProj[0][d] {
			t.Fatalf("AttnProj != the sum of HeadProj at %d", d)
		}
	}

	// Stage 6: Hidden is d_ff wide.
	if len(l.Hidden) != n || len(l.Hidden[0]) != c.DFF {
		t.Fatalf("Hidden is %dx%d, want %dx%d", len(l.Hidden), len(l.Hidden[0]), n, c.DFF)
	}

	// Stage 7: FinalNorm is T x d_model and Logits is a single vocab-wide row.
	if len(tr.FinalNorm) != n || len(tr.FinalNorm[0]) != c.DModel {
		t.Fatalf("FinalNorm is %dx%d", len(tr.FinalNorm), len(tr.FinalNorm[0]))
	}
	if len(tr.Logits) != c.Vocab {
		t.Fatalf("Logits is %d wide, want vocab %d", len(tr.Logits), c.Vocab)
	}
}
