package model

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// parityFixture mirrors tools/parity.py: every intermediate tensor PyTorch
// produced for one prompt, using the same rounded weights the binary embeds.
type parityFixture struct {
	Prompt         string        `json:"prompt"`
	IDs            []int         `json:"ids"`
	Embed          [][]float64   `json:"embed"`
	Pos            [][]float64   `json:"pos"`
	Input          [][]float64   `json:"input"`
	HeadsPerLayer  []parityLayer `json:"heads_per_layer"`
	FinalNorm      [][]float64   `json:"final_norm"`
	Logits         []float64     `json:"logits"`
	HeadsPerLayerN int           `json:"-"`
}

type parityLayer struct {
	In        [][]float64  `json:"in"`
	Norm1     [][]float64  `json:"norm1"`
	Heads     []parityHead `json:"heads"`
	AttnProj  [][]float64  `json:"attn_proj"`
	AfterAttn [][]float64  `json:"after_attn"`
	Norm2     [][]float64  `json:"norm2"`
	HiddenPre [][]float64  `json:"hidden_pre"`
	Hidden    [][]float64  `json:"hidden"`
	FFNOut    [][]float64  `json:"ffn_out"`
	AfterFFN  [][]float64  `json:"after_ffn"`
}

type parityHead struct {
	Q      [][]float64  `json:"q"`
	K      [][]float64  `json:"k"`
	V      [][]float64  `json:"v"`
	Scores [][]*float64 `json:"scores"`
	Attn   [][]float64  `json:"attn"`
	Out    [][]float64  `json:"out"`
}

// maxRel reports the largest absolute difference, and where it happened.
func diffMatrix(name string, got [][]float32, want [][]float64, tol float64) (float64, string) {
	if len(got) != len(want) {
		return math.Inf(1), fmt.Sprintf("%s: %d rows, want %d", name, len(got), len(want))
	}
	var worst float64
	var where string
	for i := range want {
		if len(got[i]) != len(want[i]) {
			return math.Inf(1), fmt.Sprintf("%s row %d: %d cols, want %d", name, i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			d := math.Abs(float64(got[i][j]) - want[i][j])
			if d > worst {
				worst, where = d, fmt.Sprintf("%s[%d][%d]", name, i, j)
			}
		}
	}
	return worst, where
}

func loadParity(t *testing.T) *parityFixture {
	t.Helper()
	path := filepath.Join("testdata", "parity.json.gz")
	fh, err := os.Open(path)
	if err != nil {
		// This used to skip, which meant the strongest correctness check in the
		// repository could disappear without a trace: delete or omit the fixture and the
		// suite still reports PASS, with the only remaining coverage of the per-tensor
		// layout being whatever the render tests happen to notice. The fixture is
		// tracked, so its absence is a broken checkout, not a configuration choice.
		t.Fatalf("parity fixture missing: %v\n"+
			"it is tracked in git; restore it with `git checkout internal/model/testdata`,"+
			" or regenerate it with `python3 tools/parity.py`", err)
	}
	defer func() {
		if err := fh.Close(); err != nil {
			t.Errorf("close parity fixture: %v", err)
		}
	}()
	zr, err := gzip.NewReader(fh)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer func() {
		if err := zr.Close(); err != nil {
			t.Errorf("close parity reader: %v", err)
		}
	}()
	data, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var f parityFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return &f
}

// TestParityEveryTensor re-derives each stage in Go and compares it with the
// PyTorch reference tensor by tensor. The point is to catch mistakes that the
// final logits alone could hide: head slicing order, the concat layout before
// Wo, the GELU variant, and LayerNorm epsilon.
func TestParityEveryTensor(t *testing.T) {
	w, err := Bundled()
	if err != nil {
		t.Fatalf("bundled weights: %v", err)
	}
	f := loadParity(t)
	if len(w.Layers) != len(f.HeadsPerLayer) {
		t.Fatalf("fixture has %d layers, weights have %d", len(f.HeadsPerLayer), len(w.Layers))
	}
	tr := Forward(w, f.IDs)
	const tol = 2e-4

	check := func(name string, got [][]float32, want [][]float64) {
		t.Helper()
		d, where := diffMatrix(name, got, want, tol)
		if d > tol {
			t.Errorf("%s: max diff %.6g at %s (tol %g)", name, d, where, tol)
		} else if testing.Verbose() {
			t.Logf("%-28s max diff %.3e", name, d)
		}
	}

	check("embed", tr.Embed, f.Embed)
	check("pos", tr.Pos, f.Pos)
	check("input", tr.Input, f.Input)

	for li, want := range f.HeadsPerLayer {
		got := tr.Layers[li]
		p := fmt.Sprintf("L%d.", li)
		check(p+"in", got.In, want.In)
		check(p+"norm1", got.Norm1, want.Norm1)
		check(p+"attn_proj", got.AttnProj, want.AttnProj)
		check(p+"after_attn", got.AfterAttn, want.AfterAttn)
		check(p+"norm2", got.Norm2, want.Norm2)
		check(p+"hidden", got.Hidden, want.Hidden)
		check(p+"ffn_out", got.FFNOut, want.FFNOut)
		check(p+"after_ffn", got.AfterFFN, want.AfterFFN)

		for hi, wh := range want.Heads {
			hp := fmt.Sprintf("%sH%d.", p, hi)
			gh := got.Heads[hi]
			check(hp+"q", gh.Q, wh.Q)
			check(hp+"k", gh.K, wh.K)
			check(hp+"v", gh.V, wh.V)
			check(hp+"attn", gh.Attn, wh.Attn)
			check(hp+"out", gh.Out, wh.Out)
			// Scores carry -Inf above the diagonal; compare them separately.
			for i := range wh.Scores {
				for j := range wh.Scores[i] {
					want := wh.Scores[i][j]
					gs := float64(gh.Scores[i][j])
					if want == nil {
						if !math.IsInf(gs, -1) {
							t.Errorf("%s scores[%d][%d] = %v, want -Inf", hp, i, j, gs)
						}
						continue
					}
					if d := math.Abs(gs - *want); d > tol {
						t.Errorf("%s scores[%d][%d] diff %.6g", hp, i, j, d)
					}
				}
			}
		}
	}

	check("final_norm", tr.FinalNorm, f.FinalNorm)

	var worst float64
	for i := range f.Logits {
		if d := math.Abs(float64(tr.Logits[i]) - f.Logits[i]); d > worst {
			worst = d
		}
	}
	if worst > 2e-3 {
		t.Errorf("logits: max diff %.6g", worst)
	}
	t.Logf("all tensors match; worst logit diff %.3e", worst)
}
