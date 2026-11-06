package model

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// ByteVocab is the vocabulary size of a byte-level tokenizer. Prompts produce
// ids in 0..ByteVocab-1 only, so a weights file with a different vocabulary
// cannot be used by this app without changing the tokenizer and sampler.
const ByteVocab = 256

// Config is the shape of the toy model. It is small enough for a full forward
// pass to run in a few milliseconds on one core, so every keystroke can
// recompute the trace.
type Config struct {
	Vocab  int `json:"vocab"`
	DModel int `json:"d_model"`
	NLayer int `json:"n_layer"`
	NHead  int `json:"n_head"`
	DFF    int `json:"d_ff"`
	Ctx    int `json:"ctx"`
}

// DefaultConfig is the configuration the bundled weights use.
var DefaultConfig = Config{Vocab: ByteVocab, DModel: 96, NLayer: 2, NHead: 4, DFF: 384, Ctx: 64}

func (c Config) DHead() int {
	if c.NHead <= 0 {
		return 0
	}
	return c.DModel / c.NHead
}

// LayerWeights holds one transformer block. Linear layers carry no bias, as in
// LLaMA, which keeps the visualizations readable.
type LayerWeights struct {
	LN1G []float32 `json:"ln1_g"`
	LN1B []float32 `json:"ln1_b"`
	WQ   []float32 `json:"wq"` // d_model × d_model
	WK   []float32 `json:"wk"`
	WV   []float32 `json:"wv"`
	WO   []float32 `json:"wo"`
	LN2G []float32 `json:"ln2_g"`
	LN2B []float32 `json:"ln2_b"`
	W1   []float32 `json:"w1"` // d_model × d_ff
	W2   []float32 `json:"w2"` // d_ff × d_model
}

// Weights is everything the model needs, plus the provenance shown in the UI.
type Weights struct {
	Config   Config         `json:"config"`
	TokEmbed []float32      `json:"tok_embed"` // vocab × d_model
	PosEmbed []float32      `json:"pos_embed"` // ctx × d_model
	Layers   []LayerWeights `json:"layers"`
	LNFG     []float32      `json:"lnf_g"`
	LNFB     []float32      `json:"lnf_b"`

	Reference *Reference `json:"reference,omitempty"`
	Corpus    string     `json:"corpus,omitempty"`
	Steps     int        `json:"steps,omitempty"`
	Loss      float32    `json:"loss,omitempty"`

	Source string `json:"-"` // where these weights came from, for the UI
}

// Reference is a forward pass recorded by the training script, so the Go
// implementation can be checked against PyTorch.
type Reference struct {
	Prompt string    `json:"prompt"`
	IDs    []int     `json:"ids"`
	Logits []float32 `json:"logits"`
}

//go:embed weights.json
var bundled []byte

// Bundled returns the weights compiled into the binary.
func Bundled() (*Weights, error) {
	return Parse(bundled, "bundled")
}

// Load reads weights from a JSON file at path.
func Load(path string) (*Weights, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data, path)
}

// Parse decodes and validates weights.
func Parse(data []byte, source string) (*Weights, error) {
	var w Weights
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("parse weights: %w", err)
	}
	w.Source = source
	if err := w.validate(); err != nil {
		return nil, err
	}
	return &w, nil
}

func (w *Weights) validate() error {
	c := w.Config
	if c.Vocab != ByteVocab {
		return fmt.Errorf("weights: vocab is %d, want %d for byte-level tokens", c.Vocab, ByteVocab)
	}
	if c.DModel <= 0 {
		return fmt.Errorf("weights: d_model must be positive, got %d", c.DModel)
	}
	if c.NLayer <= 0 {
		return fmt.Errorf("weights: n_layer must be positive, got %d", c.NLayer)
	}
	if c.NHead <= 0 {
		return fmt.Errorf("weights: n_head must be positive, got %d", c.NHead)
	}
	if c.DFF <= 0 {
		return fmt.Errorf("weights: d_ff must be positive, got %d", c.DFF)
	}
	if c.Ctx <= 0 {
		return fmt.Errorf("weights: ctx must be positive, got %d", c.Ctx)
	}
	if c.DModel%c.NHead != 0 {
		return fmt.Errorf("weights: d_model %d is not divisible by n_head %d", c.DModel, c.NHead)
	}
	if len(w.LNFG) != c.DModel {
		return fmt.Errorf("weights: lnf_g has %d values, want %d", len(w.LNFG), c.DModel)
	}
	if len(w.LNFB) != c.DModel {
		return fmt.Errorf("weights: lnf_b has %d values, want %d", len(w.LNFB), c.DModel)
	}
	if len(w.TokEmbed) != c.Vocab*c.DModel {
		return fmt.Errorf("weights: tok_embed has %d values, want %d", len(w.TokEmbed), c.Vocab*c.DModel)
	}
	if len(w.PosEmbed) != c.Ctx*c.DModel {
		return fmt.Errorf("weights: pos_embed has %d values, want %d", len(w.PosEmbed), c.Ctx*c.DModel)
	}
	if len(w.Layers) != c.NLayer {
		return fmt.Errorf("weights: %d layers, want %d", len(w.Layers), c.NLayer)
	}
	for i, l := range w.Layers {
		want := map[string]struct {
			n    int
			data []float32
		}{
			"ln1_g": {c.DModel, l.LN1G}, "ln1_b": {c.DModel, l.LN1B},
			"wq": {c.DModel * c.DModel, l.WQ}, "wk": {c.DModel * c.DModel, l.WK},
			"wv": {c.DModel * c.DModel, l.WV}, "wo": {c.DModel * c.DModel, l.WO},
			"ln2_g": {c.DModel, l.LN2G}, "ln2_b": {c.DModel, l.LN2B},
			"w1": {c.DModel * c.DFF, l.W1}, "w2": {c.DFF * c.DModel, l.W2},
		}
		for name, e := range want {
			if len(e.data) != e.n {
				return fmt.Errorf("weights: layer %d %s has %d values, want %d", i, name, len(e.data), e.n)
			}
		}
	}
	return nil
}

// Random returns deterministic random weights, used when no trained weights
// are available. The forward pass is meaningful; the text it produces is not.
func Random(c Config, seed int64) *Weights {
	rng := rand.New(rand.NewSource(seed))
	w := &Weights{Config: c, Source: fmt.Sprintf("random(seed=%d)", seed)}
	w.TokEmbed = normalSlice(rng, c.Vocab*c.DModel, 0.02)
	w.PosEmbed = normalSlice(rng, c.Ctx*c.DModel, 0.01)
	w.LNFG = ones(c.DModel)
	w.LNFB = make([]float32, c.DModel)
	for i := 0; i < c.NLayer; i++ {
		l := LayerWeights{
			LN1G: ones(c.DModel), LN1B: make([]float32, c.DModel),
			LN2G: ones(c.DModel), LN2B: make([]float32, c.DModel),
			WQ: normalSlice(rng, c.DModel*c.DModel, 0.02),
			WK: normalSlice(rng, c.DModel*c.DModel, 0.02),
			WV: normalSlice(rng, c.DModel*c.DModel, 0.02),
			WO: normalSlice(rng, c.DModel*c.DModel, 0.02),
			W1: normalSlice(rng, c.DModel*c.DFF, 0.02),
			W2: normalSlice(rng, c.DFF*c.DModel, 0.02),
		}
		w.Layers = append(w.Layers, l)
	}
	return w
}

func ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func normalSlice(rng *rand.Rand, n int, std float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * std)
	}
	return v
}

// Probabilities applies the temperature to the logits and normalizes them.
func (t *Trace) Probabilities(temperature float64) []float32 {
	return softmaxCopy(t.Logits, temperature)
}

// TopK returns the k token ids with the highest probability, most likely
// first. A non-positive k returns no ids.
func TopK(probs []float32, k int) []int {
	if k <= 0 {
		return nil
	}
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	// Partial selection keeps this cheap for a 256-wide vocabulary.
	for i := 0; i < k && i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if probs[idx[j]] > probs[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// Entropy is the Shannon entropy of a distribution, in nats.
func Entropy(probs []float32) float32 {
	var h float64
	for _, p := range probs {
		if p > 1e-9 {
			h -= float64(p) * math.Log(float64(p))
		}
	}
	return float32(h)
}
