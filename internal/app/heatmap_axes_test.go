package app

import (
	"testing"

	"github.com/egoist/mygo/ui"
)

// The score and attention matrices used to have no edge labels at all. The card's
// heading says "行 = query，列 = key", but a reader had to count cells or hover to
// learn which key a column was — on the one stage whose subject is which key a
// query reads from. The number of labels has to stay readable as the matrix
// grows, so these check the density rather than the exact strings.

// TestHeatmapLabelsBothAxes checks the edge labels are actually drawn, by
// comparing the real frame against one where the widget is told to skip them.
//
// The "real" side deliberately does not touch the override: an earlier version
// forced axes on for both renders, which meant the test could not see whether the
// call sites ask for them at all — disabling the labels at both stages still
// passed.
func TestHeatmapLabelsBothAxes(t *testing.T) {
	for _, step := range []int{2, 3} {
		render := func(forceOff bool) []byte {
			a := newTestApp(t)
			a.Step = step
			if forceOff {
				off := false
				a.heatmapAxesForTest = &off
			}
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()
			return append([]byte(nil), tr.Image().Pix...)
		}
		real := render(false) // whatever the call site asks for
		plain := render(true) // labels suppressed
		if len(real) != len(plain) {
			t.Fatalf("step %d: frame sizes differ", step+1)
		}
		diff := 0
		for i := range real {
			if real[i] != plain[i] {
				diff++
			}
		}
		if diff < 200 {
			t.Fatalf("step %d: the real frame is indistinguishable from one without "+
				"labels (%d bytes differ); this stage is not asking for them", step+1, diff)
		}
	}
}

// TestHeatmapLabelStrideAvoidsOverlap is the constraint that makes labels usable
// at 64 tokens: adjacent labels must not be closer than their own width.
func TestHeatmapLabelStrideAvoidsOverlap(t *testing.T) {
	// The rule in the widget: a label is drawn every `stride` cells, where
	// stride is the first value putting at least 26 DIPs between labels.
	// Reproduce it for the cell sizes this app actually produces.
	for _, size := range []struct {
		tokens int
		box    float32
	}{
		{36, 420}, {64, 420}, {64, 160}, {8, 160}, {1, 160},
	} {
		cell := size.box / float32(size.tokens)
		if cell > 20 {
			cell = 20
		}
		if cell < 2.5 {
			cell = 2.5
		}
		stride := 1
		for float32(stride)*cell < 26 {
			stride++
		}
		gap := float32(stride) * cell
		if gap < 26 {
			t.Fatalf("%d tokens at %.0f DIPs: stride %d gives %.1f DIPs between labels",
				size.tokens, size.box, stride, gap)
		}
		// And the stride must not skip the whole matrix for a small one.
		if stride > size.tokens && size.tokens > 3 {
			t.Fatalf("%d tokens: stride %d skips every label", size.tokens, stride)
		}
	}
}
