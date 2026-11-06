package app

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"strings"
	"testing"

	"llmlab/internal/model"

	"github.com/egoist/mygo/ui"
)

// TestRenderSizes draws the busiest stages at the window's minimum and at a
// large size, so overflow can be spotted without launching the app.
func TestRenderSizes(t *testing.T) {
	if os.Getenv("RENDER") == "" {
		t.Skip("set RENDER=1 to write snapshots")
	}
	dir := os.Getenv("RENDER_DIR")
	if dir == "" {
		dir = "/tmp"
	}
	sizes := []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}}
	a := newTestApp(t)
	for _, s := range sizes {
		for _, step := range []int{0, 3, 5, 6} {
			a.Step = step
			writeSnapshot(t, dir, fmt.Sprintf("size-%dx%d-step%d.png", s.w, s.h, step),
				ui.Render(a.View, s.w, s.h, 1))
		}
	}

	// The layer comparison row: only layer 1 has a previous layer to differ
	// from, and the default snapshots above all sit on layer 0, so this state
	// would otherwise be covered by no frame at all.
	a.Layer = 1
	a.Step = 3
	writeSnapshot(t, dir, "size-layer1-1180.png", ui.Render(a.View, 1180, 720, 1))
	a.Layer = 0

	// Long prompt: chips and the heatmap at the 64-token ceiling.
	a.Prompt = strings.Repeat("attention is a weighted sum ", 3)
	a.recompute()
	a.Step = 3
	a.SelToken = len(a.Trace.Tokens) - 1
	for _, w := range []int{1180, 1500} {
		writeSnapshot(t, dir, fmt.Sprintf("size-longprompt-%d.png", w), ui.Render(a.View, w, 940, 1))
	}
	t.Logf("tokens=%d", len(a.Trace.Tokens))

	// Empty prompt: the strip's empty state, the recovery card on a stage that
	// is not the tokenising one, and the inspector's empty copy.
	a.Prompt = ""
	a.recompute()
	for _, step := range []int{0, 4} {
		a.Step = step
		writeSnapshot(t, dir, fmt.Sprintf("size-emptyprompt-1180-step%d.png", step),
			ui.Render(a.View, 1180, 720, 1))
	}

	// Truncation: the tokenising card states both the kept length and the
	// original, which is the only place the drop is visible.
	a.Prompt = strings.Repeat("注意力", 30)
	a.recompute()
	a.Step = 0
	writeSnapshot(t, dir, "size-truncated-1180.png", ui.Render(a.View, 1180, 720, 1))

	// Untrained weights: the banner that stops a reader from trusting a random
	// model's perfectly plausible triangle.
	b := New(model.Random(model.DefaultConfig, 42))
	b.Step = 3
	writeSnapshot(t, dir, "size-untrained-1180.png", ui.Render(b.View, 1180, 720, 1))
}

// writeSnapshot encodes img to dir/name, failing the test on either error.
func writeSnapshot(t *testing.T, dir, name string, img *image.RGBA) {
	t.Helper()
	f, err := os.Create(dir + "/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", name)
}
