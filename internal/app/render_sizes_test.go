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
	// Every stage, not a sample of them: the sizes are the window's own range, and
	// "no overflow from 1180×720 to 1920×1080" is a claim about all seven. Stages 2,
	// 3 and 5 were skipped for a long time — stage 5 is the one with six vector
	// strips stacked, so it is the tallest card and the likeliest to overflow, and it
	// had no frame at the smallest window at all.
	for _, s := range sizes {
		for _, step := range []int{0, 1, 2, 3, 4, 5, 6} {
			a.Step = step
			writeSnapshot(t, dir, fmt.Sprintf("size-%dx%d-step%d.png", s.w, s.h, step),
				ui.Render(a.View, s.w, s.h, 1))
		}
	}

	// The LayerNorm histograms: the one card whose two charts share an axis, and
	// the stage-4 frame at 1920 is the only size that shows it above the fold.
	a.Step = 4
	a.SelToken = 10
	writeSnapshot(t, dir, "size-1920x1080-step5-layernorm.png", ui.Render(a.View, 1920, 1080, 1))
	a.Step = 0
	a.SelToken = len(a.Trace.Tokens) - 1

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
	// Steps 2 and 3 are the matrix stages: with no tokens they must not advertise the
	// cell keys, and nothing else renders them empty.
	for _, step := range []int{0, 2, 4} {
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

	// The same random weights on the LayerNorm stage, where the untrained case
	// changes what the prose says: γ is 1 and β is 0, so the notes must not claim
	// they were learned. Nothing else in the snapshot set reaches that branch.
	b.Step = 4
	writeSnapshot(t, dir, "size-untrained-1180-layernorm.png", ui.Render(b.View, 1180, 720, 1))

	// A head whose scores are all negative, which is what a two-token prompt gives: the
	// legend must list only the blue swatch, and the card has to say why an all-negative
	// matrix is ordinary. No other snapshot reaches that branch — every bundled prompt has
	// both signs in every head.
	neg := newTestApp(t)
	neg.Prompt = "hi"
	neg.recompute()
	neg.Step = 2
	writeSnapshot(t, dir, "size-allnegative-1180.png", ui.Render(neg.View, 1180, 900, 1))

	// A one-token prompt, which is the only state where step 2's position table has nothing to
	// compare: the heading has to state the rule instead of claiming variation, and the note has
	// to stop promising four rows. No other snapshot has fewer than two tokens.
	one := newTestApp(t)
	one.Prompt = "a"
	one.recompute()
	one.Step = 1
	writeSnapshot(t, dir, "size-onetoken-1180.png", ui.Render(one.View, 1180, 900, 1))

	// The other way to reach an untrained model: -weights pointed at a file with no training
	// metadata. untrainedWhy gives that case different advice — it must not claim the bundled
	// file was missing — and no snapshot reached it. Source is display-only (json:"-"), so
	// setting it here reproduces exactly the state main.go builds from a custom file.
	custom := model.Random(model.DefaultConfig, 42)
	custom.Source = "/tmp/custom-weights.json"
	c := New(custom)
	c.Step = 0
	writeSnapshot(t, dir, "size-untrained-custom-1180.png", ui.Render(c.View, 1180, 720, 1))
}

// writeSnapshot encodes img to dir/name, failing the test on either error.
func writeSnapshot(t *testing.T, dir, name string, img *image.RGBA) {
	t.Helper()
	// Create the output directory rather than requiring it to exist. The point of
	// RENDER_DIR is to keep a "before" and an "after" set side by side, and a
	// fresh path is the obvious way to do that; failing with "open
	// /tmp/after/x.png: no such file or directory" reads as a broken test rather
	// than a missing directory.
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("create snapshot directory %s: %v", dir, err)
	}
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
