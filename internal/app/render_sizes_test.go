package app

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
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

	// Row 0 of the attention matrix, where the causal mask leaves one key and every head's
	// entropy is exactly 0. The panel states that instead of drawing four empty bars, and no
	// other snapshot selects the first row.
	row0 := newTestApp(t)
	row0.Step = 3
	row0.SelToken = 0
	writeSnapshot(t, dir, "size-rowzero-1180.png", ui.Render(row0.View, 1180, 900, 1))

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

// TestRenderDocs refreshes the screenshots the README embeds.
//
// They are part of the repository rather than generated per run, so nothing updates them when the interface changes —
// and in round 229 all six were found to be showing an interface that no longer existed: the sidebar numbered stages
// from 0, there was no data-flow card and no misconception block, and the hero image predated the heatmap's axis
// labels. Rendering them by hand took a throwaway test; this makes it `just docs`.
//
// The four stage shots use the default window, and `window` uses the minimum the app supports, so the gallery answers
// both "what does this look like" and "does it still fit on a small screen".
func TestRenderDocs(t *testing.T) {
	if os.Getenv("RENDER_DOCS") == "" {
		t.Skip("set RENDER_DOCS=1 to refresh docs/")
	}
	dir := "../../docs"
	if d := os.Getenv("RENDER_DIR"); d != "" {
		dir = d
	}
	for _, s := range []struct {
		name string
		step int
		w, h int
	}{
		{"tokens", 0, 1500, 940},
		{"scores", 2, 1500, 940},
		{"residual", 4, 1500, 940},
		{"output", 6, 1500, 940},
		{"window", 0, 1180, 720},
		// The hero image is the guide's first of six: the softmax stage, where the heatmap and the
		// weighted-sum card are both on screen.
		{"attention", 3, 1500, 940},
	} {
		a := newTestApp(t)
		a.Step = s.step
		writeSnapshot(t, dir, s.name+".png", ui.Render(a.View, s.w, s.h, 1))
	}
}

// TestDocsImagesAreCurrent fails when docs/ has fallen behind the code.
//
// The images are repository assets, so nothing regenerates them when the interface changes — which is how all six came to show
// an interface that no longer existed, and how two of them went stale again within twenty rounds of being fixed. CONTRIBUTING
// says to run `just docs` after an interface or prose change; this makes forgetting it a test failure instead of something a
// reader notices.
//
// It renders the same six frames TestRenderDocs writes and compares them byte for byte. Rendering is deterministic here, so an
// exact match is the right bar; the cost is six frames, well under a second.
func TestDocsImagesAreCurrent(t *testing.T) {
	frames := []struct {
		name string
		step int
		w, h int
	}{
		{"tokens", 0, 1500, 940},
		{"scores", 2, 1500, 940},
		{"residual", 4, 1500, 940},
		{"output", 6, 1500, 940},
		{"window", 0, 1180, 720},
		{"attention", 3, 1500, 940},
	}
	for _, f := range frames {
		want, err := os.ReadFile(filepath.Join("..", "..", "docs", f.name+".png"))
		if err != nil {
			t.Fatalf("docs/%s.png is missing: %v", f.name, err)
		}
		var buf bytes.Buffer
		a := newTestApp(t)
		a.Step = f.step
		if err := png.Encode(&buf, ui.Render(a.View, f.w, f.h, 1)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(buf.Bytes(), want) {
			t.Fatalf("docs/%s.png no longer matches what the app renders; run `just docs` and commit the result "+
				"(step %d at %d×%d)", f.name, f.step+1, f.w, f.h)
		}
	}
}

// TestPerPositionCardIsVisibleWithoutScrolling is the guard for the mistake this
// round made: the output stage's per-position card was first appended after the
// sampling card, which put its header at y=1091 in a 940-DIP window — entirely
// below the fold, so no reader would ever see it, while every content test stayed
// green because they ask whether a card exists, not whether it is on screen.
//
// It stays a position check rather than a scrolling one because of what this card
// is: the stage's whole point is that the model predicts at every position, and a
// reader who has to scroll to discover that has already formed their impression
// from the single-row card above. The two other cards below it may scroll.
func TestPerPositionCardIsVisibleWithoutScrolling(t *testing.T) {
	// The two sizes the project renders and looks at: the minimum, and the default.
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}} {
		a := newTestApp(t)
		a.Step = 6
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		r, ok := tr.Find("每个位置都在预测")
		if !ok {
			t.Fatalf("%dx%d: the per-position card is missing from the output stage", size.w, size.h)
		}
		if r.Y < 0 || r.Y+r.H > float32(size.h) {
			t.Fatalf("%dx%d: the per-position card's header is at y=%.0f, outside a window "+
				"%d DIP tall — the stage's own point would be below the fold",
				size.w, size.h, r.Y, size.h)
		}
	}
}

// TestLogitLensCardIsReachable replaces an earlier version of this test that required
// the card to be above the fold.
//
// That rule was written when the ladder was the only analytical card, and it does not
// survive contact with a stage that has several: a stage cannot keep every card on the
// first screen, and demanding it pushes the stage's own figure off instead — which is
// what happened, measured: with the ladder and the layer-rewrite cards leading the
// residual stage, the residual-stream card itself began at y=1071 in a 720-DIP window.
// The invariant worth keeping is that the stage opens on its own figure
// (TestEveryStageOpensOnItsOwnFigure), and that every card is reachable by scrolling.
//
// This one keeps the narrower promise for the ladder specifically: it must not be
// lost, and the reader must be able to get to it.
func TestLogitLensCardIsReachable(t *testing.T) {
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
		a := newTestApp(t)
		a.Step = 4
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		if _, ok := tr.Find("每一层会说什么"); !ok {
			t.Fatalf("%dx%d: the logit lens card is missing from the residual stage", size.w, size.h)
		}
		// Scroll to the end and require the card to have come into view on the way.
		seen := false
		for i := 0; i < 40; i++ {
			r, ok := tr.Find("每一层会说什么")
			if ok && r.Y >= 0 && r.Y+r.H <= float32(size.h) {
				seen = true
				break
			}
			tr.Move(float32(size.w)/2, float32(size.h)/2)
			tr.Wheel(float32(size.w)/2, float32(size.h)/2, 0, 200)
			tr.Advance(2e9)
			tr.Frame()
		}
		if !seen {
			t.Fatalf("%dx%d: the logit lens card never came into view while scrolling the stage",
				size.w, size.h)
		}
	}
}

// TestEveryStageOpensOnItsOwnFigure is the contract that keeps the stages usable as
// they grow: whatever else a stage accumulates, the figure it is named for has to be on
// screen when the reader arrives, at the smallest supported window.
//
// This became worth pinning after several rounds of adding analytical cards. Measured,
// the residual stage reached 4.4 screens of scrolling at 1180×720 — long is acceptable
// for a tool whose whole method is "look at this number, then that one", but only if
// the first thing a reader sees is the subject. It regressed once: with the logit-lens
// and layer-rewrite cards placed at the top of the residual stage, the residual-stream
// card itself started at y=1071 in a 720-DIP window, so a stage named 残差流 opened on
// two explanation cards and never showed the residual stream.
//
// The entries name the first figure each stage is built around, not a card that merely
// appears: "加权求和" is on the Softmax stage but the matrix is what the stage is for.
func TestEveryStageOpensOnItsOwnFigure(t *testing.T) {
	primary := map[int]string{
		0: "提示词变成 token 序列",
		1: "token embedding",
		2: "分数矩阵",
		3: "注意力矩阵",
		4: "残差流",
		5: "升维 → GELU → 降维",
		6: "每个位置都在预测",
	}
	if len(primary) != len(lessons) {
		t.Fatalf("this test names %d stages, the app has %d", len(primary), len(lessons))
	}
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}} {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			want, ok := primary[step]
			if !ok {
				t.Fatalf("stage %d has no entry in this test", step+1)
			}
			r, found := tr.Find(want)
			if !found {
				t.Fatalf("%dx%d stage %d: the stage's figure %q is not rendered",
					size.w, size.h, step+1, want)
			}
			if r.Y < 0 || r.Y+r.H > float32(size.h) {
				t.Fatalf("%dx%d stage %d: the figure the stage is named for (%q) is at y=%.0f, "+
					"outside a window %d DIP tall — the stage opens on something else",
					size.w, size.h, step+1, want, r.Y, size.h)
			}
		}
	}
}
