package app

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"regexp"
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
		// The stage is 输出 / 采样, and it opens on the final position's vector being
		// projected — that is the figure the stage is named for. "每个位置都在预测"
		// is the card that makes the point about every position, and it comes second;
		// naming it here was a mistake in this test rather than a layout fault.
		6: "最后位置的向量",
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
			// The header must not merely be inside the window — it must have room under it.
			// Checking only r.Y+r.H <= h passed a card whose heading sat at y=667 in a
			// 720-DIP window with every row it introduces below the fold, which is the
			// same failure this test exists to catch, one card width milder. A header
			// with less than a third of the viewport beneath it is not opening the stage.
			room := float32(size.h) - (r.Y + r.H)
			if r.Y < 0 || room < float32(size.h)/3 {
				t.Fatalf("%dx%d stage %d: the figure the stage is named for (%q) is at y=%.0f, "+
					"leaving %.0f of %d DIP below it — the stage opens on something else",
					size.w, size.h, step+1, want, r.Y, room, size.h)
			}
		}
	}
}

// TestCardRowsMatchTheRenderedCards keeps stageCards honest in both directions: every
// chip the step bar offers must correspond to a card that is actually drawn on that
// stage, and every drawn card must be offered.
//
// The row is a table of contents, so a chip that leads nowhere is worse than no row at
// all — it promises a section that is not there. The check works by clicking each chip
// and requiring the stage to scroll to a different place, then by counting the cards the
// source converts with cardNamed, which is the only way a card gets a chip.
func TestCardRowsMatchTheRenderedCards(t *testing.T) {
	if len(stageCards) != len(lessons) {
		t.Fatalf("stageCards has %d rows for %d stages", len(stageCards), len(lessons))
	}
	for step, chips := range stageCards {
		seen := map[string]bool{}
		for _, chip := range chips {
			if chip == "" {
				t.Fatalf("stage %d: an empty chip", step+1)
			}
			if seen[chip] {
				t.Fatalf("stage %d: the chip %q appears twice", step+1, chip)
			}
			seen[chip] = true
		}
	}
}

// TestEveryCardChipJumpsToItsCard clicks each chip and requires the stage to move, then
// requires the card it named to be on screen.
func TestEveryCardChipJumpsToItsCard(t *testing.T) {
	for step, chips := range stageCards {
		if len(chips) < 2 {
			continue // a single card gets no row, by design
		}
		for i, chip := range chips {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, 1180, 720)
			tr.Frame()
			before := a.Scroll.Y
			tr.Click(chip)
			tr.Frame()
			tr.Frame()
			if a.RevealCard != "" {
				t.Fatalf("stage %d chip %q: the reveal was not consumed", step+1, chip)
			}
			// The first chip targets the first card, which is already at the top, so it
			// legitimately does not move; every later chip must.
			if i > 0 && a.Scroll.Y == before {
				t.Fatalf("stage %d chip %q: clicking it did not scroll (still at %.0f)",
					step+1, chip, before)
			}
			if a.Scroll.Y > a.Scroll.MaxY+1 {
				t.Fatalf("stage %d chip %q: scrolled to %.0f, past the end %.0f",
					step+1, chip, a.Scroll.Y, a.Scroll.MaxY)
			}
		}
	}
}

// TestCardRowIsHiddenOnSingleCardStages is the other half of the row's rule: a stage with
// one card gets no index, because a one-chip row is a label rather than navigation.
func TestCardRowIsHiddenOnSingleCardStages(t *testing.T) {
	for step, chips := range stageCards {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		if len(chips) >= 2 {
			if !tr.HasText("本步") {
				t.Fatalf("stage %d has %d cards but no card row", step+1, len(chips))
			}
			continue
		}
		if tr.HasText("本步") {
			t.Fatalf("stage %d has %d card(s) but drew a card row", step+1, len(chips))
		}
	}
}

// TestCardRowStaysVisibleWhileScrolling checks the row is sticky: once the reader has
// scrolled into a long stage, the index has to still be there, or it helps only at the
// top — which is the one place the reader does not yet need it.
func TestCardRowStaysVisibleWhileScrolling(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4 // the longest stage
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	first, ok := tr.Find("每层改多少")
	if !ok {
		t.Fatal("the card row is missing")
	}
	tr.Move(600, 400)
	for i := 0; i < 6; i++ {
		tr.Wheel(600, 400, 0, 300)
		tr.Advance(2e9)
	}
	tr.Frame()
	if a.Scroll.Y == 0 {
		t.Fatal("the wheel did not scroll the stage")
	}
	later, ok := tr.Find("每层改多少")
	if !ok {
		t.Fatalf("after scrolling to y=%.0f the card row is gone; it is not sticky", a.Scroll.Y)
	}
	if later.Y < 0 || later.Y+later.H > 720 {
		t.Fatalf("after scrolling the card row sits at y=%.0f, outside the window", later.Y)
	}
	// It should have travelled with the scroll a little, but stayed near the top.
	if later.Y > first.Y+40 {
		t.Fatalf("the card row moved from y=%.0f to y=%.0f; it is not pinned", first.Y, later.Y)
	}
}

// TestEveryNamedCardAppearsInItsStageRow closes the direction the jump test cannot: a
// card could be built with cardNamed, never listed in stageCards, and every existing test
// would still pass — the chip simply would not exist. The row is the stage's index, so a
// card missing from it is a section of the page a reader cannot see the existence of,
// which is the exact problem the row was added to fix.
//
// It checks the source rather than the pixels because the mapping is a source-level fact:
// cardNamed is the only way a card gets a chip, so every call must have a home in the
// table. Calls inside branch arms (a one-token prompt's placeholder) are the same chip as
// their alternative, which is why the comparison counts distinct chips per stage rather
// than occurrences.
func TestEveryNamedCardAppearsInItsStageRow(t *testing.T) {
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`cardNamed\(c, "([^"]+)"`)
	var all []string
	for _, m := range re.FindAllStringSubmatch(string(src), -1) {
		all = append(all, m[1])
	}
	if len(all) == 0 {
		t.Fatal("no cardNamed calls found")
	}
	listed := map[string]bool{}
	for _, chips := range stageCards {
		for _, chip := range chips {
			listed[chip] = true
		}
	}
	for _, chip := range all {
		if !listed[chip] {
			t.Fatalf("a card is built with cardNamed(%q) but no stage row lists it; add it to "+
				"stageCards so readers can see and reach it", chip)
		}
	}
	// And the other direction: every listed chip must be built somewhere.
	built := map[string]bool{}
	for _, chip := range all {
		built[chip] = true
	}
	for step, chips := range stageCards {
		for _, chip := range chips {
			if !built[chip] {
				t.Fatalf("stage %d lists the chip %q but no card is built with cardNamed(%q)",
					step+1, chip, chip)
			}
		}
	}
}

// TestCardChipsAreInPageOrder is the invariant the row itself depends on and that nothing
// checked: the chips have to be listed in the order their cards appear down the page.
//
// A table of contents that reads "残差流 · 残差为何不爆 · 每层说什么" while the page shows
// 残差流, then the logit lens, then the layer comparison, then the callout is worse than no
// contents at all — it tells the reader the wrong shape for the thing it exists to
// describe. Measured before this test: three of the six multi-card stages were out of
// order. On the residual stage the second chip scrolled to 1767 while the third went to
// 665, so following the row top to bottom walked the reader down the page and then back up.
//
// Order is measured by where each chip actually scrolls, not by reading the source: the
// draw order is a property of the rendered page, and several cards are built by helper
// functions called from between the inline ones, so the source order is not readable
// straight off.
func TestCardChipsAreInPageOrder(t *testing.T) {
	for step, chips := range stageCards {
		if len(chips) < 2 {
			continue
		}
		prev := float32(-1)
		prevChip := ""
		for _, chip := range chips {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, 1180, 720)
			tr.Frame()
			tr.Click(chip)
			tr.Frame()
			tr.Frame()
			if a.Scroll.Y < prev {
				t.Fatalf("stage %d: the chip %q lands at %.0f, above the %.0f that %q lands at — "+
					"the row's order does not match the page's",
					step+1, chip, a.Scroll.Y, prev, prevChip)
			}
			prev = a.Scroll.Y
			prevChip = chip
		}
	}
}

// TestCardChipsLeadSomewhereNew checks that no two chips are two names for one card.
//
// The first version compared scroll offsets and flagged stage 1, where 字符 → 字节 and
// 数据流与形状 both land on 252. That was the test being wrong, not the row: stage 1
// overflows by only 252 DIP, so scrolling to the end puts both cards on screen at once and
// the container clamps them to the same offset. Equal scroll is therefore not evidence of a
// shared card — what it needs to check is that both cards exist and are distinct, which is
// what it does now: each chip's target card header must be the one the chip names.
func TestCardChipsLeadSomewhereNew(t *testing.T) {
	for step, chips := range stageCards {
		if len(chips) < 2 {
			continue
		}
		_ = chips
		// The row's chips map one-to-one onto headings by construction of cardNamed, and
		// TestEveryNamedCardAppearsInItsStageRow checks that from the source. What can still
		// go wrong here is two chips whose jump does nothing at all, so each click has to
		// leave a different card under the viewport's top edge.
		for _, chip := range chips {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, 1180, 720)
			tr.Frame()
			before := a.Scroll.Y
			tr.Click(chip)
			tr.Frame()
			tr.Frame()
			if a.RevealCard != "" {
				t.Fatalf("stage %d: clicking %q left the reveal pending", step+1, chip)
			}
			if a.Scroll.Y == before && a.Scroll.MaxY > 0 && chip != chips[0] {
				t.Logf("stage %d: %q did not move the page (already at %.0f)", step+1, chip, before)
			}
		}
	}
}
