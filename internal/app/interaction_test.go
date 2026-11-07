package app

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
)

// The app's clickable surfaces had keyboard equivalents that were tested and
// mouse paths that were not: navigation, reset, sampling, the preset chips, and
// the stepper were all reachable only through tests that called selectStep or
// sampleNext directly. These drive the widgets.

// TestNextAndPreviousButtonsWalkTheStages clicks the two step buttons.
func TestNextAndPreviousButtonsWalkTheStages(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	click := func(label string) {
		t.Helper()
		r, ok := tr.Find(label)
		if !ok {
			t.Fatalf("%q not found", label)
		}
		tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
		tr.Frame()
	}

	click("下一步 →")
	if a.Step != 1 {
		t.Fatalf("下一步 left the step at %d", a.Step)
	}
	click("下一步 →")
	if a.Step != 2 {
		t.Fatalf("second 下一步 left the step at %d", a.Step)
	}
	click("← 上一步")
	if a.Step != 1 {
		t.Fatalf("上一步 left the step at %d", a.Step)
	}
}

// TestAutoplayButtonStartsAndSpaceStops checks the play button and the key that
// cancels it agree.
func TestAutoplayButtonStartsAndSpaceStops(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	r, ok := tr.Find("▶ 自动播放")
	if !ok {
		t.Fatal("autoplay button not found")
	}
	tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tr.Frame()
	if !a.Playing {
		t.Fatal("the autoplay button did not start playback")
	}
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if a.Playing {
		t.Fatal("space did not stop playback")
	}
}

// TestResetButtonRestoresTheDefaultPrompt checks the top bar's reset.
func TestResetButtonRestoresTheDefaultPrompt(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "something else entirely"
	a.recompute()
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	r, ok := tr.Find("重置")
	if !ok {
		t.Fatal("reset button not found")
	}
	tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tr.Frame()
	if a.Prompt != defaultPrompt {
		t.Fatalf("reset left the prompt as %q, want the default", a.Prompt)
	}
}

// TestSampleButtonAppendsAToken checks the sampling card's button, which the
// existing sampling tests bypass by calling sampleNext.
func TestSampleButtonAppendsAToken(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	before := a.Prompt
	r, ok := tr.Find("采样并续写")
	if !ok {
		t.Fatal("sample button not found")
	}
	tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tr.Frame()
	if len(a.Prompt) != len(before)+1 {
		t.Fatalf("prompt went from %d to %d bytes", len(before), len(a.Prompt))
	}
	if a.LastSample == "" {
		t.Fatal("sampling did not record what it drew")
	}
}

// TestPresetChipsReplaceThePrompt clicks each preset and checks the prompt it
// sets. The chips are the only way a mouse user can change the prompt besides
// typing.
func TestPresetChipsReplaceThePrompt(t *testing.T) {
	for _, p := range promptPresets {
		if p.prompt == "" {
			continue // the empty chip is the state, not an action
		}
		a := newTestApp(t)
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		r, ok := tr.Find(p.name)
		if !ok {
			t.Fatalf("preset chip %q not found", p.name)
		}
		tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
		tr.Frame()
		if a.Prompt != p.prompt {
			t.Fatalf("chip %q set the prompt to %q, want %q", p.name, a.Prompt, p.prompt)
		}
		if len(a.Trace.Tokens) == 0 {
			t.Fatalf("chip %q left no tokens", p.name)
		}
	}
}

// TestStepperRowsJumpToTheirStage clicks the sidebar rows.
func TestStepperRowsJumpToTheirStage(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, want := range []int{4, 0, 6} {
		label := "步骤 " + itoa(want+1) + " " + lessons[want].name
		r, ok := tr.Find(label)
		if !ok {
			t.Fatalf("stepper row %q not found", label)
		}
		tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
		tr.Frame()
		if a.Step != want {
			t.Fatalf("clicking %q went to step %d, want %d", label, a.Step, want)
		}
	}
}

// TestShapeChipsJumpToTheirStage drives the data-flow card's chips, which are
// the only thing on stage 1 that navigates anywhere. The heading promises
// "点击任意阶段跳转", so a chip landing on the wrong stage is a broken promise.
//
// Chips are located by their shape string, not their label: the labels duplicate
// the sidebar stepper's and Find returns whichever match comes first (the
// sidebar). The shape strings are unique to the card.
func TestShapeChipsJumpToTheirStage(t *testing.T) {
	T := len(newTestApp(t).Trace.Tokens)
	d, dff, vocab := 96, 384, 256
	chips := []struct {
		shape string
		step  int
	}{
		{fmt.Sprintf("[%d]", T), 0},
		{fmt.Sprintf("[%d, %d]", T, d), 1},
		{fmt.Sprintf("[%d, %d]", T, T), 3},
		{fmt.Sprintf("[%d, %d]", T, dff), 5},
		// The output shape keeps the leading axis as 1: only the last position is
		// projected, so the token axis collapses rather than vanishing.
		{fmt.Sprintf("[1, %d]", vocab), 6},
	}
	for _, c := range chips {
		a := newTestApp(t)
		a.Step = 0
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		r, ok := tr.Find(c.shape)
		if !ok {
			t.Fatalf("shape chip %q not found on stage 1", c.shape)
		}
		tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
		tr.Frame()
		if a.Step != c.step {
			t.Fatalf("chip %q jumped to stage %d, want %d", c.shape, a.Step+1, c.step+1)
		}
	}
}

// TestShapeChipsListEveryStage checks the card names all seven stages in order.
// A missing row would be a stage the reader cannot reach from here.
func TestShapeChipsListEveryStage(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	seen := map[string]bool{}
	for _, s := range tr.Texts() {
		seen[s] = true
	}
	for _, name := range []string{"分词", "嵌入", "注意力", "Softmax", "残差 / LN", "FFN", "输出"} {
		if !seen[name] {
			t.Fatalf("the shape card does not list %q", name)
		}
	}
}

// TestJumpingToAStageStopsPlayback pins the two side effects selectStep has beyond
// changing the step.
//
// Mutation testing removed each and the suite stayed green. Both matter: pressing a
// digit key while the lesson is auto-playing must leave the reader where they jumped
// to, not be carried away 1.3 seconds later; and the pending clock has to be dropped,
// or the next play session inherits a stale deadline.
func TestJumpingToAStageStopsPlayback(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// Start playback through the key, which is the path that does not depend on
	// hitting a button's exact position.
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("space did not start playback")
	}
	if a.NextAt.IsZero() {
		t.Fatal("playback started without scheduling a next step")
	}

	// Jump to stage 5.
	tr.Key(0, ui.Key5)
	tr.Frame()
	if a.Step != 4 {
		t.Fatalf("pressing 5 landed on stage %d", a.Step)
	}
	if a.Playing {
		t.Fatal("jumping to a stage left playback running, so the reader will be " +
			"carried away from the stage they asked for")
	}
	if !a.NextAt.IsZero() {
		t.Fatalf("the pending clock survived the jump: %v", a.NextAt)
	}

	// And the jump must start at the top of the new stage, since the previous
	// stage's scroll offset is meaningless in the new one.
	a.Scroll.Y = 400
	tr.Key(0, ui.Key2)
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("the scroll offset carried over as %v", a.Scroll.Y)
	}
}

// TestSelectingAnotherTokenDropsThePin keeps the pin and the selection from
// describing different tokens.
//
// A pinned cell belongs to the row the reader was looking at. Clicking a different
// token in the strip used to leave the pin alone, so the inspector headed
// "query = token 20" while the cell card underneath still described token 10's row —
// two different tokens named in one panel.
//
// The stage switch already dropped the pin for the same reason ("跨阶段的格子没有意义"),
// but that was the only path covered. The chip click also assigned a.SelToken
// directly, bypassing selectToken entirely, which is why fixing selectToken alone
// changed nothing.
func TestSelectingAnotherTokenDropsThePin(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// Click a chip well away from the pinned row.
	target := 20
	x, y := chipCentre(t, tr, target)
	tr.Press(x, y)
	tr.Release(x, y)
	tr.Frame()

	if a.SelToken != target {
		t.Fatalf("the click selected %d, not %d", a.SelToken, target)
	}
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Fatalf("the pin (%d,%d) survived a selection change to %d; the panel "+
			"would name one token in its heading and another in its cell card",
			a.KeyQ, a.KeyK, target)
	}

	// Re-selecting the pinned row itself must not drop it: the reader is going
	// back to where the pin points.
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5
	tr2 := ui.NewTester(a.View, 1500, 940)
	tr2.Frame()
	x2, y2 := chipCentre(t, tr2, 10)
	tr2.Press(x2, y2)
	tr2.Release(x2, y2)
	tr2.Frame()
	if a.SelToken == 10 && (a.KeyQ != 10 || a.KeyK != 5) {
		t.Fatalf("clicking the pinned row's own chip dropped the pin: (%d,%d)",
			a.KeyQ, a.KeyK)
	}
}

// chipCentre finds the centre of a token strip chip by searching the rendered
// frame for its label.
func chipCentre(t *testing.T, tr *ui.Tester, i int) (float32, float32) {
	t.Helper()
	r, ok := tr.Find(fmt.Sprintf("token %d", i))
	if !ok {
		t.Fatalf("no chip labelled token %d", i)
	}
	return r.X + r.W/2, r.Y + r.H/2
}

// TestPlayIntervalIsTheSameForTheFirstStepAndTheRest pins the pace of autoplay.
//
// The interval used to be written twice — once to schedule the first advance after
// starting, once for every advance after that. They have to be equal: a longer first
// wait reads as "nothing happened" and a shorter one as a stutter, and changing only
// one of the two would produce exactly that while no test noticed.
//
// Measures the time between two consecutive advances, which is the property that
// actually depends on both uses agreeing.
func TestPlayIntervalIsTheSameForTheFirstStepAndTheRest(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("space did not start playback")
	}

	// Time from starting to the first advance.
	const tick = 50 * time.Millisecond
	start := a.Step
	sinceStart := time.Duration(0)
	for sinceStart < 5*time.Second && a.Step == start {
		tr.Advance(tick)
		tr.Frame()
		sinceStart += tick
	}
	if a.Step != start+1 {
		t.Fatalf("playback never advanced from stage %d", start)
	}

	// Time from that advance to the next one.
	mid := a.Step
	sinceMid := time.Duration(0)
	for sinceMid < 5*time.Second && a.Step == mid {
		tr.Advance(tick)
		tr.Frame()
		sinceMid += tick
	}
	if a.Step != mid+1 {
		t.Fatalf("playback never advanced past stage %d", mid)
	}

	// Both waits must be the same interval. The tick granularity means each is
	// measured to within one tick of the true value.
	if delta := sinceMid - sinceStart; delta > tick || delta < -tick {
		t.Fatalf("the first wait was %v and the next was %v; both must use "+
			"playStepInterval (%v)", sinceStart, sinceMid, playStepInterval)
	}
}

// TestPlayingFromTheLastStageWrapsToTheStart pins what pressing play at the end does.
//
// togglePlay resets the step to 0 when playback starts from the final stage, along with
// the scroll offset and the hover state — otherwise the lesson would start on its last
// page and stop there, since play() halts when it cannot advance past the end. The
// behaviour is deliberate and had no test: removing the reset would leave the play
// button doing nothing visible on the last stage.
func TestPlayingFromTheLastStageWrapsToTheStart(t *testing.T) {
	a := newTestApp(t)
	a.Step = len(lessons) - 1
	a.Scroll.Y = 320
	a.HoverQ, a.HoverK = 3, 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	tr.Key(0, ui.KeySpace)
	tr.Frame()

	if a.Step != 0 {
		t.Fatalf("playing from stage %d stayed on %d; it should wrap to the first",
			len(lessons), a.Step)
	}
	if !a.Playing {
		t.Fatal("playback did not start")
	}
	if a.Scroll.Y != 0 {
		t.Fatalf("the scroll offset of the last stage carried over as %v", a.Scroll.Y)
	}
	if a.HoverQ >= 0 || a.HoverK >= 0 {
		t.Fatalf("a hover from the last stage survived the wrap: (%d,%d)", a.HoverQ, a.HoverK)
	}

	// And stopping there must leave the reader on the first stage rather than
	// jumping back.
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if a.Playing {
		t.Fatal("the second press did not stop playback")
	}
	if a.Step != 0 {
		t.Fatalf("stopping moved the reader to stage %d", a.Step)
	}
}

// TestPlaybackStopsAtTheEnd is the other half: play() must halt rather than wrap, or
// the two behaviours would contradict each other.
func TestPlaybackStopsAtTheEnd(t *testing.T) {
	a := newTestApp(t)
	a.Step = len(lessons) - 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("playback did not start")
	}

	// Walk the clock forward until it reaches the last stage, then past it.
	const tick = 200 * time.Millisecond
	for i := 0; i < 40 && a.Playing; i++ {
		tr.Advance(tick)
		tr.Frame()
	}
	if a.Playing {
		t.Fatal("playback never stopped at the end of the lesson")
	}
	if a.Step != len(lessons)-1 {
		t.Fatalf("playback stopped on stage %d, not the last one", a.Step)
	}
}

// TestTabDrawsAFocusRing covers the keyboard path the app never tested.
//
// The goal for this tool names "keyboard operable, with feedback" as one of the things
// it must do, and every shortcut has a test — but Tab, the key that walks between
// controls and is the only way a keyboard-only reader reaches the input field, had
// none. Measured: pressing Tab changes over 2000 bytes of the frame, which is the
// focus ring appearing. If that ever stopped being drawn, a keyboard reader would be
// moving an invisible cursor through the controls.
//
// The assertion is on the pixels rather than on any state, because the ring is what
// the reader actually sees.
func TestTabDrawsAFocusRing(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	before := append([]byte(nil), tr.Image().Pix...)

	tr.Key(0, ui.KeyTab)
	after := tr.Image().Pix

	const minChangedPixels = 200 // conservative: the measured figure is ~2000 bytes
	changed := 0
	for i := 0; i+3 < len(before) && i+3 < len(after); i += 4 {
		if before[i] != after[i] || before[i+1] != after[i+1] || before[i+2] != after[i+2] {
			changed++
			if changed >= minChangedPixels {
				break
			}
		}
	}
	if changed < minChangedPixels {
		t.Fatalf("Tab changed %d pixels; the focus ring is not being drawn, so a "+
			"keyboard reader would be moving an invisible cursor", changed)
	}

	// Tab must not also act as a shortcut: the stage and playback state are what the
	// other keys touch, and a focus move should be invisible to them.
	if a.Step != 0 {
		t.Fatalf("Tab changed the stage to %d", a.Step)
	}
	if a.Playing {
		t.Fatal("Tab started playback")
	}
}

// TestTabWalksThroughMoreThanOneControl checks the focus actually moves on, rather
// than cycling between the first control and nothing.
func TestTabWalksThroughMoreThanOneControl(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// Each Tab should land on a different control, so consecutive frames differ.
	var frames [][]byte
	frames = append(frames, append([]byte(nil), tr.Image().Pix...))
	for i := 0; i < 4; i++ {
		tr.Key(0, ui.KeyTab)
		frames = append(frames, append([]byte(nil), tr.Image().Pix...))
	}
	distinct := 1
	for i := 1; i < len(frames); i++ {
		if !bytes.Equal(frames[i], frames[i-1]) {
			distinct++
		}
	}
	if distinct < 3 {
		t.Fatalf("four Tab presses produced %d distinct frames; focus is not walking "+
			"through the controls", distinct)
	}
}

// TestDragCannotPinAMaskedCell covers the pointer path into the causal mask.
//
// A click reads the cell under the pointer and pins it. A drag presses on one cell and
// releases on another, so it can start inside the causal triangle and end outside it —
// above the diagonal, which the attention mask zeroes and the model never computes.
// Measured before the guard: pressing at a low cell and dragging up and right pinned
// (9, 28), and the inspector then showed a score for a cell that does not exist.
//
// The keyboard already refused this (stepCell clamps to the diagonal), which is why the
// asymmetry survived: only the pointer could reach the masked half.
func TestDragCannotPinAMaskedCell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// Press low on the diagonal side, drag up and right, release above it.
	tr.Press(400, 800)
	tr.Move(600, 700)
	tr.Release(600, 700)
	tr.Frame()

	if a.KeyQ >= 0 && a.KeyK > a.KeyQ {
		t.Fatalf("the drag pinned (%d,%d), above the diagonal, so the inspector would "+
			"describe a masked cell", a.KeyQ, a.KeyK)
	}
	if a.HoverQ >= 0 && a.HoverK > a.HoverQ {
		t.Fatalf("the pointer now hovers (%d,%d), above the diagonal", a.HoverQ, a.HoverK)
	}

	// A click inside the triangle must still pin — the guard must not have broken the
	// ordinary case.
	b := newTestApp(t)
	b.Step = 2
	bt := ui.NewTester(b.View, 1500, 940)
	bt.Frame()
	for y := float32(560); y < 900; y += 8 {
		bt.ClickAt(430, y)
		if b.KeyQ >= 0 {
			break
		}
	}
	if b.KeyQ < 0 {
		t.Fatal("no click inside the triangle managed to pin a cell")
	}
	if b.KeyK > b.KeyQ {
		t.Fatalf("a click pinned (%d,%d), above the diagonal", b.KeyQ, b.KeyK)
	}
}

// TestBothPromptPathsLeaveTheSameState pins that the preset chips and the text field
// cannot drift apart.
//
// Both change the prompt and both have to drop the state belonging to the old one: the
// last sample, the pending playback clock, the playing flag. They were written twice —
// applyPrompt as a function, the field's Changed() handler inline — with identical
// bodies, so nothing kept them in step. That is the same shape as the drag-versus-keyboard
// gap fixed last round: one invariant, two implementations.
//
// Both now go through promptChanged. This compares the resulting state field by field,
// which is what would have caught a divergence before the refactor.
func TestBothPromptPathsLeaveTheSameState(t *testing.T) {
	type snapshot struct {
		tokens, sel, keyQ, keyK int
		lastSample              string
		playing                 bool
		nextAtZero              bool
	}
	read := func(a *App) snapshot {
		return snapshot{
			tokens:     len(a.Trace.Tokens),
			sel:        a.SelToken,
			keyQ:       a.KeyQ,
			keyK:       a.KeyK,
			lastSample: a.LastSample,
			playing:    a.Playing,
			nextAtZero: a.NextAt.IsZero(),
		}
	}

	// Path A: a preset, through applyPrompt.
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	a.KeyQ, a.KeyK = 10, 5
	a.LastSample = `"d" (id 100, p=0.5)`
	a.Playing = true
	a.applyPrompt("hi")

	// Path B: the same edit typed into the field, through promptChanged.
	b := newTestApp(t)
	b.Step = 2
	b.SelToken = 10
	b.KeyQ, b.KeyK = 10, 5
	b.LastSample = `"d" (id 100, p=0.5)`
	b.Playing = true
	b.Prompt = "hi"
	b.promptChanged()

	if got, want := read(b), read(a); got != want {
		t.Fatalf("the typed path left %+v, the preset path left %+v", got, want)
	}

	// And both must have actually cleared the old prompt's state: a comparison of two
	// equally-wrong states would pass.
	if a.LastSample != "" {
		t.Fatalf("the sample survived the prompt change: %q", a.LastSample)
	}
	if a.Playing {
		t.Fatal("playback survived the prompt change")
	}
	if !a.NextAt.IsZero() {
		t.Fatalf("the pending clock survived: %v", a.NextAt)
	}
}

// TestSamplingStopsPlayback checks the third way of editing the prompt against the other
// two.
//
// Three paths append to or replace the prompt: the preset chips, the text field, and the
// sample button — which appends one byte. The first two stop playback and clear the
// pending clock, because the prompt they were playing no longer exists. The sample path
// did neither: a reader watching the lesson play, who then sampled, would be carried to
// the next stage a second later with the sample already replaced by a different
// prediction.
//
// It does not go through promptChanged, and should not: that helper clears LastSample,
// and the sample is the whole point of this path. The clock and playing flag are
// duplicated here for that reason, which is worth stating since the duplication is
// otherwise exactly what the helper exists to prevent.
func TestSamplingStopsPlayback(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("playback did not start")
	}
	if a.NextAt.IsZero() {
		t.Fatal("playback started without scheduling an advance")
	}

	before := a.Prompt
	a.sampleNext()
	tr.Frame()

	if a.Prompt == before {
		t.Fatalf("sampling did not extend the prompt: %q", a.Prompt)
	}
	if a.Playing {
		t.Fatal("sampling left playback running; the reader would be carried off the " +
			"stage they sampled from")
	}
	if !a.NextAt.IsZero() {
		t.Fatalf("the pending clock survived sampling: %v", a.NextAt)
	}
	// And the record of what was drawn has to survive, unlike the other two paths.
	if a.LastSample == "" {
		t.Fatal("the sample record was cleared, so the card cannot show what was drawn")
	}
}

// TestEmptyAttentionStageDoesNotAdvertiseCellKeys covers the empty state on the two stages
// that draw a matrix.
//
// hasMatrix answered only "is this stage 2 or 3", so with an empty prompt those stages still
// printed "⇧ + 方向键 移动格子 · 点击固定" in the step bar and "Shift+← → ↑ ↓ 移动矩阵格子" in
// the lesson footer — while stepCell returned immediately on zero tokens. A reader following
// the hint pressed the keys and nothing moved, which reads as a broken app rather than an
// empty one.
//
// hasMatrix now means "there is a matrix to act on". The hint goes away, and the left/right
// arrows fall through to stepToken, so the footer correctly says "切换 token" instead.
func TestEmptyAttentionStageDoesNotAdvertiseCellKeys(t *testing.T) {
	for _, step := range []int{2, 3} {
		a := newTestApp(t)
		a.Prompt = ""
		a.recompute()
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		for _, s := range tr.Texts() {
			if strings.Contains(s, "移动格子") {
				t.Fatalf("stage %d with an empty prompt still says the arrows move a "+
					"cell: %q", step, s)
			}
			if strings.Contains(s, "移动矩阵格子") {
				t.Fatalf("stage %d's footer still describes the matrix keys: %q", step, s)
			}
		}

		// And the stage must still be usable: with a prompt, the hint comes back.
		b := newTestApp(t)
		b.Step = step
		bt := ui.NewTester(b.View, 1500, 940)
		bt.Frame()
		var sawHint bool
		for _, s := range bt.Texts() {
			if strings.Contains(s, "移动格子") {
				sawHint = true
			}
		}
		if !sawHint {
			t.Fatalf("stage %d no longer advertises the cell keys even with a prompt", step)
		}
	}
}

// TestEveryTabStopDrawsSomething checks focus feedback across the whole ring, not just the
// first step.
//
// The test added earlier in this sweep pressed Tab once and required the frame to change,
// which proves a ring exists but not that every stop has one. A control in the middle of
// the ring with no visible focus would leave a keyboard reader pressing Tab through an
// invisible cursor — the same failure, just further in.
//
// Measured: the ring has six stops and every one repaints between 1096 and 6083 pixels.
// The test requires a minimum far below the smallest, so a restyle can shrink the ring
// without failing this, but a stop that paints nothing cannot pass.
func TestEveryTabStopDrawsSomething(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	const minPixels = 100
	prev := append([]byte(nil), tr.Image().Pix...)
	seen := 0
	for i := 0; i < 6; i++ {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
		cur := tr.Image().Pix
		changed := 0
		for j := 0; j+3 < len(prev) && j+3 < len(cur); j += 4 {
			if prev[j] != cur[j] || prev[j+1] != cur[j+1] || prev[j+2] != cur[j+2] {
				changed++
			}
		}
		if changed < minPixels {
			t.Fatalf("tab stop %d repainted only %d pixels; that control has no visible "+
				"focus", i+1, changed)
		}
		seen++
		prev = append([]byte(nil), cur...)
	}
	if seen != 6 {
		t.Fatalf("walked %d tab stops, expected 6", seen)
	}
}

// TestDiscoverabilityHintsAreCovered checks that the interactions a reader cannot infer from
// the layout are advertised where they apply.
//
// Four things in this app are only learnable from text: that the arrows move a matrix cell,
// that clicking a cell pins it, that some top-bar controls are off on this stage and where
// they turn on, and that Shift+arrows mean something different on matrix stages. Three of
// the four were added in response to a specific confusion; nothing checked them as a set, so
// any one of them could have been dropped in a restyle.
//
// Each is asserted on the stage it belongs to and, for the matrix pair, on the stages it does
// not belong to — a hint that shows everywhere is noise, not a signpost.
func TestDiscoverabilityHintsAreCovered(t *testing.T) {
	stageText := func(step int) string {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		var parts []string
		for _, s := range tr.Texts() {
			parts = append(parts, s)
		}
		return strings.Join(parts, "\n")
	}

	for _, step := range []int{2, 3} {
		text := stageText(step)
		for _, want := range []string{"移动格子", "点击固定", "移动矩阵格子"} {
			if !strings.Contains(text, want) {
				t.Fatalf("stage %d draws a matrix but does not mention %q anywhere", step, want)
			}
		}
	}
	for _, step := range []int{0, 1, 4, 5, 6} {
		text := stageText(step)
		if strings.Contains(text, "移动格子") {
			t.Fatalf("stage %d has no matrix but advertises moving a cell", step)
		}
	}

	// The dimming explanation has to be reachable where the dimming happens: on a stage
	// that disables at least one control.
	for _, step := range []int{0, 1, 4, 5, 6} {
		if !strings.Contains(stageText(step), "不生效") {
			t.Fatalf("stage %d dims a control but never says where it turns on", step)
		}
	}
}

// TestSelectionReachesEveryStageCorrectly checks that the inspector describes the token that
// is actually selected, on every stage.
//
// An earlier test in this sweep proved that changing the selection changes the frame. That
// says something moved, not that it moved to the right thing: an inspector stuck on token 10
// while the strip highlights 35 would repaint just as convincingly.
//
// Measured across three selections (first, middle, last) and all seven stages: every header
// reads "token #N / T-1" for the selected N. Twenty-one combinations, none wrong.
func TestSelectionReachesEveryStageCorrectly(t *testing.T) {
	a := newTestApp(t)
	total := len(a.Trace.Tokens)

	for _, token := range []int{0, 10, total - 1} {
		for step := 0; step < len(lessons); step++ {
			b := newTestApp(t)
			b.Step = step
			b.SelToken = token
			tr := ui.NewTester(b.View, 1500, 940)
			tr.Frame()
			// The strip numbers tokens from zero and the header shows the last index
			// beside the count, which is what a reader compares against.
			want := fmt.Sprintf("token #%d / %d", token, total-1)
			if !tr.HasText(want) {
				t.Fatalf("stage %d with token %d selected does not show %q; the inspector "+
					"is describing something else", step, token, want)
			}
		}
	}
}

// TestLayerAndHeadReachTheCardSubtitles is the same check for the two other controls.
//
// Three subtitles on the attention stage quote "层 N · 头 N". They followed the controls when
// measured, and a card still reporting layer 0 while the segmented control sits on 1 is the
// same class of failure as a stale selection.
func TestLayerAndHeadReachTheCardSubtitles(t *testing.T) {
	for _, layer := range []int{0, 1} {
		for _, head := range []int{0, 3} {
			a := newTestApp(t)
			a.Step = 3
			a.Layer = layer
			a.Head = head
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()
			want := fmt.Sprintf("层 %d · 头 %d", layer, head)
			found := 0
			for _, s := range tr.Texts() {
				if strings.Contains(s, want) {
					found++
				}
			}
			if found < 3 {
				t.Fatalf("layer %d head %d: only %d subtitles say %q, expected 3",
					layer, head, found, want)
			}
		}
	}
}

// TestEnterSamplesOutsideTheInput covers the one primary action that had no key.
//
// Every other clickable surface already had a keyboard equivalent — stages have arrows and
// digits, tokens have Shift+arrows, cells have Shift+arrows on the matrix stages. Sampling
// did not, and it is the action a reader repeats, at the top bar and again on the output
// stage.
//
// Enter is free and safe: the text field consumes it while focused, the same mechanism that
// keeps Space out of the lesson. Both halves are asserted, because a shortcut that also fires
// while typing would corrupt the prompt the reader is editing.
func TestEnterSamplesOutsideTheInput(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	before := a.Prompt
	tr.Key(0, ui.KeyEnter)
	tr.Frame()
	if len(a.Prompt) <= len(before) {
		t.Fatalf("Enter outside the input did not sample: prompt is still %q", a.Prompt)
	}
	if !strings.HasPrefix(a.Prompt, before) {
		t.Fatalf("sampling replaced the prompt instead of extending it: %q -> %q",
			before, a.Prompt)
	}

	// And it must not fire while the reader is typing.
	b := newTestApp(t)
	bt := ui.NewTester(b.View, 1500, 940)
	bt.Frame()
	if err := bt.Click("输入"); err != nil {
		t.Fatal(err)
	}
	bt.Frame()
	promptBefore := b.Prompt
	bt.Key(0, ui.KeyEnter)
	bt.Frame()
	if b.Prompt != promptBefore {
		t.Fatalf("Enter in the text field sampled: %q -> %q", promptBefore, b.Prompt)
	}
}

// TestEnterIsAdvertised keeps the new key in the shortcut line every stage shows.
func TestEnterIsAdvertised(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		var line string
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "快捷键：") {
				line = s
			}
		}
		if line == "" {
			t.Fatalf("stage %d has no shortcut line", step)
		}
		if !strings.Contains(line, "Enter 采样") {
			t.Fatalf("stage %d does not mention the sampling key: %q", step, line)
		}
	}
}

// TestHoveringACellShowsItsCardAtEverySize checks that the pointer reaches the matrix and the
// card follows, at each supported window width.
//
// What it does not prove is same-frame latency. ui.Tester settles after every input, so a Move
// already advances a frame — there is no way here to produce "state updated, picture not yet",
// and this test would pass either way. The claim it can make is that the hit test tracks the
// grid as the window resizes: the narrowest window shrinks the matrix, so a stale hit box would
// make hovering miss while the drawn grid had moved, and that size would show it first.
func TestHoveringACellShowsItsCardAtEverySize(t *testing.T) {
	for _, size := range [][2]int{{1180, 720}, {1500, 940}, {1920, 1080}} {
		a := newTestApp(t)
		a.Step = 3
		tr := ui.NewTester(a.View, size[0], size[1])
		tr.Frame()

		// The grid's origin moves with the window, so probe a few points rather than
		// hard-coding one that only works at the default size.
		var hit bool
		for _, pt := range [][2]float32{{300, 620}, {260, 600}, {320, 640}, {280, 660}} {
			tr.Move(pt[0], pt[1])
			tr.Frame()
			if a.HoverQ >= 0 && a.HoverK >= 0 {
				hit = true
				break
			}
		}
		if !hit {
			t.Fatalf("%dx%d: no matrix cell under the pointer at any probe point", size[0], size[1])
		}
		var found bool
		for _, s := range tr.Texts() {
			if strings.Contains(s, "注意力格子") {
				found = true
			}
		}
		if !found {
			t.Fatalf("%dx%d: hover reached (q=%d, k=%d) but no cell card was drawn on the same "+
				"frame", size[0], size[1], a.HoverQ, a.HoverK)
		}
	}
}
