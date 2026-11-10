package app

import (
	"bytes"
	"fmt"
	"image"
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
//
// The card sits below the fold on the output stage, and the per-position strip
// pushed it further down, so the frame has to scroll to it first. That is the
// contract this test should state: the button is reachable by scrolling, and the
// keyboard's Enter is the always-visible alternative (TestEveryShortcutHasA-
// VisibleAlternative pins that the top bar's own button is on screen).
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
	// Find reports layout coordinates that move with the scroll (measured: 1148
	// before scrolling, 593 after), so this has to scroll until the button is
	// actually inside the window, not just wheel a fixed number of times.
	for i := 0; i < 12 && r.Y+r.H > 940; i++ {
		tr.Wheel(600, 400, 0, 400)
		tr.Advance(2e9)
		tr.Frame()
		r, _ = tr.Find("采样并续写")
	}
	if r.Y+r.H > 940 {
		t.Fatalf("the sampling card's button is still below the window after scrolling "+
			"(y=%.0f, Scroll.Y=%.0f); the action has no visible target", r.Y, a.Scroll.Y)
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
// Measured: the smallest repaint along the ring is about 1100 pixels. The test requires a
// minimum far below that, so a restyle can shrink the ring without failing this, but a stop
// that paints nothing cannot pass.
//
// The length of the ring comes from the pixels rather than from a literal. It used to walk a
// fixed six while the default stage's cycle is eight, so the last two stops went unchecked —
// and the closing "walked %d tab stops, expected 6" could not fail either, since it counted
// its own loop iterations. Verified by mutation: making the theme's focus colour the panel
// colour turns this red at the first stop with "repainted only 0 pixels".
//
// What this still cannot see is a control that stops taking focus at all. Then the cycle gets
// shorter and every remaining stop is checked correctly, so the walk adapts instead of
// complaining. That case belongs to TestContributingFocusTableMatchesTheApp, which compares
// the cycle against the guide's independent table.
func TestEveryTabStopDrawsSomething(t *testing.T) {
	a := newTestApp(t)
	rings := focusRings(t, a.Step, 1500, 940)
	if len(rings) < 2 {
		t.Fatalf("Tab found %d focus rings; there is no ring to walk", len(rings))
	}

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	const minPixels = 100
	prev := tr.Image()
	var first *image.RGBA
	for i := range rings {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
		cur := tr.Image()
		if changed := diffCount(prev, cur); changed < minPixels {
			t.Fatalf("tab stop %d of a %d-stop ring repainted only %d pixels; that control "+
				"has no visible focus", i+1, len(rings), changed)
		}
		if i == 0 {
			first = cur
		}
		prev = cur
	}

	// One more press has to return to the first stop, so the ring really is as long as the
	// number of stops this test checked. focusRings stops at the first repeated bounding
	// box, and a box is a coarser signal than the frame: two stops could in principle share
	// one. Comparing whole frames closes that gap.
	tr.Key(0, ui.KeyTab)
	tr.Frame()
	if got := diffCount(first, tr.Image()); got != 0 {
		t.Fatalf("after %d Tab presses the ring did not return to its first stop (%d pixels "+
			"differ), so the cycle is longer than the %d stops this test walked",
			len(rings), got, len(rings))
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

// TestTypingRecomputesWithoutAKey covers the empty state's hint.
//
// The hint used to read "输入后按任意阶段键（1–7）继续，顶部的 token 条会立刻出现", which tells a
// reader that the view waits for a stage key. It does not: promptChanged recomputes on every
// keystroke, so the token strip appears as they type. The wording sent them to press a key that
// changes what they are looking at rather than what is computed.
//
// This asserts the behaviour the new wording describes — typing alone reaches the forward pass —
// so the sentence cannot drift away from it again.
func TestTypingRecomputesWithoutAKey(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	a.Step = 2
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	if len(a.Trace.Tokens) != 0 {
		t.Fatalf("the empty prompt already has %d tokens", len(a.Trace.Tokens))
	}

	if err := tr.Click("输入"); err != nil {
		t.Fatal(err)
	}
	tr.Frame()
	tr.TypeKey(0, ui.KeyH, "h")
	tr.TypeKey(0, ui.KeyI, "i")
	tr.Frame()

	if a.Prompt != "hi" {
		t.Fatalf("typing produced %q", a.Prompt)
	}
	if len(a.Trace.Tokens) != 2 {
		t.Fatalf("typing \"hi\" left %d tokens; the view does not recompute without a key",
			len(a.Trace.Tokens))
	}
	if a.Step != 2 {
		t.Fatalf("the stage changed to %d while typing; the key handler should not fire here",
			a.Step)
	}
}

// TestUndoRestoresEverythingTheActionReplaced covers the state a destructive prompt
// action replaces: the text, the token selection and the sample record.
//
// The scope widened this round. It began as "undo the last sample", which is the
// smallest loss the app can cause — one byte. The presets and 重置 cause a much
// larger one: they replace a sentence the reader typed. Measured before the change,
// typing "my own sentence here" and clicking 中文 left the field at the preset with
// undo == nil, so the reader's own text was gone with no way back.
func TestUndoRestoresEverythingTheActionReplaced(t *testing.T) {
	// Sampling: one byte, and the selection moves to what was produced.
	a := newTestApp(t)
	prompt, sel := a.Prompt, a.SelToken
	if a.undo != nil {
		t.Fatal("a fresh app already offers an undo")
	}
	if a.undoPrompt() {
		t.Fatal("undo returned true with nothing to undo")
	}
	a.sampleNext()
	if a.Prompt == prompt {
		t.Fatal("sampling did not change the prompt; the rest of this test would be vacuous")
	}
	if a.undo == nil {
		t.Fatal("sampling did not record an undo point")
	}
	if a.undo.what != "采样续写" {
		t.Fatalf("the sample's undo point is labelled %q", a.undo.what)
	}
	if a.SelToken == sel {
		t.Fatalf("the sample left the selection on token %d, the same one it started from", sel)
	}
	if !a.undoPrompt() {
		t.Fatal("undo returned false with a sample to take back")
	}
	if a.Prompt != prompt {
		t.Fatalf("undo left the prompt as %q, want %q", a.Prompt, prompt)
	}
	if a.SelToken != sel {
		t.Fatalf("undo left the selection on token %d, want %d — the token the sample "+
			"selected no longer exists, so leaving it would point past the end",
			a.SelToken, sel)
	}
	if a.LastSample != "" {
		t.Fatalf("undo left the sample record %q; it describes a byte that is gone", a.LastSample)
	}
	// One step, not a stack: a second undo has nothing to take back.
	if a.undoPrompt() {
		t.Fatal("undo kept going after the only sample it had")
	}

	// A preset replacing typed text — the case this round added.
	b := newTestApp(t)
	b.Prompt = "my own sentence here"
	b.recompute()
	typed := b.Prompt
	b.applyPrompt("注意力就是加权求和")
	if b.Prompt == typed {
		t.Fatal("the preset did not change the prompt")
	}
	if b.undo == nil || b.undo.what != "换成预设提示词" {
		t.Fatalf("a preset click left no labelled undo point: %+v", b.undo)
	}
	if !b.undoPrompt() {
		t.Fatal("a preset click could not be undone")
	}
	if b.Prompt != typed {
		t.Fatalf("undoing a preset gave %q, want the typed text %q", b.Prompt, typed)
	}

	// 重置, the other way to throw away a whole prompt.
	c := newTestApp(t)
	c.Prompt = "another sentence entirely"
	c.recompute()
	c.applyPrompt(defaultPrompt)
	if !c.undoPrompt() {
		t.Fatal("重置 could not be undone")
	}
	if c.Prompt != "another sentence entirely" {
		t.Fatalf("undoing 重置 gave %q", c.Prompt)
	}
}

// TestUndoSurvivesTypingButNotAnotherReplacement pins the rule that decides which
// actions touch the undo point.
//
// Replacing the whole prompt records one; typing does not. That asymmetry is the
// point: an undo point on every keystroke would make Ctrl+Z walk back through the
// reader's editing one character at a time, which is a text editor's behaviour and
// not this app's.
//
// Typing must also not *destroy* an existing point. The earlier version of this test
// asserted the opposite — that any prompt change dropped it — which was right while
// the undo only meant "the sample", and is wrong now: a reader who pasted a preset
// over their paragraph should still be able to take that back after typing one more
// character.
func TestUndoSurvivesTypingButNotAnotherReplacement(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "the sentence I want back"
	a.recompute()
	typed := a.Prompt
	a.applyPrompt("注意力就是加权求和")
	if a.undo == nil {
		t.Fatal("the preset click recorded no undo point")
	}
	// Typing on top of the preset must leave the point alone.
	a.Prompt += "x"
	a.promptChanged()
	if a.undo == nil {
		t.Fatal("typing destroyed the undo point, so the reader's own sentence is " +
			"unrecoverable after a single keystroke on top of the preset")
	}
	if !a.undoPrompt() {
		t.Fatal("the point was there but undo refused")
	}
	if a.Prompt != typed {
		t.Fatalf("undo gave %q, want the sentence that was replaced %q", a.Prompt, typed)
	}

	// A second replacement overwrites the point rather than stacking: the reader
	// takes back the last thing that happened, which is the only one they can reason
	// about.
	b := newTestApp(t)
	b.Prompt = "one"
	b.recompute()
	b.applyPrompt("two")
	b.applyPrompt("three")
	if !b.undoPrompt() {
		t.Fatal("no undo point after two replacements")
	}
	if b.Prompt != "two" {
		t.Fatalf("after two replacements undo gave %q, want the previous one, two", b.Prompt)
	}
	if b.undoPrompt() {
		t.Fatal("undo went back further than one step")
	}
}

// TestUndoButtonNamesTheActionItTakesBack keeps the control honest twice over: it
// appears only when there is something to take back, and it says which action that
// is.
//
// The fixed label "撤销上次采样" stopped being true when the undo grew to cover the
// presets: after a preset click it would have offered to undo a sample that was
// never drawn.
func TestUndoButtonNamesTheActionItTakesBack(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if tr.HasText("撤销「") || tr.HasText("撤销上次采样") {
		t.Fatal("the undo button is on the card before anything destructive has happened")
	}
	a.sampleNext()
	tr.Frame()
	if !tr.HasText("撤销「采样续写」") {
		t.Fatal("the undo button does not name the sample it takes back")
	}
	a.undoPrompt()
	tr.Frame()
	if tr.HasText("撤销「") {
		t.Fatal("the undo button is still there after the state it undoes is gone")
	}

	// After a preset the label names the preset, not a sample.
	b := newTestApp(t)
	b.Step = 6
	trb := ui.NewTester(b.View, 1500, 940)
	trb.Frame()
	b.applyPrompt("中文 preset")
	trb.Frame()
	if !trb.HasText("撤销「换成预设提示词」") {
		t.Fatal("after a preset click the undo button does not name the preset")
	}
}

// TestCtrlZUndoesTheSampleFromAnyStage: the shortcut is global, so a reader who
// sampled from the top bar while standing on the tokenising stage can take it back
// without first navigating to the output stage to find the button.
func TestCtrlZUndoesTheSampleFromAnyStage(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, step := range []int{0, 3, 6} {
		a.selectStep(step)
		tr.Frame()
		prompt := a.Prompt
		a.sampleNext()
		tr.Frame()
		if a.Prompt == prompt {
			t.Fatalf("step %d: sampling did not change the prompt", step+1)
		}
		tr.Key(ui.Ctrl, ui.KeyZ)
		tr.Frame()
		if a.Prompt != prompt {
			t.Fatalf("step %d: Ctrl+Z left the prompt as %q, want %q", step+1, a.Prompt, prompt)
		}
	}
}

// TestShortcutLineAdvertisesUndoOnlyWithSomethingToUndo keeps the in-app line in
// step with the key, the same way the card's button is kept in step with it.
//
// The line under every lesson is a promise; listing Ctrl+Z while it does nothing
// teaches the reader that this app ignores keys pressed.
func TestShortcutLineAdvertisesUndoOnlyWithSomethingToUndo(t *testing.T) {
	line := func(a *App) string {
		tr := ui.NewTester(a.View, 1500, 2000)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "快捷键：") {
				return s
			}
		}
		return ""
	}
	a := newTestApp(t)
	if got := line(a); strings.Contains(got, "Ctrl+Z") {
		t.Fatalf("the shortcut line advertises Ctrl+Z with nothing to undo: %q", got)
	}
	a.sampleNext()
	if got := line(a); !strings.Contains(got, "Ctrl+Z") {
		t.Fatalf("the shortcut line does not advertise Ctrl+Z after a sample: %q", got)
	}
	a.undoPrompt()
	if got := line(a); strings.Contains(got, "Ctrl+Z") {
		t.Fatalf("the shortcut line still advertises Ctrl+Z after the sample was undone: %q", got)
	}
}

// TestHeatmapPointerLandsOnTheDrawnCell is the regression test for a defect that
// made every hover and every click-to-pin name a neighbouring cell.
//
// The painter draws the cells offset by the axis-label gutter (20 DIP right, 9 DIP
// down) and sizes them from the box *minus* that gutter; the hit test mapped the
// pointer through the whole box with no offset. Measured at 1500×940 on the score
// matrix with T=36: pointing at the drawn centre of cell (6,0) reported (8,2), and
// one drawn cell reported no cell at all. 119 of 119 queried cells were wrong.
//
// The existing tests could not catch it: they assert that *a* cell is reported, or
// that the reported cell is inside the causal triangle, both of which remain true
// when every answer is shifted by two columns. This one asks each drawn centre for
// its own coordinates.
func TestHeatmapPointerLandsOnTheDrawnCell(t *testing.T) {
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
		for _, step := range []int{2, 3} { // score matrix, attention matrix
			a := newTestApp(t)
			a.Step = step
			a.SelToken = 30
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			// At the smallest window the matrix starts below the fold (measured: the
			// grid begins at y=622 in a 720-tall window and its last cells are past
			// the bottom edge), so scroll it into view before asking where its cells
			// are. Find and the painter both report scrolled coordinates.
			for i := 0; i < 8; i++ {
				r := a.lastHeatmapRect
				if r.Y >= 0 && r.Y+r.H <= float32(size.h) {
					break
				}
				tr.Move(float32(size.w)/2, float32(size.h)/2)
				tr.Wheel(float32(size.w)/2, float32(size.h)/2, 0, 200)
				tr.Advance(2e9)
				tr.Frame()
			}

			r := a.lastHeatmapRect
			if r.W <= 0 || r.H <= 0 {
				t.Fatalf("%dx%d step %d: the heatmap painter did not run", size.w, size.h, step+1)
			}
			tokens := len(a.Trace.Tokens)
			grid := heatmapGrid(r, heatmapGutter)
			cw, ch := grid.W/float32(tokens), grid.H/float32(tokens)

			// A spread of cells rather than all of them: enough to catch a shift on
			// either axis and on both ends of the matrix, without a 90-second sweep.
			checked := 0
			for q := 1; q < tokens && q < 14; q += 3 {
				for _, k := range []int{0, q / 2, q} {
					x := grid.X + (float32(k)+0.5)*cw
					y := grid.Y + (float32(q)+0.5)*ch
					tr.Move(x, y)
					tr.Frame()
					if a.HoverQ != q || a.HoverK != k {
						t.Fatalf("%dx%d step %d: the pointer at the centre of drawn cell "+
							"(%d,%d) reports (%d,%d); the hit test and the painter disagree "+
							"about where the grid is",
							size.w, size.h, step+1, q, k, a.HoverQ, a.HoverK)
					}
					checked++
				}
			}
			// And the gutter itself is not a cell: pointing into the label margin must
			// report nothing rather than cell 0.
			//
			// This is the assertion that caught the truncation bug: int(-18/cw) is -1 while cw
			// is under 18 DIP and 0 once it is over, so a wider matrix silently turned the label
			// margin into cell 0. The wider sizes in this test's list are what exercise it —
			// at 1180x720 the same probe could never fail. Keep a large size in that list.
			if grid.X > r.X {
				tr.Move(r.X+2, grid.Y+ch/2)
				tr.Frame()
				if a.HoverQ >= 0 && a.HoverK >= 0 {
					t.Fatalf("%dx%d step %d: a pointer in the axis-label gutter reports cell (%d,%d)",
						size.w, size.h, step+1, a.HoverQ, a.HoverK)
				}
				checked++
			}
			if checked < 4 {
				t.Fatalf("%dx%d step %d: only %d positions checked", size.w, size.h, step+1, checked)
			}
		}
	}
}

// TestKeyboardScrollsTheStageColumn covers the reachability gap this round found:
// at 1180×720 the tallest stage column puts over 1800 DIP of content below the
// fold — more than three full viewports — and mygo only routes scroll keys to a
// scroller that is under the pointer or in the focus chain.
//
// Measured before this change: with the pointer parked in a corner, PageDown,
// Down, End, Space and the wheel all left Scroll.Y at 0, so a keyboard-only reader
// could not reach the lesson card, the workings blocks or the shortcut line.
func TestKeyboardScrollsTheStageColumn(t *testing.T) {
	const w, h = 1180, 720
	// The residual stage is the tallest at this size (measured MaxY 1805, which is what
	// TestReadmeViewportOverflowMatchesTheLayout re-derives for the README's figure; this
	// comment used to say 856, from before the stage gained its analysis cards).
	scrollable := func(a *App, tr *ui.Tester) {
		tr.Move(0, 0) // pointer out of the column, so nothing else can scroll it
		tr.Frame()
	}
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, w, h)
	tr.Frame()
	scrollable(a, tr)
	if a.Scroll.MaxY <= 0 {
		t.Fatalf("stage 5 does not scroll at %dx%d, so this test cannot measure anything", w, h)
	}
	if a.StageViewportH <= 0 {
		t.Fatal("the stage column's viewport height was never measured, so a page cannot be sized")
	}

	// PageDown moves by roughly a page — a fraction of the viewport rather than all
	// of it, so the next page overlaps and the reader keeps their place.
	tr.Key(0, ui.KeyPageDown)
	tr.Frame()
	tr.Frame()
	first := a.Scroll.Y
	if first <= 0 {
		t.Fatalf("PageDown left the column at Y=0; the keyboard still cannot scroll it")
	}
	if want := a.StageViewportH * 0.9; first < want*0.9 || first > want*1.1 {
		t.Fatalf("PageDown moved %.0f DIP, want about %.0f (0.9 of the %.0f-DIP viewport)",
			first, want, a.StageViewportH)
	}

	// End reaches the bottom and Home returns to the top.
	tr.Key(0, ui.KeyEnd)
	tr.Frame()
	tr.Frame()
	if a.Scroll.Y != a.Scroll.MaxY {
		t.Fatalf("End left the column at %.0f, the content ends at %.0f", a.Scroll.Y, a.Scroll.MaxY)
	}
	tr.Key(0, ui.KeyHome)
	tr.Frame()
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("Home left the column at %.0f, want its top", a.Scroll.Y)
	}

	// PageUp from the top stays at the top rather than going negative.
	tr.Key(0, ui.KeyPageUp)
	tr.Frame()
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("PageUp from the top left the column at %.0f", a.Scroll.Y)
	}

	// And the content really becomes reachable: the lesson card is what sits at the
	// bottom of every stage, and before this change Find still reported it while no
	// key could bring it into the window.
	tr.Key(0, ui.KeyEnd)
	tr.Frame()
	tr.Frame()
	if !tr.HasText("这一段在讲什么") {
		t.Fatal("after scrolling to the end the lesson card is still not rendered")
	}
}

// TestScrollKeysReachTheAppFromThePromptField: the four scroll keys must not be
// eaten by the text field, since the reader may have just been typing.
//
// This is the half of the rule that differs from Space and Enter, which a focused
// input does consume — they edit or submit the text, while PageDown does not.
func TestScrollKeysReachTheAppFromThePromptField(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	tr.Move(0, 0)
	tr.Frame()
	tr.Key(0, ui.KeyTab) // focus the prompt field
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("the column was already scrolled to %.0f before the key", a.Scroll.Y)
	}
	tr.Key(0, ui.KeyPageDown)
	tr.Frame()
	tr.Frame()
	if a.Scroll.Y <= 0 {
		t.Fatal("PageDown with the prompt field focused did not scroll; the reader who just " +
			"typed cannot reach the rest of the stage without moving the pointer")
	}
}

// TestWorkingsBlocksCanBeCopied covers the one capability the app had none of: the
// numbers on screen could not be got off the screen.
//
// Every workings block exists to be re-added by hand — each says so about itself
// ("照着手算并得到卡片上那个值") — so a reader doing that needs the figures somewhere
// they can work with. The text is painted rather than selectable (no Selectable or
// text-selection API anywhere in this app), so before this the only way to move
// "0.932751" into a note was to read it off the screen and retype it.
//
// The assertion is deliberately against the clipboard *and* the rendered text: the
// clipboard has to hold exactly the lines the reader can see, or the button copies
// something other than what it sits under.
func TestWorkingsBlocksCanBeCopied(t *testing.T) {
	// The embed stage is the simplest block: two lines, and both are on screen at
	// the default window.
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 2200)
	tr.Frame()

	r, ok := tr.Find("复制算式 2 行")
	if !ok {
		t.Fatal("the embed stage's workings block has no copy control")
	}
	tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tr.Frame()
	got := tr.Clipboard()
	if got == "" {
		t.Fatal("clicking 复制 left the clipboard empty")
	}
	lines := strings.Split(got, "\n")
	if len(lines) != 2 {
		t.Fatalf("the block says 2 行 but the clipboard holds %d line(s): %q", len(lines), got)
	}
	// Each copied line must be one the tester renders, so the button cannot copy a
	// stale or differently-formatted string.
	for _, line := range lines {
		if !tr.HasText(line) {
			t.Fatalf("the clipboard holds a line the screen does not render: %q", line)
		}
	}
	// And the copy is the derivation, not a summary: the second line is the one
	// that lands on the answer, which is what a reader checks by hand.
	if !strings.Contains(lines[1], "=") {
		t.Fatalf("the copied block's last line is not an equation: %q", lines[1])
	}
}

// TestCopyControlExistsOnEveryStageWithAWorkingsBlock: every stage carries a workings
// block, and a reader should find 复制 on each of them rather than on some.
//
// This counts the control per stage instead of spot-checking one, because the
// failure mode is a call site left on the old helper — which renders fine and
// simply has no way to copy.
//
// The header used to say "eleven blocks across the seven stages", which was a count of
// call sites rather than of rendered blocks: several call sites are inside loops, so the
// bundled prompt draws far more. A number in a comment that nothing re-derives drifts, so
// the header no longer carries one.
func TestCopyControlExistsOnEveryStageWithAWorkingsBlock(t *testing.T) {
	// Every stage renders at least one workings block, but not unconditionally: the
	// score stage's block only exists once a cell is selected, and says so in its
	// otherwise-visible place ("把指针移到矩阵的某个格子上…"). So the cell is pinned
	// here — the block and its copy control appear together, which is the property
	// worth recording.
	want := map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true, 6: true}
	for step := range lessons {
		a := newTestApp(t)
		a.Step = step
		a.KeyQ, a.KeyK = 20, 5 // pin a cell so the score stage's block is conditional-true
		// A tall window so every block is laid out: the control is inside the
		// scroll view and Find reports scrolled coordinates.
		tr := ui.NewTester(a.View, 1500, 3000)
		tr.Frame()
		found := 0
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "复制算式 ") {
				found++
			}
		}
		if found > 0 != want[step] {
			t.Fatalf("stage %d renders %d copy controls; this test records that it should %v",
				step+1, found, want[step])
		}
	}
}

// TestCopiedWorkingsMatchThePinnedCell ties the copied arithmetic to the cell the
// reader pinned, which is the case the block exists for: pin a cell, copy its
// derivation, check it on paper.
func TestCopiedWorkingsMatchThePinnedCell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2 // score matrix
	a.SelToken = 20
	a.KeyQ, a.KeyK = 20, 5
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()

	// Find the copy control whose block belongs to the pinned cell: the score
	// workings is three lines.
	r, ok := tr.Find("复制算式 3 行")
	if !ok {
		t.Fatal("the pinned cell's workings block has no copy control")
	}
	tr.ClickAt(r.X+r.W/2, r.Y+r.H/2)
	tr.Frame()
	got := tr.Clipboard()
	// The cell's own numbers have to be in it, read from the trace rather than
	// hard-coded, so the check follows the weights and the prompt.
	h := a.layer().Heads[a.Head]
	want := fmt.Sprintf("%+.4f", h.Scores[a.KeyQ][a.KeyK])
	if !strings.Contains(got, want) {
		t.Fatalf("the copied derivation for the pinned cell does not contain its score %s:\n%s",
			want, got)
	}
}

// TestReadmeRailLimitationIsStillTrue keeps the two token-rail limitations the
// README now states from quietly becoming false.
//
// The README says the strip "超出视口后只能用指针拖": it scrolls horizontally, draws
// no scrollbar, has no horizontal key (←/→ switch stage) and the vertical wheel does
// not move it. It also says a keyboard reader can still reach every chip because
// Shift+←/→ moves the selection and the strip follows it. Both halves matter — the
// second is what makes the first tolerable — and neither was recorded anywhere.
func TestReadmeRailLimitationIsStillTrue(t *testing.T) {
	long := func() *App {
		a := newTestApp(t)
		a.Prompt = strings.Repeat("attention ", 12)
		a.recompute()
		return a
	}

	a := long()
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	if a.RailScroll.MaxX <= 0 {
		t.Fatalf("the long prompt no longer overflows the strip (MaxX=%.0f), so this test "+
			"cannot measure the limitation", a.RailScroll.MaxX)
	}
	tr.Move(0, 0)
	tr.Frame()

	// The vertical wheel over the strip does not scroll it horizontally.
	x0 := a.RailScroll.X
	tr.Move(400, 160)
	tr.Frame()
	tr.Wheel(400, 160, 0, 400)
	tr.Advance(2e9)
	tr.Frame()
	if a.RailScroll.X != x0 {
		t.Fatalf("the vertical wheel now scrolls the strip horizontally (%.0f -> %.0f); "+
			"the README says it does not", x0, a.RailScroll.X)
	}

	// And the escape hatch the README promises: Shift+← moves the selection and the
	// strip keeps the selected chip on screen, so a keyboard reader can walk the
	// whole sequence without a pointer.
	//
	// The thing to assert is the chip's position, not the scroll offset. The first
	// version of this test required RailScroll.X to decrease on every press and
	// failed — ScrollIntoView scrolls the least it can, so a chip that is already
	// visible needs no scroll at all, and the strip only moves once the selection
	// nears the left edge. Measured: walking 63 → 48 moves the chip from x=657 to
	// x=277 while the offset stays at 1321 for the first fourteen presses.
	tr.Move(0, 0)
	tr.Frame()
	startSel := a.SelToken
	for i := 0; i < 15; i++ {
		tr.Key(ui.Shift, ui.KeyLeft)
		tr.Frame()
		tr.Frame()
		r, ok := tr.Find("token " + itoa(a.SelToken))
		if !ok {
			t.Fatalf("press %d: the selected chip (token %d) is not in the frame at all",
				i+1, a.SelToken)
		}
		if r.W <= 0 {
			t.Fatalf("press %d: the selected chip (token %d) has zero width, so it is "+
				"clipped out of the strip", i+1, a.SelToken)
		}
		if r.X < 0 || r.X+22 > 1180 {
			t.Fatalf("press %d: the selected chip (token %d) is off screen at x=%.0f; the "+
				"README says a keyboard reader can reach every chip", i+1, a.SelToken, r.X)
		}
	}
	if a.SelToken >= startSel {
		t.Fatalf("Shift+Left did not move the selection (still %d)", a.SelToken)
	}
}

// TestShortcutLineExplainsFocusPrecedence keeps the shortcut line from promising
// something a focused control takes away.
//
// The line says "空格 播放/暂停 · Enter 采样" unconditionally, which is true while
// nothing is focused and false the moment the reader tabs to a button: measured, with
// 重置 focused Enter activates the button instead of sampling. That precedence is
// correct — the reader chose to operate that control — but an advertised key that
// silently does something else needs its exception stated where the key is listed.
func TestShortcutLineExplainsFocusPrecedence(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 2000)
	tr.Frame()
	var line, note string
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "快捷键：") {
			line = s
		}
		if strings.HasPrefix(s, "（用 Tab 选中某个按钮时") {
			note = s
		}
	}
	if line == "" {
		t.Fatal("the shortcut line is missing")
	}
	for _, want := range []string{"空格", "Enter"} {
		if !strings.Contains(line, want) {
			t.Fatalf("the line no longer advertises %q", want)
		}
	}
	if note == "" {
		t.Fatal("the shortcut line no longer explains that a focused button takes Space and Enter")
	}

	// And the exception is real, not decorative: with 重置 focused, Enter must
	// activate the button rather than sample.
	b := newTestApp(t)
	b.Step = 3
	b.Prompt = "zzz"
	b.recompute()
	trb := ui.NewTester(b.View, 1500, 940)
	trb.Frame()
	trb.Move(0, 0)
	trb.Frame()
	trb.Key(0, ui.KeyTab) // prompt field
	trb.Frame()
	trb.Key(0, ui.KeyTab) // 重置
	trb.Frame()
	before := len(b.Prompt)
	trb.Key(0, ui.KeyEnter)
	trb.Frame()
	trb.Frame()
	if len(b.Prompt) == before+1 {
		t.Fatal("Enter with 重置 focused sampled a byte; the note says a focused button " +
			"takes Enter, so the exception it describes is not the one that happens")
	}
	if b.Prompt != defaultPrompt {
		t.Fatalf("Enter with 重置 focused left the prompt at %q; the note says the button "+
			"is what gets pressed", b.Prompt)
	}
}

// TestEscReleasesEverythingItClaims pins what the README says Esc does.
//
// The row used to read only "停止自动播放", which understated it by two thirds: Esc
// also releases the pinned matrix cell and the pinned per-position prediction. A
// reader looking for how to let go of a pin would read that row and conclude Esc was
// not the key, then hunt for the cell to click again.
func TestEscReleasesEverythingItClaims(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.KeyQ, a.KeyK = 5, 3 // pinned matrix cell
	a.PredSel = 2         // pinned prediction
	a.Playing = true
	a.NextAt = a.NextAt.Add(1)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Key(0, ui.KeyEscape)
	tr.Frame()
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Fatalf("Esc left the matrix cell pinned at (%d,%d)", a.KeyQ, a.KeyK)
	}
	if a.PredSel >= 0 {
		t.Fatalf("Esc left the position pinned at %d", a.PredSel)
	}
	if a.Playing {
		t.Fatal("Esc left autoplay running")
	}
	// And the README says all three.
	doc := readREADME(t)
	var row string
	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "| `Esc` |") {
			row = line
		}
	}
	if row == "" {
		t.Fatal("the README no longer documents Esc")
	}
	for _, want := range []string{"自动播放", "格子", "预测"} {
		if !strings.Contains(row, want) {
			t.Fatalf("the Esc row does not mention %q, which Esc releases:\n%s", want, row)
		}
	}
}

// TestUndoButtonSurvivesItsOwnClick pins a crash that shipped for several rounds: the
// 撤销 button dereferenced its own undo record after the click had cleared it.
//
// The top bar built the button, handled the click, and then read a.undo.what to build the
// tooltip. undoPrompt() sets a.undo = nil, but the closure keeps running, so the tooltip
// line dereferenced a nil pointer and the whole window panicked. Measured: clicking 撤销
// after sampling took the app down at app.go:883.
//
// Why nothing caught it: TestEveryAdvertisedKeyHasAClickablePathOnEveryStage checks that
// the button is *on screen* and never clicks it, and the Ctrl+Z tests go through
// undoPrompt() directly, which never reaches the tooltip line. A control can be visible,
// correctly labelled, and still fatal, so the click has to be exercised.
func TestUndoButtonSurvivesItsOwnClick(t *testing.T) {
	// Each of the three actions that create an undo point, undone through the button.
	cases := []struct {
		name  string
		setup func(*ui.Tester, *App)
	}{
		{"after sampling", func(tr *ui.Tester, _ *App) { tr.Click("采样下一个 token") }},
		{"after a preset", func(tr *ui.Tester, _ *App) { tr.Click("中文") }},
		{"after reset", func(tr *ui.Tester, _ *App) {
			// 重置 only records an undo point when the prompt is not already the
			// default, so a preset has to come first.
			tr.Click("中文")
			tr.Frame()
			tr.Click("重置")
		}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			a := newTestApp(t)
			tr := ui.NewTester(a.View, 1180, 720)
			tr.Frame()
			tc.setup(tr, a)
			tr.Frame()
			if a.undo == nil {
				t.Fatalf("%s: no undo point was recorded, so this case tests nothing", tc.name)
			}
			before := a.Prompt
			// The click must not panic. A panic here fails the test rather than the process.
			tr.Click("撤销")
			tr.Frame()
			if a.undo != nil {
				t.Fatalf("%s: the undo point survived its own undo", tc.name)
			}
			if a.Prompt == before {
				t.Logf("%s: the prompt was already the value being restored", tc.name)
			}
			// And the button is gone, not left behind pointing at nothing.
			if tr.HasText("撤销") {
				t.Fatalf("%s: the 撤销 button is still drawn with no undo point left", tc.name)
			}
		})
	}
}

// TestUndoTooltipNamesTheActionBeforeTheClick is the narrower assertion behind the crash:
// the tooltip text has to be computed from the record while the record still exists.
func TestUndoTooltipNamesTheActionBeforeTheClick(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	tr.Click("中文")
	tr.Frame()
	if a.undo == nil {
		t.Fatal("the preset recorded no undo point")
	}
	what := a.undo.what
	// Park the pointer away, then hover the button so its tooltip has to render.
	tr.Move(600, 400)
	tr.Advance(2 * time.Second)
	r, ok := tr.Find("撤销")
	if !ok {
		t.Fatal("the 撤销 button is missing")
	}
	tr.Move(r.X+r.W/2, r.Y+r.H/2)
	var found bool
	for i := 0; i < 40; i++ {
		tr.Advance(50 * time.Millisecond)
		for _, s := range tr.Texts() {
			if strings.Contains(s, "撤销「"+what+"」") {
				found = true
			}
		}
		if found {
			break
		}
	}
	if !found {
		t.Fatalf("hovering 撤销 does not show the action it would take back (%q)", what)
	}
}

// TestUndoLabelNamesTheActionThatJustHappened checks the label against the action, not
// just against the record existing.
//
// The README promises the button "会写明撤销的是哪一次（例如「撤销『换成预设提示词』」）".
// 重置 went through the same code path as a preset chip, so after clicking 重置 the button
// offered to undo "换成预设提示词" — an action the reader had not taken. The undo record
// was right and the label was wrong, which is the kind of wrong a test that only checks
// "the button appears" cannot see.
func TestUndoLabelNamesTheActionThatJustHappened(t *testing.T) {
	cases := []struct {
		name string
		do   func(tr *ui.Tester)
		want string
	}{
		{"preset", func(tr *ui.Tester) { tr.Click("中文") }, "换成预设提示词"},
		{"reset", func(tr *ui.Tester) {
			tr.Click("中文")
			tr.Frame()
			tr.Click("重置")
		}, "重置"},
		{"sampling", func(tr *ui.Tester) { tr.Click("采样下一个 token") }, "采样续写"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			a := newTestApp(t)
			tr := ui.NewTester(a.View, 1180, 720)
			tr.Frame()
			tc.do(tr)
			tr.Frame()
			if a.undo == nil {
				t.Fatalf("%s: no undo point recorded", tc.name)
			}
			if a.undo.what != tc.want {
				t.Fatalf("%s: the undo label says %q; the action taken was %q",
					tc.name, a.undo.what, tc.want)
			}
		})
	}
}

// TestHeatmapCanBeScrolledFullyIntoView records the constraint that blocks enlarging the matrix.
//
// Two attempts to make the matrix grow on wide windows broke
// TestHeatmapPointerLandsOnTheDrawnCell rather than any size assertion, so the invariant they
// violated was never written down. It is not "the matrix fits the window" — at 1180x720 it never
// has, ending near y=999 in a 720-tall window, and the stage scrolls to reach it. The real
// invariant is that some scroll position puts the whole matrix inside the viewport.
//
// The mechanism, measured at 1500x940 with T=36: growing the matrix to 653 DIP put its top at
// y=159, while the stage viewport begins at y=228 because the step bar and its controls are fixed
// above it. A matrix taller than the viewport cannot satisfy that at any offset, so its top rows
// stay under the chrome and PointerPosition reports no element there.
func TestHeatmapCanBeScrolledFullyIntoView(t *testing.T) {
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
		for step := 2; step <= 3; step++ {
			a := newTestApp(t)
			a.Prompt = "The transformer reads the tokens and"
			a.recompute()
			a.Step = step
			a.SelToken = 30
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()

			fits := func() bool {
				r := a.lastHeatmapRect
				return r.W > 0 && r.H > 0 && r.Y >= 0 && r.Y+r.H <= float32(size.h)
			}
			ok := false
			for i := 0; i < 12 && !ok; i++ {
				if fits() {
					ok = true
					break
				}
				tr.Move(float32(size.w)/2, float32(size.h)/2)
				tr.Wheel(float32(size.w)/2, float32(size.h)/2, 0, 200)
				tr.Advance(2e9)
				tr.Frame()
			}
			r := a.lastHeatmapRect
			if !ok {
				t.Fatalf("%dx%d step %d: no scroll position puts the %.0f-DIP matrix inside the "+
					"%d-DIP viewport (it ends at y=%.0f); a matrix taller than the viewport cannot "+
					"be fully shown, so its top rows sit under the fixed step bar and report no "+
					"cell on hover", size.w, size.h, step+1, r.H, size.h, r.Y+r.H)
			}
		}
	}
}
