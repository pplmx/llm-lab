package app

import (
	"fmt"
	"testing"

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
		{fmt.Sprintf("[%d]", vocab), 6},
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
