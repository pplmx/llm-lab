package app

import (
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
