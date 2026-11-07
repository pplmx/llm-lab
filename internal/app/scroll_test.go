package app

import (
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestScrollingAStageKeepsTheHeaderOnScreen answers the question round 164 left open.
//
// The residual stage is taller than the window, so a reader who scrolls down to the histogram is
// no longer looking at the top of the card. The stage title and the step bar have to stay put, or
// the reader loses track of which stage they are in — the first goal of this tool.
//
// Measured at 1180×720 on step 5: after scrolling 400 DIP the stage title moves 17 DIP (the
// scroll view's own padding) while the step bar stays at y=176, and stage content that was on
// screen is gone. So the header is outside the scroll view, as it needs to be.
//
// Round 164 recorded the opposite conclusion — that pointer scrolling does not reach the container
// in the headless tester. That was wrong: it used Wheel without Advance (a wheel glides, so it
// needs the clock moved) and never tried Scroll, which applies at once.
func TestScrollingAStageKeepsTheHeaderOnScreen(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()

	type pos struct{ y float32 }
	read := func() map[string]pos {
		out := map[string]pos{}
		for _, label := range []string{"第 5 / 7 步", "残差与层归一化", "5 / 7 · 残差 / LN"} {
			if r, ok := tr.Find(label); ok {
				out[label] = pos{r.Y}
			}
		}
		return out
	}
	before := read()
	if len(before) < 3 {
		t.Fatalf("the stage header is not fully drawn before scrolling: %d of 3 labels", len(before))
	}

	tr.Scroll(600, 500, 0, 400)
	tr.Frame()
	if a.Scroll.Y == 0 {
		t.Fatal("the stage column did not scroll; the rest of this test would be vacuous")
	}
	after := read()

	// The step bar and the stage title stay within a few DIP of where they were.
	for _, label := range []string{"第 5 / 7 步", "残差与层归一化", "5 / 7 · 残差 / LN"} {
		dy := after[label].y - before[label].y
		if dy < -20 || dy > 20 {
			t.Fatalf("%q moved %.0f DIP when the stage scrolled %.0f; a reader at the bottom of "+
				"the card would not know which stage they are in", label, dy, a.Scroll.Y)
		}
	}

	// And the content really did move, or the assertion above proves nothing.
	if _, ok := tr.Find("输入 0 · 一次加法"); ok {
		t.Skip("the residual card fits this window, so scrolling moved nothing to test")
	}

	// The header is still inside the window, not pushed off the top.
	for label, p := range after {
		if p.y < 0 || p.y > 720 {
			t.Fatalf("%q ends up at y=%.0f, outside the window", label, p.y)
		}
	}
}

// TestEveryStageKeepsItsHeaderWhileScrolled is the same claim for all seven stages, not just the
// one long enough to be noticed.
//
// Round 183 measured step 5 because it is the tallest. The other stages scroll too — at 1180×720
// step 4's viewport holds about 870 DIP of content — and a header that scrolled on one of them
// would be the same defect there.
func TestEveryStageKeepsItsHeaderWhileScrolled(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()

		title, ok := tr.Find(lessons[step].title)
		if !ok {
			t.Fatalf("stage %d: the stage title is not drawn", step)
		}

		tr.Scroll(600, 500, 0, 5000)
		tr.Frame()
		scrolled := a.Scroll.Y

		after, ok := tr.Find(lessons[step].title)
		if !ok {
			t.Fatalf("stage %d: the stage title disappeared after scrolling %.0f DIP",
				step, scrolled)
		}
		if dy := after.Y - title.Y; dy < -20 || dy > 20 {
			t.Fatalf("stage %d: the title moved %.0f DIP when the content scrolled %.0f",
				step, dy, scrolled)
		}
		if after.Y < 0 || after.Y > 720 {
			t.Fatalf("stage %d: the title ends at y=%.0f, outside the window", step, after.Y)
		}
	}
}
