package app

import (
	"fmt"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// The stages are taller than the window at every size the app supports, which is
// fine as long as the overflow is reachable and obviously so. These record that
// contract, because a layout change that buried a callout with no way to scroll
// to it would still pass every content test.
const (
	// The callout on each stage that carries a number, by its opening phrase.
	calloutPrefix = "「"
)

// TestEveryStageCanBeScrolledToItsEnd checks the scroll offset actually moves on
// every stage and window size. Find returns layout coordinates that do not shift
// with the scroll, so the offset is the honest measurement here.
func TestEveryStageCanBeScrolledToItsEnd(t *testing.T) {
	// The margin check that used to live in a separate test is collected here instead: this one
	// already wheels every stage at both sizes, so repeating the sweep cost six more seconds of
	// scroll animation for the same numbers.
	const perWheel = 400
	tallest, tallestWhere := float32(0), ""
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}} {
		for step := 0; step < 7; step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			tr.Move(600, 400)
			tr.Frame()
			// Eight wheels, not twelve: measured, the tallest stage at the smallest window
			// needs six 400-DIP steps to reach MaxY (stage 5, MaxY 1805), so eight clears it
			// on every stage at both sizes with margin. Twelve was a round number that
			// happened to work. The saving here is small — the scroll animation, not the
			// wheel count, is what this test spends its time on — but it is nine fewer frame
			// renders per stage per size for the same assertion.
			//
			// The MaxY figure here was 1777 and drifted to 1805 as cards gained content. Six
			// steps of 400 still clear it, so the choice of eight was never wrong — but nothing
			// was watching the number that justifies the choice. The margin is asserted at the
			// end of this test rather than in a separate one, since this sweep already has the
			// numbers and a second sweep only repeats the scroll animation.
			for i := 0; i < 8; i++ {
				tr.Wheel(600, 400, 0, 400)
				tr.Advance(2e9)
			}
			tr.Frame()
			forward := a.Scroll.Y
			// The name of this test is "can be scrolled to its end", so that is what must be
			// asserted. It used to check only that the offset was non-negative, which passes
			// with the column stuck at the top — measured: every stage does reach MaxY exactly,
			// so the weaker assertion had never been asked a question it could fail.
			if forward != a.Scroll.MaxY {
				t.Fatalf("%dx%d step %d: wheeled to %.0f of a possible %.0f, so the end of the "+
					"stage is unreachable", size.w, size.h, step+1, forward, a.Scroll.MaxY)
			}
			if a.Scroll.MaxY > tallest {
				tallest = a.Scroll.MaxY
				tallestWhere = fmt.Sprintf("%dx%d step %d", size.w, size.h, step+1)
			}
			// Scrolling back up returns to the top, so the offset is a real
			// position and not a one-way ratchet.
			for i := 0; i < 8; i++ {
				tr.Wheel(600, 400, 0, -400)
				tr.Advance(2e9)
			}
			tr.Frame()
			if a.Scroll.Y != 0 {
				t.Fatalf("%dx%d step %d: scrolled back to %.0f, not the top",
					size.w, size.h, step+1, a.Scroll.Y)
			}
		}
	}
	// The eight wheels above have to clear the tallest stage with room to spare. Asserting the
	// margin rather than the MaxY figure keeps a card that gains a line from failing the test:
	// what matters is that the stage still fits inside the wheels used.
	if float32(8*perWheel) < tallest+perWheel/2 {
		t.Fatalf("the tallest stage (%s, MaxY %.0f) needs %.1f wheels of %d DIP; this test drives "+
			"8, so it would stop short", tallestWhere, tallest, float64(tallest)/perWheel, perWheel)
	}
	if float32(6*perWheel) < tallest {
		t.Fatalf("the comment says six 400-DIP steps reach the tallest stage, but %s needs %.0f",
			tallestWhere, tallest)
	}
	t.Logf("tallest stage: %s, MaxY %.0f", tallestWhere, tallest)
}

// TestMinimumWindowStillScrolls checks the tightest supported window: 1180x720 is
// where the layout is most likely to clip something outright.
func TestMinimumWindowStillScrolls(t *testing.T) {
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		tr.Move(600, 400)
		tr.Frame()
		tr.Wheel(600, 400, 0, 400)
		tr.Advance(2e9)
		tr.Frame()
		if a.Scroll.Y <= 0 {
			t.Fatalf("step %d: nothing scrolls at the minimum window size", step+1)
		}
	}
}

// TestCalloutsAreReachableAtEverySize checks the specific texts a reader is meant
// to read can be brought into the viewport. A callout below the fold is fine; one
// that cannot be reached is not.
func TestCalloutsAreReachableAtEverySize(t *testing.T) {
	want := map[int]string{
		2: "「负分就是把这个 key",
		3: "「每行和为 1，那每列",
		4: "「残差是加法，所以每",
		5: "「GELU 只是 ReLU",
	}
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
		for step, prefix := range want {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			if hasPrefix(tr, prefix) {
				continue // already visible
			}
			tr.Move(600, 400)
			tr.Frame()
			for i := 0; i < 12 && !hasPrefix(tr, prefix); i++ {
				tr.Wheel(600, 400, 0, 400)
				tr.Advance(2e9)
				tr.Frame()
			}
			if !hasPrefix(tr, prefix) {
				t.Fatalf("%dx%d step %d: %q cannot be scrolled into view",
					size.w, size.h, step+1, prefix)
			}
		}
	}
}

func hasPrefix(tr *ui.Tester, prefix string) bool {
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}
	return false
}

// TestNoHorizontalClipping covers the other half of "不溢出、不裁切". The other
// layout tests only check that vertical overflow can be scrolled to; content
// running off the right edge would be cut off with no scroll to find it, since
// only the token rail scrolls sideways.
//
// Limitation worth stating: I could not construct a break that this catches. A
// NoWrap text inside the fixed-width panels is clipped by the layout rather than
// reported past the edge, so the assertion holds trivially for the panels that
// exist today. It is kept because it costs one render pass per frame and states
// the constraint, not because it has been shown to fail. The rail's overflow is
// excluded deliberately and is covered by TestRailScrollsToItsLastToken.
func TestNoHorizontalClipping(t *testing.T) {
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
		for step := 0; step < 7; step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			for _, s := range tr.Texts() {
				r, ok := tr.Find(s)
				if !ok {
					continue
				}
				if r.X+r.W <= float32(size.w)+0.5 {
					continue
				}
				// The token rail scrolls horizontally on purpose: its chips run
				// past the window and TrackScroll(&a.RailScroll) brings them
				// back. Anything else past the edge would be cut off with no way
				// to reach it, since only the rail scrolls sideways.
				if r.Y < railBottomForTest {
					continue
				}
				label := s
				if n := len([]rune(label)); n > 24 {
					label = string([]rune(s)[:24]) + "…"
				}
				t.Fatalf("%dx%d step %d: %q ends at x=%.0f, past the %dpx window",
					size.w, size.h, step+1, label, r.X+r.W, size.w)
			}
		}
	}
}

// railBottomForTest is below the token strip and its readout: anything in the
// first ~110 DIPs scrolls horizontally by design.
const railBottomForTest = 115

// TestRailScrollsToItsLastToken covers the overflow that TestNoHorizontalClipping
// deliberately skips. The token strip is a horizontal scroller, so its chips
// legitimately run past the right edge; the promise there is that the last one is
// reachable, which a long prompt makes non-trivial.
func TestRailScrollsToItsLastToken(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("ab", 40) // 80 bytes, truncated to 64 tokens
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	n := len(a.Trace.Tokens)
	if n < 40 {
		t.Fatalf("only %d tokens; the prompt was meant to overflow the rail", n)
	}
	// The rail scrolls the selection into view, so a long prompt opens already
	// offset. Scroll back to the start, then forward again: both directions have
	// to work, and where it happened to open says nothing either way.
	//
	// The pointer has to sit on the strip itself. Wheel events are delivered by
	// hit test, and a pointer in the empty space below the chips reaches the
	// vertical scroller instead — which is what made my first attempt at this
	// look like a broken rail.
	r, ok := tr.Find("token") // the strip's fixed label, always present
	if !ok {
		t.Fatal("the token strip is not on screen")
	}
	y := r.Y + r.H/2
	tr.Move(600, y)
	tr.Frame()
	for i := 0; i < 12; i++ {
		tr.Scroll(600, y, -300, 0)
		tr.Frame()
	}
	tr.Frame()
	atStart := a.RailScroll.X
	for i := 0; i < 12; i++ {
		tr.Scroll(600, y, 300, 0)
		tr.Frame()
	}
	tr.Frame()
	if a.RailScroll.X <= atStart {
		t.Fatalf("the rail did not scroll horizontally for a %d-token prompt "+
			"(stayed at %.0f)", n, a.RailScroll.X)
	}
}

// TestWideWindowsUseTheirWidth covers 2560 and 3840, which no other test reaches.
//
// Every size list in this package stops at 1920x1080. The heatmap grew from a hard 420-DIP bound
// and only ever shrank from it, so it stayed 440 DIP wide at 1500, 2560 and 3840 alike — 13% of
// the stage at 4K — and nothing failed, because no test rendered a window that wide. Widening the
// matrix then exposed a truncation bug in the gutter hit test that the 1920 case happened to catch.
//
// Two properties are asserted here, both of which the ultrawide sizes are the first to exercise:
// the content reaches most of the available width rather than clustering on the left, and the
// heatmap keeps growing with the window until the cell cap stops it.
func TestWideWindowsUseTheirWidth(t *testing.T) {
	for _, size := range []struct{ w, h int }{{2560, 1440}, {3840, 2160}} {
		a := newTestApp(t)
		a.Prompt = "The transformer reads the tokens and"
		a.recompute()
		a.Step = 3
		a.SelToken = 20
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()

		// The stage's content should reach into the right-hand part of the window. The inspector
		// is a fixed column on the right, so "most of the width" means up to roughly where it
		// starts.
		right := float32(0)
		for _, s := range tr.Texts() {
			if r, ok := tr.Find(s); ok && r.X >= 240 && r.X+r.W > right {
				right = r.X + r.W
			}
		}
		wantRight := float32(size.w) * 0.95
		if right < wantRight {
			t.Fatalf("%dx%d: the stage's widest row ends at x=%.0f, only %.0f%% of the window — "+
				"the content is clustering on the left instead of using the width",
				size.w, size.h, right, 100*right/float32(size.w))
		}

		// And the matrix itself must be wider than the old fixed bound allowed.
		if r := a.lastHeatmapRect; r.W < 500 {
			t.Fatalf("%dx%d: the heatmap is only %.0f DIP wide; it used to cap at 440 whatever the "+
				"window did, and this test exists to keep it from capping again", size.w, size.h, r.W)
		}
	}
}
