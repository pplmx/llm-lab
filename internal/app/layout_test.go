package app

import (
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
			// was watching the number that justifies the choice. TestWheelCountClearsTheTallestStage
			// below checks the margin rather than the figure, since a card that grows past eight
			// wheels is the failure that matters.
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

// TestWheelCountClearsTheTallestStage checks the scroll margin rather than the MaxY figure.
//
// The comment above TestEveryStageCanBeScrolledToItsEnd justifies its eight wheels with a
// measurement: the tallest stage needs six 400-DIP steps. That figure was quoted as "stage 5,
// MaxY 1777" and had drifted to 1805 as cards gained content — the number was wrong while the
// conclusion it supported was still right, which is exactly why it went unnoticed.
//
// Asserting the margin instead of the number is deliberate. A specific MaxY would fail every time
// a card gains a line, which trains a reader to update the constant without asking whether the
// choice still holds. What matters is that the tallest stage fits inside the wheels used.
func TestWheelCountClearsTheTallestStage(t *testing.T) {
	const wheels, perWheel = 8, 400
	tallest, tallestStep := float32(0), 0
	// Only the smallest window: a taller one has a larger viewport and therefore a smaller
	// MaxY, so the margin is set here. Measuring both sizes doubled this test's time, most of
	// which is the scroll animation, for a comparison that cannot fail on the larger one.
	const w, h = 1180, 720
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, w, h)
		tr.Frame()
		tr.Move(600, 400)
		tr.Frame()
		for i := 0; i < wheels; i++ {
			tr.Wheel(600, 400, 0, perWheel)
			tr.Advance(2e9)
		}
		tr.Frame()
		if a.Scroll.MaxY > tallest {
			tallest, tallestStep = a.Scroll.MaxY, step+1
		}
	}
	if tallest == 0 {
		t.Fatal("no stage reported a scroll extent; the measurement is not reaching the stage area")
	}
	// Half a wheel of slack: the point is a comfortable margin, not an exact fit.
	if float32(wheels*perWheel) < tallest+perWheel/2 {
		t.Fatalf("the tallest stage (step %d, MaxY %.0f) needs %.1f wheels of %d DIP; the scroll "+
			"test drives %d, so it would stop short — raise the count there and update this margin",
			tallestStep, tallest, float64(tallest)/perWheel, perWheel, wheels)
	}
	// And the claim in that comment: six 400-DIP steps reach the tallest stage.
	if float32(6*perWheel) < tallest {
		t.Fatalf("the comment above TestEveryStageCanBeScrolledToItsEnd says six 400-DIP steps "+
			"reach the tallest stage, but step %d has MaxY %.0f", tallestStep, tallest)
	}
	t.Logf("tallest stage: step %d, MaxY %.0f, inside %d wheels of %d",
		tallestStep, tallest, wheels, perWheel)
}
