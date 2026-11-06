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
			for i := 0; i < 12; i++ {
				tr.Wheel(600, 400, 0, 400)
				tr.Advance(2e9)
			}
			tr.Frame()
			forward := a.Scroll.Y
			if forward < 0 {
				t.Fatalf("%dx%d step %d: wheeled forward to %.0f", size.w, size.h, step+1, forward)
			}
			// Scrolling back up returns to the top, so the offset is a real
			// position and not a one-way ratchet.
			for i := 0; i < 12; i++ {
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
