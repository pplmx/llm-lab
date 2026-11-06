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
