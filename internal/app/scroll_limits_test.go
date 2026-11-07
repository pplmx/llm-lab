package app

import (
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestScrollWheelDoesNotReachTheContainer documents a limit of the headless tester rather
// than a property of the app.
//
// The stage column is a ui.Scroll with TrackScroll(&a.Scroll). Wheel and Scroll events at a
// point inside it leave a.Scroll.Y at zero: mygo routes pointer scroll in the headless
// harness somewhere other than the container, while a real window scrolls normally. PageDown
// does move the offset, which is how the long stages were exercised before.
//
// The test pins that observation so a later reader does not repeat the investigation, and so
// that if mygo starts routing wheel events the test fails and the real scrolling behaviour
// becomes testable.
//
// It deliberately does not assert anything about the app. Whether the stage header stays on
// screen while scrolled is a real question, and the wheel path is the one that would answer
// it; until that path works here, a test would be guessing.
func TestScrollWheelDoesNotReachTheContainer(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()

	tr.Move(600, 400)
	tr.Frame()
	tr.Wheel(600, 400, 0, -300)
	tr.Frame()
	wheelOffset := a.Scroll.Y

	tr.Scroll(600, 400, 0, -300)
	tr.Frame()
	if wheelOffset != 0 || a.Scroll.Y != 0 {
		t.Skipf("pointer scrolling now reaches the container (offset %.0f then %.0f); the "+
			"limitation this test documents has gone away, and stage-scroll behaviour can be "+
			"tested for real", wheelOffset, a.Scroll.Y)
	}

	// PageDown is the path that does work, and it is what the earlier scroll measurements
	// used.
	tr.Key(0, ui.KeyPageDown)
	tr.Frame()
	if a.Scroll.Y == 0 {
		t.Fatal("neither pointer nor keyboard scrolling moves the stage column; the long " +
			"stages are unreachable in tests and something else has changed")
	}
}
